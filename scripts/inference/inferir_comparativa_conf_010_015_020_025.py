# -*- coding: utf-8 -*-

"""
Ejecuta una comparación controlada del mismo modelo YOLO utilizando
cuatro umbrales de confianza: 0.10, 0.15, 0.20 y 0.25.

Todos los demás parámetros permanecen constantes para que las
diferencias observadas se deban únicamente al umbral de confianza.
"""

from pathlib import Path
import csv
import os
import time

from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = REPO_ROOT / "models" / "best.pt"

SOURCE_DIR = (
    REPO_ROOT
    / "data"
    / "external"
    / "pnoa_tiles"
    / "png"
)

PROJECT_DIR = REPO_ROOT / "outputs" / "inference"
SUMMARY_CSV = PROJECT_DIR / "resumen_inferencias_conf.csv"

CONFIDENCIAS = [0.10, 0.15, 0.20, 0.25]

IMGSZ = 1280
MAX_DET = 3000
IOU_NMS = 0.70
DEVICE = 0


# ============================================================
# FUNCIONES
# ============================================================

def contar_detecciones(labels_dir: Path):
    """
    Cuenta:
    - archivos TXT generados;
    - detecciones totales;
    - máximo de detecciones en un recorte;
    - recortes que alcanzan MAX_DET.
    """

    txt_files = sorted(labels_dir.glob("*.txt"))

    total_detecciones = 0
    max_detecciones_recorte = 0
    nombre_recorte_maximo = ""
    recortes_saturados = []

    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8") as f:
            numero_detecciones = sum(
                1 for linea in f if linea.strip()
            )

        total_detecciones += numero_detecciones

        if numero_detecciones > max_detecciones_recorte:
            max_detecciones_recorte = numero_detecciones
            nombre_recorte_maximo = txt_path.stem

        if numero_detecciones >= MAX_DET:
            recortes_saturados.append(txt_path.stem)

    return {
        "archivos_labels": len(txt_files),
        "detecciones_totales": total_detecciones,
        "max_detecciones_recorte": max_detecciones_recorte,
        "recorte_maximo": nombre_recorte_maximo,
        "numero_recortes_saturados": len(recortes_saturados),
        "recortes_saturados": ";".join(recortes_saturados),
    }


def escribir_resumen(resultados):
    campos = [
        "confianza",
        "nombre_experimento",
        "imagenes_procesadas",
        "archivos_labels",
        "imagenes_sin_detecciones",
        "detecciones_totales",
        "max_detecciones_recorte",
        "recorte_maximo",
        "max_det_configurado",
        "numero_recortes_saturados",
        "recortes_saturados",
        "tiempo_inferencia_s",
        "detecciones_por_segundo",
        "ruta_resultados",
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 75)
    print("COMPARACIÓN HOMOGÉNEA DE UMBRALES YOLO")
    print("=" * 75)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{MODEL_PATH}"
        )

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta de imágenes:\n{SOURCE_DIR}"
        )

    imagenes = sorted(SOURCE_DIR.glob("*.png"))
    numero_imagenes = len(imagenes)

    if numero_imagenes == 0:
        raise RuntimeError(
            f"No se encontraron PNG en:\n{SOURCE_DIR}"
        )

    print(f"[INFO] Modelo: {MODEL_PATH}")
    print(f"[INFO] Origen: {SOURCE_DIR}")
    print(f"[INFO] Imágenes encontradas: {numero_imagenes}")
    print(f"[INFO] imgsz: {IMGSZ}")
    print(f"[INFO] max_det: {MAX_DET}")
    print(f"[INFO] IoU NMS: {IOU_NMS}")
    print(f"[INFO] Dispositivo: CUDA:{DEVICE}")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    resultados = []

    for conf in CONFIDENCIAS:
        etiqueta = f"conf{int(round(conf * 100)):03d}"
        output_dir = PROJECT_DIR / etiqueta

        print("")
        print("-" * 75)
        print(f"[INFO] Iniciando experimento {etiqueta}")
        print(f"[INFO] Confianza: {conf:.2f}")
        print(f"[INFO] Salida: {output_dir}")
        print("-" * 75)

        if output_dir.exists():
            raise FileExistsError(
                "\nLa carpeta de salida ya existe:\n"
                f"{output_dir}\n\n"
                "Elimínala o cambia el nombre antes de repetir "
                "la comparación para no mezclar resultados."
            )

        inicio = time.perf_counter()

        resultados_stream = model.predict(
            source=str(SOURCE_DIR),
            conf=conf,
            iou=IOU_NMS,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            device=DEVICE,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(PROJECT_DIR),
            name=etiqueta,
            exist_ok=True,
            stream=True,
            verbose=False,
        )

        # Es necesario recorrer el generador para ejecutar toda la inferencia.
        for _ in resultados_stream:
            pass

        tiempo_s = time.perf_counter() - inicio

        labels_dir = output_dir / "labels"
        conteo = contar_detecciones(labels_dir)

        imagenes_sin_detecciones = (
            numero_imagenes - conteo["archivos_labels"]
        )

        detecciones_por_segundo = (
            conteo["detecciones_totales"] / tiempo_s
            if tiempo_s > 0
            else 0.0
        )

        fila = {
            "confianza": conf,
            "nombre_experimento": etiqueta,
            "imagenes_procesadas": numero_imagenes,
            "archivos_labels": conteo["archivos_labels"],
            "imagenes_sin_detecciones": imagenes_sin_detecciones,
            "detecciones_totales": conteo["detecciones_totales"],
            "max_detecciones_recorte": (
                conteo["max_detecciones_recorte"]
            ),
            "recorte_maximo": conteo["recorte_maximo"],
            "max_det_configurado": MAX_DET,
            "numero_recortes_saturados": (
                conteo["numero_recortes_saturados"]
            ),
            "recortes_saturados": conteo["recortes_saturados"],
            "tiempo_inferencia_s": round(tiempo_s, 3),
            "detecciones_por_segundo": round(
                detecciones_por_segundo, 3
            ),
            "ruta_resultados": str(output_dir),
        }

        resultados.append(fila)
        escribir_resumen(resultados)

        print(f"[OK] Umbral terminado: {conf:.2f}")
        print(
            f"[OK] Detecciones: "
            f"{conteo['detecciones_totales']}"
        )
        print(
            f"[OK] Labels: "
            f"{conteo['archivos_labels']} / {numero_imagenes}"
        )
        print(
            f"[OK] Máximo en un recorte: "
            f"{conteo['max_detecciones_recorte']}"
        )
        print(
            f"[OK] Recorte más denso: "
            f"{conteo['recorte_maximo']}"
        )
        print(
            f"[OK] Recortes que alcanzaron max_det: "
            f"{conteo['numero_recortes_saturados']}"
        )
        print(f"[OK] Tiempo: {tiempo_s:.2f} s")

    print("")
    print("=" * 75)
    print("[FIN] COMPARACIÓN COMPLETADA")
    print(f"[OK] Resumen CSV: {SUMMARY_CSV}")
    print("=" * 75)

    saturados_totales = sum(
        int(r["numero_recortes_saturados"])
        for r in resultados
    )

    if saturados_totales > 0:
        print("")
        print("[ADVERTENCIA]")
        print(
            "Algún recorte alcanzó max_det. En ese caso habría "
            "que repetir los experimentos afectados con un valor "
            "superior para evitar truncar las detecciones."
        )


if __name__ == "__main__":
    main()