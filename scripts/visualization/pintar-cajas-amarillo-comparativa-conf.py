# -*- coding: utf-8 -*-

"""
Generación de evidencias visuales con cajas amarillas para comparar
las inferencias YOLO realizadas con distintos umbrales de confianza.

El script:

1. Utiliza los PNG originales sin anotaciones.
2. Lee los archivos TXT generados por YOLO.
3. Convierte las cajas normalizadas a coordenadas de píxel.
4. Dibuja cajas amarillas sin clase, confianza ni texto.
5. Guarda también los recortes sin detecciones.
6. Genera un conteo por recorte y un resumen global.
7. Comprueba que el número de cajas coincide con las detecciones
   registradas durante la inferencia.
8. Admite rutas de Windows con caracteres Unicode, como la letra ñ.
"""

from pathlib import Path
import csv
import os
import shutil

import cv2
import numpy as np


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

def _env_path(name, default):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default

IMAGENES_DIR = _env_path(
    "TFG_PNOA_PNG_DIR",
    REPO_ROOT / "data" / "external" / "pnoa_tiles" / "png",
)

COMPARATIVA_DIR = _env_path(
    "TFG_INFERENCE_DIR",
    REPO_ROOT / "results" / "inference",
)

OUTPUT_ROOT = REPO_ROOT / "outputs" / "boxes"

EXPERIMENTOS = [
    "conf010",
    "conf015",
    "conf020",
    "conf025",
]

RESUMEN_INFERENCIA_CSV = COMPARATIVA_DIR / "resumen_inferencias_conf.csv"
RESUMEN_CAJAS_CSV = OUTPUT_ROOT / "resumen_cajas_amarillas.csv"

# OpenCV utiliza el orden BGR.
COLOR_AMARILLO = (0, 255, 255)

# Grosor de las cajas.
GROSOR_CAJA = 2

# Calidad de salida JPG.
CALIDAD_JPG = 95

# Se guardan también los recortes que no tienen detecciones.
GUARDAR_IMAGENES_SIN_DETECCIONES = True

# Elimina la carpeta de cajas amarillas anterior antes de regenerarla.
# No elimina labels ni resultados YOLO.
LIMPIAR_SALIDA_ANTERIOR = True


# ============================================================
# LECTURA Y ESCRITURA DE IMÁGENES CON RUTAS UNICODE
# ============================================================

def leer_imagen_unicode(imagen_path: Path):
    """
    Lee una imagen con OpenCV admitiendo rutas de Windows que contienen
    caracteres Unicode, como la letra ñ.
    """

    if not imagen_path.exists():
        raise FileNotFoundError(
            "No existe la imagen:\n"
            f"{imagen_path}"
        )

    try:
        datos = np.fromfile(
            str(imagen_path),
            dtype=np.uint8,
        )

    except OSError as exc:
        raise RuntimeError(
            "No se pudo leer el archivo de imagen:\n"
            f"{imagen_path}\n"
            f"Error del sistema: {exc}"
        ) from exc

    if datos.size == 0:
        raise RuntimeError(
            "El archivo está vacío o no pudo leerse:\n"
            f"{imagen_path}"
        )

    imagen = cv2.imdecode(
        datos,
        cv2.IMREAD_COLOR,
    )

    if imagen is None:
        raise RuntimeError(
            "OpenCV no pudo decodificar la imagen:\n"
            f"{imagen_path}"
        )

    return imagen


def guardar_imagen_unicode(
    salida_path: Path,
    imagen,
    calidad_jpg: int = 95,
):
    """
    Guarda una imagen admitiendo rutas de Windows que contienen
    caracteres Unicode.
    """

    salida_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = salida_path.suffix.lower()

    if extension in (".jpg", ".jpeg"):
        extension_codificacion = ".jpg"

        parametros = [
            cv2.IMWRITE_JPEG_QUALITY,
            calidad_jpg,
        ]

    elif extension == ".png":
        extension_codificacion = ".png"

        parametros = [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ]

    else:
        raise ValueError(
            "Formato de salida no admitido: "
            f"{salida_path.suffix}"
        )

    correcto, datos_codificados = cv2.imencode(
        extension_codificacion,
        imagen,
        parametros,
    )

    if not correcto:
        raise RuntimeError(
            "OpenCV no pudo codificar la imagen:\n"
            f"{salida_path}"
        )

    try:
        datos_codificados.tofile(
            str(salida_path)
        )

    except OSError as exc:
        raise RuntimeError(
            "No se pudo guardar la imagen:\n"
            f"{salida_path}\n"
            f"Error del sistema: {exc}"
        ) from exc


# ============================================================
# FUNCIONES DE LECTURA DE DATOS
# ============================================================

def cargar_conteos_esperados():
    """
    Lee el resumen de las inferencias y obtiene el número esperado
    de detecciones para cada experimento.

    Resultado:

        {
            "conf010": número,
            "conf015": número,
            "conf020": número,
            "conf025": número,
        }
    """

    esperados = {}

    if not RESUMEN_INFERENCIA_CSV.exists():
        print("")
        print("[AVISO] No existe el resumen de inferencias:")
        print(f"        {RESUMEN_INFERENCIA_CSV}")
        print(
            "[AVISO] Se generarán las cajas, pero no será posible "
            "comprobar automáticamente los totales."
        )

        return esperados

    with open(
        RESUMEN_INFERENCIA_CSV,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            nombre = str(
                row.get("nombre_experimento", "")
            ).strip()

            total_texto = str(
                row.get("detecciones_totales", "")
            ).strip()

            if not nombre or not total_texto:
                continue

            try:
                esperados[nombre] = int(
                    float(total_texto)
                )

            except ValueError:
                print(
                    "[AVISO] No se pudo interpretar el número "
                    f"de detecciones de {nombre}: {total_texto}"
                )

    return esperados


def crear_indice_imagenes():
    """
    Relaciona el nombre base de cada recorte con su PNG original.
    """

    imagenes = {}

    for imagen_path in sorted(IMAGENES_DIR.iterdir()):
        if not imagen_path.is_file():
            continue

        if imagen_path.suffix.lower() != ".png":
            continue

        imagenes[imagen_path.stem] = imagen_path

    return imagenes


def leer_detecciones_yolo(label_path: Path):
    """
    Lee un archivo de labels YOLO.

    Formatos admitidos:

        clase x_centro y_centro ancho alto

    o:

        clase x_centro y_centro ancho alto confianza

    La confianza se ignora porque no debe aparecer en la imagen.
    """

    detecciones = []
    lineas_malformadas = 0

    if not label_path.exists():
        return detecciones, lineas_malformadas

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        for numero_linea, linea in enumerate(
            f,
            start=1,
        ):
            linea = linea.strip()

            if not linea:
                continue

            valores = linea.split()

            if len(valores) < 5:
                lineas_malformadas += 1

                print(
                    "[AVISO] Línea incompleta en "
                    f"{label_path.name}, línea {numero_linea}"
                )

                continue

            try:
                clase = int(float(valores[0]))
                x_centro = float(valores[1])
                y_centro = float(valores[2])
                ancho = float(valores[3])
                alto = float(valores[4])

            except ValueError:
                lineas_malformadas += 1

                print(
                    "[AVISO] Valores no numéricos en "
                    f"{label_path.name}, línea {numero_linea}"
                )

                continue

            detecciones.append(
                {
                    "clase": clase,
                    "x_centro": x_centro,
                    "y_centro": y_centro,
                    "ancho": ancho,
                    "alto": alto,
                }
            )

    return detecciones, lineas_malformadas


# ============================================================
# CONVERSIÓN Y DIBUJO DE CAJAS
# ============================================================

def convertir_caja_a_pixeles(
    deteccion,
    ancho_img,
    alto_img,
):
    """
    Convierte una caja YOLO normalizada a píxeles.

    Entrada YOLO:

        x_centro, y_centro, ancho, alto

    Salida:

        x_min, y_min, x_max, y_max
    """

    x_centro_px = (
        deteccion["x_centro"]
        * ancho_img
    )

    y_centro_px = (
        deteccion["y_centro"]
        * alto_img
    )

    ancho_px = (
        deteccion["ancho"]
        * ancho_img
    )

    alto_px = (
        deteccion["alto"]
        * alto_img
    )

    x_min = int(
        round(
            x_centro_px
            - ancho_px / 2.0
        )
    )

    y_min = int(
        round(
            y_centro_px
            - alto_px / 2.0
        )
    )

    x_max = int(
        round(
            x_centro_px
            + ancho_px / 2.0
        )
    )

    y_max = int(
        round(
            y_centro_px
            + alto_px / 2.0
        )
    )

    # Ajustar la caja a los límites reales de la imagen.
    x_min = max(
        0,
        min(x_min, ancho_img - 1),
    )

    y_min = max(
        0,
        min(y_min, alto_img - 1),
    )

    x_max = max(
        0,
        min(x_max, ancho_img - 1),
    )

    y_max = max(
        0,
        min(y_max, alto_img - 1),
    )

    return x_min, y_min, x_max, y_max


def dibujar_detecciones(
    imagen,
    detecciones,
):
    """
    Dibuja cajas amarillas sin texto, clase ni confianza.
    """

    alto_img, ancho_img = imagen.shape[:2]

    cajas_dibujadas = 0

    for deteccion in detecciones:
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = convertir_caja_a_pixeles(
            deteccion,
            ancho_img,
            alto_img,
        )

        # Evitar cajas degeneradas o sin superficie.
        if x_max <= x_min or y_max <= y_min:
            continue

        cv2.rectangle(
            imagen,
            (x_min, y_min),
            (x_max, y_max),
            COLOR_AMARILLO,
            GROSOR_CAJA,
            lineType=cv2.LINE_AA,
        )

        cajas_dibujadas += 1

    return cajas_dibujadas


# ============================================================
# ESCRITURA DE CSV
# ============================================================

def escribir_csv_por_recorte(
    csv_path: Path,
    filas,
):
    campos = [
        "experimento",
        "recorte",
        "imagen_origen",
        "label",
        "tiene_label",
        "detecciones_leidas",
        "cajas_dibujadas",
        "lineas_malformadas",
        "imagen_salida",
    ]

    with open(
        csv_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=campos,
        )

        writer.writeheader()
        writer.writerows(filas)


def escribir_resumen_global(filas):
    campos = [
        "experimento",
        "imagenes_origen",
        "imagenes_guardadas",
        "imagenes_con_detecciones",
        "imagenes_sin_detecciones",
        "archivos_labels",
        "detecciones_leidas",
        "cajas_dibujadas",
        "lineas_malformadas",
        "detecciones_esperadas_inferencia",
        "diferencia_con_inferencia",
        "comprobacion",
        "ruta_salida",
    ]

    with open(
        RESUMEN_CAJAS_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=campos,
        )

        writer.writeheader()
        writer.writerows(filas)


# ============================================================
# PROCESAMIENTO DE UN EXPERIMENTO
# ============================================================

def procesar_experimento(
    experimento,
    indice_imagenes,
    detecciones_esperadas,
):
    experimento_dir = (
        COMPARATIVA_DIR
        / experimento
    )

    labels_dir = (
        experimento_dir
        / "labels"
    )

    salida_dir = (
        OUTPUT_ROOT
        / experimento
        / "cajas_amarillas_sin_labels"
    )

    csv_por_recorte = (
        OUTPUT_ROOT
        / experimento
        / f"conteo_cajas_por_recorte_{experimento}.csv"
    )

    print("")
    print("=" * 75)
    print(f"[INFO] Procesando experimento: {experimento}")
    print(f"[INFO] Labels: {labels_dir}")
    print(f"[INFO] Salida: {salida_dir}")
    print("=" * 75)

    if not experimento_dir.exists():
        raise FileNotFoundError(
            "No existe la carpeta del experimento:\n"
            f"{experimento_dir}\n\n"
            "Ejecuta primero el script de inferencia homogénea."
        )

    if not labels_dir.exists():
        raise FileNotFoundError(
            "No existe la carpeta de labels:\n"
            f"{labels_dir}"
        )

    if (
        LIMPIAR_SALIDA_ANTERIOR
        and salida_dir.exists()
    ):
        print(
            "[INFO] Eliminando únicamente la salida "
            "anterior de cajas amarillas..."
        )

        shutil.rmtree(
            salida_dir
        )

    salida_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filas_recortes = []

    imagenes_guardadas = 0
    imagenes_con_detecciones = 0
    imagenes_sin_detecciones = 0

    total_detecciones_leidas = 0
    total_cajas_dibujadas = 0
    total_lineas_malformadas = 0

    for recorte, imagen_path in sorted(
        indice_imagenes.items()
    ):
        label_path = (
            labels_dir
            / f"{recorte}.txt"
        )

        (
            detecciones,
            lineas_malformadas,
        ) = leer_detecciones_yolo(
            label_path
        )

        numero_detecciones = len(
            detecciones
        )

        if numero_detecciones > 0:
            imagenes_con_detecciones += 1

        else:
            imagenes_sin_detecciones += 1

        total_detecciones_leidas += (
            numero_detecciones
        )

        total_lineas_malformadas += (
            lineas_malformadas
        )

        if (
            numero_detecciones == 0
            and not GUARDAR_IMAGENES_SIN_DETECCIONES
        ):
            filas_recortes.append(
                {
                    "experimento": experimento,
                    "recorte": recorte,
                    "imagen_origen": str(imagen_path),
                    "label": (
                        str(label_path)
                        if label_path.exists()
                        else ""
                    ),
                    "tiene_label": int(
                        label_path.exists()
                    ),
                    "detecciones_leidas": 0,
                    "cajas_dibujadas": 0,
                    "lineas_malformadas": (
                        lineas_malformadas
                    ),
                    "imagen_salida": "",
                }
            )

            continue

        # Lectura Unicode segura.
        imagen = leer_imagen_unicode(
            imagen_path
        )

        cajas_dibujadas = dibujar_detecciones(
            imagen,
            detecciones,
        )

        salida_path = (
            salida_dir
            / f"{recorte}.jpg"
        )

        # Escritura Unicode segura.
        guardar_imagen_unicode(
            salida_path=salida_path,
            imagen=imagen,
            calidad_jpg=CALIDAD_JPG,
        )

        imagenes_guardadas += 1

        total_cajas_dibujadas += (
            cajas_dibujadas
        )

        filas_recortes.append(
            {
                "experimento": experimento,
                "recorte": recorte,
                "imagen_origen": str(imagen_path),
                "label": (
                    str(label_path)
                    if label_path.exists()
                    else ""
                ),
                "tiene_label": int(
                    label_path.exists()
                ),
                "detecciones_leidas": (
                    numero_detecciones
                ),
                "cajas_dibujadas": (
                    cajas_dibujadas
                ),
                "lineas_malformadas": (
                    lineas_malformadas
                ),
                "imagen_salida": str(
                    salida_path
                ),
            }
        )

        print(
            f"[OK] {salida_path.name} "
            f"-> {cajas_dibujadas} cajas"
        )

    escribir_csv_por_recorte(
        csv_por_recorte,
        filas_recortes,
    )

    archivos_labels = len(
        list(
            labels_dir.glob("*.txt")
        )
    )

    esperado = detecciones_esperadas.get(
        experimento
    )

    if esperado is None:
        diferencia = ""
        comprobacion = "SIN_RESUMEN_DE_REFERENCIA"

    else:
        diferencia = (
            total_cajas_dibujadas
            - esperado
        )

        if diferencia == 0:
            comprobacion = "CORRECTO"

        else:
            comprobacion = "REVISAR"

    print("")
    print(
        f"[RESUMEN] Experimento:             "
        f"{experimento}"
    )
    print(
        f"[RESUMEN] Imágenes origen:          "
        f"{len(indice_imagenes)}"
    )
    print(
        f"[RESUMEN] Imágenes guardadas:       "
        f"{imagenes_guardadas}"
    )
    print(
        f"[RESUMEN] Imágenes con detecciones: "
        f"{imagenes_con_detecciones}"
    )
    print(
        f"[RESUMEN] Imágenes sin detecciones: "
        f"{imagenes_sin_detecciones}"
    )
    print(
        f"[RESUMEN] Archivos labels:          "
        f"{archivos_labels}"
    )
    print(
        f"[RESUMEN] Detecciones leídas:       "
        f"{total_detecciones_leidas}"
    )
    print(
        f"[RESUMEN] Cajas dibujadas:          "
        f"{total_cajas_dibujadas}"
    )
    print(
        f"[RESUMEN] Líneas malformadas:       "
        f"{total_lineas_malformadas}"
    )
    print(
        f"[RESUMEN] Comprobación:             "
        f"{comprobacion}"
    )
    print(
        f"[OK] CSV por recorte: "
        f"{csv_por_recorte}"
    )

    return {
        "experimento": experimento,
        "imagenes_origen": len(indice_imagenes),
        "imagenes_guardadas": imagenes_guardadas,
        "imagenes_con_detecciones": (
            imagenes_con_detecciones
        ),
        "imagenes_sin_detecciones": (
            imagenes_sin_detecciones
        ),
        "archivos_labels": archivos_labels,
        "detecciones_leidas": (
            total_detecciones_leidas
        ),
        "cajas_dibujadas": (
            total_cajas_dibujadas
        ),
        "lineas_malformadas": (
            total_lineas_malformadas
        ),
        "detecciones_esperadas_inferencia": (
            esperado
            if esperado is not None
            else ""
        ),
        "diferencia_con_inferencia": diferencia,
        "comprobacion": comprobacion,
        "ruta_salida": str(salida_dir),
    }


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 75)
    print("CAJAS AMARILLAS - COMPARATIVA DE UMBRALES")
    print("=" * 75)

    if not IMAGENES_DIR.exists():
        raise FileNotFoundError(
            "No existe la carpeta de imágenes limpias:\n"
            f"{IMAGENES_DIR}"
        )

    if not COMPARATIVA_DIR.exists():
        raise FileNotFoundError(
            "No existe la carpeta de inferencias homogéneas:\n"
            f"{COMPARATIVA_DIR}\n\n"
            "Ejecuta primero el script de inferencia."
        )

    indice_imagenes = crear_indice_imagenes()

    if not indice_imagenes:
        raise RuntimeError(
            "No se encontraron imágenes PNG en:\n"
            f"{IMAGENES_DIR}"
        )

    print(
        f"[INFO] Imágenes limpias encontradas: "
        f"{len(indice_imagenes)}"
    )
    print(
        f"[INFO] Experimentos: "
        f"{', '.join(EXPERIMENTOS)}"
    )
    print("[INFO] Color: amarillo")
    print(
        f"[INFO] Grosor: "
        f"{GROSOR_CAJA} píxeles"
    )
    print(
        "[INFO] Sin labels, clases ni porcentajes"
    )

    detecciones_esperadas = (
        cargar_conteos_esperados()
    )

    resumen_global = []

    for experimento in EXPERIMENTOS:
        resultado = procesar_experimento(
            experimento,
            indice_imagenes,
            detecciones_esperadas,
        )

        resumen_global.append(
            resultado
        )

        # Se guarda después de cada experimento.
        escribir_resumen_global(
            resumen_global
        )

    print("")
    print("=" * 75)
    print("[FIN] PROCESO COMPLETADO")
    print(
        f"[OK] Resumen global: "
        f"{RESUMEN_CAJAS_CSV}"
    )
    print("=" * 75)

    problemas = [
        fila
        for fila in resumen_global
        if fila["comprobacion"] == "REVISAR"
    ]

    if problemas:
        print("")
        print("[ADVERTENCIA]")
        print(
            "Algún experimento no coincide con el "
            "conteo registrado durante la inferencia."
        )

        for fila in problemas:
            print(
                f"  {fila['experimento']}: "
                f"diferencia = "
                f"{fila['diferencia_con_inferencia']}"
            )

    else:
        print(
            "[OK] Todos los conteos disponibles "
            "coinciden con las inferencias."
        )


if __name__ == "__main__":
    main()