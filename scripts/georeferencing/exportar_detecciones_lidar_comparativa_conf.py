# -*- coding: utf-8 -*-

"""
Georreferenciación homogénea de detecciones YOLO y extracción de
altura LiDAR para distintos umbrales de confianza.

Experimentos procesados:
    conf010
    conf015
    conf020
    conf025

Para cada experimento:

1. Lee los labels YOLO.
2. Localiza el GeoTIFF equivalente de cada recorte.
3. Convierte las coordenadas normalizadas a píxeles.
4. Convierte los píxeles a coordenadas UTM EPSG:25830.
5. Extrae estadísticas de altura del raster LiDAR.
6. Filtra alturas válidas entre 2 y 40 m.
7. Deduplica detecciones mediante distancia entre centros de 1,5 m.
8. Genera CSV de detecciones brutas, detecciones únicas y resumen.

El raster LiDAR completo se carga una sola vez en memoria para acelerar
el procesamiento.
"""

from collections import defaultdict
from pathlib import Path
import csv
import math
import os
import statistics

import numpy as np
from osgeo import gdal, osr


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

gdal.UseExceptions()

REPO_ROOT = Path(__file__).resolve().parents[2]

def _env_path(name, default):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default

# Por defecto se utilizan los labels congelados publicados en results/inference.
# Para encadenar una inferencia recién generada puede definirse TFG_INFERENCE_DIR.
COMPARATIVA_DIR = _env_path(
    "TFG_INFERENCE_DIR",
    REPO_ROOT / "results" / "inference",
)

GEOTIFF_DIR = _env_path(
    "TFG_GEOTIFF_DIR",
    REPO_ROOT / "data" / "external" / "pnoa_tiles" / "geotiff",
)

LIDAR_DIR = _env_path(
    "TFG_LIDAR_DIR",
    REPO_ROOT / "data" / "external" / "lidar",
)

OUT_ROOT = REPO_ROOT / "outputs" / "georeferencing"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RESUMEN_GLOBAL_CSV = OUT_ROOT / "resumen_georreferenciacion_comparativa_conf.csv"

EXPERIMENTOS = {
    "conf010": 0.10,
    "conf015": 0.15,
    "conf020": 0.20,
    "conf025": 0.25,
}

# Filtro de alturas.
ALTURA_MIN_M = 2.0
ALTURA_MAX_M = 40.0

# Deduplicación por distancia entre centros.
RADIO_DEDUPLICACION_M = 1.5


# ============================================================
# CAMPOS DE LOS CSV
# ============================================================

CAMPOS_CSV = [
    "experimento",
    "umbral_confianza",
    "id_deteccion",
    "id_unica",
    "recorte_origen",
    "archivo_label",
    "archivo_geotiff",
    "clase",
    "confianza",
    "ancho_imagen_px",
    "alto_imagen_px",
    "x_centro_normalizado",
    "y_centro_normalizado",
    "ancho_normalizado",
    "alto_normalizado",
    "x_centro_px",
    "y_centro_px",
    "x_min_px",
    "y_min_px",
    "x_max_px",
    "y_max_px",
    "x_centro_utm",
    "y_centro_utm",
    "x_min_utm",
    "y_min_utm",
    "x_max_utm",
    "y_max_utm",
    "ancho_caja_m",
    "alto_caja_m",
    "altura_media_m",
    "altura_max_m",
    "altura_p95_m",
    "numero_pixeles_lidar_validos",
    "altura_valida",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def buscar_raster_altura():
    """
    Localiza el raster de altura LiDAR.
    """

    candidatos = sorted(
        LIDAR_DIR.glob("Altura_*Biomasa.tif")
    )

    if not candidatos:
        raise FileNotFoundError(
            "No se encontró el raster Altura_*Biomasa.tif en:\n"
            f"{LIDAR_DIR}"
        )

    if len(candidatos) > 1:
        print(
            "[AVISO] Se encontraron varios raster de altura. "
            "Se utilizará el primero:"
        )

        for candidato in candidatos:
            print(f"        {candidato}")

    return candidatos[0]


def obtener_epsg(proyeccion_wkt):
    """
    Trata de recuperar el código EPSG de una proyección WKT.
    """

    if not proyeccion_wkt:
        return None

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proyeccion_wkt)

    try:
        srs.AutoIdentifyEPSG()
    except RuntimeError:
        pass

    codigo = srs.GetAuthorityCode(None)

    if codigo is None:
        codigo = srs.GetAuthorityCode(
            "PROJCS"
        )

    return codigo


def construir_indice_geotiff():
    """
    Carga los metadatos de todos los GeoTIFF una única vez.

    El índice se relaciona mediante el nombre base del archivo.
    """

    indice = {}

    archivos = []

    for extension in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        archivos.extend(
            GEOTIFF_DIR.glob(extension)
        )

    archivos = sorted(
        set(archivos)
    )

    if not archivos:
        raise FileNotFoundError(
            "No se encontraron recortes GeoTIFF en:\n"
            f"{GEOTIFF_DIR}"
        )

    for tif_path in archivos:
        ds = gdal.Open(
            str(tif_path),
            gdal.GA_ReadOnly,
        )

        if ds is None:
            raise RuntimeError(
                "No se pudo abrir el GeoTIFF:\n"
                f"{tif_path}"
            )

        indice[tif_path.stem] = {
            "path": tif_path,
            "ancho": ds.RasterXSize,
            "alto": ds.RasterYSize,
            "geotransformacion": ds.GetGeoTransform(),
            "proyeccion": ds.GetProjection(),
        }

        ds = None

    return indice


def cargar_raster_altura(altura_path):
    """
    Carga el raster LiDAR completo en memoria.
    """

    ds = gdal.Open(
        str(altura_path),
        gdal.GA_ReadOnly,
    )

    if ds is None:
        raise RuntimeError(
            "No se pudo abrir el raster LiDAR:\n"
            f"{altura_path}"
        )

    band = ds.GetRasterBand(1)

    arr = band.ReadAsArray()

    if arr is None:
        raise RuntimeError(
            "No se pudo leer el raster LiDAR."
        )

    arr = arr.astype(
        np.float32,
        copy=False,
    )

    geotransformacion = ds.GetGeoTransform()
    proyeccion = ds.GetProjection()
    nodata = band.GetNoDataValue()

    inversa = gdal.InvGeoTransform(
        geotransformacion
    )

    if inversa is None:
        raise RuntimeError(
            "No se pudo invertir la geotransformación LiDAR."
        )

    resultado = {
        "path": altura_path,
        "array": arr,
        "ancho": ds.RasterXSize,
        "alto": ds.RasterYSize,
        "geotransformacion": geotransformacion,
        "geotransformacion_inversa": inversa,
        "proyeccion": proyeccion,
        "nodata": nodata,
    }

    ds = None

    return resultado


def comprobar_sistemas_referencia(
    indice_geotiff,
    lidar,
):
    """
    Comprueba que los GeoTIFF y el raster LiDAR usan el mismo CRS.
    """

    primer_recorte = next(
        iter(indice_geotiff.values())
    )

    srs_tif = osr.SpatialReference()
    srs_lidar = osr.SpatialReference()

    srs_tif.ImportFromWkt(
        primer_recorte["proyeccion"]
    )

    srs_lidar.ImportFromWkt(
        lidar["proyeccion"]
    )

    epsg_tif = obtener_epsg(
        primer_recorte["proyeccion"]
    )

    epsg_lidar = obtener_epsg(
        lidar["proyeccion"]
    )

    print(
        f"[INFO] EPSG recortes PNOA: {epsg_tif}"
    )

    print(
        f"[INFO] EPSG raster LiDAR:   {epsg_lidar}"
    )

    if not bool(srs_tif.IsSame(srs_lidar)):
        raise RuntimeError(
            "Los recortes PNOA y el raster LiDAR no utilizan "
            "el mismo sistema de referencia.\n"
            f"EPSG PNOA: {epsg_tif}\n"
            f"EPSG LiDAR: {epsg_lidar}"
        )


def aplicar_geotransformacion(
    gt,
    x_px,
    y_px,
):
    """
    Convierte una coordenada de píxel a coordenada geográfica.
    """

    x_geo = (
        gt[0]
        + x_px * gt[1]
        + y_px * gt[2]
    )

    y_geo = (
        gt[3]
        + x_px * gt[4]
        + y_px * gt[5]
    )

    return x_geo, y_geo


def convertir_bbox_a_utm(
    gt,
    x_min_px,
    y_min_px,
    x_max_px,
    y_max_px,
):
    """
    Convierte las cuatro esquinas de una caja a coordenadas UTM.
    """

    esquinas = [
        aplicar_geotransformacion(
            gt,
            x_min_px,
            y_min_px,
        ),
        aplicar_geotransformacion(
            gt,
            x_max_px,
            y_min_px,
        ),
        aplicar_geotransformacion(
            gt,
            x_min_px,
            y_max_px,
        ),
        aplicar_geotransformacion(
            gt,
            x_max_px,
            y_max_px,
        ),
    ]

    xs = [
        punto[0]
        for punto in esquinas
    ]

    ys = [
        punto[1]
        for punto in esquinas
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def leer_label_yolo(
    label_path,
    confianza_defecto,
):
    """
    Lee las detecciones de un archivo YOLO.

    Admite:

        clase x y ancho alto

    o:

        clase x y ancho alto confianza
    """

    detecciones = []
    lineas_malformadas = 0

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

            partes = linea.split()

            if len(partes) < 5:
                lineas_malformadas += 1

                print(
                    "[AVISO] Línea incompleta en "
                    f"{label_path.name}, línea {numero_linea}"
                )

                continue

            try:
                clase = int(
                    float(partes[0])
                )

                x_centro = float(partes[1])
                y_centro = float(partes[2])
                ancho = float(partes[3])
                alto = float(partes[4])

                if len(partes) >= 6:
                    confianza = float(partes[5])
                else:
                    confianza = confianza_defecto

            except ValueError:
                lineas_malformadas += 1

                print(
                    "[AVISO] Valor no numérico en "
                    f"{label_path.name}, línea {numero_linea}"
                )

                continue

            detecciones.append(
                {
                    "clase": clase,
                    "x_centro_normalizado": x_centro,
                    "y_centro_normalizado": y_centro,
                    "ancho_normalizado": ancho,
                    "alto_normalizado": alto,
                    "confianza": confianza,
                }
            )

    return detecciones, lineas_malformadas


def extraer_estadisticas_altura(
    lidar,
    x_min_utm,
    y_min_utm,
    x_max_utm,
    y_max_utm,
):
    """
    Extrae estadísticas LiDAR dentro de la caja UTM.
    """

    inv_gt = lidar[
        "geotransformacion_inversa"
    ]

    esquinas_px = [
        gdal.ApplyGeoTransform(
            inv_gt,
            x_min_utm,
            y_min_utm,
        ),
        gdal.ApplyGeoTransform(
            inv_gt,
            x_min_utm,
            y_max_utm,
        ),
        gdal.ApplyGeoTransform(
            inv_gt,
            x_max_utm,
            y_min_utm,
        ),
        gdal.ApplyGeoTransform(
            inv_gt,
            x_max_utm,
            y_max_utm,
        ),
    ]

    columnas = [
        punto[0]
        for punto in esquinas_px
    ]

    filas = [
        punto[1]
        for punto in esquinas_px
    ]

    col_min = int(
        math.floor(min(columnas))
    )

    col_max = int(
        math.ceil(max(columnas))
    )

    fila_min = int(
        math.floor(min(filas))
    )

    fila_max = int(
        math.ceil(max(filas))
    )

    col_min = max(
        0,
        min(col_min, lidar["ancho"] - 1),
    )

    col_max = max(
        col_min + 1,
        min(col_max, lidar["ancho"]),
    )

    fila_min = max(
        0,
        min(fila_min, lidar["alto"] - 1),
    )

    fila_max = max(
        fila_min + 1,
        min(fila_max, lidar["alto"]),
    )

    ventana = lidar["array"][
        fila_min:fila_max,
        col_min:col_max,
    ]

    if ventana.size == 0:
        return {
            "altura_media_m": "",
            "altura_max_m": "",
            "altura_p95_m": "",
            "numero_pixeles_lidar_validos": 0,
            "altura_valida": 0,
        }

    mascara = np.isfinite(
        ventana
    )

    nodata = lidar["nodata"]

    if nodata is not None:
        mascara &= (
            ventana != nodata
        )

    mascara &= (
        ventana >= ALTURA_MIN_M
    )

    mascara &= (
        ventana <= ALTURA_MAX_M
    )

    valores = ventana[mascara]

    if valores.size == 0:
        return {
            "altura_media_m": "",
            "altura_max_m": "",
            "altura_p95_m": "",
            "numero_pixeles_lidar_validos": 0,
            "altura_valida": 0,
        }

    return {
        "altura_media_m": float(
            np.mean(valores)
        ),
        "altura_max_m": float(
            np.max(valores)
        ),
        "altura_p95_m": float(
            np.percentile(valores, 95)
        ),
        "numero_pixeles_lidar_validos": int(
            valores.size
        ),
        "altura_valida": 1,
    }


def deduplicar_por_distancia(
    detecciones,
    radio_m,
):
    """
    Deduplicación espacial mediante un índice de cuadrícula.

    Las detecciones se ordenan priorizando:

    1. Altura LiDAR válida.
    2. Mayor confianza.

    Una detección se elimina cuando su centro queda a una distancia
    menor o igual al radio respecto a una detección ya conservada.
    """

    if not detecciones:
        return []

    ordenadas = sorted(
        detecciones,
        key=lambda fila: (
            int(fila["altura_valida"]),
            float(fila["confianza"]),
        ),
        reverse=True,
    )

    tamano_celda = radio_m

    rejilla = defaultdict(list)
    conservadas = []

    radio_cuadrado = (
        radio_m * radio_m
    )

    for deteccion in ordenadas:
        x = float(
            deteccion["x_centro_utm"]
        )

        y = float(
            deteccion["y_centro_utm"]
        )

        celda_x = int(
            math.floor(x / tamano_celda)
        )

        celda_y = int(
            math.floor(y / tamano_celda)
        )

        duplicada = False

        for dx in (-1, 0, 1):
            if duplicada:
                break

            for dy in (-1, 0, 1):
                vecinos = rejilla.get(
                    (
                        celda_x + dx,
                        celda_y + dy,
                    ),
                    [],
                )

                for indice_conservada in vecinos:
                    anterior = conservadas[
                        indice_conservada
                    ]

                    diferencia_x = (
                        x
                        - float(
                            anterior["x_centro_utm"]
                        )
                    )

                    diferencia_y = (
                        y
                        - float(
                            anterior["y_centro_utm"]
                        )
                    )

                    distancia_cuadrado = (
                        diferencia_x * diferencia_x
                        + diferencia_y * diferencia_y
                    )

                    if (
                        distancia_cuadrado
                        <= radio_cuadrado
                    ):
                        duplicada = True
                        break

                if duplicada:
                    break

        if duplicada:
            continue

        nueva = deteccion.copy()

        nueva["id_unica"] = (
            f"{deteccion['experimento']}_"
            f"{len(conservadas) + 1:07d}"
        )

        indice_nuevo = len(
            conservadas
        )

        conservadas.append(
            nueva
        )

        rejilla[
            (
                celda_x,
                celda_y,
            )
        ].append(
            indice_nuevo
        )

    return conservadas


def escribir_csv(
    path,
    filas,
):
    """
    Escribe un CSV en UTF-8 con BOM para facilitar su apertura en Excel.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CAMPOS_CSV,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(filas)


def estadisticas_lista(
    valores,
):
    """
    Devuelve estadísticas básicas de una lista numérica.
    """

    if not valores:
        return {
            "n": 0,
            "media": None,
            "minimo": None,
            "maximo": None,
            "mediana": None,
        }

    return {
        "n": len(valores),
        "media": statistics.mean(valores),
        "minimo": min(valores),
        "maximo": max(valores),
        "mediana": statistics.median(valores),
    }


# ============================================================
# PROCESAMIENTO DE UN EXPERIMENTO
# ============================================================

def procesar_experimento(
    experimento,
    umbral,
    indice_geotiff,
    lidar,
):
    labels_dir = (
        COMPARATIVA_DIR
        / experimento
        / "labels"
    )

    out_dir = (
        OUT_ROOT
        / experimento
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_brutas = (
        out_dir
        / f"detecciones_brutas_{experimento}.csv"
    )

    csv_unicas = (
        out_dir
        / f"detecciones_unicas_{experimento}.csv"
    )

    resumen_txt = (
        out_dir
        / f"resumen_{experimento}.txt"
    )

    print("")
    print("=" * 75)
    print(f"[INFO] Experimento: {experimento}")
    print(f"[INFO] Umbral: {umbral:.2f}")
    print(f"[INFO] Labels: {labels_dir}")
    print(f"[INFO] Salida: {out_dir}")
    print("=" * 75)

    if not labels_dir.exists():
        raise FileNotFoundError(
            "No existe la carpeta de labels:\n"
            f"{labels_dir}"
        )

    labels = sorted(
        labels_dir.glob("*.txt")
    )

    detecciones_brutas = []

    labels_sin_geotiff = []
    lineas_malformadas_total = 0

    contador_deteccion = 0

    for label_path in labels:
        recorte = label_path.stem

        metadatos = indice_geotiff.get(
            recorte
        )

        if metadatos is None:
            labels_sin_geotiff.append(
                recorte
            )

            print(
                "[AVISO] No se encontró GeoTIFF para:"
                f" {recorte}"
            )

            continue

        detecciones_label, malformadas = (
            leer_label_yolo(
                label_path,
                confianza_defecto=umbral,
            )
        )

        lineas_malformadas_total += (
            malformadas
        )

        ancho_img = metadatos["ancho"]
        alto_img = metadatos["alto"]
        gt = metadatos[
            "geotransformacion"
        ]

        for deteccion in detecciones_label:
            contador_deteccion += 1

            x_centro_px = (
                deteccion["x_centro_normalizado"]
                * ancho_img
            )

            y_centro_px = (
                deteccion["y_centro_normalizado"]
                * alto_img
            )

            ancho_px = (
                deteccion["ancho_normalizado"]
                * ancho_img
            )

            alto_px = (
                deteccion["alto_normalizado"]
                * alto_img
            )

            x_min_px = max(
                0.0,
                x_centro_px - ancho_px / 2.0,
            )

            y_min_px = max(
                0.0,
                y_centro_px - alto_px / 2.0,
            )

            x_max_px = min(
                float(ancho_img),
                x_centro_px + ancho_px / 2.0,
            )

            y_max_px = min(
                float(alto_img),
                y_centro_px + alto_px / 2.0,
            )

            (
                x_centro_utm,
                y_centro_utm,
            ) = aplicar_geotransformacion(
                gt,
                x_centro_px,
                y_centro_px,
            )

            (
                x_min_utm,
                y_min_utm,
                x_max_utm,
                y_max_utm,
            ) = convertir_bbox_a_utm(
                gt,
                x_min_px,
                y_min_px,
                x_max_px,
                y_max_px,
            )

            ancho_caja_m = abs(
                x_max_utm - x_min_utm
            )

            alto_caja_m = abs(
                y_max_utm - y_min_utm
            )

            alturas = extraer_estadisticas_altura(
                lidar,
                x_min_utm,
                y_min_utm,
                x_max_utm,
                y_max_utm,
            )

            fila = {
                "experimento": experimento,
                "umbral_confianza": umbral,
                "id_deteccion": (
                    f"{experimento}_raw_"
                    f"{contador_deteccion:07d}"
                ),
                "id_unica": "",
                "recorte_origen": recorte,
                "archivo_label": str(label_path),
                "archivo_geotiff": str(
                    metadatos["path"]
                ),
                "clase": deteccion["clase"],
                "confianza": deteccion[
                    "confianza"
                ],
                "ancho_imagen_px": ancho_img,
                "alto_imagen_px": alto_img,
                "x_centro_normalizado": deteccion[
                    "x_centro_normalizado"
                ],
                "y_centro_normalizado": deteccion[
                    "y_centro_normalizado"
                ],
                "ancho_normalizado": deteccion[
                    "ancho_normalizado"
                ],
                "alto_normalizado": deteccion[
                    "alto_normalizado"
                ],
                "x_centro_px": x_centro_px,
                "y_centro_px": y_centro_px,
                "x_min_px": x_min_px,
                "y_min_px": y_min_px,
                "x_max_px": x_max_px,
                "y_max_px": y_max_px,
                "x_centro_utm": x_centro_utm,
                "y_centro_utm": y_centro_utm,
                "x_min_utm": x_min_utm,
                "y_min_utm": y_min_utm,
                "x_max_utm": x_max_utm,
                "y_max_utm": y_max_utm,
                "ancho_caja_m": ancho_caja_m,
                "alto_caja_m": alto_caja_m,
                **alturas,
            }

            detecciones_brutas.append(
                fila
            )

    print(
        f"[INFO] Detecciones brutas: "
        f"{len(detecciones_brutas)}"
    )

    print(
        "[INFO] Aplicando deduplicación espacial "
        f"de {RADIO_DEDUPLICACION_M:.1f} m..."
    )

    detecciones_unicas = (
        deduplicar_por_distancia(
            detecciones_brutas,
            RADIO_DEDUPLICACION_M,
        )
    )

    escribir_csv(
        csv_brutas,
        detecciones_brutas,
    )

    escribir_csv(
        csv_unicas,
        detecciones_unicas,
    )

    unicas_validas = [
        fila
        for fila in detecciones_unicas
        if int(fila["altura_valida"]) == 1
    ]

    alturas_p95 = [
        float(fila["altura_p95_m"])
        for fila in unicas_validas
        if fila["altura_p95_m"] != ""
    ]

    alturas_max = [
        float(fila["altura_max_m"])
        for fila in unicas_validas
        if fila["altura_max_m"] != ""
    ]

    estadisticas_p95 = estadisticas_lista(
        alturas_p95
    )

    estadisticas_max = estadisticas_lista(
        alturas_max
    )

    eliminadas = (
        len(detecciones_brutas)
        - len(detecciones_unicas)
    )

    porcentaje_eliminado = (
        eliminadas
        / len(detecciones_brutas)
        * 100.0
        if detecciones_brutas
        else 0.0
    )

    with open(
        resumen_txt,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"RESUMEN {experimento}\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Umbral de confianza: {umbral:.2f}\n"
        )

        f.write(
            f"Carpeta de labels: {labels_dir}\n"
        )

        f.write(
            f"Raster PNOA GeoTIFF: {GEOTIFF_DIR}\n"
        )

        f.write(
            f"Raster LiDAR: {lidar['path']}\n"
        )

        f.write(
            "Radio deduplicacion: "
            f"{RADIO_DEDUPLICACION_M:.1f} m\n"
        )

        f.write(
            "Altura minima valida: "
            f"{ALTURA_MIN_M:.1f} m\n"
        )

        f.write(
            "Altura maxima valida: "
            f"{ALTURA_MAX_M:.1f} m\n\n"
        )

        f.write(
            f"Archivos label procesados: {len(labels)}\n"
        )

        f.write(
            "Labels sin GeoTIFF equivalente: "
            f"{len(labels_sin_geotiff)}\n"
        )

        f.write(
            "Lineas malformadas descartadas: "
            f"{lineas_malformadas_total}\n"
        )

        f.write(
            f"Detecciones brutas: "
            f"{len(detecciones_brutas)}\n"
        )

        f.write(
            "Detecciones unicas tras deduplicacion: "
            f"{len(detecciones_unicas)}\n"
        )

        f.write(
            "Detecciones eliminadas por deduplicacion: "
            f"{eliminadas} "
            f"({porcentaje_eliminado:.3f} %)\n"
        )

        f.write(
            "Detecciones unicas con altura valida: "
            f"{len(unicas_validas)}\n\n"
        )

        f.write(
            "Estadisticas de altura usando altura_p95_m:\n"
        )

        if estadisticas_p95["n"] > 0:
            f.write(
                "  Altura media: "
                f"{estadisticas_p95['media']:.3f} m\n"
            )

            f.write(
                "  Altura minima: "
                f"{estadisticas_p95['minimo']:.3f} m\n"
            )

            f.write(
                "  Altura maxima: "
                f"{estadisticas_p95['maximo']:.3f} m\n"
            )

            f.write(
                "  Mediana: "
                f"{estadisticas_p95['mediana']:.3f} m\n"
            )

        else:
            f.write(
                "  Sin alturas validas.\n"
            )

        f.write(
            "\nEstadisticas de altura usando altura_max_m:\n"
        )

        if estadisticas_max["n"] > 0:
            f.write(
                "  Altura media maxima: "
                f"{estadisticas_max['media']:.3f} m\n"
            )

            f.write(
                "  Altura maxima absoluta: "
                f"{estadisticas_max['maximo']:.3f} m\n"
            )

        else:
            f.write(
                "  Sin alturas validas.\n"
            )

        if labels_sin_geotiff:
            f.write(
                "\nLabels sin GeoTIFF:\n"
            )

            for nombre in labels_sin_geotiff:
                f.write(
                    f"  {nombre}\n"
                )

    print(f"[OK] CSV brutas: {csv_brutas}")
    print(f"[OK] CSV únicas: {csv_unicas}")
    print(f"[OK] Resumen:    {resumen_txt}")

    print(
        f"[RESUMEN] Brutas:       "
        f"{len(detecciones_brutas)}"
    )

    print(
        f"[RESUMEN] Únicas:       "
        f"{len(detecciones_unicas)}"
    )

    print(
        f"[RESUMEN] Altura válida:"
        f" {len(unicas_validas)}"
    )

    return {
        "experimento": experimento,
        "umbral_confianza": umbral,
        "archivos_labels": len(labels),
        "labels_sin_geotiff": len(
            labels_sin_geotiff
        ),
        "lineas_malformadas": (
            lineas_malformadas_total
        ),
        "detecciones_brutas": len(
            detecciones_brutas
        ),
        "detecciones_unicas": len(
            detecciones_unicas
        ),
        "detecciones_eliminadas": eliminadas,
        "porcentaje_eliminado": (
            porcentaje_eliminado
        ),
        "detecciones_altura_valida": len(
            unicas_validas
        ),
        "altura_p95_media_m": (
            estadisticas_p95["media"]
            if estadisticas_p95["n"] > 0
            else ""
        ),
        "altura_p95_mediana_m": (
            estadisticas_p95["mediana"]
            if estadisticas_p95["n"] > 0
            else ""
        ),
        "altura_p95_min_m": (
            estadisticas_p95["minimo"]
            if estadisticas_p95["n"] > 0
            else ""
        ),
        "altura_p95_max_m": (
            estadisticas_p95["maximo"]
            if estadisticas_p95["n"] > 0
            else ""
        ),
        "ruta_csv_unicas": str(
            csv_unicas
        ),
        "ruta_resumen": str(
            resumen_txt
        ),
    }


def escribir_resumen_global(
    resultados,
):
    campos = [
        "experimento",
        "umbral_confianza",
        "archivos_labels",
        "labels_sin_geotiff",
        "lineas_malformadas",
        "detecciones_brutas",
        "detecciones_unicas",
        "detecciones_eliminadas",
        "porcentaje_eliminado",
        "detecciones_altura_valida",
        "altura_p95_media_m",
        "altura_p95_mediana_m",
        "altura_p95_min_m",
        "altura_p95_max_m",
        "ruta_csv_unicas",
        "ruta_resumen",
    ]

    with open(
        RESUMEN_GLOBAL_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=campos,
        )

        writer.writeheader()
        writer.writerows(
            resultados
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 75)
    print("GEORREFERENCIACIÓN YOLO + ALTURA LIDAR")
    print("COMPARATIVA HOMOGÉNEA DE UMBRALES")
    print("=" * 75)

    if not COMPARATIVA_DIR.exists():
        raise FileNotFoundError(
            "No existe la carpeta de inferencias:\n"
            f"{COMPARATIVA_DIR}"
        )

    if not GEOTIFF_DIR.exists():
        raise FileNotFoundError(
            "No existe la carpeta de GeoTIFF:\n"
            f"{GEOTIFF_DIR}"
        )

    altura_path = buscar_raster_altura()

    print(
        f"[INFO] Raster LiDAR: {altura_path}"
    )

    print(
        "[INFO] Cargando GeoTIFF y raster LiDAR..."
    )

    indice_geotiff = (
        construir_indice_geotiff()
    )

    lidar = cargar_raster_altura(
        altura_path
    )

    print(
        f"[INFO] GeoTIFF indexados: "
        f"{len(indice_geotiff)}"
    )

    print(
        f"[INFO] Raster LiDAR: "
        f"{lidar['ancho']} × {lidar['alto']}"
    )

    comprobar_sistemas_referencia(
        indice_geotiff,
        lidar,
    )

    resultados = []

    for experimento, umbral in EXPERIMENTOS.items():
        resultado = procesar_experimento(
            experimento,
            umbral,
            indice_geotiff,
            lidar,
        )

        resultados.append(
            resultado
        )

        # Guardado progresivo.
        escribir_resumen_global(
            resultados
        )

    print("")
    print("=" * 75)
    print("[FIN] PROCESO COMPLETADO")
    print(
        f"[OK] Resumen global:\n"
        f"     {RESUMEN_GLOBAL_CSV}"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()