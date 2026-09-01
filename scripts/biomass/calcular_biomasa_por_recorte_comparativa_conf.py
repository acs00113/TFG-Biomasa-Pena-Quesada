# -*- coding: utf-8 -*-

"""
Calcula biomasa global y por recorte para una comparativa homogénea
con varios umbrales de confianza YOLO.

El análisis por recorte NO utiliza directamente toda la extensión de
cada tesela original, porque las teselas se solapan 192 píxeles. En su
lugar, se construyen 180 zonas núcleo no solapadas. Cada detección
única se asigna a una sola zona según su centro UTM.

Experimentos:
    conf010
    conf015
    conf020
    conf025

Para cada experimento se generan:
    - detecciones individuales con biomasa y recorte asignado;
    - tabla con 180 recortes no solapados;
    - resumen global;
    - comparación conjunta de los cuatro umbrales.
"""

from bisect import bisect_right
from pathlib import Path
import csv
import math
import os
import statistics

import numpy as np
from osgeo import gdal


gdal.UseExceptions()


# ============================================================
# CONFIGURACIÓN
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

def _env_path(name, default):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default

DETECCIONES_ROOT = _env_path(
    "TFG_GEOREF_DIR",
    REPO_ROOT / "outputs" / "georeferencing",
)

GEOTIFF_DIR = _env_path(
    "TFG_GEOTIFF_DIR",
    REPO_ROOT / "data" / "external" / "pnoa_tiles" / "geotiff",
)

LIDAR_DIR = _env_path(
    "TFG_LIDAR_DIR",
    REPO_ROOT / "data" / "external" / "lidar",
)

OUT_ROOT = REPO_ROOT / "outputs" / "biomass"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPERIMENTOS = {
    "conf010": 0.10,
    "conf015": 0.15,
    "conf020": 0.20,
    "conf025": 0.25,
}

ALTURA_MIN_M = 2.0
ALTURA_MAX_M = 40.0

DIAMETRO_COPA_MIN_M = 0.4
DIAMETRO_COPA_MAX_M = 20.0

FACTOR_CORRECCION_GLOBAL = 1.0

CLASES_ALTURA = [
    ("2_5", 2.0, 5.0),
    ("5_10", 5.0, 10.0),
    ("10_15", 10.0, 15.0),
    ("15_20", 15.0, 20.0),
    ("20_30", 20.0, 30.0),
    ("30_40", 30.0, 40.0),
]

RESUMEN_GLOBAL_CSV = OUT_ROOT / "resumen_global_comparativa_biomasa.csv"
TABLA_LARGA_CSV = OUT_ROOT / "biomasa_por_recorte_todos_largo.csv"
TABLA_COMPARACION_CSV = OUT_ROOT / "comparacion_biomasa_por_recorte.csv"
RESUMEN_TXT = OUT_ROOT / "resumen_comparativa_biomasa.txt"


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def safe_float(valor):
    if valor is None:
        return None

    texto = str(valor).strip().replace(",", ".")

    if texto == "":
        return None

    try:
        return float(texto)
    except ValueError:
        return None


def safe_int(valor, defecto=0):
    numero = safe_float(valor)

    if numero is None:
        return defecto

    return int(round(numero))


def media(lista):
    return statistics.mean(lista) if lista else ""


def mediana(lista):
    return statistics.median(lista) if lista else ""


def minimo(lista):
    return min(lista) if lista else ""


def maximo(lista):
    return max(lista) if lista else ""


def buscar_raster_altura():
    candidatos = sorted(LIDAR_DIR.glob("Altura_*Biomasa.tif"))

    if not candidatos:
        raise FileNotFoundError(
            "No se encontró Altura_*Biomasa.tif en:\n"
            f"{LIDAR_DIR}"
        )

    return candidatos[0]


def aplicar_gt(gt, x_px, y_px):
    x_geo = gt[0] + x_px * gt[1] + y_px * gt[2]
    y_geo = gt[3] + x_px * gt[4] + y_px * gt[5]
    return x_geo, y_geo


def escribir_csv(path, filas, campos):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=campos,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(filas)


# ============================================================
# ECUACIÓN DE BIOMASA UTILIZADA EN EL FLUJO ACTUAL
# ============================================================

def agb_altura_copa(altura_m, diametro_copa_m, grupo):
    """
    Estimación exploratoria de biomasa aérea individual en kg.

    grupo:
        1 -> gimnosperma/conífera
        2 -> angiosperma/frondosa

    Mantiene exactamente la formulación utilizada en el script global
    anterior para que los resultados sean comparables.
    """

    if grupo == 1:
        a = 0.093
        b = -0.223
    elif grupo == 2:
        a = 0.0
        b = 0.0
    else:
        raise ValueError("grupo debe ser 1 o 2")

    biomasa_kg = (
        (0.016 + a)
        * ((altura_m * diametro_copa_m) ** (2.013 + b))
        * math.exp((0.204 ** 2) / 2.0)
    )

    return biomasa_kg * FACTOR_CORRECCION_GLOBAL


# ============================================================
# CONSTRUCCIÓN DE LAS 180 ZONAS NÚCLEO NO SOLAPADAS
# ============================================================

def leer_teselas_geotiff():
    archivos = []

    for patron in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        archivos.extend(GEOTIFF_DIR.glob(patron))

    archivos = sorted(set(archivos))

    if not archivos:
        raise FileNotFoundError(
            "No se encontraron GeoTIFF en:\n"
            f"{GEOTIFF_DIR}"
        )

    teselas = []

    for tif_path in archivos:
        ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)

        if ds is None:
            raise RuntimeError(f"No se pudo abrir:\n{tif_path}")

        gt = ds.GetGeoTransform()
        ancho = ds.RasterXSize
        alto = ds.RasterYSize

        esquinas = [
            aplicar_gt(gt, 0, 0),
            aplicar_gt(gt, ancho, 0),
            aplicar_gt(gt, 0, alto),
            aplicar_gt(gt, ancho, alto),
        ]

        xs = [p[0] for p in esquinas]
        ys = [p[1] for p in esquinas]

        teselas.append(
            {
                "recorte": tif_path.stem,
                "path": tif_path,
                "xmin": min(xs),
                "xmax": max(xs),
                "ymin": min(ys),
                "ymax": max(ys),
                "ancho_px": ancho,
                "alto_px": alto,
            }
        )

        ds = None

    return teselas


def construir_zonas_nucleo(teselas):
    """
    Divide la extensión completa en 180 zonas no solapadas.

    La frontera entre dos teselas vecinas se sitúa en el punto medio
    de su franja de solape. Así, la mitad del solape se asigna a cada
    recorte y la suma de todas las zonas cubre una única vez el área.
    """

    columnas_xmin = sorted({round(t["xmin"], 6) for t in teselas})
    filas_ymax = sorted(
        {round(t["ymax"], 6) for t in teselas},
        reverse=True,
    )

    columnas = []

    for xmin_clave in columnas_xmin:
        grupo = [
            t for t in teselas
            if round(t["xmin"], 6) == xmin_clave
        ]
        columnas.append(
            {
                "xmin_tesela": min(t["xmin"] for t in grupo),
                "xmax_tesela": max(t["xmax"] for t in grupo),
            }
        )

    filas = []

    for ymax_clave in filas_ymax:
        grupo = [
            t for t in teselas
            if round(t["ymax"], 6) == ymax_clave
        ]
        filas.append(
            {
                "ymin_tesela": min(t["ymin"] for t in grupo),
                "ymax_tesela": max(t["ymax"] for t in grupo),
            }
        )

    x_total_min = min(t["xmin"] for t in teselas)
    x_total_max = max(t["xmax"] for t in teselas)
    y_total_min = min(t["ymin"] for t in teselas)
    y_total_max = max(t["ymax"] for t in teselas)

    x_bordes = [x_total_min]

    for i in range(len(columnas) - 1):
        limite = (
            columnas[i]["xmax_tesela"]
            + columnas[i + 1]["xmin_tesela"]
        ) / 2.0
        x_bordes.append(limite)

    x_bordes.append(x_total_max)

    # Bordes de norte a sur.
    y_bordes_desc = [y_total_max]

    for i in range(len(filas) - 1):
        limite = (
            filas[i]["ymin_tesela"]
            + filas[i + 1]["ymax_tesela"]
        ) / 2.0
        y_bordes_desc.append(limite)

    y_bordes_desc.append(y_total_min)

    indice_columna = {
        round(col["xmin_tesela"], 6): i
        for i, col in enumerate(columnas)
    }

    indice_fila = {
        round(fila["ymax_tesela"], 6): i
        for i, fila in enumerate(filas)
    }

    zonas = []
    mapa_indices = {}

    for tesela in teselas:
        col = indice_columna[round(tesela["xmin"], 6)]
        fila = indice_fila[round(tesela["ymax"], 6)]

        zona = {
            "recorte": tesela["recorte"],
            "fila": fila + 1,
            "columna": col + 1,
            "xmin": x_bordes[col],
            "xmax": x_bordes[col + 1],
            "ymax": y_bordes_desc[fila],
            "ymin": y_bordes_desc[fila + 1],
            "geotiff_origen": str(tesela["path"]),
        }

        zona["x_centro"] = (zona["xmin"] + zona["xmax"]) / 2.0
        zona["y_centro"] = (zona["ymin"] + zona["ymax"]) / 2.0
        zona["area_total_ha_geometrica"] = (
            (zona["xmax"] - zona["xmin"])
            * (zona["ymax"] - zona["ymin"])
            / 10000.0
        )

        zonas.append(zona)
        mapa_indices[(fila, col)] = zona["recorte"]

    zonas.sort(key=lambda z: (z["fila"], z["columna"]))

    if len(zonas) != len(teselas):
        raise RuntimeError(
            "El número de zonas no coincide con el número de teselas."
        )

    return {
        "zonas": zonas,
        "x_bordes": x_bordes,
        "y_bordes_desc": y_bordes_desc,
        "mapa_indices": mapa_indices,
        "numero_filas": len(filas),
        "numero_columnas": len(columnas),
    }


def localizar_zona(x, y, particion):
    x_bordes = particion["x_bordes"]
    y_bordes = particion["y_bordes_desc"]

    col = bisect_right(x_bordes, x) - 1
    col = max(0, min(col, particion["numero_columnas"] - 1))

    fila_encontrada = None

    for fila in range(particion["numero_filas"]):
        y_superior = y_bordes[fila]
        y_inferior = y_bordes[fila + 1]

        es_ultima = fila == particion["numero_filas"] - 1

        if y <= y_superior and (y > y_inferior or es_ultima and y >= y_inferior):
            fila_encontrada = fila
            break

    if fila_encontrada is None:
        raise RuntimeError(
            "No se pudo asignar la coordenada a una zona:\n"
            f"X={x}, Y={y}"
        )

    recorte = particion["mapa_indices"].get((fila_encontrada, col))

    if recorte is None:
        raise RuntimeError(
            "No existe zona para los índices "
            f"fila={fila_encontrada}, columna={col}"
        )

    return recorte


# ============================================================
# ESTADÍSTICAS LIDAR POR ZONA
# ============================================================

def cargar_lidar(altura_path):
    ds = gdal.Open(str(altura_path), gdal.GA_ReadOnly)

    if ds is None:
        raise RuntimeError(f"No se pudo abrir:\n{altura_path}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    if arr is None:
        raise RuntimeError("No se pudo leer el raster LiDAR.")

    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)

    if inv_gt is None:
        raise RuntimeError("No se pudo invertir la geotransformación LiDAR.")

    resultado = {
        "path": altura_path,
        "array": arr.astype(np.float32, copy=False),
        "ancho": ds.RasterXSize,
        "alto": ds.RasterYSize,
        "gt": gt,
        "inv_gt": inv_gt,
        "nodata": band.GetNoDataValue(),
        "pixel_area_m2": abs(gt[1] * gt[5] - gt[2] * gt[4]),
    }

    ds = None
    return resultado


def indice_casi_entero(valor):
    redondeado = round(valor)

    if abs(valor - redondeado) < 1e-6:
        return int(redondeado)

    return None


def ventana_lidar_para_zona(zona, lidar):
    inv_gt = lidar["inv_gt"]

    p_superior_izquierda = gdal.ApplyGeoTransform(
        inv_gt,
        zona["xmin"],
        zona["ymax"],
    )

    p_inferior_derecha = gdal.ApplyGeoTransform(
        inv_gt,
        zona["xmax"],
        zona["ymin"],
    )

    c0f, r0f = p_superior_izquierda
    c1f, r1f = p_inferior_derecha

    c0 = indice_casi_entero(c0f)
    r0 = indice_casi_entero(r0f)
    c1 = indice_casi_entero(c1f)
    r1 = indice_casi_entero(r1f)

    if None in (c0, r0, c1, r1):
        # Caso general si alguna frontera no coincide exactamente con píxel.
        c0 = int(math.floor(min(c0f, c1f)))
        c1 = int(math.ceil(max(c0f, c1f)))
        r0 = int(math.floor(min(r0f, r1f)))
        r1 = int(math.ceil(max(r0f, r1f)))
    else:
        c0, c1 = sorted((c0, c1))
        r0, r1 = sorted((r0, r1))

    c0 = max(0, min(c0, lidar["ancho"]))
    c1 = max(c0, min(c1, lidar["ancho"]))
    r0 = max(0, min(r0, lidar["alto"]))
    r1 = max(r0, min(r1, lidar["alto"]))

    return r0, r1, c0, c1


def calcular_lidar_por_zona(zonas, lidar):
    total_pixeles = 0

    for zona in zonas:
        r0, r1, c0, c1 = ventana_lidar_para_zona(zona, lidar)

        ventana = lidar["array"][r0:r1, c0:c1]

        total_pixeles += int(ventana.size)

        valida = np.isfinite(ventana)

        if lidar["nodata"] is not None:
            valida &= ventana != lidar["nodata"]

        arbolada = (
            valida
            & (ventana >= ALTURA_MIN_M)
            & (ventana <= ALTURA_MAX_M)
        )

        pixeles_arbolados = int(np.sum(arbolada))
        pixel_area = lidar["pixel_area_m2"]

        zona["pixeles_lidar_total"] = int(ventana.size)
        zona["area_total_ha"] = ventana.size * pixel_area / 10000.0
        zona["pixeles_lidar_arbolados"] = pixeles_arbolados
        zona["area_arbolada_ha"] = pixeles_arbolados * pixel_area / 10000.0
        zona["porcentaje_arbolado"] = (
            zona["area_arbolada_ha"] / zona["area_total_ha"] * 100.0
            if zona["area_total_ha"] > 0
            else 0.0
        )

        for nombre, h_min, h_max in CLASES_ALTURA:
            if h_max == ALTURA_MAX_M:
                mascara_clase = valida & (ventana >= h_min) & (ventana <= h_max)
            else:
                mascara_clase = valida & (ventana >= h_min) & (ventana < h_max)

            pixeles_clase = int(np.sum(mascara_clase))
            zona[f"area_altura_{nombre}_ha"] = (
                pixeles_clase * pixel_area / 10000.0
            )

    esperado = lidar["ancho"] * lidar["alto"]

    if total_pixeles != esperado:
        raise RuntimeError(
            "Las zonas núcleo no cubren exactamente el raster LiDAR una vez.\n"
            f"Píxeles asignados: {total_pixeles}\n"
            f"Píxeles esperados: {esperado}"
        )


# ============================================================
# PROCESAMIENTO DE BIOMASA POR EXPERIMENTO
# ============================================================

def crear_acumulador(zonas):
    acumulador = {}

    for zona in zonas:
        acumulador[zona["recorte"]] = {
            "detecciones_unicas": 0,
            "detecciones_altura_valida": 0,
            "detecciones_usadas_biomasa": 0,
            "detecciones_excluidas_biomasa": 0,
            "confianzas": [],
            "alturas": [],
            "diametros": [],
            "biomasa_gimnosperma_kg": 0.0,
            "biomasa_angiosperma_kg": 0.0,
        }

    return acumulador


def calcular_diametro_copa(row):
    ancho = safe_float(row.get("ancho_caja_m"))
    alto = safe_float(row.get("alto_caja_m"))

    if ancho is None or alto is None:
        x_min = safe_float(row.get("x_min_utm"))
        x_max = safe_float(row.get("x_max_utm"))
        y_min = safe_float(row.get("y_min_utm"))
        y_max = safe_float(row.get("y_max_utm"))

        if None in (x_min, x_max, y_min, y_max):
            return None, ancho, alto

        ancho = abs(x_max - x_min)
        alto = abs(y_max - y_min)

    if ancho <= 0 or alto <= 0:
        return None, ancho, alto

    return math.sqrt(ancho * alto), ancho, alto


def procesar_experimento(nombre, umbral, particion, zonas):
    csv_in = (
        DETECCIONES_ROOT
        / nombre
        / f"detecciones_unicas_{nombre}.csv"
    )

    if not csv_in.exists():
        raise FileNotFoundError(
            "No existe el CSV de detecciones únicas:\n"
            f"{csv_in}"
        )

    out_dir = OUT_ROOT / nombre
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_individual = out_dir / f"detecciones_biomasa_por_recorte_{nombre}.csv"
    csv_recortes = out_dir / f"biomasa_por_recorte_{nombre}.csv"
    resumen_txt = out_dir / f"resumen_biomasa_por_recorte_{nombre}.txt"

    acumulador = crear_acumulador(zonas)
    zonas_por_nombre = {z["recorte"]: z for z in zonas}

    filas_individuales = []
    campos_originales = []
    total_entrada = 0

    with open(csv_in, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        campos_originales = reader.fieldnames or []

        for row in reader:
            total_entrada += 1

            x = safe_float(row.get("x_centro_utm"))
            y = safe_float(row.get("y_centro_utm"))

            if x is None or y is None:
                raise RuntimeError(
                    "Una detección no contiene centro UTM válido.\n"
                    f"Experimento: {nombre}\n"
                    f"ID: {row.get('id_unica', row.get('id_deteccion', ''))}"
                )

            recorte_analisis = localizar_zona(x, y, particion)
            zona = zonas_por_nombre[recorte_analisis]
            acc = acumulador[recorte_analisis]

            acc["detecciones_unicas"] += 1

            confianza = safe_float(row.get("confianza"))
            if confianza is not None:
                acc["confianzas"].append(confianza)

            altura_valida = safe_int(row.get("altura_valida"), 0) == 1
            altura = safe_float(row.get("altura_p95_m"))

            diametro, ancho_copa, alto_copa = calcular_diametro_copa(row)

            motivo_exclusion = ""
            usada_biomasa = 0
            biomasa_gym_kg = None
            biomasa_ang_kg = None

            if not altura_valida:
                motivo_exclusion = "altura_no_valida"
            elif altura is None:
                motivo_exclusion = "altura_p95_vacia"
            elif altura < ALTURA_MIN_M or altura > ALTURA_MAX_M:
                motivo_exclusion = "altura_fuera_de_rango"
            elif diametro is None:
                motivo_exclusion = "diametro_no_calculable"
            elif diametro < DIAMETRO_COPA_MIN_M or diametro > DIAMETRO_COPA_MAX_M:
                motivo_exclusion = "diametro_fuera_de_rango"
            else:
                usada_biomasa = 1
                biomasa_gym_kg = agb_altura_copa(altura, diametro, grupo=1)
                biomasa_ang_kg = agb_altura_copa(altura, diametro, grupo=2)

            if altura_valida and altura is not None:
                acc["detecciones_altura_valida"] += 1
                acc["alturas"].append(altura)

            if usada_biomasa:
                acc["detecciones_usadas_biomasa"] += 1
                acc["diametros"].append(diametro)
                acc["biomasa_gimnosperma_kg"] += biomasa_gym_kg
                acc["biomasa_angiosperma_kg"] += biomasa_ang_kg
            else:
                acc["detecciones_excluidas_biomasa"] += 1

            fila_salida = row.copy()
            fila_salida.update(
                {
                    "recorte_analisis": recorte_analisis,
                    "fila_recorte": zona["fila"],
                    "columna_recorte": zona["columna"],
                    "core_xmin_utm": zona["xmin"],
                    "core_ymin_utm": zona["ymin"],
                    "core_xmax_utm": zona["xmax"],
                    "core_ymax_utm": zona["ymax"],
                    "diametro_copa_m": diametro if diametro is not None else "",
                    "ancho_copa_m": ancho_copa if ancho_copa is not None else "",
                    "alto_copa_m": alto_copa if alto_copa is not None else "",
                    "usada_para_biomasa": usada_biomasa,
                    "motivo_exclusion_biomasa": motivo_exclusion,
                    "agb_gimnosperma_kg": biomasa_gym_kg if biomasa_gym_kg is not None else "",
                    "agb_gimnosperma_t": biomasa_gym_kg / 1000.0 if biomasa_gym_kg is not None else "",
                    "agb_angiosperma_kg": biomasa_ang_kg if biomasa_ang_kg is not None else "",
                    "agb_angiosperma_t": biomasa_ang_kg / 1000.0 if biomasa_ang_kg is not None else "",
                }
            )

            filas_individuales.append(fila_salida)

    campos_nuevos_individual = [
        "recorte_analisis",
        "fila_recorte",
        "columna_recorte",
        "core_xmin_utm",
        "core_ymin_utm",
        "core_xmax_utm",
        "core_ymax_utm",
        "diametro_copa_m",
        "ancho_copa_m",
        "alto_copa_m",
        "usada_para_biomasa",
        "motivo_exclusion_biomasa",
        "agb_gimnosperma_kg",
        "agb_gimnosperma_t",
        "agb_angiosperma_kg",
        "agb_angiosperma_t",
    ]

    campos_individual = list(campos_originales)
    for campo in campos_nuevos_individual:
        if campo not in campos_individual:
            campos_individual.append(campo)

    escribir_csv(csv_individual, filas_individuales, campos_individual)

    filas_recortes = []

    for zona in zonas:
        acc = acumulador[zona["recorte"]]

        bio_gym_t = acc["biomasa_gimnosperma_kg"] / 1000.0
        bio_ang_t = acc["biomasa_angiosperma_kg"] / 1000.0

        fila = {
            "experimento": nombre,
            "umbral_confianza": umbral,
            "recorte_analisis": zona["recorte"],
            "fila": zona["fila"],
            "columna": zona["columna"],
            "x_centro_utm": zona["x_centro"],
            "y_centro_utm": zona["y_centro"],
            "xmin_utm": zona["xmin"],
            "ymin_utm": zona["ymin"],
            "xmax_utm": zona["xmax"],
            "ymax_utm": zona["ymax"],
            "area_total_ha": zona["area_total_ha"],
            "area_arbolada_ha": zona["area_arbolada_ha"],
            "porcentaje_arbolado": zona["porcentaje_arbolado"],
            "area_altura_2_5_ha": zona["area_altura_2_5_ha"],
            "area_altura_5_10_ha": zona["area_altura_5_10_ha"],
            "area_altura_10_15_ha": zona["area_altura_10_15_ha"],
            "area_altura_15_20_ha": zona["area_altura_15_20_ha"],
            "area_altura_20_30_ha": zona["area_altura_20_30_ha"],
            "area_altura_30_40_ha": zona["area_altura_30_40_ha"],
            "detecciones_unicas": acc["detecciones_unicas"],
            "detecciones_altura_valida": acc["detecciones_altura_valida"],
            "detecciones_usadas_biomasa": acc["detecciones_usadas_biomasa"],
            "detecciones_excluidas_biomasa": acc["detecciones_excluidas_biomasa"],
            "confianza_media": media(acc["confianzas"]),
            "altura_p95_media_m": media(acc["alturas"]),
            "altura_p95_mediana_m": mediana(acc["alturas"]),
            "altura_p95_min_m": minimo(acc["alturas"]),
            "altura_p95_max_m": maximo(acc["alturas"]),
            "diametro_copa_medio_m": media(acc["diametros"]),
            "diametro_copa_mediano_m": mediana(acc["diametros"]),
            "biomasa_gimnosperma_t": bio_gym_t,
            "biomasa_angiosperma_t": bio_ang_t,
            "biomasa_gimnosperma_t_ha_total": (
                bio_gym_t / zona["area_total_ha"]
                if zona["area_total_ha"] > 0
                else ""
            ),
            "biomasa_angiosperma_t_ha_total": (
                bio_ang_t / zona["area_total_ha"]
                if zona["area_total_ha"] > 0
                else ""
            ),
            "biomasa_gimnosperma_t_ha_arbolada": (
                bio_gym_t / zona["area_arbolada_ha"]
                if zona["area_arbolada_ha"] > 0
                else ""
            ),
            "biomasa_angiosperma_t_ha_arbolada": (
                bio_ang_t / zona["area_arbolada_ha"]
                if zona["area_arbolada_ha"] > 0
                else ""
            ),
            "detecciones_ha_total": (
                acc["detecciones_usadas_biomasa"] / zona["area_total_ha"]
                if zona["area_total_ha"] > 0
                else ""
            ),
            "detecciones_ha_arbolada": (
                acc["detecciones_usadas_biomasa"] / zona["area_arbolada_ha"]
                if zona["area_arbolada_ha"] > 0
                else ""
            ),
        }

        filas_recortes.append(fila)

    campos_recortes = list(filas_recortes[0].keys())
    escribir_csv(csv_recortes, filas_recortes, campos_recortes)

    total_area = sum(f["area_total_ha"] for f in filas_recortes)
    total_arbolada = sum(f["area_arbolada_ha"] for f in filas_recortes)
    total_unicas = sum(f["detecciones_unicas"] for f in filas_recortes)
    total_altura = sum(f["detecciones_altura_valida"] for f in filas_recortes)
    total_usadas = sum(f["detecciones_usadas_biomasa"] for f in filas_recortes)
    total_excluidas = sum(f["detecciones_excluidas_biomasa"] for f in filas_recortes)
    total_gym_t = sum(f["biomasa_gimnosperma_t"] for f in filas_recortes)
    total_ang_t = sum(f["biomasa_angiosperma_t"] for f in filas_recortes)

    if total_unicas != total_entrada:
        raise RuntimeError(
            "No todas las detecciones fueron asignadas exactamente una vez.\n"
            f"Entrada: {total_entrada}\n"
            f"Asignadas: {total_unicas}"
        )

    resumen = {
        "experimento": nombre,
        "umbral_confianza": umbral,
        "numero_recortes": len(filas_recortes),
        "area_total_ha": total_area,
        "area_arbolada_ha": total_arbolada,
        "porcentaje_arbolado": total_arbolada / total_area * 100.0,
        "detecciones_unicas": total_unicas,
        "detecciones_altura_valida": total_altura,
        "detecciones_usadas_biomasa": total_usadas,
        "detecciones_excluidas_biomasa": total_excluidas,
        "biomasa_gimnosperma_t": total_gym_t,
        "biomasa_angiosperma_t": total_ang_t,
        "biomasa_gimnosperma_t_ha_total": total_gym_t / total_area,
        "biomasa_angiosperma_t_ha_total": total_ang_t / total_area,
        "biomasa_gimnosperma_t_ha_arbolada": total_gym_t / total_arbolada,
        "biomasa_angiosperma_t_ha_arbolada": total_ang_t / total_arbolada,
        "detecciones_ha_total": total_usadas / total_area,
        "detecciones_ha_arbolada": total_usadas / total_arbolada,
        "ruta_csv_recortes": str(csv_recortes),
        "ruta_csv_individual": str(csv_individual),
    }

    with open(resumen_txt, "w", encoding="utf-8") as f:
        f.write(f"RESUMEN BIOMASA POR RECORTE {nombre}\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Umbral de confianza: {umbral:.2f}\n")
        f.write(f"CSV de entrada: {csv_in}\n")
        f.write(f"Número de zonas no solapadas: {len(filas_recortes)}\n")
        f.write(f"Área total: {total_area:.3f} ha\n")
        f.write(f"Área arbolada: {total_arbolada:.3f} ha\n")
        f.write(f"Porcentaje arbolado: {resumen['porcentaje_arbolado']:.3f} %\n\n")
        f.write(f"Detecciones únicas: {total_unicas}\n")
        f.write(f"Detecciones con altura válida: {total_altura}\n")
        f.write(f"Detecciones usadas para biomasa: {total_usadas}\n")
        f.write(f"Detecciones excluidas: {total_excluidas}\n\n")
        f.write(f"Biomasa conífera: {total_gym_t:.3f} t\n")
        f.write(f"Biomasa frondosa: {total_ang_t:.3f} t\n")
        f.write(
            "Biomasa conífera por ha total: "
            f"{resumen['biomasa_gimnosperma_t_ha_total']:.3f} t/ha\n"
        )
        f.write(
            "Biomasa conífera por ha arbolada: "
            f"{resumen['biomasa_gimnosperma_t_ha_arbolada']:.3f} t/ha\n"
        )
        f.write(f"\nTabla por recorte: {csv_recortes}\n")
        f.write(f"Tabla individual: {csv_individual}\n")

    print(f"[OK] {nombre}: {total_gym_t:.3f} t de biomasa conífera")
    print(f"[OK] {nombre}: {resumen['biomasa_gimnosperma_t_ha_arbolada']:.3f} t/ha arbolada")
    print(f"[OK] Tabla por recorte: {csv_recortes}")

    return resumen, filas_recortes


# ============================================================
# TABLAS CONJUNTAS
# ============================================================

def escribir_tabla_comparacion(resultados_por_experimento, zonas):
    indice = {}

    for nombre, filas in resultados_por_experimento.items():
        indice[nombre] = {
            fila["recorte_analisis"]: fila
            for fila in filas
        }

    filas_salida = []

    for zona in zonas:
        fila = {
            "recorte_analisis": zona["recorte"],
            "fila": zona["fila"],
            "columna": zona["columna"],
            "x_centro_utm": zona["x_centro"],
            "y_centro_utm": zona["y_centro"],
            "xmin_utm": zona["xmin"],
            "ymin_utm": zona["ymin"],
            "xmax_utm": zona["xmax"],
            "ymax_utm": zona["ymax"],
            "area_total_ha": zona["area_total_ha"],
            "area_arbolada_ha": zona["area_arbolada_ha"],
            "porcentaje_arbolado": zona["porcentaje_arbolado"],
        }

        for nombre in EXPERIMENTOS:
            origen = indice[nombre][zona["recorte"]]
            fila[f"{nombre}_detecciones_unicas"] = origen["detecciones_unicas"]
            fila[f"{nombre}_detecciones_altura_valida"] = origen["detecciones_altura_valida"]
            fila[f"{nombre}_detecciones_usadas_biomasa"] = origen["detecciones_usadas_biomasa"]
            fila[f"{nombre}_altura_p95_media_m"] = origen["altura_p95_media_m"]
            fila[f"{nombre}_diametro_copa_medio_m"] = origen["diametro_copa_medio_m"]
            fila[f"{nombre}_biomasa_gimnosperma_t"] = origen["biomasa_gimnosperma_t"]
            fila[f"{nombre}_biomasa_gimnosperma_t_ha_arbolada"] = origen["biomasa_gimnosperma_t_ha_arbolada"]

        filas_salida.append(fila)

    escribir_csv(
        TABLA_COMPARACION_CSV,
        filas_salida,
        list(filas_salida[0].keys()),
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 75)
    print("BIOMASA GLOBAL Y POR RECORTE")
    print("COMPARATIVA HOMOGÉNEA DE UMBRALES")
    print("=" * 75)

    if not DETECCIONES_ROOT.exists():
        raise FileNotFoundError(
            "No existe la carpeta de detecciones georreferenciadas:\n"
            f"{DETECCIONES_ROOT}"
        )

    if not GEOTIFF_DIR.exists():
        raise FileNotFoundError(
            "No existe la carpeta de recortes GeoTIFF:\n"
            f"{GEOTIFF_DIR}"
        )

    altura_path = buscar_raster_altura()

    print(f"[INFO] Raster LiDAR: {altura_path}")
    print("[INFO] Construyendo 180 zonas núcleo no solapadas...")

    teselas = leer_teselas_geotiff()
    particion = construir_zonas_nucleo(teselas)
    zonas = particion["zonas"]

    print(
        f"[INFO] Rejilla detectada: "
        f"{particion['numero_filas']} filas × "
        f"{particion['numero_columnas']} columnas"
    )
    print(f"[INFO] Zonas núcleo: {len(zonas)}")

    lidar = cargar_lidar(altura_path)
    calcular_lidar_por_zona(zonas, lidar)

    area_total = sum(z["area_total_ha"] for z in zonas)
    area_arbolada = sum(z["area_arbolada_ha"] for z in zonas)

    print(f"[INFO] Área total comprobada: {area_total:.3f} ha")
    print(f"[INFO] Área arbolada comprobada: {area_arbolada:.3f} ha")

    resumenes = []
    resultados_por_experimento = {}
    filas_largas = []

    for nombre, umbral in EXPERIMENTOS.items():
        print("")
        print("-" * 75)
        print(f"[INFO] Procesando {nombre}")
        print("-" * 75)

        resumen, filas_recortes = procesar_experimento(
            nombre,
            umbral,
            particion,
            zonas,
        )

        resumenes.append(resumen)
        resultados_por_experimento[nombre] = filas_recortes
        filas_largas.extend(filas_recortes)

        escribir_csv(
            RESUMEN_GLOBAL_CSV,
            resumenes,
            list(resumenes[0].keys()),
        )

    escribir_csv(
        TABLA_LARGA_CSV,
        filas_largas,
        list(filas_largas[0].keys()),
    )

    escribir_tabla_comparacion(resultados_por_experimento, zonas)

    with open(RESUMEN_TXT, "w", encoding="utf-8") as f:
        f.write("COMPARATIVA GLOBAL DE BIOMASA\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Área total: {area_total:.3f} ha\n")
        f.write(f"Área arbolada: {area_arbolada:.3f} ha\n")
        f.write(f"Número de zonas: {len(zonas)}\n\n")

        for r in resumenes:
            f.write(f"{r['experimento']}\n")
            f.write(f"  Detecciones únicas: {r['detecciones_unicas']}\n")
            f.write(f"  Usadas para biomasa: {r['detecciones_usadas_biomasa']}\n")
            f.write(f"  Biomasa conífera: {r['biomasa_gimnosperma_t']:.3f} t\n")
            f.write(
                "  Biomasa conífera por ha arbolada: "
                f"{r['biomasa_gimnosperma_t_ha_arbolada']:.3f} t/ha\n\n"
            )

    print("")
    print("=" * 75)
    print("[FIN] PROCESO COMPLETADO")
    print(f"[OK] Resumen global: {RESUMEN_GLOBAL_CSV}")
    print(f"[OK] Tabla larga: {TABLA_LARGA_CSV}")
    print(f"[OK] Comparación por recorte: {TABLA_COMPARACION_CSV}")
    print("=" * 75)


if __name__ == "__main__":
    main()
