# -*- coding: utf-8 -*-

"""
Genera visualizaciones para analizar la biomasa estimada recorte por recorte
y comparar los umbrales YOLO 0,10, 0,15, 0,20 y 0,25.

Entradas esperadas:
- comparacion_biomasa_por_recorte.csv
- resumen_global_comparativa_biomasa.csv

Salidas:
- curvas globales;
- mapas por recorte en coordenadas UTM;
- distribución de biomasa;
- relaciones entre superficie, detecciones y biomasa;
- mapa y ranking de sensibilidad al umbral;
- selección automática de recortes para revisión visual o trabajo de campo;
- tablas auxiliares CSV;
- resumen automático TXT.

El script usa únicamente pandas, numpy y matplotlib.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

CSV_COMPARACION_DEFECTO = (
    REPO_ROOT
    / "results"
    / "spatial_analysis"
    / "comparacion_biomasa_por_recorte.csv"
)

CSV_RESUMEN_DEFECTO = (
    REPO_ROOT
    / "results"
    / "biomass"
    / "resumen_global_comparativa_biomasa.csv"
)

SALIDA_DEFECTO = REPO_ROOT / "outputs" / "figures"

EXPERIMENTOS = {
    "conf010": "10 %",
    "conf015": "15 %",
    "conf020": "20 %",
    "conf025": "25 %",
}

COLUMNAS_BIOMASA_HA = {
    conf: f"{conf}_biomasa_gimnosperma_t_ha_arbolada"
    for conf in EXPERIMENTOS
}

COLUMNAS_BIOMASA_TOTAL = {
    conf: f"{conf}_biomasa_gimnosperma_t"
    for conf in EXPERIMENTOS
}

COLUMNAS_DETECCIONES = {
    conf: f"{conf}_detecciones_usadas_biomasa"
    for conf in EXPERIMENTOS
}

COLUMNAS_ALTURA = {
    conf: f"{conf}_altura_p95_media_m"
    for conf in EXPERIMENTOS
}

DPI = 300
GUARDAR_PDF = True

# Para el ranking de sensibilidad se evita que zonas con una superficie
# arbolada casi nula dominen la tabla por divisiones poco representativas.
AREA_ARBOLADA_MIN_RANKING_HA = 1.0

# Número máximo de recortes mostrados en rankings.
TOP_N = 15


# ============================================================================
# UTILIDADES
# ============================================================================

def resolver_archivo(ruta: Path, patron: str) -> Path:
    """
    Usa la ruta exacta si existe. En caso contrario, busca un archivo
    compatible en la misma carpeta.
    """
    if ruta.exists():
        return ruta

    candidatos = sorted(ruta.parent.glob(patron))

    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ruta}\n"
            f"Ni coincidencias con el patrón: {patron}"
        )

    print(
        f"[AVISO] No se encontró el nombre exacto. "
        f"Se utilizará:\n        {candidatos[0]}"
    )
    return candidatos[0]


def leer_csv(ruta: Path) -> pd.DataFrame:
    """
    Lee CSV UTF-8/UTF-8 con BOM y ofrece un error legible.
    """
    try:
        return pd.read_csv(ruta, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="latin-1")


def comprobar_columnas(
    df: pd.DataFrame,
    columnas: Iterable[str],
    nombre_archivo: str,
) -> None:
    faltantes = [c for c in columnas if c not in df.columns]

    if faltantes:
        raise KeyError(
            f"Faltan columnas obligatorias en {nombre_archivo}:\n"
            + "\n".join(f"  - {c}" for c in faltantes)
        )


def nombre_corto_recorte(nombre: str) -> str:
    """
    Convierte el identificador largo del recorte en una etiqueta breve Fxx-Cxx.
    """
    partes = str(nombre).split("_")

    if len(partes) >= 2:
        try:
            fila = int(partes[-2])
            columna = int(partes[-1])
            return f"F{fila:02d}-C{columna:02d}"
        except ValueError:
            pass

    return str(nombre)


def guardar_figura(fig: plt.Figure, salida_base: Path) -> None:
    """
    Guarda la figura en PNG y, opcionalmente, PDF.
    """
    salida_base.parent.mkdir(parents=True, exist_ok=True)

    png = salida_base.with_suffix(".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")

    if GUARDAR_PDF:
        pdf = salida_base.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] {png.name}")


def anotar_puntos(
    ax: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    etiquetas: Iterable[str],
) -> None:
    for xi, yi, etiqueta in zip(x, y, etiquetas):
        ax.annotate(
            etiqueta,
            (xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )


def preparar_datos(
    df_comp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Añade variables derivadas y crea una tabla larga:
    una fila por recorte y experimento.
    """
    df = df_comp.copy()

    biomasa_ha_cols = list(COLUMNAS_BIOMASA_HA.values())
    biomasa_total_cols = list(COLUMNAS_BIOMASA_TOTAL.values())

    df["recorte_corto"] = df["recorte_analisis"].map(nombre_corto_recorte)

    df["biomasa_ha_media_umbrales"] = df[biomasa_ha_cols].mean(
        axis=1,
        skipna=True,
    )
    df["biomasa_ha_min_umbrales"] = df[biomasa_ha_cols].min(
        axis=1,
        skipna=True,
    )
    df["biomasa_ha_max_umbrales"] = df[biomasa_ha_cols].max(
        axis=1,
        skipna=True,
    )
    df["rango_biomasa_ha"] = (
        df["biomasa_ha_max_umbrales"]
        - df["biomasa_ha_min_umbrales"]
    )

    media = df["biomasa_ha_media_umbrales"].replace(0, np.nan)
    df["sensibilidad_relativa_pct"] = (
        df["rango_biomasa_ha"]
        / media
        * 100.0
    )

    df["diferencia_conf010_conf025_t_ha"] = (
        df[COLUMNAS_BIOMASA_HA["conf010"]]
        - df[COLUMNAS_BIOMASA_HA["conf025"]]
    )

    base_025 = df[COLUMNAS_BIOMASA_HA["conf025"]].replace(0, np.nan)
    df["incremento_conf010_vs_conf025_pct"] = (
        df["diferencia_conf010_conf025_t_ha"]
        / base_025
        * 100.0
    )

    df["biomasa_total_media_umbrales_t"] = df[
        biomasa_total_cols
    ].mean(axis=1, skipna=True)

    registros = []

    for _, fila in df.iterrows():
        for conf, etiqueta in EXPERIMENTOS.items():
            registros.append(
                {
                    "recorte_analisis": fila["recorte_analisis"],
                    "recorte_corto": fila["recorte_corto"],
                    "fila": fila["fila"],
                    "columna": fila["columna"],
                    "x_centro_utm": fila["x_centro_utm"],
                    "y_centro_utm": fila["y_centro_utm"],
                    "xmin_utm": fila["xmin_utm"],
                    "ymin_utm": fila["ymin_utm"],
                    "xmax_utm": fila["xmax_utm"],
                    "ymax_utm": fila["ymax_utm"],
                    "area_total_ha": fila["area_total_ha"],
                    "area_arbolada_ha": fila["area_arbolada_ha"],
                    "porcentaje_arbolado": fila["porcentaje_arbolado"],
                    "experimento": conf,
                    "umbral": etiqueta,
                    "detecciones_usadas_biomasa": fila[
                        COLUMNAS_DETECCIONES[conf]
                    ],
                    "altura_p95_media_m": fila[COLUMNAS_ALTURA[conf]],
                    "biomasa_gimnosperma_t": fila[
                        COLUMNAS_BIOMASA_TOTAL[conf]
                    ],
                    "biomasa_gimnosperma_t_ha_arbolada": fila[
                        COLUMNAS_BIOMASA_HA[conf]
                    ],
                    "sensibilidad_relativa_pct": fila[
                        "sensibilidad_relativa_pct"
                    ],
                }
            )

    df_largo = pd.DataFrame(registros)

    return df, df_largo


# ============================================================================
# MAPAS DE RECTÁNGULOS UTM
# ============================================================================

def mapa_rectangulos_utm(
    df: pd.DataFrame,
    columna_valor: str,
    titulo: str,
    etiqueta_barra: str,
    salida_base: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    mostrar_codigo: bool = False,
) -> None:
    """
    Representa cada unidad espacial usando sus límites UTM reales.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    patches = []
    valores = []

    for _, fila in df.iterrows():
        valor = fila[columna_valor]

        if pd.isna(valor):
            continue

        ancho = float(fila["xmax_utm"] - fila["xmin_utm"])
        alto = float(fila["ymax_utm"] - fila["ymin_utm"])

        patches.append(
            Rectangle(
                (float(fila["xmin_utm"]), float(fila["ymin_utm"])),
                ancho,
                alto,
            )
        )
        valores.append(float(valor))

    if not patches:
        plt.close(fig)
        print(
            f"[AVISO] No hay datos válidos para el mapa: {titulo}"
        )
        return

    coleccion = PatchCollection(
        patches,
        edgecolor="black",
        linewidth=0.25,
    )
    coleccion.set_array(np.asarray(valores, dtype=float))

    if vmin is not None:
        coleccion.set_clim(vmin=vmin, vmax=vmax)

    ax.add_collection(coleccion)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")

    if mostrar_codigo:
        for _, fila in df.iterrows():
            ax.text(
                fila["x_centro_utm"],
                fila["y_centro_utm"],
                f"{int(fila['fila']):02d}-{int(fila['columna']):02d}",
                ha="center",
                va="center",
                fontsize=5,
            )

    barra = fig.colorbar(coleccion, ax=ax)
    barra.set_label(etiqueta_barra)

    ax.set_title(titulo)
    ax.set_xlabel("Coordenada UTM X (m)")
    ax.set_ylabel("Coordenada UTM Y (m)")
    ax.ticklabel_format(style="plain", useOffset=False)

    guardar_figura(fig, salida_base)


# ============================================================================
# GRÁFICOS
# ============================================================================

def grafico_biomasa_global(
    df_resumen: pd.DataFrame,
    out_dir: Path,
) -> None:
    df = df_resumen.sort_values("umbral_confianza")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["umbral_confianza"] * 100,
        df["biomasa_gimnosperma_t"],
        marker="o",
    )
    anotar_puntos(
        ax,
        df["umbral_confianza"] * 100,
        df["biomasa_gimnosperma_t"],
        [f"{v:,.0f} t" for v in df["biomasa_gimnosperma_t"]],
    )
    ax.set_title("Biomasa total estimada según el umbral YOLO")
    ax.set_xlabel("Umbral de confianza (%)")
    ax.set_ylabel("Biomasa de gimnospermas (t)")
    ax.grid(True, alpha=0.3)
    guardar_figura(
        fig,
        out_dir / "01_biomasa_total_segun_umbral",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["umbral_confianza"] * 100,
        df["biomasa_gimnosperma_t_ha_arbolada"],
        marker="o",
    )
    anotar_puntos(
        ax,
        df["umbral_confianza"] * 100,
        df["biomasa_gimnosperma_t_ha_arbolada"],
        [
            f"{v:.2f} t/ha"
            for v in df["biomasa_gimnosperma_t_ha_arbolada"]
        ],
    )
    ax.set_title(
        "Biomasa estimada por hectárea arbolada según el umbral"
    )
    ax.set_xlabel("Umbral de confianza (%)")
    ax.set_ylabel("Biomasa de gimnospermas (t/ha arbolada)")
    ax.grid(True, alpha=0.3)
    guardar_figura(
        fig,
        out_dir / "02_biomasa_ha_arbolada_segun_umbral",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["umbral_confianza"] * 100,
        df["detecciones_usadas_biomasa"],
        marker="o",
    )
    anotar_puntos(
        ax,
        df["umbral_confianza"] * 100,
        df["detecciones_usadas_biomasa"],
        [f"{int(v):,}" for v in df["detecciones_usadas_biomasa"]],
    )
    ax.set_title(
        "Detecciones utilizadas para biomasa según el umbral"
    )
    ax.set_xlabel("Umbral de confianza (%)")
    ax.set_ylabel("Detecciones utilizadas")
    ax.grid(True, alpha=0.3)
    guardar_figura(
        fig,
        out_dir / "03_detecciones_usadas_segun_umbral",
    )


def boxplot_biomasa(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    datos = [
        df[COLUMNAS_BIOMASA_HA[conf]].dropna().to_numpy()
        for conf in EXPERIMENTOS
    ]
    etiquetas = list(EXPERIMENTOS.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(datos, labels=etiquetas, showfliers=True)
    ax.set_title(
        "Distribución de biomasa por unidad espacial de análisis"
    )
    ax.set_xlabel("Umbral de confianza YOLO")
    ax.set_ylabel("Biomasa de gimnospermas (t/ha arbolada)")
    ax.grid(True, axis="y", alpha=0.3)
    guardar_figura(
        fig,
        out_dir / "04_distribucion_biomasa_por_umbral",
    )


def dispersion_area_biomasa(
    df_largo: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for conf, etiqueta in EXPERIMENTOS.items():
        sub = df_largo[
            (df_largo["experimento"] == conf)
            & df_largo["biomasa_gimnosperma_t"].notna()
        ]

        ax.scatter(
            sub["area_arbolada_ha"],
            sub["biomasa_gimnosperma_t"],
            label=etiqueta,
            alpha=0.65,
            s=22,
        )

    ax.set_title("Relación entre superficie arbolada y biomasa")
    ax.set_xlabel("Superficie arbolada de la unidad (ha)")
    ax.set_ylabel("Biomasa estimada de gimnospermas (t)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Confianza")
    guardar_figura(
        fig,
        out_dir / "05_area_arbolada_frente_biomasa",
    )


def dispersion_detecciones_biomasa(
    df_largo: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for conf, etiqueta in EXPERIMENTOS.items():
        sub = df_largo[
            (df_largo["experimento"] == conf)
            & df_largo["biomasa_gimnosperma_t"].notna()
        ]

        ax.scatter(
            sub["detecciones_usadas_biomasa"],
            sub["biomasa_gimnosperma_t"],
            label=etiqueta,
            alpha=0.65,
            s=22,
        )

    ax.set_title(
        "Relación entre detecciones utilizadas y biomasa estimada"
    )
    ax.set_xlabel("Detecciones utilizadas para biomasa")
    ax.set_ylabel("Biomasa estimada de gimnospermas (t)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Confianza")
    guardar_figura(
        fig,
        out_dir / "06_detecciones_frente_biomasa",
    )


def ranking_sensibilidad(
    df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    candidatos = df[
        (df["area_arbolada_ha"] >= AREA_ARBOLADA_MIN_RANKING_HA)
        & df["sensibilidad_relativa_pct"].notna()
    ].copy()

    ranking = candidatos.nlargest(
        TOP_N,
        "sensibilidad_relativa_pct",
    ).sort_values("sensibilidad_relativa_pct")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        ranking["recorte_corto"],
        ranking["sensibilidad_relativa_pct"],
    )
    ax.set_title(
        f"Unidades más sensibles al umbral "
        f"(área arbolada ≥ {AREA_ARBOLADA_MIN_RANKING_HA:g} ha)"
    )
    ax.set_xlabel("Sensibilidad relativa de la biomasa (%)")
    ax.set_ylabel("Unidad espacial")
    ax.grid(True, axis="x", alpha=0.3)
    guardar_figura(
        fig,
        out_dir / "07_top_sensibilidad_umbral",
    )

    return ranking.sort_values(
        "sensibilidad_relativa_pct",
        ascending=False,
    )


def grafico_recortes_seleccionados(
    df: pd.DataFrame,
    seleccion: pd.DataFrame,
    out_dir: Path,
) -> None:
    if seleccion.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    eje_x = [10, 15, 20, 25]

    for _, sel in seleccion.iterrows():
        fila = df[
            df["recorte_analisis"] == sel["recorte_analisis"]
        ].iloc[0]

        valores = [
            fila[COLUMNAS_BIOMASA_HA[conf]]
            for conf in EXPERIMENTOS
        ]

        if all(pd.isna(v) for v in valores):
            continue

        ax.plot(
            eje_x,
            valores,
            marker="o",
            label=f"{fila['recorte_corto']} · {sel['motivo_seleccion']}",
        )

    ax.set_title(
        "Comportamiento de recortes representativos según el umbral"
    )
    ax.set_xlabel("Umbral de confianza (%)")
    ax.set_ylabel("Biomasa de gimnospermas (t/ha arbolada)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    guardar_figura(
        fig,
        out_dir / "08_recortes_representativos_segun_umbral",
    )


# ============================================================================
# SELECCIÓN AUTOMÁTICA DE RECORTES
# ============================================================================

def seleccionar_recortes_para_revision(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Elige casos representativos para mostrar imágenes con cajas amarillas,
    revisar sobre ortofoto o comprobar en campo.
    """
    candidatos = df[df["area_arbolada_ha"] > 0].copy()
    selecciones: list[tuple[str, pd.Series]] = []

    def agregar(motivo: str, fila: pd.Series) -> None:
        if fila is None:
            return

        nombre = fila["recorte_analisis"]

        if any(nombre == existente["recorte_analisis"] for _, existente in selecciones):
            return

        selecciones.append((motivo, fila))

    agregar(
        "Mayor superficie arbolada",
        candidatos.loc[candidatos["area_arbolada_ha"].idxmax()],
    )
    agregar(
        "Mayor biomasa media",
        candidatos.loc[
            candidatos["biomasa_ha_media_umbrales"].idxmax()
        ],
    )

    candidatos_sens = candidatos[
        candidatos["area_arbolada_ha"] >= AREA_ARBOLADA_MIN_RANKING_HA
    ]

    if not candidatos_sens.empty:
        agregar(
            "Mayor sensibilidad",
            candidatos_sens.loc[
                candidatos_sens["sensibilidad_relativa_pct"].idxmax()
            ],
        )
        agregar(
            "Mayor estabilidad",
            candidatos_sens.loc[
                candidatos_sens["sensibilidad_relativa_pct"].idxmin()
            ],
        )

    q25 = candidatos["porcentaje_arbolado"].quantile(0.25)
    q50 = candidatos["porcentaje_arbolado"].quantile(0.50)
    q75 = candidatos["porcentaje_arbolado"].quantile(0.75)

    agregar(
        "Cobertura baja",
        candidatos.loc[
            (candidatos["porcentaje_arbolado"] - q25).abs().idxmin()
        ],
    )
    agregar(
        "Cobertura media",
        candidatos.loc[
            (candidatos["porcentaje_arbolado"] - q50).abs().idxmin()
        ],
    )
    agregar(
        "Cobertura alta",
        candidatos.loc[
            (candidatos["porcentaje_arbolado"] - q75).abs().idxmin()
        ],
    )
    agregar(
        "Mayor diferencia 10 %-25 %",
        candidatos.loc[
            candidatos["diferencia_conf010_conf025_t_ha"].idxmax()
        ],
    )

    filas_salida = []

    for motivo, fila in selecciones:
        filas_salida.append(
            {
                "motivo_seleccion": motivo,
                "recorte_analisis": fila["recorte_analisis"],
                "recorte_corto": fila["recorte_corto"],
                "fila": int(fila["fila"]),
                "columna": int(fila["columna"]),
                "x_centro_utm": fila["x_centro_utm"],
                "y_centro_utm": fila["y_centro_utm"],
                "area_arbolada_ha": fila["area_arbolada_ha"],
                "porcentaje_arbolado": fila["porcentaje_arbolado"],
                "sensibilidad_relativa_pct": fila[
                    "sensibilidad_relativa_pct"
                ],
                "conf010_t_ha": fila[COLUMNAS_BIOMASA_HA["conf010"]],
                "conf015_t_ha": fila[COLUMNAS_BIOMASA_HA["conf015"]],
                "conf020_t_ha": fila[COLUMNAS_BIOMASA_HA["conf020"]],
                "conf025_t_ha": fila[COLUMNAS_BIOMASA_HA["conf025"]],
            }
        )

    return pd.DataFrame(filas_salida)


# ============================================================================
# RESUMEN AUTOMÁTICO
# ============================================================================

def generar_resumen_txt(
    df: pd.DataFrame,
    df_resumen: pd.DataFrame,
    ranking: pd.DataFrame,
    seleccion: pd.DataFrame,
    salida: Path,
) -> None:
    resumen = df_resumen.sort_values("umbral_confianza")

    biomasa_010 = float(
        resumen.loc[
            resumen["experimento"] == "conf010",
            "biomasa_gimnosperma_t",
        ].iloc[0]
    )
    biomasa_025 = float(
        resumen.loc[
            resumen["experimento"] == "conf025",
            "biomasa_gimnosperma_t",
        ].iloc[0]
    )

    incremento = (
        (biomasa_010 - biomasa_025)
        / biomasa_025
        * 100.0
    )

    validas = df["sensibilidad_relativa_pct"].dropna()
    n_sin_area = int((df["area_arbolada_ha"] == 0).sum())

    with open(salida, "w", encoding="utf-8") as f:
        f.write("RESUMEN VISUAL DEL ANÁLISIS POR RECORTE\n")
        f.write("=" * 72 + "\n\n")

        f.write(
            f"Unidades espaciales analizadas: {len(df)}\n"
        )
        f.write(
            f"Unidades sin superficie arbolada LiDAR: {n_sin_area}\n"
        )
        f.write(
            f"Superficie total: {df['area_total_ha'].sum():.3f} ha\n"
        )
        f.write(
            f"Superficie arbolada: {df['area_arbolada_ha'].sum():.3f} ha\n\n"
        )

        f.write("RESULTADOS GLOBALES\n")
        f.write("-" * 72 + "\n")

        for _, fila in resumen.iterrows():
            f.write(
                f"{fila['experimento']}: "
                f"{fila['biomasa_gimnosperma_t']:.3f} t; "
                f"{fila['biomasa_gimnosperma_t_ha_arbolada']:.3f} "
                f"t/ha arbolada; "
                f"{int(fila['detecciones_usadas_biomasa'])} "
                f"detecciones utilizadas.\n"
            )

        f.write(
            f"\nEl resultado de conf010 supera al de conf025 en "
            f"{incremento:.2f} %.\n\n"
        )

        if not validas.empty:
            f.write("SENSIBILIDAD POR UNIDAD ESPACIAL\n")
            f.write("-" * 72 + "\n")
            f.write(
                f"Media: {validas.mean():.3f} %\n"
            )
            f.write(
                f"Mediana: {validas.median():.3f} %\n"
            )
            f.write(
                f"Mínima: {validas.min():.3f} %\n"
            )
            f.write(
                f"Máxima: {validas.max():.3f} %\n\n"
            )

        if not ranking.empty:
            f.write(
                "RECORTES PRIORITARIOS POR SENSIBILIDAD "
                f"(área arbolada ≥ {AREA_ARBOLADA_MIN_RANKING_HA:g} ha)\n"
            )
            f.write("-" * 72 + "\n")

            for _, fila in ranking.head(10).iterrows():
                f.write(
                    f"{fila['recorte_corto']}: "
                    f"{fila['sensibilidad_relativa_pct']:.2f} %; "
                    f"área arbolada {fila['area_arbolada_ha']:.3f} ha; "
                    f"cobertura {fila['porcentaje_arbolado']:.2f} %.\n"
                )

        if not seleccion.empty:
            f.write("\nRECORTES SELECCIONADOS PARA REVISIÓN\n")
            f.write("-" * 72 + "\n")

            for _, fila in seleccion.iterrows():
                f.write(
                    f"{fila['recorte_corto']} — "
                    f"{fila['motivo_seleccion']}\n"
                )

        f.write("\nINTERPRETACIÓN\n")
        f.write("-" * 72 + "\n")
        f.write(
            "Las diferencias entre umbrales deben interpretarse como "
            "sensibilidad del procedimiento de detección, no como un "
            "intervalo de confianza estadístico. Los mapas permiten "
            "localizar las unidades donde el resultado es estable y "
            "aquellas donde debe priorizarse la revisión visual o la "
            "comparación con datos de campo.\n"
        )


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera visualizaciones de biomasa por recorte y "
            "comparativa de umbrales YOLO."
        )
    )
    parser.add_argument(
        "--comparacion",
        type=Path,
        default=CSV_COMPARACION_DEFECTO,
        help="Ruta a comparacion_biomasa_por_recorte.csv",
    )
    parser.add_argument(
        "--resumen",
        type=Path,
        default=CSV_RESUMEN_DEFECTO,
        help="Ruta a resumen_global_comparativa_biomasa.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=SALIDA_DEFECTO,
        help="Carpeta de salida",
    )

    args = parser.parse_args()

    csv_comparacion = resolver_archivo(
        args.comparacion,
        "comparacion_biomasa_por_recorte*.csv",
    )
    csv_resumen = resolver_archivo(
        args.resumen,
        "resumen_global_comparativa_biomasa*.csv",
    )

    out_dir = args.salida
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("VISUALIZACIONES DE BIOMASA POR RECORTE")
    print("=" * 76)
    print(f"[INFO] Comparación: {csv_comparacion}")
    print(f"[INFO] Resumen:     {csv_resumen}")
    print(f"[INFO] Salida:      {out_dir}")

    df_comp = leer_csv(csv_comparacion)
    df_resumen = leer_csv(csv_resumen)

    columnas_comp = [
        "recorte_analisis",
        "fila",
        "columna",
        "x_centro_utm",
        "y_centro_utm",
        "xmin_utm",
        "ymin_utm",
        "xmax_utm",
        "ymax_utm",
        "area_total_ha",
        "area_arbolada_ha",
        "porcentaje_arbolado",
        *COLUMNAS_BIOMASA_HA.values(),
        *COLUMNAS_BIOMASA_TOTAL.values(),
        *COLUMNAS_DETECCIONES.values(),
        *COLUMNAS_ALTURA.values(),
    ]

    columnas_resumen = [
        "experimento",
        "umbral_confianza",
        "numero_recortes",
        "area_total_ha",
        "area_arbolada_ha",
        "detecciones_usadas_biomasa",
        "biomasa_gimnosperma_t",
        "biomasa_gimnosperma_t_ha_arbolada",
    ]

    comprobar_columnas(
        df_comp,
        columnas_comp,
        csv_comparacion.name,
    )
    comprobar_columnas(
        df_resumen,
        columnas_resumen,
        csv_resumen.name,
    )

    if len(df_comp) != 180:
        print(
            f"[AVISO] Se esperaban 180 unidades, "
            f"pero el CSV contiene {len(df_comp)}."
        )

    df, df_largo = preparar_datos(df_comp)

    df.to_csv(
        out_dir / "resumen_visual_por_recorte.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df_largo.to_csv(
        out_dir / "datos_por_recorte_formato_largo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[INFO] Generando curvas globales...")
    grafico_biomasa_global(df_resumen, out_dir)

    print("[INFO] Generando distribución y relaciones...")
    boxplot_biomasa(df, out_dir)
    dispersion_area_biomasa(df_largo, out_dir)
    dispersion_detecciones_biomasa(df_largo, out_dir)

    print("[INFO] Generando mapa de cobertura arbolada...")
    mapa_rectangulos_utm(
        df,
        "porcentaje_arbolado",
        "Porcentaje de superficie arbolada por unidad espacial",
        "Superficie arbolada (%)",
        out_dir / "09_mapa_porcentaje_arbolado",
        mostrar_codigo=True,
    )

    print("[INFO] Generando mapas de biomasa con escala común...")
    todos_biomasa_ha = pd.concat(
        [df[col] for col in COLUMNAS_BIOMASA_HA.values()],
        ignore_index=True,
    ).dropna()

    if todos_biomasa_ha.empty:
        vmin_biomasa = None
        vmax_biomasa = None
    else:
        vmin_biomasa = float(todos_biomasa_ha.min())
        vmax_biomasa = float(todos_biomasa_ha.max())

    numero_mapa = 10

    for conf, etiqueta in EXPERIMENTOS.items():
        mapa_rectangulos_utm(
            df,
            COLUMNAS_BIOMASA_HA[conf],
            f"Biomasa estimada por unidad — confianza {etiqueta}",
            "Biomasa de gimnospermas (t/ha arbolada)",
            out_dir / (
                f"{numero_mapa:02d}_mapa_biomasa_"
                f"{conf}_t_ha_arbolada"
            ),
            vmin=vmin_biomasa,
            vmax=vmax_biomasa,
            mostrar_codigo=True,
        )
        numero_mapa += 1

    print("[INFO] Generando mapa de sensibilidad...")
    mapa_rectangulos_utm(
        df,
        "sensibilidad_relativa_pct",
        "Sensibilidad de la biomasa al umbral YOLO",
        "Sensibilidad relativa (%)",
        out_dir / "14_mapa_sensibilidad_relativa",
        mostrar_codigo=True,
    )

    mapa_rectangulos_utm(
        df,
        "diferencia_conf010_conf025_t_ha",
        "Diferencia de biomasa entre confianza 10 % y 25 %",
        "Diferencia (t/ha arbolada)",
        out_dir / "15_mapa_diferencia_conf010_conf025",
        mostrar_codigo=True,
    )

    print("[INFO] Generando ranking y selección de recortes...")
    ranking = ranking_sensibilidad(df, out_dir)
    ranking.to_csv(
        out_dir / "ranking_recortes_sensibilidad.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seleccion = seleccionar_recortes_para_revision(df)
    seleccion.to_csv(
        out_dir / "recortes_seleccionados_para_revision.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grafico_recortes_seleccionados(
        df,
        seleccion,
        out_dir,
    )

    generar_resumen_txt(
        df,
        df_resumen,
        ranking,
        seleccion,
        out_dir / "resumen_interpretacion_visual.txt",
    )

    print("")
    print("=" * 76)
    print("[FIN] VISUALIZACIONES GENERADAS")
    print(f"[OK] Carpeta: {out_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
