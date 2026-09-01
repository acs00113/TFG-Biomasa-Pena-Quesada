import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIGURACIÓN
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_EVAL_DIR = REPO_ROOT / "results" / "detector_evaluation"
GENERATED_EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"

def _prefer_generated(filename):
    generated = GENERATED_EVAL_DIR / filename
    if generated.exists():
        return generated
    return PUBLISHED_EVAL_DIR / filename

SUMMARY_CSV = _prefer_generated("resumen_experimentos_todos_los_iou.csv")
DETAIL_CSV = _prefer_generated("detalle_por_imagen_todos_los_iou.csv")

OUTPUT_TXT = GENERATED_EVAL_DIR / "analisis_interpretado_resultados.txt"
OUTPUT_RANKING_CSV = GENERATED_EVAL_DIR / "ranking_global_interpretado.csv"
OUTPUT_BEST_BY_IMAGE_CSV = GENERATED_EVAL_DIR / "mejor_modelo_por_imagen.csv"

FRIENDLY_NAMES = {
    "baseline_original_img1280_conf025": "Baseline original (VHRTrees) conf=0.25",
    "cpu_best_conf025": "Fine-tuning CPU best.pt conf=0.25",
    "cpu_best_conf040": "Fine-tuning CPU best.pt conf=0.40",
    "gpu_best_conf025_maxdet1000": "Fine-tuning GPU best.pt conf=0.25",
    "gpu_best_conf040_maxdet1000": "Fine-tuning GPU best.pt conf=0.40",
    "gpu_last_conf025_maxdet1000": "Fine-tuning GPU last.pt conf=0.25",
    "gpu_last_conf030_maxdet1000": "Fine-tuning GPU last.pt conf=0.30",
}

BASELINE_NAME = "baseline_original_img1280_conf025"

COMPARISON_PAIRS = [
    ("cpu_best_conf025", "cpu_best_conf040", "Comparativa CPU best.pt: conf=0.25 vs conf=0.40"),
    ("gpu_best_conf025_maxdet1000", "gpu_best_conf040_maxdet1000", "Comparativa GPU best.pt: conf=0.25 vs conf=0.40"),
    ("gpu_last_conf025_maxdet1000", "gpu_last_conf030_maxdet1000", "Comparativa GPU last.pt: conf=0.25 vs conf=0.30"),
]


# ============================================================
# UTILIDADES
# ============================================================

def friendly_name(exp):
    return FRIENDLY_NAMES.get(exp, exp)


def fmt_float(x, nd=4):
    return f"{float(x):.{nd}f}"


def fmt_pct(x):
    return f"{float(x)*100:.2f}%"


def to_float(value):
    return float(value)


def to_int(value):
    return int(float(value))


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def group_by(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def get_metric(row, metric):
    if metric in {"gt", "pred", "tp", "fp", "fn"}:
        return to_int(row[metric])
    return to_float(row[metric])


def sort_rows(rows, metric, reverse=True):
    return sorted(rows, key=lambda r: get_metric(r, metric), reverse=reverse)


def safe_get_baseline(rows):
    for r in rows:
        if r["experiment"] == BASELINE_NAME:
            return r
    return None


def diff_str(a, b, metric, percentage=False):
    va = get_metric(a, metric)
    vb = get_metric(b, metric)
    diff = vb - va
    sign = "+" if diff >= 0 else ""
    if percentage:
        return f"{sign}{diff*100:.2f} puntos"
    if isinstance(diff, float) and metric not in {"gt", "pred", "tp", "fp", "fn"}:
        return f"{sign}{diff:.4f}"
    return f"{sign}{int(diff)}"


def build_summary_index(summary_rows):
    idx = {}
    for r in summary_rows:
        idx[(r["iou_threshold"], r["experiment"])] = r
    return idx


# ============================================================
# ANÁLISIS
# ============================================================

def analyze_global(summary_rows):
    """
    Devuelve:
    - text_lines: líneas de análisis global
    - ranking_rows: filas para CSV
    """
    text_lines = []
    ranking_rows = []

    by_iou = group_by(summary_rows, "iou_threshold")

    for iou in sorted(by_iou.keys(), key=float, reverse=True):
        rows = by_iou[iou]
        baseline = safe_get_baseline(rows)

        # Ranking por F1, y desempate por recall y precision
        ranked = sorted(
            rows,
            key=lambda r: (to_float(r["f1"]), to_float(r["recall"]), to_float(r["precision"])),
            reverse=True
        )

        best_f1 = max(rows, key=lambda r: to_float(r["f1"]))
        best_recall = max(rows, key=lambda r: to_float(r["recall"]))
        best_precision = max(rows, key=lambda r: to_float(r["precision"]))
        lowest_fn = min(rows, key=lambda r: to_int(r["fn"]))
        lowest_fp = min(rows, key=lambda r: to_int(r["fp"]))

        text_lines.append("")
        text_lines.append("=" * 90)
        text_lines.append(f"ANÁLISIS GLOBAL PARA IoU >= {iou}")
        text_lines.append("=" * 90)
        text_lines.append("")
        text_lines.append("1) Ranking global por F1")
        text_lines.append("")

        for pos, r in enumerate(ranked, start=1):
            exp = r["experiment"]
            line = (
                f"{pos:>2}. {friendly_name(exp)} | "
                f"F1={fmt_float(r['f1'])} | "
                f"Precision={fmt_float(r['precision'])} | "
                f"Recall={fmt_float(r['recall'])} | "
                f"TP={r['tp']} | FP={r['fp']} | FN={r['fn']}"
            )
            text_lines.append(line)

            ranking_rows.append({
                "iou_threshold": iou,
                "rank_f1": pos,
                "experiment": exp,
                "friendly_name": friendly_name(exp),
                "f1": r["f1"],
                "precision": r["precision"],
                "recall": r["recall"],
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "pred": r["pred"],
                "gt": r["gt"],
            })

        text_lines.append("")
        text_lines.append("2) Mejores experimentos por criterio")
        text_lines.append("")
        text_lines.append(f"- Mejor F1:        {friendly_name(best_f1['experiment'])} (F1={fmt_float(best_f1['f1'])})")
        text_lines.append(f"- Mayor recall:    {friendly_name(best_recall['experiment'])} (Recall={fmt_float(best_recall['recall'])})")
        text_lines.append(f"- Mayor precision: {friendly_name(best_precision['experiment'])} (Precision={fmt_float(best_precision['precision'])})")
        text_lines.append(f"- Menor FN:        {friendly_name(lowest_fn['experiment'])} (FN={lowest_fn['fn']})")
        text_lines.append(f"- Menor FP:        {friendly_name(lowest_fp['experiment'])} (FP={lowest_fp['fp']})")

        if baseline is not None:
            text_lines.append("")
            text_lines.append("3) Mejora respecto al baseline original")
            text_lines.append("")

            best_vs_baseline = best_f1
            text_lines.append(
                f"Tomando como referencia el mejor experimento por F1 "
                f"({friendly_name(best_vs_baseline['experiment'])}), la mejora frente al baseline es:"
            )
            text_lines.append(f"- Î”TP = {diff_str(baseline, best_vs_baseline, 'tp')}")
            text_lines.append(f"- Î”FP = {diff_str(baseline, best_vs_baseline, 'fp')}")
            text_lines.append(f"- Î”FN = {diff_str(baseline, best_vs_baseline, 'fn')}")
            text_lines.append(f"- Î”Precision = {diff_str(baseline, best_vs_baseline, 'precision', percentage=True)}")
            text_lines.append(f"- Î”Recall = {diff_str(baseline, best_vs_baseline, 'recall', percentage=True)}")
            text_lines.append(f"- Î”F1 = {diff_str(baseline, best_vs_baseline, 'f1', percentage=True)}")

        text_lines.append("")
        text_lines.append("4) Interpretación rápida")
        text_lines.append("")

        text_lines.append(
            f"El mejor equilibrio global para IoU >= {iou} lo ofrece "
            f"{friendly_name(best_f1['experiment'])}, al presentar el F1 más alto."
        )
        text_lines.append(
            f"Si se prioriza no dejar árboles sin detectar, el modelo con mayor recall es "
            f"{friendly_name(best_recall['experiment'])}."
        )
        text_lines.append(
            f"Si se prioriza una detección más conservadora y con menos falsas alarmas, conviene "
            f"observar especialmente la precision y el número de FP."
        )

    return text_lines, ranking_rows


def analyze_by_image(detail_rows):
    """
    Devuelve:
    - text_lines
    - best_by_image_rows
    """
    text_lines = []
    best_by_image_rows = []

    by_iou = group_by(detail_rows, "iou_threshold")

    for iou in sorted(by_iou.keys(), key=float, reverse=True):
        rows_iou = by_iou[iou]
        by_image = group_by(rows_iou, "image")

        text_lines.append("")
        text_lines.append("=" * 90)
        text_lines.append(f"ANÁLISIS POR IMAGEN PARA IoU >= {iou}")
        text_lines.append("=" * 90)

        image_difficulty = []

        for image_name in sorted(by_image.keys()):
            image_rows = by_image[image_name]

            best_f1 = max(image_rows, key=lambda r: to_float(r["f1"]))
            best_recall = max(image_rows, key=lambda r: to_float(r["recall"]))
            best_precision = max(image_rows, key=lambda r: to_float(r["precision"]))
            lowest_fn = min(image_rows, key=lambda r: to_int(r["fn"]))
            lowest_fp = min(image_rows, key=lambda r: to_int(r["fp"]))

            avg_recall = sum(to_float(r["recall"]) for r in image_rows) / len(image_rows)
            avg_f1 = sum(to_float(r["f1"]) for r in image_rows) / len(image_rows)
            avg_fn = sum(to_int(r["fn"]) for r in image_rows) / len(image_rows)
            avg_fp = sum(to_int(r["fp"]) for r in image_rows) / len(image_rows)

            image_difficulty.append({
                "image": image_name,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
                "avg_fn": avg_fn,
                "avg_fp": avg_fp,
            })

            text_lines.append("")
            text_lines.append("-" * 90)
            text_lines.append(f"Imagen: {image_name}")
            text_lines.append("-" * 90)
            text_lines.append(f"- Mejor F1:        {friendly_name(best_f1['experiment'])} | F1={fmt_float(best_f1['f1'])}")
            text_lines.append(f"- Mayor recall:    {friendly_name(best_recall['experiment'])} | Recall={fmt_float(best_recall['recall'])}")
            text_lines.append(f"- Mayor precision: {friendly_name(best_precision['experiment'])} | Precision={fmt_float(best_precision['precision'])}")
            text_lines.append(f"- Menor FN:        {friendly_name(lowest_fn['experiment'])} | FN={lowest_fn['fn']}")
            text_lines.append(f"- Menor FP:        {friendly_name(lowest_fp['experiment'])} | FP={lowest_fp['fp']}")
            text_lines.append(
                f"- Promedio entre modelos: Recall medio={avg_recall:.4f}, "
                f"F1 medio={avg_f1:.4f}, FN medio={avg_fn:.2f}, FP medio={avg_fp:.2f}"
            )

            # guardar CSV de mejores por imagen
            best_by_image_rows.append({
                "iou_threshold": iou,
                "image": image_name,
                "criterion": "best_f1",
                "experiment": best_f1["experiment"],
                "friendly_name": friendly_name(best_f1["experiment"]),
                "value": best_f1["f1"],
            })
            best_by_image_rows.append({
                "iou_threshold": iou,
                "image": image_name,
                "criterion": "best_recall",
                "experiment": best_recall["experiment"],
                "friendly_name": friendly_name(best_recall["experiment"]),
                "value": best_recall["recall"],
            })
            best_by_image_rows.append({
                "iou_threshold": iou,
                "image": image_name,
                "criterion": "best_precision",
                "experiment": best_precision["experiment"],
                "friendly_name": friendly_name(best_precision["experiment"]),
                "value": best_precision["precision"],
            })
            best_by_image_rows.append({
                "iou_threshold": iou,
                "image": image_name,
                "criterion": "lowest_fn",
                "experiment": lowest_fn["experiment"],
                "friendly_name": friendly_name(lowest_fn["experiment"]),
                "value": lowest_fn["fn"],
            })
            best_by_image_rows.append({
                "iou_threshold": iou,
                "image": image_name,
                "criterion": "lowest_fp",
                "experiment": lowest_fp["experiment"],
                "friendly_name": friendly_name(lowest_fp["experiment"]),
                "value": lowest_fp["fp"],
            })

        # dificultad de imágenes
        hardest_by_recall = min(image_difficulty, key=lambda x: x["avg_recall"])
        hardest_by_f1 = min(image_difficulty, key=lambda x: x["avg_f1"])
        most_fn = max(image_difficulty, key=lambda x: x["avg_fn"])
        most_fp = max(image_difficulty, key=lambda x: x["avg_fp"])

        text_lines.append("")
        text_lines.append("Resumen de dificultad de las imágenes:")
        text_lines.append(f"- Imagen más difícil por recall medio: {hardest_by_recall['image']} (Recall medio={hardest_by_recall['avg_recall']:.4f})")
        text_lines.append(f"- Imagen más difícil por F1 medio:     {hardest_by_f1['image']} (F1 medio={hardest_by_f1['avg_f1']:.4f})")
        text_lines.append(f"- Imagen con más FN medios:            {most_fn['image']} (FN medio={most_fn['avg_fn']:.2f})")
        text_lines.append(f"- Imagen con más FP medios:            {most_fp['image']} (FP medio={most_fp['avg_fp']:.2f})")

    return text_lines, best_by_image_rows


def analyze_comparison_pairs(summary_rows):
    text_lines = []
    by_iou = group_by(summary_rows, "iou_threshold")
    summary_index = build_summary_index(summary_rows)

    for iou in sorted(by_iou.keys(), key=float, reverse=True):
        text_lines.append("")
        text_lines.append("=" * 90)
        text_lines.append(f"COMPARATIVAS DIRIGIDAS PARA IoU >= {iou}")
        text_lines.append("=" * 90)

        for exp_a, exp_b, title in COMPARISON_PAIRS:
            row_a = summary_index.get((iou, exp_a))
            row_b = summary_index.get((iou, exp_b))

            text_lines.append("")
            text_lines.append(title)

            if row_a is None or row_b is None:
                text_lines.append("- No se ha podido realizar la comparativa porque falta algún experimento.")
                continue

            text_lines.append(f"- Modelo A: {friendly_name(exp_a)}")
            text_lines.append(
                f"  Precision={fmt_float(row_a['precision'])} | Recall={fmt_float(row_a['recall'])} | "
                f"F1={fmt_float(row_a['f1'])} | TP={row_a['tp']} | FP={row_a['fp']} | FN={row_a['fn']}"
            )
            text_lines.append(f"- Modelo B: {friendly_name(exp_b)}")
            text_lines.append(
                f"  Precision={fmt_float(row_b['precision'])} | Recall={fmt_float(row_b['recall'])} | "
                f"F1={fmt_float(row_b['f1'])} | TP={row_b['tp']} | FP={row_b['fp']} | FN={row_b['fn']}"
            )

            text_lines.append("  Diferencias (B - A):")
            text_lines.append(f"  Î”Precision = {diff_str(row_a, row_b, 'precision', percentage=True)}")
            text_lines.append(f"  Î”Recall    = {diff_str(row_a, row_b, 'recall', percentage=True)}")
            text_lines.append(f"  Î”F1        = {diff_str(row_a, row_b, 'f1', percentage=True)}")
            text_lines.append(f"  Î”TP        = {diff_str(row_a, row_b, 'tp')}")
            text_lines.append(f"  Î”FP        = {diff_str(row_a, row_b, 'fp')}")
            text_lines.append(f"  Î”FN        = {diff_str(row_a, row_b, 'fn')}")

            # interpretación básica
            prec_a = to_float(row_a["precision"])
            rec_a = to_float(row_a["recall"])
            f1_a = to_float(row_a["f1"])

            prec_b = to_float(row_b["precision"])
            rec_b = to_float(row_b["recall"])
            f1_b = to_float(row_b["f1"])

            if prec_b > prec_a and rec_b < rec_a:
                text_lines.append("  Interpretación: el modelo B es más conservador: mejora precision pero empeora recall.")
            elif prec_b < prec_a and rec_b > rec_a:
                text_lines.append("  Interpretación: el modelo B es más permisivo: recupera más árboles, pero introduce más ruido.")
            elif f1_b > f1_a:
                text_lines.append("  Interpretación: el modelo B mejora el equilibrio global medido por F1.")
            elif f1_b < f1_a:
                text_lines.append("  Interpretación: el modelo A mantiene mejor equilibrio global medido por F1.")
            else:
                text_lines.append("  Interpretación: ambos modelos ofrecen un comportamiento muy similar.")

    return text_lines


def build_final_conclusions(summary_rows):
    text_lines = []
    by_iou = group_by(summary_rows, "iou_threshold")

    text_lines.append("")
    text_lines.append("=" * 90)
    text_lines.append("CONCLUSIONES AUTOMÁTICAS")
    text_lines.append("=" * 90)

    # IoU 0.50
    rows_05 = by_iou.get("0.5", []) or by_iou.get("0.50", [])
    if rows_05:
        best_f1_05 = max(rows_05, key=lambda r: to_float(r["f1"]))
        best_recall_05 = max(rows_05, key=lambda r: to_float(r["recall"]))
        best_precision_05 = max(rows_05, key=lambda r: to_float(r["precision"]))

        text_lines.append("")
        text_lines.append("Para el criterio estricto (IoU >= 0.50):")
        text_lines.append(
            f"- El mejor experimento por F1 es {friendly_name(best_f1_05['experiment'])} "
            f"(F1={fmt_float(best_f1_05['f1'])})."
        )
        text_lines.append(
            f"- El experimento con mayor recall es {friendly_name(best_recall_05['experiment'])} "
            f"(Recall={fmt_float(best_recall_05['recall'])})."
        )
        text_lines.append(
            f"- El experimento con mayor precision es {friendly_name(best_precision_05['experiment'])} "
            f"(Precision={fmt_float(best_precision_05['precision'])})."
        )

    # IoU 0.30
    rows_03 = by_iou.get("0.3", []) or by_iou.get("0.30", [])
    if rows_03:
        best_f1_03 = max(rows_03, key=lambda r: to_float(r["f1"]))
        best_recall_03 = max(rows_03, key=lambda r: to_float(r["recall"]))
        best_precision_03 = max(rows_03, key=lambda r: to_float(r["precision"]))

        text_lines.append("")
        text_lines.append("Para el criterio flexible (IoU >= 0.30):")
        text_lines.append(
            f"- El mejor experimento por F1 es {friendly_name(best_f1_03['experiment'])} "
            f"(F1={fmt_float(best_f1_03['f1'])})."
        )
        text_lines.append(
            f"- El experimento con mayor recall es {friendly_name(best_recall_03['experiment'])} "
            f"(Recall={fmt_float(best_recall_03['recall'])})."
        )
        text_lines.append(
            f"- El experimento con mayor precision es {friendly_name(best_precision_03['experiment'])} "
            f"(Precision={fmt_float(best_precision_03['precision'])})."
        )

    text_lines.append("")
    text_lines.append("Lectura general sugerida:")
    text_lines.append(
        "- Si el objetivo principal del TFG es el conteo de árboles para estimar biomasa, "
        "conviene priorizar modelos con recall alto y menor número de FN, aunque acepten más FP."
    )
    text_lines.append(
        "- Si el objetivo es una detección visual más conservadora o una menor tasa de falsas alarmas, "
        "conviene prestar más atención a la precision y al número de FP."
    )
    text_lines.append(
        "- En este tipo de problema, la elección del umbral de confianza afecta de forma muy importante "
        "al equilibrio entre recall y precision, por lo que no debe fijarse de forma arbitraria."
    )

    return text_lines


# ============================================================
# MAIN
# ============================================================

def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"No se encuentra: {SUMMARY_CSV}")
    if not DETAIL_CSV.exists():
        raise FileNotFoundError(f"No se encuentra: {DETAIL_CSV}")

    summary_rows = read_csv(SUMMARY_CSV)
    detail_rows = read_csv(DETAIL_CSV)

    global_text, ranking_rows = analyze_global(summary_rows)
    image_text, best_by_image_rows = analyze_by_image(detail_rows)
    comparison_text = analyze_comparison_pairs(summary_rows)
    conclusions_text = build_final_conclusions(summary_rows)

    full_text = []
    full_text.append("ANÁLISIS INTERPRETADO DE RESULTADOS - TFG BIOMASA")
    full_text.append("=" * 90)
    full_text.append("")
    full_text.append("Este documento resume e interpreta automáticamente los resultados")
    full_text.append("de evaluación TP/FP/FN generados para los distintos experimentos.")
    full_text.append("")
    full_text.extend(global_text)
    full_text.extend(image_text)
    full_text.extend(comparison_text)
    full_text.extend(conclusions_text)
    full_text.append("")
    full_text.append("=" * 90)
    full_text.append("FIN DEL INFORME")
    full_text.append("=" * 90)

    GENERATED_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

    write_csv(
        OUTPUT_RANKING_CSV,
        ranking_rows,
        fieldnames=[
            "iou_threshold",
            "rank_f1",
            "experiment",
            "friendly_name",
            "f1",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "pred",
            "gt",
        ],
    )

    write_csv(
        OUTPUT_BEST_BY_IMAGE_CSV,
        best_by_image_rows,
        fieldnames=[
            "iou_threshold",
            "image",
            "criterion",
            "experiment",
            "friendly_name",
            "value",
        ],
    )

    print("")
    print("[OK] Análisis interpretado generado correctamente")
    print(f"[OK] TXT: {OUTPUT_TXT}")
    print(f"[OK] CSV ranking: {OUTPUT_RANKING_CSV}")
    print(f"[OK] CSV mejor por imagen: {OUTPUT_BEST_BY_IMAGE_CSV}")


if __name__ == "__main__":
    main()
