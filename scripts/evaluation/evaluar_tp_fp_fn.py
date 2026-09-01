import os
import csv
from pathlib import Path


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

GT_LABELS_DIR = REPO_ROOT / "data" / "dataset" / "labels" / "val"
PREDICTIONS_ROOT = REPO_ROOT / "data" / "evaluation_predictions"
OUTPUT_DIR = REPO_ROOT / "outputs" / "evaluation"

# Umbrales IoU:
# 0.50 = criterio estándar habitual en detección
# 0.30 = criterio complementario más flexible para copas densas o cajas no perfectamente ajustadas
IOU_THRESHOLDS = [0.50, 0.30]

# Experimentos evaluados en la memoria.
EXPERIMENTS = [
    {
        "name": "baseline_original_img1280_conf025",
        "labels": PREDICTIONS_ROOT / "baseline_original_img1280_conf025" / "labels",
    },
    {
        "name": "cpu_best_conf025",
        "labels": PREDICTIONS_ROOT / "cpu_best_conf025" / "labels",
    },
    {
        "name": "cpu_best_conf040",
        "labels": PREDICTIONS_ROOT / "cpu_best_conf040" / "labels",
    },
    {
        "name": "gpu_best_conf025_maxdet1000",
        "labels": PREDICTIONS_ROOT / "gpu_best_conf025_maxdet1000" / "labels",
    },
    {
        "name": "gpu_best_conf040_maxdet1000",
        "labels": PREDICTIONS_ROOT / "gpu_best_conf040_maxdet1000" / "labels",
    },
    {
        "name": "gpu_last_conf025_maxdet1000",
        "labels": PREDICTIONS_ROOT / "gpu_last_conf025_maxdet1000" / "labels",
    },
    {
        "name": "gpu_last_conf030_maxdet1000",
        "labels": PREDICTIONS_ROOT / "gpu_last_conf030_maxdet1000" / "labels",
    },
]

# ============================================================
# FUNCIONES DE LECTURA
# ============================================================

def read_yolo_file(path, has_conf=False):
    """
    Lee un archivo YOLO.

    Ground truth:
        class x_center y_center width height

    Predicción con save_conf=True:
        class x_center y_center width height confidence

    Devuelve lista de diccionarios:
        {
            "cls": int,
            "x": float,
            "y": float,
            "w": float,
            "h": float,
            "conf": float or None
        }
    """
    boxes = []

    if not path.exists():
        return boxes

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            conf = None
            if has_conf and len(parts) >= 6:
                conf = float(parts[5])

            boxes.append({
                "cls": cls,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
            })

        except ValueError:
            continue

    return boxes


def yolo_to_xyxy(box):
    """
    Convierte YOLO normalizado:
        x_center, y_center, width, height

    a:
        x1, y1, x2, y2

    Se mantiene en coordenadas normalizadas, por lo que no hace falta conocer
    el tamaño en píxeles de la imagen. Esto es válido porque GT y predicciones
    están normalizadas sobre la misma imagen.
    """
    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return x1, y1, x2, y2


# ============================================================
# IOU Y MATCHING
# ============================================================

def compute_iou(box_a, box_b):
    """
    Calcula IoU entre dos cajas YOLO normalizadas.
    """
    ax1, ay1, ax2, ay2 = yolo_to_xyxy(box_a)
    bx1, by1, bx2, by2 = yolo_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def evaluate_image(gt_boxes, pred_boxes, iou_threshold):
    """
    Evalúa una imagen mediante matching greedy:

    1. Ordena predicciones por confianza de mayor a menor.
    2. Para cada predicción, busca el GT no usado con mayor IoU.
    3. Si IoU >= threshold y la clase coincide, cuenta como TP.
    4. Si no encuentra match, cuenta como FP.
    5. Los GT no emparejados son FN.

    Esto evita que varias predicciones coincidan con el mismo árbol real.
    """
    matched_gt = set()
    tp = 0
    fp = 0

    # Si no hay confianza, se deja a 0.0. En tus predicciones sí debería haberla.
    sorted_preds = sorted(
        pred_boxes,
        key=lambda b: b["conf"] if b["conf"] is not None else 0.0,
        reverse=True
    )

    ious_tp = []

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            # Matching por clase. En este TFG solo debería existir clase 0 = Tree.
            if pred["cls"] != gt["cls"]:
                continue

            iou = compute_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious_tp.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou_tp = sum(ious_tp) / len(ious_tp) if ious_tp else 0.0

    return {
        "gt": len(gt_boxes),
        "pred": len(pred_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_tp": mean_iou_tp,
    }


# ============================================================
# UTILIDADES DE SALIDA
# ============================================================

def safe_float(value):
    return round(float(value), 6)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, title):
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)

    header = (
        f"{'Experimento':38s} "
        f"{'IoU':>5s} "
        f"{'GT':>6s} "
        f"{'Pred':>6s} "
        f"{'TP':>6s} "
        f"{'FP':>6s} "
        f"{'FN':>6s} "
        f"{'Prec':>8s} "
        f"{'Recall':>8s} "
        f"{'F1':>8s}"
    )

    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['experiment']:38s} "
            f"{r['iou_threshold']:5.2f} "
            f"{r['gt']:6d} "
            f"{r['pred']:6d} "
            f"{r['tp']:6d} "
            f"{r['fp']:6d} "
            f"{r['fn']:6d} "
            f"{r['precision']:8.4f} "
            f"{r['recall']:8.4f} "
            f"{r['f1']:8.4f}"
        )


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GT_LABELS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de GT: {GT_LABELS_DIR}")

    gt_files = sorted(GT_LABELS_DIR.glob("*.txt"))

    if not gt_files:
        raise RuntimeError(f"No se encontraron .txt de ground truth en: {GT_LABELS_DIR}")

    all_detail_rows = []
    all_summary_rows = []

    print("")
    print("[INFO] Evaluación TP / FP / FN")
    print(f"[INFO] GT labels: {GT_LABELS_DIR}")
    print(f"[INFO] Salida:    {OUTPUT_DIR}")
    print(f"[INFO] Imágenes GT encontradas: {len(gt_files)}")

    for iou_threshold in IOU_THRESHOLDS:
        detail_rows_iou = []
        summary_rows_iou = []

        print("")
        print(f"[INFO] Evaluando con IoU >= {iou_threshold:.2f}")

        for exp in EXPERIMENTS:
            exp_name = exp["name"]
            pred_labels_dir = exp["labels"]

            if not pred_labels_dir.exists():
                print(f"[WARN] No existe carpeta de predicciones para {exp_name}:")
                print(f"       {pred_labels_dir}")
                print("       Se evaluará como 0 predicciones.")
            
            total_gt = 0
            total_pred = 0
            total_tp = 0
            total_fp = 0
            total_fn = 0
            weighted_iou_sum = 0.0
            weighted_iou_count = 0

            for gt_file in gt_files:
                image_name = gt_file.stem

                pred_file = pred_labels_dir / f"{image_name}.txt"

                gt_boxes = read_yolo_file(gt_file, has_conf=False)
                pred_boxes = read_yolo_file(pred_file, has_conf=True)

                result = evaluate_image(gt_boxes, pred_boxes, iou_threshold)

                detail_row = {
                    "experiment": exp_name,
                    "iou_threshold": iou_threshold,
                    "image": image_name,
                    "gt": result["gt"],
                    "pred": result["pred"],
                    "tp": result["tp"],
                    "fp": result["fp"],
                    "fn": result["fn"],
                    "precision": safe_float(result["precision"]),
                    "recall": safe_float(result["recall"]),
                    "f1": safe_float(result["f1"]),
                    "mean_iou_tp": safe_float(result["mean_iou_tp"]),
                    "pred_labels_dir": str(pred_labels_dir),
                }

                detail_rows_iou.append(detail_row)
                all_detail_rows.append(detail_row)

                total_gt += result["gt"]
                total_pred += result["pred"]
                total_tp += result["tp"]
                total_fp += result["fp"]
                total_fn += result["fn"]

                if result["tp"] > 0:
                    weighted_iou_sum += result["mean_iou_tp"] * result["tp"]
                    weighted_iou_count += result["tp"]

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            mean_iou_tp = weighted_iou_sum / weighted_iou_count if weighted_iou_count > 0 else 0.0

            summary_row = {
                "experiment": exp_name,
                "iou_threshold": iou_threshold,
                "gt": total_gt,
                "pred": total_pred,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": safe_float(precision),
                "recall": safe_float(recall),
                "f1": safe_float(f1),
                "mean_iou_tp": safe_float(mean_iou_tp),
                "pred_labels_dir": str(pred_labels_dir),
            }

            summary_rows_iou.append(summary_row)
            all_summary_rows.append(summary_row)

        # CSV separados por umbral IoU
        detail_path = OUTPUT_DIR / f"detalle_por_imagen_iou_{str(iou_threshold).replace('.', '')}.csv"
        summary_path = OUTPUT_DIR / f"resumen_experimentos_iou_{str(iou_threshold).replace('.', '')}.csv"

        write_csv(
            detail_path,
            detail_rows_iou,
            fieldnames=[
                "experiment",
                "iou_threshold",
                "image",
                "gt",
                "pred",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_iou_tp",
                "pred_labels_dir",
            ],
        )

        write_csv(
            summary_path,
            summary_rows_iou,
            fieldnames=[
                "experiment",
                "iou_threshold",
                "gt",
                "pred",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_iou_tp",
                "pred_labels_dir",
            ],
        )

        print_table(summary_rows_iou, f"RESUMEN GLOBAL - IoU >= {iou_threshold:.2f}")
        print(f"[OK] Guardado detalle: {detail_path}")
        print(f"[OK] Guardado resumen: {summary_path}")

    # CSV globales con todos los IoU
    all_detail_path = OUTPUT_DIR / "detalle_por_imagen_todos_los_iou.csv"
    all_summary_path = OUTPUT_DIR / "resumen_experimentos_todos_los_iou.csv"

    write_csv(
        all_detail_path,
        all_detail_rows,
        fieldnames=[
            "experiment",
            "iou_threshold",
            "image",
            "gt",
            "pred",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_iou_tp",
            "pred_labels_dir",
        ],
    )

    write_csv(
        all_summary_path,
        all_summary_rows,
        fieldnames=[
            "experiment",
            "iou_threshold",
            "gt",
            "pred",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_iou_tp",
            "pred_labels_dir",
        ],
    )

    print("")
    print("[FIN] Evaluación completada.")
    print(f"[OK] CSV global detalle: {all_detail_path}")
    print(f"[OK] CSV global resumen: {all_summary_path}")


if __name__ == "__main__":
    main()