"""Evaluation harness - accuracy, per-class precision/recall/F1, confusion matrix.

Pandas only, no sklearn (per project constraint). Every ratio is guarded
against division by zero so metrics are always plain floats, never NaN.
"""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from config import BASE, DATA_DIR
from schema import RequestType
from classifier import classify

LABELED_CSV = DATA_DIR / "labeled_requests.csv"
REPORT_PATH = BASE / "eval_report.md"

ALL_CLASSES = [rt.value for rt in RequestType]


def compute_metrics(y_true: list, y_pred: list) -> dict:
    y_true = [str(t) for t in y_true]
    y_pred = [str(p) for p in y_pred]
    n = len(y_true)

    accuracy = (
        sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n > 0 else 0.0
    )

    true_series = pd.Series(y_true, dtype="object")
    pred_series = pd.Series(y_pred, dtype="object")

    if n > 0:
        cm = pd.crosstab(true_series, pred_series)
    else:
        cm = pd.DataFrame()
    cm = cm.reindex(index=ALL_CLASSES, columns=ALL_CLASSES, fill_value=0)

    rows = []
    for c in ALL_CLASSES:
        tp = int(((true_series == c) & (pred_series == c)).sum())
        fp = int(((true_series != c) & (pred_series == c)).sum())
        fn = int(((true_series == c) & (pred_series != c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append({"class": c, "precision": precision, "recall": recall, "f1": f1})

    per_class = pd.DataFrame(rows).set_index("class")

    return {"accuracy": accuracy, "confusion_matrix": cm, "per_class": per_class}


def run_evaluation(
    llm_call: Optional[Callable[[str, str], str]] = None,
    csv_path: Path = LABELED_CSV,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return compute_metrics([], []) | {"predictions": []}

    y_true = []
    y_pred = []
    predictions = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row.get("text", ""))
        true_type = str(row.get("true_type", ""))
        result = classify(text, llm_call=llm_call)
        y_true.append(true_type)
        y_pred.append(result.request_type.value)
        predictions.append(
            {
                "text": text,
                "true_type": true_type,
                "predicted_type": result.request_type.value,
                "confidence": result.confidence,
            }
        )
        if progress is not None:
            progress(i + 1, total)

    metrics = compute_metrics(y_true, y_pred)
    metrics["predictions"] = predictions
    return metrics


def _df_to_markdown_table(df: pd.DataFrame, index_label: str = "") -> str:
    headers = [index_label] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export_report(results: dict, path: Path = REPORT_PATH) -> None:
    lines = ["# TriageSense Evaluation Report", ""]
    lines.append(f"**Accuracy:** {results.get('accuracy', 0.0):.2%}")
    lines.append("")
    per_class = results.get("per_class")
    if per_class is not None:
        lines.append("## Per-class Precision / Recall / F1")
        lines.append("")
        lines.append(_df_to_markdown_table(per_class.round(3), index_label="class"))
        lines.append("")
    cm = results.get("confusion_matrix")
    if cm is not None:
        lines.append("## Confusion Matrix (rows=true, columns=predicted)")
        lines.append("")
        lines.append(_df_to_markdown_table(cm, index_label="true \\ pred"))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
