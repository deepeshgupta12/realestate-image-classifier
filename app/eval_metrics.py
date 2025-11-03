from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def load_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"filename", "parent", "child"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground truth missing columns: {missing}")
    # normalized label column
    df["gt_label"] = df["parent"].astype(str).str.strip() + " ▸ " + df["child"].astype(str).str.strip()
    # only keep needed columns
    return df[["filename", "gt_label"]]

def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else None
    if df is None:
        raise ValueError("Predictions must be a CSV (logs/predictions.csv).")
    required = {"filename", "parent", "child", "confidence", "action", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    df["pred_label"] = df["parent"].astype(str).str.strip() + " ▸ " + df["child"].astype(str).str.strip()
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True, help="Path to ground_truth.csv")
    p.add_argument("--preds", default=str(Path("logs") / "predictions.csv"),
                   help="Path to logs/predictions.csv")
    p.add_argument("--out", default="report.txt", help="Where to write the report")
    args = p.parse_args()

    truth = load_truth(Path(args.truth))
    preds = load_preds(Path(args.preds))

    # join on filename
    df = preds.merge(truth, on="filename", how="inner")
    if df.empty:
        raise SystemExit("No filename overlap between truth and predictions. " 
                         "Ensure filenames in ground truth match uploaded file names exactly.")

    y_true = df["gt_label"].tolist()
    y_pred = df["pred_label"].tolist()

    # metrics
    report = classification_report(y_true, y_pred, digits=3)
    labels_sorted = sorted(sorted(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    # action quality: how precise is auto_publish?
    auto_df = df[df["action"] == "auto_publish"]
    autopub_prec = None
    if not auto_df.empty:
        autopub_prec = (auto_df["pred_label"] == auto_df["gt_label"]).mean()

    # per-parent accuracy (optional view)
    df["gt_parent"] = df["gt_label"].str.split(" ▸ ").str[0]
    df["pred_parent"] = df["pred_label"].str.split(" ▸ ").str[0]
    parent_acc = df.assign(ok=(df["gt_label"] == df["pred_label"])).groupby("gt_parent")["ok"].mean().sort_values(ascending=False)

    # write report
    out_path = Path(args.out)
    with out_path.open("w") as f:
        f.write("# Real Estate Classifier — Evaluation Report\n\n")
        f.write("## Overall classification report (Parent ▸ Child)\n")
        f.write(report + "\n\n")
        if autopub_prec is not None:
            f.write(f"## Auto-publish precision (@ current thresholds): {autopub_prec:.3f}\n\n")
        else:
            f.write("## Auto-publish precision: N/A (no rows auto-published in log)\n\n")

        f.write("## Per-parent accuracy (based on exact child match)\n")
        for parent, acc in parent_acc.items():
            f.write(f"- {parent}: {acc:.3f}\n")
        f.write("\n")

        f.write("## Confusion Matrix (labels in order)\n")
        f.write(", ".join(labels_sorted) + "\n")
        for row in cm:
            f.write(", ".join(str(int(x)) for x in row) + "\n")

    print(f"Saved evaluation to {out_path}")
    print("Tip: If auto-publish precision < 0.95, raise the confidence threshold or funnel more samples to training.")

if __name__ == "__main__":
    main()