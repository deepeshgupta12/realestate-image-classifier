from pathlib import Path
import csv
from datetime import datetime
from typing import Dict, Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "predictions.csv"

CSV_FIELDS = [
    "timestamp",
    "filename",
    "parent",
    "child",
    "confidence",
    "action",
    "source",
    "width",
    "height",
    "mode",
    "model_version",
]

def append_log(row: Dict[str, Any]) -> None:
    row_out = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        **{k: row.get(k) for k in CSV_FIELDS if k != "timestamp"},
    }
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row_out)