from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from .model import RealEstateZeroShotClassifier

app = FastAPI(title="Real Estate Image Classifier", version="0.5.0")

BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = BASE_DIR / "taxonomy" / "taxonomy.yaml"
PROMPTS_PATH = BASE_DIR / "taxonomy" / "prompts.yaml"

with open(TAXONOMY_PATH, "r") as f:
    TAXONOMY: Dict[str, Any] = yaml.safe_load(f)

classifier = RealEstateZeroShotClassifier(
    taxonomy_path=TAXONOMY_PATH,
    prompts_path=PROMPTS_PATH,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "taxonomy_loaded": bool(TAXONOMY),
        "num_labels": len(classifier.labels),
        "model": "open-clip ViT-B-32 (zero-shot, domain prompts)",
    }


def load_image_from_upload(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(file_bytes))
        img.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    img = load_image_from_upload(file_bytes)

    pred = classifier.predict(img)
    conf = pred["confidence"]
    parent = pred["parent"]
    child = pred["child"]

    # business rules
    # 1) plans are usually higher priority if predicted
    if parent.startswith("Plans") and conf >= 0.6:
        action = "auto_publish"
    # 2) marketing / junk needs review even if high
    elif parent.startswith("Marketing") and conf >= 0.6:
        action = "needs_review"
    # 3) generic rule
    else:
        if conf >= 0.8:
            action = "auto_publish"
        elif conf >= 0.6:
            action = "needs_review"
        else:
            action = "reject"

    return JSONResponse(
        {
            "filename": file.filename,
            "image_info": {"width": img.width, "height": img.height, "mode": img.mode},
            "parent": parent,
            "child": child,
            "grandchild": None,
            "confidence": conf,
            "alternates": pred["alternates"],
            "action": action,
            "model_version": "zero-shot-v1-prompts",
        }
    )