from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from .model import RealEstateZeroShotClassifier, RealEstateLinearHead
from .utils import append_log

app = FastAPI(title="Real Estate Image Classifier", version="0.7.0")

BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = BASE_DIR / "taxonomy" / "taxonomy.yaml"
PROMPTS_PATH = BASE_DIR / "taxonomy" / "prompts.yaml"
LINEAR_WEIGHTS = BASE_DIR / "models" / "clip_linear_realestate.pt"

with open(TAXONOMY_PATH, "r") as f:
    TAXONOMY: Dict[str, Any] = yaml.safe_load(f)

zc = RealEstateZeroShotClassifier(taxonomy_path=TAXONOMY_PATH, prompts_path=PROMPTS_PATH)

linear_head: Optional[RealEstateLinearHead] = None
try:
    if LINEAR_WEIGHTS.exists():
        linear_head = RealEstateLinearHead(LINEAR_WEIGHTS)
except Exception as e:
    print(f"[WARN] Linear head not loaded: {e}")
    linear_head = None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "taxonomy_loaded": bool(TAXONOMY),
        "num_labels_zero_shot": len(zc.labels),
        "zero_shot_model": "open-clip ViT-B-32 (domain prompts)",
        "linear_head_loaded": bool(linear_head is not None),
        "linear_head_classes": getattr(linear_head, "class_names", []),
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


def decide_action(parent: str, conf: float) -> str:
    if parent.startswith("Plans") and conf >= 0.60:
        return "auto_publish"
    if parent.startswith("Marketing") and conf >= 0.60:
        return "needs_review"
    if conf >= 0.80:
        return "auto_publish"
    if conf >= 0.60:
        return "needs_review"
    return "reject"


def classify_pil_image(img: Image.Image, filename: str) -> Dict[str, Any]:
    # 1) linear head first
    lin_pred = None
    if linear_head is not None:
        lin_pred = linear_head.predict(img)

    # 2) zero-shot generalist
    zc_pred = zc.predict(img)

    # choose
    if lin_pred and lin_pred["confidence"] >= 0.80:
        final = {
            "parent": lin_pred["parent"],
            "child": lin_pred["child"],
            "confidence": lin_pred["confidence"],
            "alternates": lin_pred["alternates"],
            "source": "linear_head",
        }
    else:
        final = {
            "parent": zc_pred["parent"],
            "child": zc_pred["child"],
            "confidence": zc_pred["confidence"],
            "alternates": zc_pred["alternates"],
            "source": "zero_shot",
        }

    action = decide_action(final["parent"], final["confidence"])

    # log
    append_log(
        {
            "filename": filename,
            "parent": final["parent"],
            "child": final["child"],
            "confidence": final["confidence"],
            "action": action,
            "source": final["source"],
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "model_version": "hybrid-v1",
        }
    )

    return {
        "filename": filename,
        "image_info": {"width": img.width, "height": img.height, "mode": img.mode},
        "parent": final["parent"],
        "child": final["child"],
        "grandchild": None,
        "confidence": final["confidence"],
        "alternates": final["alternates"],
        "action": action,
        "model_version": "hybrid-v1",
        "decision_source": final["source"],
    }


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    img = load_image_from_upload(file_bytes)
    result = classify_pil_image(img, file.filename)
    return JSONResponse(result)


@app.post("/classify-batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    results: List[Dict[str, Any]] = []
    for uf in files:
        try:
            file_bytes = await uf.read()
            img = load_image_from_upload(file_bytes)
            res = classify_pil_image(img, uf.filename)
            results.append(res)
        except HTTPException as e:
            results.append(
                {
                    "filename": uf.filename,
                    "error": True,
                    "detail": e.detail,
                }
            )
    return JSONResponse({"results": results})