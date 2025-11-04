from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from .model import RealEstateZeroShotClassifier, RealEstateLinearHead
from .utils import append_log
from .thumbnail import choose_thumbnail

app = FastAPI(title="Real Estate Image Classifier", version="0.8.0")

BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = BASE_DIR / "taxonomy" / "taxonomy.yaml"
PROMPTS_PATH = BASE_DIR / "taxonomy" / "prompts.yaml"
LINEAR_WEIGHTS = BASE_DIR / "models" / "clip_linear_realestate.pt"
THUMB_RULES_PATH = BASE_DIR / "taxonomy" / "thumbnail_rules.yaml"

# Load taxonomy/prompts
with open(TAXONOMY_PATH, "r") as f:
    TAXONOMY: Dict[str, Any] = yaml.safe_load(f)

with open(PROMPTS_PATH, "r") as f:
    PROMPTS: Dict[str, Any] = yaml.safe_load(f)

# Load thumbnail rules (with banned labels for Bathroom, etc.)
try:
    with open(THUMB_RULES_PATH, "r") as f:
        THUMB_RULES: Dict[str, Any] = yaml.safe_load(f) or {}
except FileNotFoundError:
    THUMB_RULES = {}

# Load classifiers
zc = RealEstateZeroShotClassifier(
    taxonomy_path=TAXONOMY_PATH,
    prompts_path=PROMPTS_PATH,
)

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
        "thumb_rules_loaded": bool(THUMB_RULES),
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
    """
    Business rules for CMS.
    - Plans >= 0.60 → auto_publish
    - Marketing >= 0.60 → needs_review
    - Otherwise: >=0.80 auto_publish; >=0.60 needs_review; else reject
    """
    if parent.startswith("Plans") and conf >= 0.60:
        return "auto_publish"
    if parent.startswith("Marketing") and conf >= 0.60:
        return "needs_review"
    if conf >= 0.80:
        return "auto_publish"
    if conf >= 0.60:
        return "needs_review"
    return "reject"


def hybrid_predict(img: Image.Image) -> Dict[str, Any]:
    """
    Use linear head first (if confident), otherwise fallback to zero-shot.
    Returns dict with parent/child/conf/alternates/source.
    """
    lin_pred = None
    if linear_head is not None:
        lin_pred = linear_head.predict(img)

    zc_pred = zc.predict(img)

    if lin_pred and lin_pred["confidence"] >= 0.80:
        return {
            "parent": lin_pred["parent"],
            "child": lin_pred["child"],
            "confidence": lin_pred["confidence"],
            "alternates": lin_pred["alternates"],
            "source": "linear_head",
        }
    else:
        return {
            "parent": zc_pred["parent"],
            "child": zc_pred["child"],
            "confidence": zc_pred["confidence"],
            "alternates": zc_pred["alternates"],
            "source": "zero_shot",
        }


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    img = load_image_from_upload(file_bytes)

    pred = hybrid_predict(img)
    action = decide_action(pred["parent"], pred["confidence"])

    # CSV log
    append_log(
        {
            "filename": file.filename,
            "parent": pred["parent"],
            "child": pred["child"],
            "confidence": pred["confidence"],
            "action": action,
            "source": pred["source"],
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "model_version": "hybrid-v1",
        }
    )

    return JSONResponse(
        {
            "filename": file.filename,
            "image_info": {"width": img.width, "height": img.height, "mode": img.mode},
            "parent": pred["parent"],
            "child": pred["child"],
            "grandchild": None,
            "confidence": pred["confidence"],
            "alternates": pred["alternates"],
            "action": action,
            "model_version": "hybrid-v1",
            "decision_source": pred["source"],
        }
    )


@app.post("/classify-batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    results: List[Dict[str, Any]] = []
    for uf in files:
        try:
            file_bytes = await uf.read()
            img = load_image_from_upload(file_bytes)
            pred = hybrid_predict(img)
            action = decide_action(pred["parent"], pred["confidence"])

            append_log(
                {
                    "filename": uf.filename,
                    "parent": pred["parent"],
                    "child": pred["child"],
                    "confidence": pred["confidence"],
                    "action": action,
                    "source": pred["source"],
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "model_version": "hybrid-v1",
                }
            )

            results.append(
                {
                    "filename": uf.filename,
                    "image_info": {"width": img.width, "height": img.height, "mode": img.mode},
                    "parent": pred["parent"],
                    "child": pred["child"],
                    "grandchild": None,
                    "confidence": pred["confidence"],
                    "alternates": pred["alternates"],
                    "action": action,
                    "model_version": "hybrid-v1",
                    "decision_source": pred["source"],
                }
            )
        except HTTPException as e:
            results.append({"filename": uf.filename, "error": True, "detail": e.detail})
    return JSONResponse({"results": results})


@app.post("/choose-thumbnail")
async def choose_thumbnail_endpoint(files: List[UploadFile] = File(...)):
    """
    Select the best listing thumbnail from a set of images.
    Enforces a hard 'ban' on Bathroom/Washroom/Toilet/etc. via thumbnail_rules.yaml.
    If all are banned, returns the best anyway with policy_fallback=True.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    candidates: List[Dict[str, Any]] = []
    for uf in files:
        file_bytes = await uf.read()
        img = load_image_from_upload(file_bytes)

        pred = hybrid_predict(img)
        candidates.append(
            {
                "filename": uf.filename,
                "pil": img,
                "parent": pred["parent"],
                "child": pred["child"],
                "confidence": pred["confidence"],
            }
        )

    best = choose_thumbnail(candidates, THUMB_RULES or {})

    return {
        "selected": {
            "filename": best.filename,
            "parent": best.parent,
            "child": best.child,
            "confidence": best.conf,
            "score": best.score,
            "components": best.parts,
            "suggested_crop_xywh": best.crop,   # apply on original to produce 4:3 card
            "policy_fallback": best.policy_fallback,
        },
        "alternates": sorted(
            [
                {
                    "filename": c["filename"],
                    "parent": c["parent"],
                    "child": c["child"],
                    "confidence": c["confidence"],
                }
                for c in candidates
            ],
            key=lambda x: (x["confidence"], x["filename"]),
            reverse=True,
        )[:4],
    }