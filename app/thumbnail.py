from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Set
import math
import numpy as np
from PIL import Image
import imagehash
import cv2  # opencv(-contrib)-python-headless or opencv-contrib-python-headless


@dataclass
class ThumbScore:
    filename: str
    score: float
    parts: Dict[str, float]
    parent: str
    child: str
    conf: float
    crop: Tuple[int, int, int, int]  # x, y, w, h
    policy_fallback: bool = False    # True if we relaxed size/bans to return a best-effort


def _ratio_wh(rtxt: str) -> float:
    w, h = rtxt.split(":")
    return float(w) / float(h)


def _sharpness(img: np.ndarray) -> float:
    var = cv2.Laplacian(img, cv2.CV_64F).var()
    return float(1 - math.exp(-var / 200.0))  # ~0..1


def _brightness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.clip(gray.mean() / 255.0, 0, 1))


def _contrast(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.clip(gray.std() / 64.0, 0, 1))


def _composition_saliency(img: np.ndarray) -> float:
    """Prefer focus near rule-of-thirds intersections (uses saliency if available)."""
    try:
        if hasattr(cv2, "saliency"):
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, salmap = sal.computeSaliency(img)
            if ok:
                salmap = (salmap * 255).astype("uint8")
                h, w = salmap.shape
                thirds = [
                    (int(w / 3), int(h / 3)),
                    (int(2 * w / 3), int(h / 3)),
                    (int(w / 3), int(2 * h / 3)),
                    (int(2 * w / 3), int(2 * h / 3)),
                ]
                r = max(6, min(w, h) // 15)
                vals = []
                for (cx, cy) in thirds:
                    patch = salmap[max(0, cy - r): min(h, cy + r),
                                   max(0, cx - r): min(w, cx + r)]
                    if patch.size:
                        vals.append(patch.mean() / 255.0)
                return float(np.mean(vals)) if vals else 0.5
    except Exception:
        pass
    # Fallback: edge density
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    return float(np.clip(edges.mean() / 128.0, 0, 1))


def _face_penalty(img: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.2, 6)
        return 1.0 if len(faces) > 0 else 0.0
    except Exception:
        return 0.0


def _phash(img_pil: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(img_pil)


def _duplicate_penalty(prev_hashes: List[imagehash.ImageHash],
                       current_hash: imagehash.ImageHash,
                       threshold: int = 4) -> float:
    """Penalty if near-duplicate of any prior kept image."""
    for h in prev_hashes:
        if abs(current_hash - h) <= threshold:
            return 1.0
    return 0.0


def _smart_crop(img: np.ndarray, ratio: float) -> Tuple[int, int, int, int]:
    """Center crop to target W:H ratio; returns (x, y, w, h) inside original."""
    h, w = img.shape[:2]
    target_w = w
    target_h = int(round(w / ratio))
    if target_h > h:
        target_h = h
        target_w = int(round(h * ratio))
    x = (w - target_w) // 2
    y = (h - target_h) // 2
    return (x, y, target_w, target_h)


def choose_thumbnail(
    candidates: List[Dict[str, Any]],
    rules: Dict[str, Any],
) -> ThumbScore:
    """
    candidates items must contain: filename, pil (PIL.Image), parent, child, confidence
    rules from taxonomy/thumbnail_rules.yaml
    """

    def _is_banned(p: str, c: str, banned: Set[str]) -> bool:
        label = f"{p} ▸ {c}"
        if label in banned:
            return True
        ch = (c or "").lower()
        # defensive synonyms to ensure bathrooms never pass
        return any(k in ch for k in ["bathroom", "washroom", "toilet", "powder room", "restroom"])

    # ---- config
    aspect = rules.get("aspect_ratio", "4:3")
    ar = _ratio_wh(aspect)
    min_w, min_h = rules.get("min_resolution", [900, 675])
    weights = rules.get("weights", {})
    class_w = weights.get("class", 0.55)
    qual_w = weights.get("quality", 0.30)
    comp_w = weights.get("composition", 0.15)
    pen = rules.get("penalties", {})
    low_res_pen = pen.get("low_res", 0.25)  # used when we relax size
    class_weights = rules.get("class_weights", {})
    min_quality = rules.get("thresholds", {}).get("min_quality_score", 0.35)
    banned: Set[str] = set(rules.get("banned_labels", []))

    def _score_all(skip_banned: bool, relax_size: bool = False) -> List[ThumbScore]:
        scores: List[ThumbScore] = []
        kept_hashes: List[imagehash.ImageHash] = []

        for c in candidates:
            p, child_name = c["parent"], c["child"]
            if skip_banned and _is_banned(p, child_name, banned):
                continue

            pil: Image.Image = c["pil"].convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            H, W = img.shape[:2]

            # crop and size gate
            cx, cy, cw, ch = _smart_crop(img, ar)
            size_pen = 0.0
            if not relax_size:
                if cw < min_w or ch < min_h:
                    continue
            else:
                shortfall_w = max(0.0, (min_w - cw) / max(min_w, 1))
                shortfall_h = max(0.0, (min_h - ch) / max(min_h, 1))
                size_pen = max(shortfall_w, shortfall_h)

            # quality & composition
            sharp = _sharpness(img)
            bright = _brightness(img)
            contr = _contrast(img)
            comp = _composition_saliency(img)

            too_dark = float(bright < 0.35)
            too_bright = float(bright > 0.85)
            blur = float(sharp < 0.35)
            facep = _face_penalty(img)

            # duplicate check against previously kept items
            ph = _phash(pil)
            dupp = _duplicate_penalty(kept_hashes, ph)
            kept_hashes.append(ph)

            portr = float(H > W)  # prefer landscape for listing cards

            # overall quality 0..1
            quality = float(np.clip(0.45 * sharp + 0.25 * contr + 0.30 * (1.0 - abs(bright - 0.55) / 0.55), 0, 1))

            label = f"{p} ▸ {child_name}"
            cls_w = float(class_weights.get(label, 0.5))

            raw = class_w * cls_w + qual_w * quality + comp_w * comp
            penalty = (
                pen.get("face_present", 0.15) * facep
                + pen.get("too_dark", 0.10) * too_dark
                + pen.get("too_bright", 0.10) * too_bright
                + pen.get("blur", 0.25) * blur
                + pen.get("duplicate", 0.30) * dupp
                + pen.get("portrait_orientation", 0.10) * portr
                + low_res_pen * size_pen
            )
            final = max(0.0, raw - penalty)
            if quality < min_quality:
                final *= 0.5

            scores.append(
                ThumbScore(
                    filename=c["filename"],
                    score=float(final),
                    parts={
                        "class_weight": cls_w,
                        "quality": quality,
                        "composition": float(comp),
                        "penalty": float(penalty),
                    },
                    parent=p,
                    child=child_name,
                    conf=float(c["confidence"]),
                    crop=(cx, cy, cw, ch),
                )
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    # 1) Strict: skip banned, enforce min size
    scores = _score_all(skip_banned=True, relax_size=False)
    if scores:
        return scores[0]

    # 2) Soft-size fallback: skip banned, allow low-res with penalty
    scores = _score_all(skip_banned=True, relax_size=True)
    if scores:
        best = scores[0]
        best.policy_fallback = True
        return best

    # 3) Final fallback: allow everything (incl. banned) with low-res penalty
    scores = _score_all(skip_banned=False, relax_size=True)
    if scores:
        best = scores[0]
        best.policy_fallback = True
        return best

    # 4) Nothing usable
    return ThumbScore(
        filename="",
        score=0.0,
        parts={"class_weight": 0.0, "quality": 0.0, "composition": 0.0, "penalty": 0.0},
        parent="UNKNOWN",
        child=None,
        conf=0.0,
        crop=(0, 0, 0, 0),
        policy_fallback=True,
    )