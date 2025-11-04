from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import open_clip
import yaml
from PIL import Image
import torch.nn as nn
from typing import Optional


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

class RealEstateLinearHead:
    """
    Uses CLIP image encoder + your trained linear layer to classify a small set of classes.
    """
    def __init__(self, weights_path: Path, device: Optional[str] = None):
        self.weights_path = weights_path
        if device is None:
            # prefer Apple Metal if available
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        # load CLIP image encoder + preprocess
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        self.model = model.to(self.device)
        self.preprocess = preprocess

        ckpt = torch.load(self.weights_path, map_location="cpu")
        self.class_names = ckpt["class_names"]  # e.g. ["interiors_kitchen", ...]
        # infer embedding dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            emb = self.model.encode_image(dummy)
        emb_dim = emb.shape[-1]

        head = nn.Linear(emb_dim, len(self.class_names))
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        self.head = head.to(self.device)

        # map your training class names to taxonomy labels
        self.class_to_taxonomy = {
            "interiors_kitchen_real": ("Interiors", "Kitchen (Real Photo)"),
            "interiors_bedroom": ("Interiors", "Bedroom"),
            "amenities_common_area": ("Amenities", "Common Area / Seating / Lobby"),
            "exteriors_building": ("Exteriors & Facade", "Building / Tower Exterior"),
            "marketing_rendered": ("Marketing / Creative", "Rendered Interior / Exterior"),
            "plans_master": ("Plans", "Master Plan"),
        }

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        img = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(img)
            logits = self.head(feats)
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=min(3, len(self.class_names)), dim=-1)

        best_idx = int(topk.indices[0, 0].cpu())
        best_prob = float(topk.values[0, 0].cpu())
        best_name = self.class_names[best_idx]
        parent, child = self.class_to_taxonomy.get(best_name, ("UNKNOWN", None))

        alternates = []
        for j in range(1, topk.indices.shape[1]):
            idx = int(topk.indices[0, j].cpu())
            prob = float(topk.values[0, j].cpu())
            name = self.class_names[idx]
            p, c = self.class_to_taxonomy.get(name, ("UNKNOWN", None))
            alternates.append(
                {"raw_label": name, "parent": p, "child": c, "confidence": prob}
            )

        return {
            "raw_label": best_name,
            "parent": parent,
            "child": child,
            "confidence": best_prob,
            "alternates": alternates,
        }
    
class RealEstateZeroShotClassifier:
    """
    Zero-shot classifier over our taxonomy using CLIP-like model.
    v1: domain-aware prompts + explicit 'other' bucket + rule bias.
    """

    def __init__(self, taxonomy_path: Path, prompts_path: Path | None = None):
        self.device = "cpu"

        # 1) load taxonomy
        taxonomy: Dict[str, Any] = _load_yaml(taxonomy_path)
        self.taxonomy = taxonomy

        # 2) load prompts (optional)
        self.prompts_cfg: Dict[str, Any] = {}
        if prompts_path is not None and prompts_path.exists():
            self.prompts_cfg = _load_yaml(prompts_path)

        # 3) build label list + prompt texts
        (
            self.labels,
            self.label_to_parent_child,
            self.label_to_prompts,
        ) = self._build_labels_and_prompts(taxonomy, self.prompts_cfg)

        # 4) load CLIP model
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # 5) precompute text embeddings (mean of multiple prompts per label)
        self.text_features = self._encode_label_prompts(self.labels, self.label_to_prompts)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    def _build_labels_and_prompts(
        self, taxonomy: Dict[str, Any], prompts_cfg: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, List[str]]]:
        labels: List[str] = []
        mapping: Dict[str, Dict[str, str]] = {}
        label_prompts: Dict[str, List[str]] = {}

        global_templates: List[str] = prompts_cfg.get("global_templates", [])

        for parent in taxonomy.get("parents", []):
            p_name = parent["name"]
            for child in parent.get("children", []):
                c_name = child["name"]
                label_text = f"{p_name}: {c_name}"
                labels.append(label_text)
                mapping[label_text] = {"parent": p_name, "child": c_name}

                # class specific prompts
                class_key = label_text
                class_prompts = prompts_cfg.get("class_prompts", {}).get(class_key, [])

                if class_prompts:
                    label_prompts[label_text] = class_prompts
                else:
                    # fall back to global templates
                    if global_templates:
                        label_prompts[label_text] = [
                            t.format(label=c_name) for t in global_templates
                        ]
                    else:
                        label_prompts[label_text] = [label_text]

        return labels, mapping, label_prompts

    # ------------------------------------------------------------------
    def _encode_label_prompts(
        self,
        labels: List[str],
        label_to_prompts: Dict[str, List[str]],
    ) -> torch.Tensor:
        """
        For each label we may have multiple prompt variants.
        We encode all and average.
        """
        all_label_features = []
        with torch.no_grad():
            for label in labels:
                prompts = label_to_prompts[label]
                tokens = self.tokenizer(prompts).to(self.device)
                text_feats = self.model.encode_text(tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                # average prompts → single vector
                mean_feat = text_feats.mean(dim=0)
                all_label_features.append(mean_feat)
        return torch.stack(all_label_features, dim=0)

    # ------------------------------------------------------------------
    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        sims = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        probs = sims[0].cpu().tolist()
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        best_idx = sorted_indices[0]
        best_label = self.labels[best_idx]
        best_prob = float(probs[best_idx])

        parent = self.label_to_parent_child[best_label]["parent"]
        child = self.label_to_parent_child[best_label]["child"]

        alternates = []
        for idx in sorted_indices[1:4]:
            alt_label = self.labels[idx]
            alt_parent = self.label_to_parent_child[alt_label]["parent"]
            alt_child = self.label_to_parent_child[alt_label]["child"]
            alternates.append(
                {
                    "raw_label": alt_label,
                    "parent": alt_parent,
                    "child": alt_child,
                    "confidence": float(probs[idx]),
                }
            )

        return {
            "raw_label": best_label,
            "parent": parent,
            "child": child,
            "confidence": best_prob,
            "alternates": alternates,
        }