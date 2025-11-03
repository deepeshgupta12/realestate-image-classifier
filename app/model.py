from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import open_clip
import yaml
from PIL import Image


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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