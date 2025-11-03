Real Estate Image Classifier

A FastAPI-based service that classifies real-estate images into a hierarchical taxonomy and returns structured labels with confidence and action hints for your CMS. It combines a zero-shot CLIP classifier (open-clip) with an optional trained linear head on CLIP embeddings to correct domain-specific mistakes (e.g., real kitchen vs rendered interior, clubhouse vs common area, exterior vs interior).

- API-first: POST /classify-image, POST /classify-batch
- Hybrid inference: linear head (your data) + zero-shot CLIP (generalist)
- Domain prompts + taxonomy: editable YAMLs
- Logging: CSV log for auditing and evaluation
- Local-first: runs fully on macOS CPU; supports Apple Silicon (MPS)
⸻
Table of Contents

1. Taxonomy
2. Project Structure
3. Quick Start
4. Run the API
5. API
6. Hybrid Model
7. Training the Linear Head
8. Batch + Logging
9. Evaluation
10. Performance Tips
11. Deployment
12. License
⸻
Taxonomy

Edit taxonomy/taxonomy.yaml to control the Parent → Child labels. Example (v1):

parents:
  - name: Interiors
    children:
      - name: Bedroom
      - name: Living
      - name: Kitchen (Real Photo)
      - name: Bathroom
      - name: Balcony
  - name: Amenities
    children:
      - name: Gym
      - name: Swimming Pool
      - name: Clubhouse / Indoor Recreation
      - name: Kids Play Area
      - name: Park / Landscape
      - name: Common Area / Seating / Lobby
  - name: Exteriors & Facade
    children:
      - name: Building / Tower Exterior
      - name: Entry Gate / Dropoff
      - name: Inside Compound / Driveway
  - name: Plans
    children:
      - name: Master Plan
      - name: Unit / Floor Plan
      - name: Location Map
      - name: Brochure / Document
  - name: Marketing / Creative
    children:
      - name: Rendered Interior / Exterior
      - name: Offer / Promo Tile
  - name: Junk / Other
    children:
      - name: Low Quality
      - name: Not Real Estate


Prompts to bias CLIP can be edited in taxonomy/prompts.yaml.
⸻
Project Structure

realestate-image-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI entrypoint (single/batch endpoints, logging)
│   ├── model.py            # Zero-shot CLIP + LinearHead (hybrid)
│   ├── train_linear.py     # Train a small linear head on your images
│   ├── eval_metrics.py     # Compute accuracy, confusion, auto-publish precision
│   └── utils.py            # CSV logging helper
├── taxonomy/
│   ├── taxonomy.yaml       # Parent → Child labels
│   └── prompts.yaml        # Domain prompts for CLIP
├── models/                 # Saved weights (ignored in git)
├── data/                   # Your training/validation images (ignored in git)
├── logs/                   # predictions.csv (ignored in git)
├── .gitignore
└── README.md

⸻
Quick Start

# 1) env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2) install deps (CPU wheels)
pip install fastapi uvicorn pillow python-multipart pyyaml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch pandas scikit-learn


If you are on Apple Silicon, you can keep CPU wheels to start. Later you can enable Metal (MPS) in model.py to speed up inference.
⸻
Run the API

source .venv/bin/activate
python -m uvicorn app.main:app --reload


Open:
- Health: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs
⸻
API

GET /health
Returns model/taxonomy status, number of zero-shot labels, and whether the linear head is loaded.

POST /classify-image
Multipart upload (file=@image.jpg). Returns:

{
  "filename": "kitchen_01.jpg",
  "image_info": {"width": 1920, "height": 1080, "mode": "RGB"},
  "parent": "Interiors",
  "child": "Kitchen (Real Photo)",
  "grandchild": null,
  "confidence": 0.87,
  "alternates": [
    {"raw_label": "Interiors: Living", "parent": "Interiors", "child": "Living", "confidence": 0.41}
  ],
  "action": "auto_publish",
  "model_version": "hybrid-v1",
  "decision_source": "linear_head"
}


POST /classify-batch
files=@img1.jpg & files=@img2.jpg … (multiple form fields). Returns per-file JSON array.

Business rules (action)
- Plans with conf ≥ 0.60 → auto_publish
- Marketing with conf ≥ 0.60 → needs_review
- Otherwise conf ≥ 0.80 → auto_publish, 0.60–0.80 → needs_review, else reject

You can tune thresholds in main.py.
⸻
Hybrid Model

- Zero-shot CLIP (RealEstateZeroShotClassifier): generalist across all taxonomy labels using taxonomy/prompts.yaml for domain phrasing.
- Linear head (RealEstateLinearHead): small classifier trained on your images to fix frequent mistakes; preferred when confidence ≥ 0.80 by default.

Load path for weights: models/clip_linear_realestate.pt.
⸻
Training the Linear Head

Folders (example classes):
data/train/interiors_kitchen_real/
data/train/interiors_bedroom/
data/train/amenities_common_area/
data/train/exteriors_building/
data/train/marketing_rendered/
data/train/plans_master/

Place ~12–20 images per class (more is better). Then:

source .venv/bin/activate
python -m app.train_linear

This freezes CLIP, trains a linear layer, and writes:
models/clip_linear_realestate.pt


Tip: add light augmentations (resize/crop, compression, horizontal flip where applicable) to better match WhatsApp/agent uploads.
⸻
Batch + Logging

- POST /classify-batch accepts multiple files.
- Every prediction appends a CSV row to logs/predictions.csv with:
    - filename, parent, child, confidence, action, source, width, height, mode, model_version, timestamp

This enables automated QA and A/B checks in your CMS workflow.
⸻
Evaluation

Prepare a ground-truth CSV:
filename,parent,child
kitchen_01.jpg,Interiors,Kitchen (Real Photo)
...


Run:
source .venv/bin/activate
python app/eval_metrics.py --truth /path/to/ground_truth.csv --preds logs/predictions.csv --out eval_report.txt


Outputs:
- Overall precision/recall/F1 (Parent ▸ Child)
- Auto-publish precision at current thresholds
- Per-parent accuracy
- Confusion matrix

Use this to tune thresholds and identify classes needing more samples.
⸻
Performance Tips

- Switch to Apple Metal (MPS) if available. In model.py, prefer "mps" when torch.backends.mps.is_available().
- Keep prompts specific but concise (avoid overly long sentences).
- Expand the linear head with more classes when you see recurring confusions.
- Cache model + text embeddings at startup (already handled).
⸻
Deployment

Simple systemd + Uvicorn
Run the same uvicorn command as a service and put Nginx in front for TLS and auth.

Docker (example)
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart pyyaml \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir open-clip-torch pandas scikit-learn
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

⸻
License

This repository is provided under the MIT License (see LICENSE if present). Ensure that any training images you add to data/ are cleared for commercial use within your organization.
