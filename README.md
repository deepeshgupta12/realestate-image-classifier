# Real Estate Image Classifier & Thumbnail Selector

FastAPI service to **classify real‑estate photos** (Kitchen, Bedroom, Clubhouse, Master Plan, etc.) and **auto‑select the best listing thumbnail** using configurable business rules (e.g., *never pick Bathroom*).  
Runs fully local on macOS, supports Apple Silicon (MPS), and integrates with your CMS via HTTP endpoints.

---

## Highlights

- **Hybrid classifier**
  - **Zero‑shot OpenCLIP** (ViT‑B/32) with domain prompts from YAML.
  - **Optional linear head** trained on your images for domain fixes.
- **Thumbnail selector**
  - Scores candidates by class priority, sharpness, exposure, contrast, saliency, **duplicate penalty**, and **size**.
  - **Hard ban** on Bathroom/Washroom/Toilet (configurable).
  - **Smart low‑res fallback**: if all images are below the min gate, it still picks the best and flags `policy_fallback=true`.
  - Returns a **4:3 crop** `(x,y,w,h)` for consistent cards.
- **API‑first** with `/docs` (Swagger), CSV logging, and YAML‑driven taxonomy/thresholds/rules.

---

## Project Structure

```
realestate-image-classifier/
├── app/
│   ├── main.py               # FastAPI endpoints
│   ├── model.py              # Zero-shot CLIP + optional Linear Head
│   ├── train_linear.py       # Train linear head on your images
│   ├── eval_metrics.py       # Accuracy/F1/auto-publish precision
│   ├── thumbnail.py          # Thumbnail selector (ban + low-res fallback)
│   └── utils.py              # CSV logging helper
├── taxonomy/
│   ├── taxonomy.yaml         # Parent → Child (→ Grandchild) labels
│   ├── prompts.yaml          # CLIP domain prompts per label
│   ├── thresholds.yaml       # Action thresholds (global/parent/child)
│   └── thumbnail_rules.yaml  # Aspect, min res, class weights, bans
├── models/                   # Saved linear-head weights (*.pt)
├── data/                     # Training/validation images
├── logs/                     # predictions.csv
├── .gitignore
└── README.md
```

---

## Quick Start (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# core deps
pip install fastapi uvicorn pillow python-multipart pyyaml pandas scikit-learn imagehash

# PyTorch CPU wheels (simple start); switch to MPS later in model.py
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OpenCLIP and OpenCV
pip install open-clip-torch opencv-contrib-python-headless

# run API
python -m uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for Swagger UI. Health check at **/health**.

> **Apple Silicon (MPS):** In `app/model.py`, set `self.device = "mps"` when `torch.backends.mps.is_available()`.

---

## Configuration (YAML)

### 1) `taxonomy/taxonomy.yaml`
Controls label hierarchy (Parent → Child). Example:
```yaml
parents:
  - name: Interiors
    children:
      - name: Bedroom
      - name: Living
      - name: Kitchen
      - name: Bathroom
  - name: Amenities
    children:
      - name: Gym
      - name: Swimming Pool
      - name: Clubhouse / Indoor Recreation
      - name: Common Area / Seating / Lobby
  - name: Exteriors & Facade
    children:
      - name: Building / Tower Exterior
  - name: Plans
    children:
      - name: Master Plan
      - name: Unit / Floor Plan
  - name: Marketing / Creative
    children:
      - name: Rendered Interior / Exterior
```

### 2) `taxonomy/prompts.yaml`
Defines CLIP text prompts per label to improve zero‑shot separability.

### 3) `taxonomy/thresholds.yaml`
Set publish/review thresholds globally, by parent, and by child:
```yaml
global:
  auto_publish: 0.80
  needs_review: 0.60

parents:
  Plans:
    auto_publish: 0.60

children:
  "Interiors ▸ Kitchen":
    auto_publish: 0.78
```

### 4) `taxonomy/thumbnail_rules.yaml`
Controls selector behavior:
```yaml
aspect_ratio: "4:3"
min_resolution: [900, 675]   # lower to [640,480] for UGC if needed

weights:
  class: 0.55
  quality: 0.30
  composition: 0.15

penalties:
  face_present: 0.15
  too_dark: 0.10
  too_bright: 0.10
  blur: 0.25
  duplicate: 0.30
  portrait_orientation: 0.10
  low_res: 0.25             # used in low-res fallback

class_weights:
  "Exteriors & Facade ▸ Building / Tower Exterior": 1.00
  "Interiors ▸ Living": 0.95
  "Interiors ▸ Bedroom": 0.92
  "Interiors ▸ Kitchen": 0.88
  "Amenities ▸ Swimming Pool": 0.82
  "Amenities ▸ Clubhouse / Indoor Recreation": 0.80
  "Amenities ▸ Common Area / Seating / Lobby": 0.78
  "Marketing / Creative ▸ Rendered Interior / Exterior": 0.35
  "Plans ▸ Master Plan": 0.15

banned_labels:
  - "Interiors ▸ Bathroom"
  - "Interiors ▸ Washroom"
  - "Interiors ▸ Toilet"
  - "Interiors ▸ Powder Room"
  - "Interiors ▸ Restroom"
```

---

## API Endpoints

- **`GET /health`** → model/taxonomy status.
- **`POST /classify-image`** → single image multipart, returns `{parent, child, confidence, action, alternates}`.
- **`POST /classify-batch`** → multiple files, returns array of results.
- **`POST /choose-thumbnail`** → multiple files, returns the **selected thumbnail** and `suggested_crop_xywh`.

**Decisions (`action`)**
- Default rules in `main.py`:
  - `Plans` with conf ≥ 0.60 → `auto_publish`
  - `Marketing` with conf ≥ 0.60 → `needs_review`
  - Else conf ≥ 0.80 → `auto_publish`; 0.60–0.80 → `needs_review`; otherwise `reject`
- Override per parent/child via `thresholds.yaml`.

**Example (curl):**
```bash
curl -s -X POST "http://127.0.0.1:8000/classify-image"   -F "file=@/path/to/kitchen.jpg" | jq

curl -s -X POST "http://127.0.0.1:8000/choose-thumbnail"   -F "files=@/path/1.jpg" -F "files=@/path/2.jpg" -F "files=@/path/3.jpg" | jq
```

---

## Training the Linear Head (optional)

Prepare folders (one per child class):
```
data/train/interiors_kitchen/
data/train/interiors_bedroom/
data/train/amenities_common_area/
data/train/exteriors_building/
data/train/marketing_rendered/
data/train/plans_master/
...
```
Add 12–50 images per class. Then:
```bash
source .venv/bin/activate
python -m app.train_linear
```
This writes `models/clip_linear_realestate.pt`. `main.py` auto‑loads it and prefers it if `confidence ≥ 0.80`.

---

## Evaluation

Create a ground‑truth CSV:
```
filename,parent,child
img001.jpg,Interiors,Kitchen
```
Run:
```bash
python app/eval_metrics.py   --truth /path/to/ground_truth.csv   --preds logs/predictions.csv   --out eval_report.txt
```

---

## Deployment

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Docker example:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart pyyaml pandas scikit-learn imagehash     && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu     && pip install --no-cache-dir open-clip-torch opencv-contrib-python-headless
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## License

Licensed under **Deepesh Gupta**. See [LICENSE](LICENSE) for details.
