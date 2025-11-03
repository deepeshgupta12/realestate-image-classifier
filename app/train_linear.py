from pathlib import Path
from typing import List, Tuple

import torch
import open_clip
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader


class FolderImageDataset(Dataset):
    def __init__(self, root: Path, class_names: List[str], transform=None):
        self.root = root
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        for cname in class_names:
            cdir = root / cname
            for img_path in cdir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((img_path, self.class_to_idx[cname]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    base_dir = Path(__file__).resolve().parent.parent
    train_dir = base_dir / "data" / "train"
    val_dir = base_dir / "data" / "val"  # we created it, we may use it later

    # map your folders to label names (same as we created in bash)
    class_names = [
        "interiors_kitchen_real",
        "interiors_bedroom",
        "amenities_common_area",
        "exteriors_building",
        "marketing_rendered",
        "plans_master",
    ]

    # 1) load CLIP image encoder + its preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    device = "cpu"
    model = model.to(device)

    # 2) datasets/loaders
    train_ds = FolderImageDataset(train_dir, class_names, transform=preprocess)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # (optional) val — we won’t use right now, but keep the code for future
    val_dir = FolderImageDataset(val_dir, class_names, transform=preprocess)

    # 3) freeze CLIP
    for p in model.parameters():
        p.requires_grad = False

    # 4) find embedding dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        emb = model.encode_image(dummy)
    emb_dim = emb.shape[-1]

    # 5) linear head
    clf = nn.Linear(emb_dim, len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        clf.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feats = model.encode_image(imgs)

            preds = clf(feats)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / max(len(train_ds), 1)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    # 6) save head
    out_path = base_dir / "models" / "clip_linear_realestate.pt"
    torch.save(
        {
            "state_dict": clf.state_dict(),
            "class_names": class_names,
        },
        out_path,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()