import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import timm
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ================================================================
# DATASET
# ================================================================
class RubikDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=224, augment=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment

        self.transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                   rotate_limit=15, p=0.5),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
        )

    def order_keypoints(self, pts):
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        pts = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        start = np.argmin([p[0] + p[1] for p in pts])
        return pts[start:] + pts[:start]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        image = cv2.imread(img_path)

        oh, ow = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        sx = self.img_size / ow
        sy = self.img_size / oh

        presence = 1.0 if row["x1"] != -1 else 0.0

        if presence > 0:
            kps = [
                (row["x1"] * sx, row["y1"] * sy),
                (row["x2"] * sx, row["y2"] * sy),
                (row["x3"] * sx, row["y3"] * sy),
                (row["x4"] * sx, row["y4"] * sy),
            ]
        else:
            kps = [(0, 0)] * 4

        if self.augment:
            aug = self.transform(image=image, keypoints=kps)
            image = aug["image"]
            if presence > 0:
                kps = self.order_keypoints(aug["keypoints"])

        kps = np.array(kps, dtype="float32").flatten() / self.img_size

        # image normalization
        image = (image / 127.5) - 1.0
        image = np.transpose(image, (2, 0, 1))

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(np.concatenate([kps, [presence]]), dtype=torch.float32)
        )


# ================================================================
# MODEL (MobileNetV2 features → regressor)
# ================================================================
class RubikModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv2_100.ra_in1k",
            pretrained=True,
            features_only=True
        )
        
        # last feature map dims
        backbone_dim = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9)   # 8 keypoints + 1 presence
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]   # last feature map
        pooled = self.global_pool(feats).flatten(1)
        return self.regressor(pooled)


# ================================================================
# LOSS FUNCTION (COMBINATION OF BCE FOR PRESENCE AND SMOOTH L1 FOR KEYPOINTS)
# ================================================================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        pred_kp = preds[:, :8]
        pred_presence = preds[:, 8]

        true_kp = targets[:, :8]
        true_presence = targets[:, 8]

        # presence classification loss
        loss_presence = self.bce(pred_presence, true_presence)

        # compute regression loss for all samples
        kp_loss = self.smooth_l1(pred_kp, true_kp).sum(dim=1)

        # weigh regression loss by presence
        kp_loss = (kp_loss * true_presence).mean()

        return loss_presence + kp_loss


# ================================================================
# TRAINING FUNCTIONS
# ================================================================
def train_one_epoch(model, loader, opt, loss_fn, epoch):
    model.train()
    total = 0
    pbar = tqdm(loader, desc=f"Train {epoch}")

    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total / len(loader)


def validate_one_epoch(model, loader, loss_fn, epoch):
    model.eval()
    total = 0
    pbar = tqdm(loader, desc=f"Val {epoch}")

    with torch.no_grad():
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)

            total += loss.item()
            pbar.set_postfix({"val_loss": loss.item()})

    return total / len(loader)

dataset_dir = "./"

train_dataset = RubikDataset(
    csv_file=os.path.join(dataset_dir, "image-dataset/rubik_coord_train.csv"),
    img_dir=os.path.join(dataset_dir, "image-dataset/combined"),
    augment=True
)
val_dataset = RubikDataset(
    csv_file=os.path.join(dataset_dir, "image-dataset/rubik_coord_val.csv"),
    img_dir=os.path.join(dataset_dir, "image-dataset/combined"),
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = RubikModel().to(device)
checkpoint_path = "./rubik_best.pt"
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    except RuntimeError as e:
        print(f"Warning: could not load {checkpoint_path} (shape mismatch): {e}")
        print("Continuing with randomly initialized MobileNetV2 weights.")
else:
    print("No pretrained checkpoint found; training from scratch.")

loss_fn = CombinedLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val = 9999
patience = 10
wait = 0

for epoch in range(500):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch=epoch+1)
    val_loss = validate_one_epoch(model, val_loader, loss_fn, epoch=epoch+1)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        print("New best model phase 1 found, saving...")
        torch.save(model.state_dict(), "rubik_best_phase1.pt")
    else:
        wait += 1

    if wait >= patience:
        print("Early stopping triggered.")
        break


for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch=epoch+1)
    val_loss = validate_one_epoch(model, val_loader, loss_fn, epoch=epoch+1)

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        print("New best model phase 2 found, saving...")
        torch.save(model.state_dict(), "rubik_best_phase2.pt")
    else:
        wait += 1

    if wait >= patience:
        print("Early stopping triggered.")
        break

    print(f"[FT] Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

torch.save(model.state_dict(), "rubik_model_pytorch.pt")

print("Training complete.")