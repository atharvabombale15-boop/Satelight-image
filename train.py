import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LEVIRDataset
from config import MODEL_ARCH
from model import build_model
import os
from tqdm import tqdm

# -------------------
# Config
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 256
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
HARD_NEG_RATIO = 0.0
HARD_NEG_THRESHOLD = 0.005
DATA_PATH = "data"
MODEL_SAVE_PATH = "models/best_model.pth"

os.makedirs("models", exist_ok=True)

# -------------------
# Dice Loss
# -------------------
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets, smooth=1):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        return (self.bce_weight * self.bce(preds, targets)) + (
            self.dice_weight * self.dice(preds, targets)
        )

# -------------------
# Load Dataset
# -------------------
train_dataset = LEVIRDataset(
    DATA_PATH,
    split="train",
    image_size=IMAGE_SIZE,
    hard_negative_ratio=HARD_NEG_RATIO,
    negative_threshold=HARD_NEG_THRESHOLD,
)
val_dataset = LEVIRDataset(DATA_PATH, split="val", image_size=IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------
# Model
# -------------------
model = build_model(MODEL_ARCH).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)

# -------------------
# Training Loop
# -------------------
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, leave=True)

    for t1, t2, mask in loop:
        t1, t2, mask = t1.to(DEVICE), t2.to(DEVICE), mask.to(DEVICE)

        preds = model(t1, t2)

        loss = criterion(preds, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    # -------------------
    # Validation
    # -------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for t1, t2, mask in val_loader:
            t1, t2, mask = t1.to(DEVICE), t2.to(DEVICE), mask.to(DEVICE)
            preds = model(t1, t2)
            loss = criterion(preds, mask)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Model Saved!")

print("🎉 Training Finished!")
