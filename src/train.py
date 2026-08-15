from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import ImageRestorationUNet


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "restoration_model.pth"


# --------------------------------------------------
# Settings
# --------------------------------------------------

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class RestorationDataset(Dataset):

    def __init__(self, degraded_dir, clean_dir):

        self.degraded_dir = Path(degraded_dir)
        self.clean_dir = Path(clean_dir)

        self.images = []

        for file in self.degraded_dir.iterdir():

            if file.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
            }:

                clean_file = self.clean_dir / file.name

                if clean_file.exists():
                    self.images.append(file)

        if len(self.images) == 0:
            raise RuntimeError(
                "No matching degraded/clean image pairs found."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        degraded_path = self.images[index]
        clean_path = self.clean_dir / degraded_path.name

        degraded = cv2.imread(
            str(degraded_path),
            cv2.IMREAD_COLOR
        )

        clean = cv2.imread(
            str(clean_path),
            cv2.IMREAD_COLOR
        )

        if degraded is None or clean is None:
            raise RuntimeError(
                f"Could not read {degraded_path.name}"
            )

        # Resize
        degraded = cv2.resize(
            degraded,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        clean = cv2.resize(
            clean,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # OpenCV BGR -> RGB
        degraded = cv2.cvtColor(
            degraded,
            cv2.COLOR_BGR2RGB
        )

        clean = cv2.cvtColor(
            clean,
            cv2.COLOR_BGR2RGB
        )

        # Convert [0,255] -> [0,1]
        degraded = degraded.astype(
            np.float32
        ) / 255.0

        clean = clean.astype(
            np.float32
        ) / 255.0

        # HWC -> CHW
        degraded = np.transpose(
            degraded,
            (2, 0, 1)
        )

        clean = np.transpose(
            clean,
            (2, 0, 1)
        )

        degraded = torch.tensor(
            degraded,
            dtype=torch.float32
        )

        clean = torch.tensor(
            clean,
            dtype=torch.float32
        )

        return degraded, clean


# --------------------------------------------------
# Training
# --------------------------------------------------

def train():

    # Select device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # Dataset
    dataset = RestorationDataset(
        DEGRADED_DIR,
        CLEAN_DIR
    )

    print("Training images:", len(dataset))

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Model
    model = ImageRestorationUNet().to(device)

    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print()
    print("Starting training...")
    print()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for degraded, clean in dataloader:

            degraded = degraded.to(device)
            clean = clean.to(device)

            # Forward pass
            restored = model(degraded)

            # Calculate loss
            loss = criterion(
                restored,
                clean
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.6f}"
        )

    # Create model directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model
    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print("Training completed.")
    print("Model saved to:")
    print(MODEL_PATH)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    train()
