from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Training settings
# --------------------------------------------------

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

# Weight given to edge/detail preservation
EDGE_LOSS_WEIGHT = 0.20


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

        if not self.images:
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

        degraded = cv2.resize(
            degraded,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        clean = cv2.resize(
            clean,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        degraded = cv2.cvtColor(
            degraded,
            cv2.COLOR_BGR2RGB
        )

        clean = cv2.cvtColor(
            clean,
            cv2.COLOR_BGR2RGB
        )

        degraded = (
            degraded.astype(np.float32) / 255.0
        )

        clean = (
            clean.astype(np.float32) / 255.0
        )

        degraded = np.transpose(
            degraded,
            (2, 0, 1)
        )

        clean = np.transpose(
            clean,
            (2, 0, 1)
        )

        return (
            torch.tensor(
                degraded,
                dtype=torch.float32
            ),
            torch.tensor(
                clean,
                dtype=torch.float32
            )
        )


# --------------------------------------------------
# Sobel edge extraction
# --------------------------------------------------

def sobel_edges(image):
    """
    Calculate edge magnitude using Sobel filters.
    """

    # Convert RGB image to grayscale
    gray = (
        0.299 * image[:, 0:1]
        + 0.587 * image[:, 1:2]
        + 0.114 * image[:, 2:3]
    )

    sobel_x = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        device=image.device,
        dtype=image.dtype
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        device=image.device,
        dtype=image.dtype
    ).view(1, 1, 3, 3)

    edge_x = F.conv2d(
        gray,
        sobel_x,
        padding=1
    )

    edge_y = F.conv2d(
        gray,
        sobel_y,
        padding=1
    )

    magnitude = torch.sqrt(
        edge_x ** 2
        + edge_y ** 2
        + 1e-8
    )

    return magnitude


# --------------------------------------------------
# Defect-preserving loss
# --------------------------------------------------

class DefectPreservingLoss(nn.Module):

    def __init__(
        self,
        edge_weight=0.20
    ):

        super().__init__()

        self.edge_weight = edge_weight

        self.pixel_loss = nn.L1Loss()

    def forward(
        self,
        restored,
        clean
    ):

        # Pixel reconstruction loss
        reconstruction_loss = (
            self.pixel_loss(
                restored,
                clean
            )
        )

        # Edge/detail loss
        restored_edges = sobel_edges(
            restored
        )

        clean_edges = sobel_edges(
            clean
        )

        edge_loss = (
            self.pixel_loss(
                restored_edges,
                clean_edges
            )
        )

        # Combined loss
        total_loss = (
            reconstruction_loss
            + self.edge_weight * edge_loss
        )

        return (
            total_loss,
            reconstruction_loss,
            edge_loss
        )


# --------------------------------------------------
# Training
# --------------------------------------------------

def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    dataset = RestorationDataset(
        DEGRADED_DIR,
        CLEAN_DIR
    )

    print(
        "Training images:",
        len(dataset)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = ImageRestorationUNet().to(
        device
    )

    criterion = DefectPreservingLoss(
        edge_weight=EDGE_LOSS_WEIGHT
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print()
    print(
        "Starting defect-preserving training..."
    )
    print()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_edge = 0.0

        for degraded, clean in dataloader:

            degraded = degraded.to(device)
            clean = clean.to(device)

            restored = model(
                degraded
            )

            (
                loss,
                reconstruction_loss,
                edge_loss
            ) = criterion(
                restored,
                clean
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_reconstruction += (
                reconstruction_loss.item()
            )
            total_edge += (
                edge_loss.item()
            )

        batches = len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Total Loss: "
            f"{total_loss / batches:.6f} | "
            f"Reconstruction: "
            f"{total_reconstruction / batches:.6f} | "
            f"Edge: "
            f"{total_edge / batches:.6f}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print(
        "Defect-preserving training completed."
    )

    print(
        "Model saved to:"
    )

    print(MODEL_PATH)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    train()
