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

# Using batch size 1 for CPU testing
BATCH_SIZE = 4

# One epoch for testing training speed
EPOCHS = 10

LEARNING_RATE = 0.001

# Weight given to edge/detail preservation
EDGE_LOSS_WEIGHT = 0.20


# --------------------------------------------------
# Supported file extensions
# --------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class RestorationDataset(Dataset):

    def __init__(
        self,
        degraded_dir,
        clean_dir
    ):

        self.degraded_dir = Path(
            degraded_dir
        )

        self.clean_dir = Path(
            clean_dir
        )

        self.images = []

        for file in self.degraded_dir.iterdir():

            if (
                file.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):

                clean_file = (
                    self.clean_dir
                    / file.name
                )

                if clean_file.exists():

                    self.images.append(
                        file
                    )

        self.images.sort()

        if not self.images:

            raise RuntimeError(
                "No matching degraded/clean image pairs found."
            )

    def __len__(self):

        return len(self.images)

    def load_file(self, path):

        # --------------------------------------------------
        # NumPy format
        # --------------------------------------------------

        if path.suffix.lower() == ".npy":

            image = np.load(
                path
            )

        # --------------------------------------------------
        # Normal image formats
        # --------------------------------------------------

        else:

            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR
            )

        if image is None:

            raise RuntimeError(
                f"Could not read image: {path}"
            )

        return image

    def prepare_image(self, image):

        # --------------------------------------------------
        # Handle grayscale images
        # --------------------------------------------------

        if image.ndim == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        # --------------------------------------------------
        # Handle channel-first NumPy arrays
        # --------------------------------------------------

        elif (
            image.ndim == 3
            and image.shape[0] in [1, 3]
            and image.shape[2] not in [1, 3]
        ):

            image = np.transpose(
                image,
                (1, 2, 0)
            )

        # --------------------------------------------------
        # Convert single-channel to RGB
        # --------------------------------------------------

        if (
            image.ndim == 3
            and image.shape[2] == 1
        ):

            image = np.repeat(
                image,
                3,
                axis=2
            )

        # --------------------------------------------------
        # Convert BGR to RGB
        #
        # For OpenCV images this is required.
        # For .npy RGB arrays, this may already be RGB.
        # --------------------------------------------------

        if (
            image.ndim == 3
            and image.shape[2] == 3
        ):

            # Keep NumPy arrays as they are.
            # OpenCV images are converted before normalization.
            pass

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        # --------------------------------------------------
        # Convert data type
        # --------------------------------------------------

        image = image.astype(
            np.float32
        )

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        if image.max() > 1.0:

            image = image / 255.0

        image = np.clip(
            image,
            0.0,
            1.0
        )

        # --------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        return torch.tensor(
            image,
            dtype=torch.float32
        )

    def __getitem__(self, index):

        degraded_path = (
            self.images[index]
        )

        clean_path = (
            self.clean_dir
            / degraded_path.name
        )

        degraded = self.load_file(
            degraded_path
        )

        clean = self.load_file(
            clean_path
        )

        degraded = self.prepare_image(
            degraded
        )

        clean = self.prepare_image(
            clean
        )

        return (
            degraded,
            clean
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
    ).view(
        1,
        1,
        3,
        3
    )

    sobel_y = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        device=image.device,
        dtype=image.dtype
    ).view(
        1,
        1,
        3,
        3
    )

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
            + self.edge_weight
            * edge_loss
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

    print()
    print("=" * 70)
    print("DEFECT-PRESERVING AI TRAINING")
    print("=" * 70)

    print()
    print(
        "Using device:",
        device
    )

    print(
        "Image size:",
        IMAGE_SIZE,
        "x",
        IMAGE_SIZE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print()

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = RestorationDataset(
        DEGRADED_DIR,
        CLEAN_DIR
    )

    print(
        "Training images:",
        len(dataset)
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print(
        "Batches per epoch:",
        len(dataloader)
    )

    print()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = ImageRestorationUNet().to(
        device
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = DefectPreservingLoss(
        edge_weight=EDGE_LOSS_WEIGHT
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        "Starting defect-preserving training..."
    )

    print()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_edge = 0.0

        print(
            f"Starting Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        print()

        for batch_index, (
            degraded,
            clean
        ) in enumerate(
            dataloader,
            start=1
        ):

            print(
                f"Processing batch "
                f"{batch_index}/{len(dataloader)}...",
                flush=True
            )

            # Move data to device

            degraded = degraded.to(
                device
            )

            clean = clean.to(
                device
            )

            # Forward pass

            restored = model(
                degraded
            )

            # Calculate loss

            (
                loss,
                reconstruction_loss,
                edge_loss
            ) = criterion(
                restored,
                clean
            )

            # Clear gradients

            optimizer.zero_grad()

            # Backward pass

            loss.backward()

            # Update model

            optimizer.step()

            # Accumulate losses

            total_loss += (
                loss.item()
            )

            total_reconstruction += (
                reconstruction_loss.item()
            )

            total_edge += (
                edge_loss.item()
            )

            print(
                f"Completed batch "
                f"{batch_index}/{len(dataloader)} | "
                f"Loss={loss.item():.6f}",
                flush=True
            )

            print()

        # --------------------------------------------------
        # Epoch averages
        # --------------------------------------------------

        batches = len(
            dataloader
        )

        average_loss = (
            total_loss / batches
        )

        average_reconstruction = (
            total_reconstruction
            / batches
        )

        average_edge = (
            total_edge
            / batches
        )

        print(
            "=" * 70
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"completed"
        )

        print(
            f"Total Loss: "
            f"{average_loss:.6f}"
        )

        print(
            f"Reconstruction Loss: "
            f"{average_reconstruction:.6f}"
        )

        print(
            f"Edge Loss: "
            f"{average_edge:.6f}"
        )

        print(
            "=" * 70
        )

        print()

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print("=" * 70)

    print(
        "Defect-preserving training completed."
    )

    print(
        "Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print("=" * 70)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    train()