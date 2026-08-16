from pathlib import Path

import cv2
import numpy as np
import torch

from model import ImageRestorationUNet


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "restoration_model.pth"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "degraded"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
)


# --------------------------------------------------
# Settings
# --------------------------------------------------

IMAGE_SIZE = 256


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model(device):

    model = ImageRestorationUNet().to(device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}\n\n"
            "Run train.py first."
        )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    return model


# --------------------------------------------------
# Restore image
# --------------------------------------------------

def restore_image(
    model,
    image_path,
    device
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    original_height, original_width = image.shape[:2]

    # Resize for model
    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # BGR -> RGB
    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # [0,255] -> [0,1]
    resized = resized.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    resized = np.transpose(
        resized,
        (2, 0, 1)
    )

    # Add batch dimension
    tensor = torch.tensor(
        resized,
        dtype=torch.float32
    ).unsqueeze(0)

    tensor = tensor.to(device)

    # Model inference
    with torch.no_grad():

        restored = model(tensor)

    # Remove batch dimension
    restored = restored.squeeze(0)

    # CPU -> NumPy
    restored = restored.cpu().numpy()

    # CHW -> HWC
    restored = np.transpose(
        restored,
        (1, 2, 0)
    )

    # [0,1] -> [0,255]
    restored = (
        restored * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # RGB -> BGR
    restored = cv2.cvtColor(
        restored,
        cv2.COLOR_RGB2BGR
    )

    # Return to original image size
    restored = cv2.resize(
        restored,
        (
            original_width,
            original_height
        )
    )

    return restored


# --------------------------------------------------
# Main inference
# --------------------------------------------------

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model(device)

    image_files = [
        file
        for file in INPUT_DIR.iterdir()
        if file.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }
    ]

    if not image_files:

        print(
            "No degraded images found in:"
        )

        print(INPUT_DIR)

        return

    print(
        f"Found {len(image_files)} image(s)."
    )

    for image_path in image_files:

        restored = restore_image(
            model,
            image_path,
            device
        )

        output_path = (
            OUTPUT_DIR
            / f"restored_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            restored
        )

        print(
            f"Saved: {output_path.name}"
        )

    print()
    print("Inference completed.")
    print(
        "Restored images are available in:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
