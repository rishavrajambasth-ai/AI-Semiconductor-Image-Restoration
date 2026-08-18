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
# Load input image
# --------------------------------------------------

def load_input(path):
    """
    Load either a NumPy .npy image or a normal image file.
    """

    if path.suffix.lower() == ".npy":

        image = np.load(path)

        if image is None:
            raise RuntimeError(
                f"Could not load: {path}"
            )

        return image

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    # Convert BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# --------------------------------------------------
# Prepare image
# --------------------------------------------------

def prepare_image(image):
    """
    Convert input image to a model-ready
    RGB tensor with shape [1, 3, H, W].
    """

    image = np.asarray(image)

    # Remove unnecessary dimensions
    image = np.squeeze(image)

    if image.ndim != 2 and image.ndim != 3:
        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    # Grayscale -> RGB
    if image.ndim == 2:

        image = np.stack(
            [image, image, image],
            axis=-1
        )

    # Handle CHW arrays
    elif image.shape[0] == 3 and image.shape[2] != 3:

        image = np.transpose(
            image,
            (1, 2, 0)
        )

    # If image has more than 3 channels,
    # keep the first three channels.
    if image.shape[2] > 3:

        image = image[:, :, :3]

    # Convert to uint8 safely
    if image.dtype != np.uint8:

        image = image.astype(
            np.float32
        )

        # Handle normalized [0, 1] data
        if image.max() <= 1.0:

            image = image * 255.0

        image = np.clip(
            image,
            0,
            255
        ).astype(
            np.uint8
        )

    # Resize for model
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # [0,255] -> [0,1]
    image = (
        image.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model(device):

    model = ImageRestorationUNet().to(
        device
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Run train.py first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


# --------------------------------------------------
# Restore image
# --------------------------------------------------

def restore_image(
    model,
    image,
    device
):

    original_height = image.shape[0]
    original_width = image.shape[1]

    tensor = prepare_image(
        image
    ).to(device)

    # AI inference
    with torch.no_grad():

        restored = model(
            tensor
        )

    # Remove batch dimension
    restored = restored.squeeze(
        0
    )

    # CPU -> NumPy
    restored = (
        restored
        .cpu()
        .numpy()
    )

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

    # Resize back to original dimensions
    restored = cv2.resize(
        restored,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return restored


# --------------------------------------------------
# Main inference
# --------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("AI IMAGE RESTORATION INFERENCE")
    print("=" * 70)
    print()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    if not INPUT_DIR.exists():

        print()
        print(
            "Degraded directory not found:"
        )
        print(INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    print()
    print(
        "Loading trained model..."
    )

    model = load_model(
        device
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------
    # Find degraded images
    # --------------------------------------------------

    image_files = [
        file
        for file in INPUT_DIR.iterdir()
        if file.suffix.lower() in {
            ".npy",
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }
    ]

    if not image_files:

        print()
        print(
            "No degraded images found."
        )

        print(
            INPUT_DIR
        )

        return

    print()
    print(
        f"Found {len(image_files)} degraded image(s)."
    )

    print()

    # --------------------------------------------------
    # Process images
    # --------------------------------------------------

    processed = 0

    for image_path in image_files:

        try:

            image = load_input(
                image_path
            )

            restored = restore_image(
                model,
                image,
                device
            )

            # Save as .npy so that
            # compare_methods.py can
            # evaluate the AI output.

            output_path = (
                OUTPUT_DIR
                / f"restored_{image_path.stem}.npy"
            )

            np.save(
                output_path,
                restored
            )

            processed += 1

            print(
                f"[{processed}] Saved: "
                f"{output_path.name}"
            )

        except Exception as error:

            print(
                f"[ERROR] "
                f"{image_path.name}: "
                f"{error}"
            )

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"Processed {processed} image(s)."
    )

    print()

    print(
        "AI restored images saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 70)
    print()


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()