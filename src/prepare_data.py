from pathlib import Path
import random

import cv2
import numpy as np


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def add_degradation(image):
    """
    Create a degraded version of a clean image.

    Degradations:
    1. Slight Gaussian blur
    2. Gaussian noise
    3. Small contrast/brightness variation
    """

    # 1. Random blur
    kernel_size = random.choice([3, 5])

    degraded = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    # 2. Add Gaussian noise
    noise_strength = random.uniform(5.0, 20.0)

    noise = np.random.normal(
        0,
        noise_strength,
        degraded.shape
    ).astype(np.float32)

    degraded = degraded.astype(np.float32) + noise

    # 3. Slight contrast and brightness change
    alpha = random.uniform(0.85, 1.05)
    beta = random.uniform(-10, 10)

    degraded = alpha * degraded + beta

    # Keep pixel values in valid range
    degraded = np.clip(
        degraded,
        0,
        255
    ).astype(np.uint8)

    return degraded


def process_images():
    """
    Process all images inside data/raw.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        file
        for file in RAW_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("No images found in:")
        print(RAW_DIR)
        print()
        print("Put PNG/JPG images inside data/raw and run again.")
        return

    print(f"Found {len(image_files)} image(s).")
    print()

    for index, image_path in enumerate(image_files, start=1):

        # Read image
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            print(f"[SKIP] Could not read: {image_path.name}")
            continue

        # Create degraded image
        degraded = add_degradation(image)

        # Output paths
        clean_path = CLEAN_DIR / image_path.name
        degraded_path = DEGRADED_DIR / image_path.name

        # Save clean and degraded versions
        cv2.imwrite(
            str(clean_path),
            image
        )

        cv2.imwrite(
            str(degraded_path),
            degraded
        )

        print(
            f"[{index}/{len(image_files)}] "
            f"Processed: {image_path.name}"
        )

    print()
    print("Data preparation completed.")
    print(f"Clean images:    {CLEAN_DIR}")
    print(f"Degraded images: {DEGRADED_DIR}")


if __name__ == "__main__":
    process_images()
