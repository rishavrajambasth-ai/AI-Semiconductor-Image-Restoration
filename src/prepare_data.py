from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


# --------------------------------------------------
# Settings
# --------------------------------------------------

IMAGE_SIZE = 256


# --------------------------------------------------
# Create directories
# --------------------------------------------------

def create_directories():

    DEGRADED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# Normalize NumPy image
# --------------------------------------------------

def normalize_image(image):

    image = np.asarray(image)

    # Remove unnecessary dimensions
    image = np.squeeze(image)

    # Convert to float
    image = image.astype(np.float32)

    # Handle common data ranges
    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:

        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

    else:

        image = np.zeros_like(
            image,
            dtype=np.float32
        )

    image = (
        image * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return image


# --------------------------------------------------
# Convert array to image
# --------------------------------------------------

def array_to_image(array):

    array = np.asarray(array)

    array = np.squeeze(array)

    # Grayscale image
    if array.ndim == 2:

        return normalize_image(array)

    # Channel-first image: C,H,W
    if array.ndim == 3 and array.shape[0] in (1, 3):

        array = np.transpose(
            array,
            (1, 2, 0)
        )

    # Channel-last grayscale
    if array.ndim == 3 and array.shape[2] == 1:

        array = array[:, :, 0]

    # RGB image
    if array.ndim == 3 and array.shape[2] == 3:

        image = normalize_image(array)

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    # Unknown format
    raise ValueError(
        f"Unsupported array shape: {array.shape}"
    )


# --------------------------------------------------
# Process one NPY file
# --------------------------------------------------

def process_file(npy_path):

    print()
    print(
        f"Processing: {npy_path.name}"
    )

    data = np.load(
        npy_path,
        allow_pickle=False
    )

    print(
        "Original shape:",
        data.shape
    )

    print(
        "Data type:",
        data.dtype
    )

    print(
        "Minimum:",
        data.min()
    )

    print(
        "Maximum:",
        data.max()
    )

    # --------------------------------------------------
    # Case 1: Single image
    # --------------------------------------------------

    if data.ndim == 2:

        image = array_to_image(data)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        output_name = (
            npy_path.stem + ".png"
        )

        # For now, save as degraded.
        # We will change this once we confirm
        # the exact KLA dataset structure.
        output_path = (
            DEGRADED_DIR / output_name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        print(
            "Saved:",
            output_path
        )

        return

    # --------------------------------------------------
    # Case 2: Multiple images
    # --------------------------------------------------

    if data.ndim == 3:

        # Possible format:
        # N,H,W
        if (
            data.shape[0] > 3
            and data.shape[1] > 10
            and data.shape[2] > 10
        ):

            for index in range(
                data.shape[0]
            ):

                image = array_to_image(
                    data[index]
                )

                image = cv2.resize(
                    image,
                    (IMAGE_SIZE, IMAGE_SIZE)
                )

                output_name = (
                    f"{npy_path.stem}_{index:06d}.png"
                )

                output_path = (
                    DEGRADED_DIR
                    / output_name
                )

                cv2.imwrite(
                    str(output_path),
                    image
                )

                print(
                    "Saved:",
                    output_path
                )

            return

        # Possible RGB image
        image = array_to_image(data)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        output_name = (
            npy_path.stem + ".png"
        )

        output_path = (
            DEGRADED_DIR / output_name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        print(
            "Saved:",
            output_path
        )

        return

    raise ValueError(
        f"Unsupported NPY dimensions: "
        f"{data.ndim}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("KLA NPY DATA PREPARATION")
    print("=" * 60)

    create_directories()

    npy_files = sorted(
        RAW_DIR.glob("*.npy")
    )

    if not npy_files:

        print()
        print(
            "No .npy files found in:"
        )

        print(RAW_DIR)

        print()
        print(
            "Copy the KLA .npy files into:"
        )

        print(RAW_DIR)

        return

    print()
    print(
        f"Found {len(npy_files)} NPY file(s)."
    )

    for npy_path in npy_files:

        try:

            process_file(
                npy_path
            )

        except Exception as error:

            print()
            print(
                f"[ERROR] {npy_path.name}"
            )

            print(error)

    print()
    print("=" * 60)
    print("DATA PREPARATION FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
