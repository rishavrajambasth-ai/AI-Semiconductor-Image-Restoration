from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"

RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_DIR = RESULTS_DIR / "baseline"


# --------------------------------------------------
# Settings
# --------------------------------------------------

H = 10
TEMPLATE_WINDOW = 7
SEARCH_WINDOW = 21


# --------------------------------------------------
# Baseline restoration
# --------------------------------------------------

def restore_with_baseline(image):
    """
    Traditional image restoration using
    Non-Local Means denoising.
    """

    # Grayscale image
    if len(image.shape) == 2:

        restored = cv2.fastNlMeansDenoising(
            image,
            None,
            h=H,
            templateWindowSize=TEMPLATE_WINDOW,
            searchWindowSize=SEARCH_WINDOW
        )

        return restored

    # Color image
    restored = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=H,
        hColor=H,
        templateWindowSize=TEMPLATE_WINDOW,
        searchWindowSize=SEARCH_WINDOW
    )

    return restored


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print()
    print("=" * 60)
    print("TRADITIONAL RESTORATION BASELINE")
    print("=" * 60)
    print()

    # Check degraded directory
    if not DEGRADED_DIR.exists():

        print("Degraded directory not found:")
        print(DEGRADED_DIR)

        return

    # Create baseline output directory
    BASELINE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find supported image files
    image_files = [
        file
        for file in DEGRADED_DIR.iterdir()
        if file.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".npy",
        }
    ]

    # Check whether images were found
    if not image_files:

        print("No degraded images found.")

        return

    processed = 0

    # --------------------------------------------------
    # Process each image
    # --------------------------------------------------

    for image_path in image_files:

        print(f"Processing: {image_path.name}")

        # Load NumPy files
        if image_path.suffix.lower() == ".npy":

            try:
                image = np.load(image_path)

            except Exception as e:

                print(
                    f"[SKIP] Could not load: "
                    f"{image_path.name}"
                )

                print(f"Reason: {e}")

                continue

        # Load normal image files
        else:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED
            )

        # Check image
        if image is None:

            print(
                f"[SKIP] Could not read: "
                f"{image_path.name}"
            )

            continue

        # Make sure NumPy image is uint8
        if image.dtype != np.uint8:

            image_min = image.min()
            image_max = image.max()

            if image_max > image_min:

                image = (
                    (image - image_min)
                    / (image_max - image_min)
                    * 255
                ).astype(np.uint8)

            else:

                image = np.zeros_like(
                    image,
                    dtype=np.uint8
                )

        # Restore image
        try:

            restored = restore_with_baseline(
                image
            )

        except Exception as e:

            print(
                f"[ERROR] Restoration failed: "
                f"{image_path.name}"
            )

            print(f"Reason: {e}")

            continue

        # Output path
        output_path = (
            BASELINE_DIR
            / f"baseline_{image_path.name}"
        )

        # --------------------------------------------------
        # Save NumPy files
        # --------------------------------------------------

        if image_path.suffix.lower() == ".npy":

            try:

                np.save(
                    output_path,
                    restored
                )

                success = True

            except Exception as e:

                print(
                    f"[ERROR] Could not save: "
                    f"{output_path.name}"
                )

                print(f"Reason: {e}")

                success = False

        # --------------------------------------------------
        # Save normal image files
        # --------------------------------------------------

        else:

            success = cv2.imwrite(
                str(output_path),
                restored
            )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if success:

            processed += 1

            print(
                f"[{processed}] Saved: "
                f"{output_path.name}"
            )

        else:

            print(
                f"[ERROR] Could not save: "
                f"{output_path.name}"
            )

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print()

    print(
        f"Processed {processed} image(s)."
    )

    print()

    print(
        "Baseline results saved to:"
    )

    print(BASELINE_DIR)

    print()

    print("=" * 60)


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()