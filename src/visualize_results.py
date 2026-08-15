from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
RESULTS_DIR = PROJECT_ROOT / "results"

COMPARISON_DIR = RESULTS_DIR / "comparisons"


# --------------------------------------------------
# Create comparison image
# --------------------------------------------------

def create_comparison(
    degraded,
    restored,
    clean
):
    """
    Create a side-by-side comparison:

    Degraded | Restored | Clean
    """

    # Make all images the same size
    height, width = clean.shape[:2]

    degraded = cv2.resize(
        degraded,
        (width, height)
    )

    restored = cv2.resize(
        restored,
        (width, height)
    )

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.8
    thickness = 2

    images = [
        ("DEGRADED", degraded),
        ("RESTORED", restored),
        ("CLEAN", clean),
    ]

    labeled_images = []

    for label, image in images:

        image_copy = image.copy()

        cv2.rectangle(
            image_copy,
            (0, 0),
            (width, 45),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            image_copy,
            label,
            (15, 32),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

        labeled_images.append(
            image_copy
        )

    comparison = np.hstack(
        labeled_images
    )

    return comparison


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print()
    print("=" * 60)
    print("CREATING IMAGE COMPARISONS")
    print("=" * 60)
    print()

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = [
        file
        for file in CLEAN_DIR.iterdir()
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
            "No clean images found."
        )

        return

    created = 0

    for clean_path in image_files:

        degraded_path = (
            DEGRADED_DIR
            / clean_path.name
        )

        restored_path = (
            RESULTS_DIR
            / f"restored_{clean_path.name}"
        )

        if not degraded_path.exists():

            print(
                f"[SKIP] Missing degraded image: "
                f"{clean_path.name}"
            )

            continue

        if not restored_path.exists():

            print(
                f"[SKIP] Missing restored image: "
                f"{clean_path.name}"
            )

            continue

        clean = cv2.imread(
            str(clean_path)
        )

        degraded = cv2.imread(
            str(degraded_path)
        )

        restored = cv2.imread(
            str(restored_path)
        )

        if (
            clean is None
            or degraded is None
            or restored is None
        ):

            print(
                f"[SKIP] Could not read: "
                f"{clean_path.name}"
            )

            continue

        comparison = create_comparison(
            degraded,
            restored,
            clean
        )

        output_path = (
            COMPARISON_DIR
            / f"comparison_{clean_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            comparison
        )

        created += 1

        print(
            f"[{created}] Created: "
            f"{output_path.name}"
        )

    print()
    print(
        f"Created {created} comparison image(s)."
    )

    print()
    print(
        "Saved in:"
    )

    print(COMPARISON_DIR)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
