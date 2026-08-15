from pathlib import Path

import cv2


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

    if not DEGRADED_DIR.exists():

        print(
            "Degraded directory not found:"
        )

        print(DEGRADED_DIR)

        return

    BASELINE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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
        }
    ]

    if not image_files:

        print(
            "No degraded images found."
        )

        return

    processed = 0

    for image_path in image_files:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:

            print(
                f"[SKIP] Could not read: "
                f"{image_path.name}"
            )

            continue

        restored = restore_with_baseline(
            image
        )

        output_path = (
            BASELINE_DIR
            / f"baseline_{image_path.name}"
        )

        success = cv2.imwrite(
            str(output_path),
            restored
        )

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


if __name__ == "__main__":
    main()
