from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
RESULTS_DIR = PROJECT_ROOT / "results"

RESTORED_PREFIX = "restored_"


# --------------------------------------------------
# Image loading
# --------------------------------------------------

def load_image(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# --------------------------------------------------
# Resize images to same dimensions
# --------------------------------------------------

def match_size(reference, image):

    height, width = reference.shape[:2]

    if image.shape[:2] != (height, width):

        image = cv2.resize(
            image,
            (width, height)
        )

    return image


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

def calculate_metrics(clean, test):

    test = match_size(
        clean,
        test
    )

    clean_float = (
        clean.astype(np.float32) / 255.0
    )

    test_float = (
        test.astype(np.float32) / 255.0
    )

    psnr = peak_signal_noise_ratio(
        clean_float,
        test_float,
        data_range=1.0
    )

    ssim = structural_similarity(
        clean_float,
        test_float,
        channel_axis=2,
        data_range=1.0
    )

    return psnr, ssim


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    if not CLEAN_DIR.exists():

        print(
            "Clean image directory not found:"
        )

        print(CLEAN_DIR)

        return

    if not DEGRADED_DIR.exists():

        print(
            "Degraded image directory not found:"
        )

        print(DEGRADED_DIR)

        return

    if not RESULTS_DIR.exists():

        print(
            "Results directory not found:"
        )

        print(RESULTS_DIR)

        return

    clean_files = [
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

    if not clean_files:

        print(
            "No clean images found."
        )

        return

    total_degraded_psnr = 0.0
    total_degraded_ssim = 0.0

    total_restored_psnr = 0.0
    total_restored_ssim = 0.0

    degraded_count = 0
    restored_count = 0

    print()
    print("=" * 60)
    print("SEMICONDUCTOR IMAGE RESTORATION EVALUATION")
    print("=" * 60)
    print()

    for clean_path in clean_files:

        clean = load_image(
            clean_path
        )

        # ------------------------------------------
        # Evaluate degraded image
        # ------------------------------------------

        degraded_path = (
            DEGRADED_DIR
            / clean_path.name
        )

        if degraded_path.exists():

            degraded = load_image(
                degraded_path
            )

            psnr, ssim = calculate_metrics(
                clean,
                degraded
            )

            total_degraded_psnr += psnr
            total_degraded_ssim += ssim

            degraded_count += 1

            print(
                f"{clean_path.name}"
            )

            print(
                f"  Degraded -> "
                f"PSNR: {psnr:.2f} dB | "
                f"SSIM: {ssim:.4f}"
            )

        # ------------------------------------------
        # Evaluate restored image
        # ------------------------------------------

        restored_path = (
            RESULTS_DIR
            / f"{RESTORED_PREFIX}{clean_path.name}"
        )

        if restored_path.exists():

            restored = load_image(
                restored_path
            )

            psnr, ssim = calculate_metrics(
                clean,
                restored
            )

            total_restored_psnr += psnr
            total_restored_ssim += ssim

            restored_count += 1

            print(
                f"  Restored -> "
                f"PSNR: {psnr:.2f} dB | "
                f"SSIM: {ssim:.4f}"
            )

            print()

    # --------------------------------------------------
    # Average results
    # --------------------------------------------------

    print("=" * 60)
    print("AVERAGE RESULTS")
    print("=" * 60)

    if degraded_count > 0:

        avg_degraded_psnr = (
            total_degraded_psnr
            / degraded_count
        )

        avg_degraded_ssim = (
            total_degraded_ssim
            / degraded_count
        )

        print()
        print("Before restoration:")
        print(
            f"Average PSNR: "
            f"{avg_degraded_psnr:.2f} dB"
        )

        print(
            f"Average SSIM: "
            f"{avg_degraded_ssim:.4f}"
        )

    if restored_count > 0:

        avg_restored_psnr = (
            total_restored_psnr
            / restored_count
        )

        avg_restored_ssim = (
            total_restored_ssim
            / restored_count
        )

        print()
        print("After restoration:")
        print(
            f"Average PSNR: "
            f"{avg_restored_psnr:.2f} dB"
        )

        print(
            f"Average SSIM: "
            f"{avg_restored_ssim:.4f}"
        )

        if degraded_count > 0:

            psnr_improvement = (
                avg_restored_psnr
                - avg_degraded_psnr
            )

            ssim_improvement = (
                avg_restored_ssim
                - avg_degraded_ssim
            )

            print()
            print("Improvement:")
            print(
                f"PSNR improvement: "
                f"{psnr_improvement:.2f} dB"
            )

            print(
                f"SSIM improvement: "
                f"{ssim_improvement:.4f}"
            )

    else:

        print()
        print(
            "No restored images found."
        )

        print(
            "Run inference.py first."
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
