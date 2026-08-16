from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import lpips

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_DIR = RESULTS_DIR / "baseline"

RESULTS_CSV = RESULTS_DIR / "comparison_results.csv"


# --------------------------------------------------
# LPIPS model
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"Using device for LPIPS: {DEVICE}"
)

LPIPS_MODEL = lpips.LPIPS(
    net="alex"
).to(DEVICE)

LPIPS_MODEL.eval()


# --------------------------------------------------
# Supported image extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# Load grayscale image
# --------------------------------------------------

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return image


# --------------------------------------------------
# Resize image to reference size
# --------------------------------------------------

def resize_to_reference(
    reference,
    image
):

    height, width = reference.shape[:2]

    if image.shape != reference.shape:

        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    return image


# --------------------------------------------------
# PSNR and SSIM
# --------------------------------------------------

def calculate_psnr_ssim(
    reference,
    image
):

    image = resize_to_reference(
        reference,
        image
    )

    reference_float = (
        reference.astype(np.float32)
        / 255.0
    )

    image_float = (
        image.astype(np.float32)
        / 255.0
    )

    psnr = peak_signal_noise_ratio(
        reference_float,
        image_float,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference_float,
        image_float,
        data_range=1.0
    )

    return psnr, ssim


# --------------------------------------------------
# Convert grayscale image for LPIPS
# --------------------------------------------------

def prepare_for_lpips(image):

    image = image.astype(
        np.float32
    ) / 255.0

    # Convert grayscale [H, W]
    # to RGB-like [3, H, W]

    image = np.stack(
        [image, image, image],
        axis=0
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    # LPIPS expects values in [-1, 1]

    tensor = tensor * 2.0 - 1.0

    return tensor.to(DEVICE)


# --------------------------------------------------
# Calculate LPIPS
# --------------------------------------------------

def calculate_lpips(
    reference,
    image
):

    image = resize_to_reference(
        reference,
        image
    )

    reference_tensor = prepare_for_lpips(
        reference
    )

    image_tensor = prepare_for_lpips(
        image
    )

    with torch.no_grad():

        distance = LPIPS_MODEL(
            reference_tensor,
            image_tensor
        )

    return float(
        distance.item()
    )


# --------------------------------------------------
# Calculate all metrics
# --------------------------------------------------

def calculate_all_metrics(
    reference,
    image
):

    image = resize_to_reference(
        reference,
        image
    )

    psnr, ssim = calculate_psnr_ssim(
        reference,
        image
    )

    lpips_score = calculate_lpips(
        reference,
        image
    )

    return (
        psnr,
        ssim,
        lpips_score
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("RESTORATION METHOD COMPARISON")
    print("=" * 80)
    print()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CLEAN_DIR.exists():

        print(
            "Clean directory not found:"
        )

        print(CLEAN_DIR)

        return

    # --------------------------------------------------
    # Find clean images
    # --------------------------------------------------

    clean_files = [
        file
        for file in CLEAN_DIR.iterdir()
        if file.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    if not clean_files:

        print(
            "No clean images found."
        )

        print(
            f"Expected images in: {CLEAN_DIR}"
        )

        return

    print(
        f"Found {len(clean_files)} clean image(s)."
    )

    print()

    # --------------------------------------------------
    # Storage for averages
    # --------------------------------------------------

    total = {
        "degraded_psnr": 0.0,
        "degraded_ssim": 0.0,
        "degraded_lpips": 0.0,

        "baseline_psnr": 0.0,
        "baseline_ssim": 0.0,
        "baseline_lpips": 0.0,

        "ai_psnr": 0.0,
        "ai_ssim": 0.0,
        "ai_lpips": 0.0,
    }

    counts = {
        "degraded": 0,
        "baseline": 0,
        "ai": 0,
    }

    # --------------------------------------------------
    # Process every image
    # --------------------------------------------------

    for clean_path in clean_files:

        print(
            f"Image: {clean_path.name}"
        )

        clean = load_image(
            clean_path
        )

        # --------------------------------------------------
        # Degraded image
        # --------------------------------------------------

        degraded_path = (
            DEGRADED_DIR
            / clean_path.name
        )

        if degraded_path.exists():

            degraded = load_image(
                degraded_path
            )

            psnr, ssim, lpips_score = (
                calculate_all_metrics(
                    clean,
                    degraded
                )
            )

            total[
                "degraded_psnr"
            ] += psnr

            total[
                "degraded_ssim"
            ] += ssim

            total[
                "degraded_lpips"
            ] += lpips_score

            counts[
                "degraded"
            ] += 1

            print(
                f"  Degraded : "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f} | "
                f"LPIPS={lpips_score:.4f}"
            )

        else:

            print(
                "  Degraded : image not found"
            )

        # --------------------------------------------------
        # Traditional baseline
        # --------------------------------------------------

        baseline_path = (
            BASELINE_DIR
            / f"baseline_{clean_path.name}"
        )

        if baseline_path.exists():

            baseline = load_image(
                baseline_path
            )

            psnr, ssim, lpips_score = (
                calculate_all_metrics(
                    clean,
                    baseline
                )
            )

            total[
                "baseline_psnr"
            ] += psnr

            total[
                "baseline_ssim"
            ] += ssim

            total[
                "baseline_lpips"
            ] += lpips_score

            counts[
                "baseline"
            ] += 1

            print(
                f"  Baseline : "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f} | "
                f"LPIPS={lpips_score:.4f}"
            )

        else:

            print(
                "  Baseline : image not found"
            )

        # --------------------------------------------------
        # AI restored image
        # --------------------------------------------------

        ai_path = (
            RESULTS_DIR
            / f"restored_{clean_path.name}"
        )

        if ai_path.exists():

            ai = load_image(
                ai_path
            )

            psnr, ssim, lpips_score = (
                calculate_all_metrics(
                    clean,
                    ai
                )
            )

            total[
                "ai_psnr"
            ] += psnr

            total[
                "ai_ssim"
            ] += ssim

            total[
                "ai_lpips"
            ] += lpips_score

            counts[
                "ai"
            ] += 1

            print(
                f"  AI Model : "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f} | "
                f"LPIPS={lpips_score:.4f}"
            )

        else:

            print(
                "  AI Model : image not found"
            )

        print()

    # --------------------------------------------------
    # Calculate average results
    # --------------------------------------------------

    print("=" * 80)
    print("AVERAGE RESULTS")
    print("=" * 80)

    averages = []

    methods = [
        ("Degraded Input", "degraded"),
        ("Traditional Baseline", "baseline"),
        ("Our AI Model", "ai"),
    ]

    for method_name, key in methods:

        if counts[key] == 0:

            continue

        average_psnr = (
            total[f"{key}_psnr"]
            / counts[key]
        )

        average_ssim = (
            total[f"{key}_ssim"]
            / counts[key]
        )

        average_lpips = (
            total[f"{key}_lpips"]
            / counts[key]
        )

        averages.append(
            (
                method_name,
                average_psnr,
                average_ssim,
                average_lpips
            )
        )

        print(
            f"{method_name:22s} "
            f"PSNR={average_psnr:.2f} dB | "
            f"SSIM={average_ssim:.4f} | "
            f"LPIPS={average_lpips:.4f}"
        )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Method",
            "Average PSNR (dB)",
            "Average SSIM",
            "Average LPIPS"
        ])

        for (
            method_name,
            average_psnr,
            average_ssim,
            average_lpips
        ) in averages:

            writer.writerow([
                method_name,
                f"{average_psnr:.4f}",
                f"{average_ssim:.4f}",
                f"{average_lpips:.4f}"
            ])

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    print()
    print("=" * 80)

    print(
        "Results successfully saved to:"
    )

    print(
        RESULTS_CSV
    )

    print("=" * 80)
    print()


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()
