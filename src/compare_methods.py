from pathlib import Path

import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_DIR = RESULTS_DIR / "baseline"


def load_image(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read: {path}"
        )

    return image


def resize_to_reference(reference, image):

    height, width = reference.shape[:2]

    if image.shape != reference.shape:

        image = cv2.resize(
            image,
            (width, height)
        )

    return image


def calculate_metrics(reference, image):

    image = resize_to_reference(
        reference,
        image
    )

    reference_float = (
        reference.astype("float32") / 255.0
    )

    image_float = (
        image.astype("float32") / 255.0
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


def main():

    print()
    print("=" * 75)
    print("RESTORATION METHOD COMPARISON")
    print("=" * 75)
    print()

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

        print("No clean images found.")

        return

    total = {
        "degraded_psnr": 0.0,
        "degraded_ssim": 0.0,
        "baseline_psnr": 0.0,
        "baseline_ssim": 0.0,
        "ai_psnr": 0.0,
        "ai_ssim": 0.0,
    }

    counts = {
        "degraded": 0,
        "baseline": 0,
        "ai": 0,
    }

    for clean_path in clean_files:

        clean = load_image(
            clean_path
        )

        print(
            clean_path.name
        )

        # --------------------------------------------------
        # Degraded
        # --------------------------------------------------

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

            total["degraded_psnr"] += psnr
            total["degraded_ssim"] += ssim

            counts["degraded"] += 1

            print(
                f"  Degraded : "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f}"
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

            psnr, ssim = calculate_metrics(
                clean,
                baseline
            )

            total["baseline_psnr"] += psnr
            total["baseline_ssim"] += ssim

            counts["baseline"] += 1

            print(
                f"  Baseline : "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f}"
            )

        # --------------------------------------------------
        # AI restoration
        # --------------------------------------------------

        ai_path = (
            RESULTS_DIR
            / f"restored_{clean_path.name}"
        )

        if ai_path.exists():

            ai = load_image(
                ai_path
            )

            psnr, ssim = calculate_metrics(
                clean,
                ai
            )

            total["ai_psnr"] += psnr
            total["ai_ssim"] += ssim

            counts["ai"] += 1

            print(
                f"  AI Model : "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f}"
            )

        print()

    # --------------------------------------------------
    # Average results
    # --------------------------------------------------

    print("=" * 75)
    print("AVERAGE RESULTS")
    print("=" * 75)

    for method, key in [
        ("Degraded", "degraded"),
        ("Traditional Baseline", "baseline"),
        ("AI Model", "ai"),
    ]:

        if counts[key] == 0:

            continue

        psnr = (
            total[f"{key}_psnr"]
            / counts[key]
        )

        ssim = (
            total[f"{key}_ssim"]
            / counts[key]
        )

        print(
            f"{method:22s} "
            f"PSNR: {psnr:.2f} dB | "
            f"SSIM: {ssim:.4f}"
        )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()
