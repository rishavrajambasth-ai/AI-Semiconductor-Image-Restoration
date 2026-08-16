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


# --------------------------------------------------
# Image loading
# --------------------------------------------------

def load_gray_image(path):
    """
    Load an image as grayscale.
    """

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
# Edge extraction
# --------------------------------------------------

def extract_edges(image):
    """
    Extract fine structural details using Canny edges.
    """

    edges = cv2.Canny(
        image,
        threshold1=50,
        threshold2=150
    )

    return edges


# --------------------------------------------------
# Detail preservation score
# --------------------------------------------------

def detail_preservation_score(
    reference,
    restored
):
    """
    Compare structural details between the clean
    reference image and restored image.

    Returns a score between 0 and 1.
    Higher = better preservation of details.
    """

    if reference.shape != restored.shape:

        restored = cv2.resize(
            restored,
            (
                reference.shape[1],
                reference.shape[0]
            )
        )

    reference_edges = extract_edges(
        reference
    )

    restored_edges = extract_edges(
        restored
    )

    reference_edges = (
        reference_edges > 0
    )

    restored_edges = (
        restored_edges > 0
    )

    # True positives:
    # details present in both images
    preserved = np.logical_and(
        reference_edges,
        restored_edges
    ).sum()

    # Number of reference details
    reference_count = (
        reference_edges.sum()
    )

    if reference_count == 0:
        return 1.0

    score = (
        preserved / reference_count
    )

    return float(score)


# --------------------------------------------------
# Edge similarity
# --------------------------------------------------

def edge_similarity(
    reference,
    restored
):
    """
    Calculate similarity between edge maps.
    """

    if reference.shape != restored.shape:

        restored = cv2.resize(
            restored,
            (
                reference.shape[1],
                reference.shape[0]
            )
        )

    reference_edges = extract_edges(
        reference
    )

    restored_edges = extract_edges(
        restored
    )

    reference_edges = (
        reference_edges.astype(
            np.float32
        ) / 255.0
    )

    restored_edges = (
        restored_edges.astype(
            np.float32
        ) / 255.0
    )

    intersection = (
        reference_edges * restored_edges
    ).sum()

    reference_total = (
        reference_edges.sum()
    )

    restored_total = (
        restored_edges.sum()
    )

    denominator = (
        reference_total
        + restored_total
    )

    if denominator == 0:
        return 1.0

    similarity = (
        2.0 * intersection
        / denominator
    )

    return float(similarity)


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

def main():

    print()
    print("=" * 65)
    print("DEFECT / DETAIL PRESERVATION ANALYSIS")
    print("=" * 65)
    print()

    if not CLEAN_DIR.exists():

        print(
            "Clean directory not found:"
        )
        print(CLEAN_DIR)
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

        print("No clean images found.")
        return

    total_dps = 0.0
    total_similarity = 0.0
    count = 0

    for clean_path in clean_files:

        restored_path = (
            RESULTS_DIR
            / f"restored_{clean_path.name}"
        )

        if not restored_path.exists():

            print(
                f"[SKIP] Restored image not found: "
                f"{clean_path.name}"
            )

            continue

        clean = load_gray_image(
            clean_path
        )

        restored = load_gray_image(
            restored_path
        )

        dps = detail_preservation_score(
            clean,
            restored
        )

        similarity = edge_similarity(
            clean,
            restored
        )

        total_dps += dps
        total_similarity += similarity
        count += 1

        print(
            f"{clean_path.name}"
        )

        print(
            f"  Detail Preservation Score: "
            f"{dps:.4f}"
        )

        print(
            f"  Edge Similarity: "
            f"{similarity:.4f}"
        )

        print()

    if count == 0:

        print(
            "No image pairs were available "
            "for analysis."
        )

        return

    average_dps = (
        total_dps / count
    )

    average_similarity = (
        total_similarity / count
    )

    print("=" * 65)
    print("AVERAGE RESULTS")
    print("=" * 65)

    print(
        f"Images analyzed: {count}"
    )

    print(
        f"Average Detail Preservation Score: "
        f"{average_dps:.4f}"
    )

    print(
        f"Average Edge Similarity: "
        f"{average_similarity:.4f}"
    )

    print()
    print(
        "Higher scores indicate better preservation "
        "of fine image structures."
    )

    print()
    print(
        "Note: This is a detail-preservation proxy, "
        "not a confirmed semiconductor-defect detector."
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
