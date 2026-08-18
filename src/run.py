from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from model import ImageRestorationUNet


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "restoration_model.pth",
    PROJECT_ROOT / "model" / "restoration_model.pth",
]

IMAGE_SIZE = 256


# ============================================================
# LOAD NUMPY IMAGE
# ============================================================

def load_npy(path):
    image = np.load(path)

    if not isinstance(image, np.ndarray):
        raise ValueError(f"Invalid NumPy file: {path}")

    if image.size == 0:
        raise ValueError(f"Empty image: {path}")

    if not np.isfinite(image.astype(np.float32)).all():
        raise ValueError(f"Input contains NaN or Inf: {path}")

    return image


# ============================================================
# NORMALIZE INPUT
# ============================================================

def normalize_image(image):
    """
    Convert input to float32 [0,1].
    Supports:
        (H,W)
        (H,W,1)
        (H,W,3)
    """

    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] >= 3:
            image = image[:, :, :3]
        else:
            raise ValueError(
                f"Unsupported image shape: {image.shape}"
            )

    image = image.astype(np.float32)

    # Convert common [0,255] data to [0,1]
    if image.max() > 1.0:
        image = image / 255.0

    image = np.clip(image, 0.0, 1.0)

    return image


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_tensor(image, device):
    """
    Convert grayscale image to 3-channel tensor.
    Resize to model size.
    """

    original_h, original_w = image.shape[:2]

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Model expects 3 channels
    if resized.ndim == 2:
        rgb = np.stack(
            [resized, resized, resized],
            axis=0
        )
    else:
        rgb = np.transpose(
            resized,
            (2, 0, 1)
        )

        if rgb.shape[0] == 1:
            rgb = np.repeat(
                rgb,
                3,
                axis=0
            )

        elif rgb.shape[0] > 3:
            rgb = rgb[:3]

    tensor = torch.from_numpy(
        rgb.astype(np.float32)
    ).unsqueeze(0)

    return tensor.to(device), original_h, original_w


# ============================================================
# FIND MODEL
# ============================================================

def find_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(
        str(path) for path in MODEL_CANDIDATES
    )

    raise FileNotFoundError(
        "Trained model not found.\n\n"
        "Searched:\n"
        f"{searched}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    model_path = find_model()

    print(f"Model: {model_path}")

    model = ImageRestorationUNet().to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Support both plain state_dict and checkpoint dictionaries
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# RESTORE ONE IMAGE
# ============================================================

def restore_image(model, image, device):

    tensor, original_h, original_w = prepare_tensor(
        image,
        device
    )

    with torch.no_grad():

        output = model(tensor)

    # Remove batch dimension
    output = output.squeeze(0)

    # Convert CHW -> HWC
    output = output.detach().cpu().numpy()

    if output.ndim == 3:
        output = np.transpose(
            output,
            (1, 2, 0)
        )

    # Convert output to grayscale
    if output.ndim == 3:

        if output.shape[2] == 1:
            output = output[:, :, 0]

        else:
            output = np.mean(
                output[:, :, :3],
                axis=2
            )

    elif output.ndim != 2:
        raise ValueError(
            f"Unexpected model output shape: {output.shape}"
        )

    # Ensure [0,1]
    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    output = np.clip(
        output,
        0.0,
        1.0
    )

    # Restore original resolution
    output = cv2.resize(
        output,
        (original_w, original_h),
        interpolation=cv2.INTER_CUBIC
    )

    output = np.clip(
        output,
        0.0,
        1.0
    ).astype(
        np.float32
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AI SEMICONDUCTOR IMAGE RESTORATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    if len(sys.argv) != 3:

        print()
        print(
            "Usage:"
        )
        print(
            "python run.py <input-dir> <output-dir>"
        )
        print()

        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    print(
        f"Input : {input_dir}"
    )

    print(
        f"Output: {output_dir}"
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not input_dir.exists():

        print()
        print(
            f"[ERROR] Input directory not found:"
        )
        print(input_dir)

        sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Find .npy files
    # --------------------------------------------------------

    image_files = sorted(
        input_dir.glob("*.npy")
    )

    if not image_files:

        print()
        print(
            "[ERROR] No .npy files found."
        )

        sys.exit(1)

    print()
    print(
        f"Found {len(image_files)} input file(s)."
    )

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    processed = 0
    failed = 0

    for index, input_path in enumerate(
        image_files,
        start=1
    ):

        try:

            image = load_npy(
                input_path
            )

            normalized = normalize_image(
                image
            )

            restored = restore_image(
                model,
                normalized,
                device
            )

            # ------------------------------------------------
            # IMPORTANT:
            # SAME filename as input
            # ------------------------------------------------

            output_path = (
                output_dir
                / input_path.name
            )

            # ------------------------------------------------
            # Final validation
            # ------------------------------------------------

            if restored.ndim != 2:

                raise ValueError(
                    f"Output must be grayscale (H,W), "
                    f"got {restored.shape}"
                )

            if not np.isfinite(
                restored
            ).all():

                raise ValueError(
                    "Output contains NaN or Inf"
                )

            if restored.min() < 0.0:
                raise ValueError(
                    "Output contains values below 0"
                )

            if restored.max() > 1.0:
                raise ValueError(
                    "Output contains values above 1"
                )

            # Save as float32 NumPy array
            np.save(
                output_path,
                restored.astype(np.float32)
            )

            processed += 1

            print(
                f"[{index}/{len(image_files)}] "
                f"{input_path.name} -> OK "
                f"shape={restored.shape}"
            )

        except Exception as error:

            failed += 1

            print(
                f"[ERROR] {input_path.name}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RESTORATION COMPLETED"
    )
    print(
        f"Processed files: {processed}"
    )
    print(
        f"Failed files:    {failed}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print("=" * 70)
    print()

    if failed > 0:
        sys.exit(1)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()