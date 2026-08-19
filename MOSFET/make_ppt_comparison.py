from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

INPUT_DIR = Path(r"..\data\degraded")
OUTPUT_DIR = Path(r".\test_output_256")
SAVE_DIR = Path(r".\ppt_comparison")

SAVE_DIR.mkdir(exist_ok=True)

names = [
    f"{i:06d}.npy"
    for i in range(100)
]


def to_display(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    # Contrast stretching for visualization only.
    low = np.percentile(arr, 1)
    high = np.percentile(arr, 99)

    if high > low:
        arr = (arr - low) / (high - low)

    arr = np.clip(arr, 0.0, 1.0)

    return Image.fromarray(
        (arr * 255).astype(np.uint8)
    ).convert("RGB")


for name in names:

    before = np.load(INPUT_DIR / name)
    after = np.load(OUTPUT_DIR / name)

    before_img = to_display(before).resize((512, 512))
    after_img = to_display(after).resize((512, 512))

    canvas = Image.new(
        "RGB",
        (1024, 600),
        "white"
    )

    canvas.paste(before_img, (0, 70))
    canvas.paste(after_img, (512, 70))

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (220, 20),
        "BEFORE - Degraded",
        fill="black"
    )

    draw.text(
        (700, 20),
        "AFTER - AI Restored",
        fill="black"
    )

    draw.text(
        (470, 560),
        name,
        fill="black"
    )

    output_path = SAVE_DIR / f"{Path(name).stem}_before_after.png"

    canvas.save(output_path)

    print(f"Created: {output_path}")


print("Done.")