"""
Script to build a realistic deer-head app icon from deer.jpg.
Keeps everything local/offline and outputs a 1024x1024 PNG at icon.png.
"""
from PIL import Image, ImageDraw, ImageFilter
import os

ROOT = os.path.dirname(__file__)


def _soft_mask(size: tuple[int, int], blur: int, inset: int) -> Image.Image:
    """Ellipse mask with soft edges to hide background/hands."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (inset, inset, size[0] - inset, size[1] - inset),
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=blur))


def create_realistic_icon(size: int = 1024) -> Image.Image:
    """Crop the provided deer photo to a head/antler icon."""
    src_path = os.path.join(ROOT, "deer.jpg")
    img = Image.open(src_path).convert("RGBA")
    w, h = img.size

    # Focus on head/antlers region (relative crop keeps this robust if photo changes size)
    crop_box = (
        int(w * 0.52),
        int(h * 0.22),
        int(w * 0.96),
        int(h * 0.86),
    )
    head = img.crop(crop_box)

    # Resize and apply a soft elliptical mask to trim out background/person
    head_size = int(size * 0.9)
    head = head.resize((head_size, head_size), Image.Resampling.LANCZOS)
    mask = _soft_mask(head.size, blur=int(size * 0.02), inset=int(size * 0.04))

    # Deep forest background so antlers stand out
    bg = Image.new("RGBA", (size, size), (24, 36, 26, 255))
    offset = ((size - head_size) // 2, (size - head_size) // 2)
    bg.paste(head, offset, mask)
    return bg


if __name__ == "__main__":
    icon = create_realistic_icon(1024)
    icon_path = os.path.join(ROOT, "icon.png")
    icon.save(icon_path, "PNG", optimize=True)
    print(f"Icon created: {icon_path}")
    print(f"Size: {icon.size[0]}x{icon.size[1]} pixels")
