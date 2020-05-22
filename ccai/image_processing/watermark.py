"""General image processing utilities"""

import os

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

# WATERMARK_TEXT = "For development use only."
WATERMARK_TEXT = ""


def apply_watermark(input_image_path: str, output_image_path: str) -> None:
    """Apply a watermark to an image at a given path"""
    photo = Image.open(input_image_path)
    drawing = ImageDraw.Draw(photo)
    font_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "helvetica.ttf")
    font = ImageFont.truetype(font_file, 48)
    pos = (0, 0)
    drawing.text(pos, WATERMARK_TEXT, fill=ImageColor.getrgb("red"), font=font)
    photo.save(output_image_path)
