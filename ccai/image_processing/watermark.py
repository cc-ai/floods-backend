"""General image processing utilities"""

import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

WATERMARK_TEXT = "For development use only."


def apply_watermark(input_image_path: str, output_image_path: str) -> None:
    """Apply a watermark to an image at a given path"""
    photo = Image.open(input_image_path)
    drawing = ImageDraw.Draw(photo)
    font_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "helvetica.ttf")
    font = ImageFont.truetype(font_file, 20)
    black = (3, 8, 12)
    pos = (0, 0)
    drawing.text(pos, WATERMARK_TEXT, fill=black, font=font)
    photo.save(output_image_path)
