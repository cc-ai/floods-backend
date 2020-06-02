import os, base64
from flask import jsonify
from ccai.config import CONFIG
from ccai.image_processing.streetview import fetch_street_view_image
from ccai.image_processing.watermark import apply_watermark


def create_temp_dir(images, temp_dir):

    images.download_links(temp_dir)
    files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
    if "gsv_0.jpg" not in files:
        response = jsonify({"error": "Image not found in response from Google Street View"})
        response.status_code = 200
        return response


def fetch_image(address):

    try:
        images = fetch_street_view_image(
            address, CONFIG.GEO_CODER_API_KEY, CONFIG.STREET_VIEW_API_KEY
        )
        return images
    except Exception as exception:  # pylint: disable=W0703
        response = jsonify(
            {
                "error": "An error occurred fetching the Google Street View image",
                "exception_text": str(exception),
            }
        )
        response.status_code = 500
        return response


def encode_image(temp_dir):

    path_to_gsv_image = os.path.join(temp_dir, "gsv_0.jpg")
    path_to_gsv_image_watermarked = os.path.join(temp_dir, "gsv_watermarked.jpg")
    apply_watermark(path_to_gsv_image, path_to_gsv_image_watermarked)
    with open(path_to_gsv_image_watermarked, "rb") as gsv_image_handle:
        gsv_image_data = gsv_image_handle.read()
    gsv_image_encoded = base64.b64encode(gsv_image_data)
    gsv_image_response = gsv_image_encoded.decode("ascii")

    return os.path.join(temp_dir, "gsv_0.jpg"), gsv_image_response


def decode_image(temp_dir, path_to_flooded_image):
    path_to_flooded_image_watermarked = os.path.join(temp_dir, "flooded_watermarked.jpg")
    apply_watermark(path_to_flooded_image, path_to_flooded_image_watermarked)

    with open(path_to_flooded_image_watermarked, "rb") as flooded_image_handle:
        flooded_image_data = flooded_image_handle.read()
    flooded_image_encoded = base64.b64encode(flooded_image_data)
    flooded_image_response = flooded_image_encoded.decode("ascii")

    return flooded_image_response
