from flask import Flask, request, jsonify
import requests
from PIL import Image, ImageChops, ImageEnhance
import io
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Ghartak Verifier API active"})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "Missing image_url"}), 400

    try:
        # Download image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')

        # --- ELA ANALYSIS ---
        ela_path = "temp_ela.jpg"
        image.save(ela_path, 'JPEG', quality=90)
        ela_image = Image.open(ela_path)
        diff = ImageChops.difference(image, ela_image)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff else 1
        ela_image = ImageEnhance.Brightness(diff).enhance(scale)

        # --- OpenCV NOISE CHECK ---
        np_img = np.array(image)
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        ela_score = round(max_diff, 2)
        noise_score = round(blur, 2)
        tampered = ela_score > 25 or noise_score < 100  # tweak as needed

        return jsonify({
            "tampered": tampered,
            "ela_score": ela_score,
            "noise_score": noise_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
