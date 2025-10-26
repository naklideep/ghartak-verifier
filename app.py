from flask import Flask, request, jsonify
from PIL import Image, ImageChops, ImageEnhance, UnidentifiedImageError
import io
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Thresholds
ELA_THRESHOLD = 25
NOISE_THRESHOLD = 100

def tamper_check(pil_img):
    try:
        # Save & reload for ELA
        ela_path = "temp_ela.jpg"
        pil_img.save(ela_path, 'JPEG', quality=90)
        ela_image = Image.open(ela_path)
        diff = ImageChops.difference(pil_img, ela_image)
        max_diff = max([ex[1] for ex in diff.getextrema()])
        scale = 255.0 / max_diff if max_diff else 1
        ImageEnhance.Brightness(diff).enhance(scale)
        ela_score = round(max_diff, 2)

        # Noise / blur
        np_img = np.array(pil_img)[:, :, ::-1].copy()
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = round(blur, 2)

        tampered = bool(ela_score > ELA_THRESHOLD or noise_score < NOISE_THRESHOLD)
        return tampered, float(ela_score), float(noise_score)
    except Exception as e:
        return True, -1.0, -1.0  # treat as tampered if any error


@app.route('/')
def home():
    return jsonify({"message": "🔥 Ghartak Tamper Checker Active"})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image_base64")
        if not image_b64:
            return jsonify({"error": "Missing 'image_base64'"}), 400

        # Decode image
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Image decode/open failed: {e}"}), 400

        # Tamper check
        tampered, ela_score, noise_score = tamper_check(image)

        return jsonify({
            "tampered": tampered,
            "ela_score": ela_score,
            "noise_score": noise_score
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
