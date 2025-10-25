from flask import Flask, request, jsonify
import requests
from PIL import Image, ImageChops, ImageEnhance
import io
import cv2
import numpy as np
import pytesseract
import re
import os

app = Flask(__name__)

LOGO_TEMPLATE_PATH = "templates/logo.png"  # your logo image


def extract_name(text):
    """Extract likely student name in uppercase."""
    name_pattern = re.compile(r'\b[A-Z]{2,}(?:\s+[A-Z]\.)?(?:\s+[A-Z]{2,})\b')
    matches = name_pattern.findall(text)
    for m in matches:
        if 2 <= len(m.split()) <= 3:
            return m.strip()
    return None


def extract_enrollment(text):
    """Extract a 15-digit enrollment number."""
    match = re.search(r'\b\d{15}\b', text)
    return match.group() if match else None


def logo_match(cv_img, template_path=LOGO_TEMPLATE_PATH, threshold=0.45):
    """Check if logo appears in the image."""
    if not os.path.exists(template_path):
        print("⚠️ Logo template not found.")
        return False

    try:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            return False

        img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold
    except Exception as e:
        print("logo_match error:", e)
        return False


@app.route('/')
def home():
    return jsonify({"message": "🔥 Ghartak Verifier API active"})


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "Missing image_url"}), 400

    try:
        # --- DOWNLOAD IMAGE ---
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        np_img = np.array(image)[:, :, ::-1].copy()  # PIL -> OpenCV BGR

        # --- ELA ANALYSIS ---
        ela_path = "temp_ela.jpg"
        image.save(ela_path, 'JPEG', quality=90)
        ela_image = Image.open(ela_path)
        diff = ImageChops.difference(image, ela_image)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff else 1
        ela_image = ImageEnhance.Brightness(diff).enhance(scale)
        ela_score = round(max_diff, 2)

        # --- NOISE / BLUR CHECK ---
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = round(blur, 2)
        tampered = ela_score > 25 or noise_score < 100

        # --- OCR ---
        ocr_text = pytesseract.image_to_string(image)
        ocr_text_clean = ocr_text.upper().strip()

        # --- EXTRACT DETAILS ---
        enrollment_number = extract_enrollment(ocr_text_clean)
        name_detected = extract_name(ocr_text_clean)
        has_logo = logo_match(np_img, threshold=0.45)

        # --- FINAL DECISION ---
        accepted = bool(enrollment_number and has_logo and not tampered)

        return jsonify({
            "accepted": accepted,
            "tampered": tampered,
            "ela_score": ela_score,
            "noise_score": noise_score,
            "enrollment_number": enrollment_number,
            "name_detected": name_detected,
            "has_logo": has_logo,
            "ocr_excerpt": ocr_text_clean[:300]
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
