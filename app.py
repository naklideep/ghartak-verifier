from flask import Flask, request, jsonify
import requests
from PIL import Image, ImageChops, ImageEnhance
import io
import cv2
import numpy as np
import pytesseract
import re

app = Flask(__name__)

# --- UTILITIES ---
def extract_name(text):
    """Extract likely student name like: 'DEEP H. CHAUDHARI' or 'FIRSTNAME M LASTNAME'."""
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


@app.route('/')
def home():
    return jsonify({"message": "🔥 Ghartak Verifier API active"})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 🔹 Get image from form (accept both 'file' or 'image')
        uploaded_file = request.files.get('file') or request.files.get('image')
        if not uploaded_file:
            return jsonify({"error": "No image uploaded. Field name must be 'file' or 'image'."}), 400

        # 🔹 Load image
        image = Image.open(uploaded_file.stream).convert('RGB')
        np_img = np.array(image)[:, :, ::-1].copy()  # PIL → OpenCV (BGR)

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

        # --- BLUR / NOISE CHECK ---
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = round(blur, 2)

        tampered = ela_score > 25 or noise_score < 100

        # --- OCR ---
        ocr_text = pytesseract.image_to_string(image)
        ocr_text_clean = ocr_text.upper().strip()

        # --- Extract info ---
        enrollment_number = extract_enrollment(ocr_text_clean)
        name_detected = extract_name(ocr_text_clean)

        # --- Final result ---
        accepted = bool(enrollment_number and not tampered)

        return jsonify({
            "accepted": accepted,
            "tampered": tampered,
            "ela_score": ela_score,
            "noise_score": noise_score,
            "enrollment_number": enrollment_number,
            "name_detected": name_detected,
            "ocr_excerpt": ocr_text_clean[:300]
        })

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
