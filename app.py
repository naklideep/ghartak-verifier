from flask import Flask, request, jsonify
from PIL import Image, ImageChops, ImageEnhance, UnidentifiedImageError
import io
import cv2
import numpy as np
import pytesseract
import re
import base64

app = Flask(__name__)

# College name keywords (partial match allowed)
COLLEGE_KEYWORDS = [
    "UKA", "TARSADIA", "UNIVERSITY", "SRIMCA", "COLLEGE", "COMPUTER APPLICATIONS"
]

# Thresholds for tamper detection
ELA_THRESHOLD = 25
NOISE_THRESHOLD = 100

# Extract 15-digit enrollment
def extract_enrollment(text):
    match = re.search(r'\b\d{15}\b', text)
    return match.group() if match else None

# Check if college name exists (partial match)
def check_college_name(text):
    text_upper = text.upper()
    for keyword in COLLEGE_KEYWORDS:
        if keyword in text_upper:
            return True
    return False

# Error Level Analysis (ELA) + Noise check
def tamper_check(pil_img):
    try:
        ela_path = "temp_ela.jpg"
        pil_img.save(ela_path, 'JPEG', quality=90)
        ela_image = Image.open(ela_path)
        diff = ImageChops.difference(pil_img, ela_image)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff else 1
        ela_image = ImageEnhance.Brightness(diff).enhance(scale)
        ela_score = round(max_diff, 2)

        # Noise / blur
        np_img = np.array(pil_img)[:, :, ::-1].copy()
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = round(blur, 2)

        is_tampered = ela_score > ELA_THRESHOLD or noise_score < NOISE_THRESHOLD
        return is_tampered, ela_score, noise_score
    except Exception as e:
        return True, -1, -1  # treat as tampered if error

@app.route('/')
def home():
    return jsonify({"message": "🔥 Ghartak Verifier API active"})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image_base64")
        if not image_b64:
            return jsonify({"error": "Missing 'image_base64'"}), 400

        # Decode base64
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except (UnidentifiedImageError, Exception) as e:
            return jsonify({"error": f"Image decode/open failed: {e}"}), 400

        # OCR
        try:
            ocr_text = pytesseract.image_to_string(image)
            ocr_text_clean = ocr_text.upper().strip()
        except Exception as e:
            ocr_text_clean = ""
        
        # Enrollment and college check
        enrollment_number = extract_enrollment(ocr_text_clean)
        college_ok = check_college_name(ocr_text_clean)

        # Tamper check
        tampered, ela_score, noise_score = tamper_check(image)

        accepted = bool(enrollment_number and college_ok and not tampered)

        return jsonify({
            "accepted": accepted,
            "tampered": tampered,
            "ela_score": ela_score,
            "noise_score": noise_score,
            "enrollment_number": enrollment_number,
            "college_ok": college_ok,
            "ocr_excerpt": ocr_text_clean[:300]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
