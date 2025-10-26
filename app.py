from flask import Flask, request, jsonify
from PIL import Image, ImageChops, ImageEnhance
import io
import cv2
import numpy as np
import pytesseract
import re
import base64

app = Flask(__name__)

# ✅ College names (uppercased for comparison)
COLLEGE_NAMES = [
    "UKA TARSADIA UNIVERSITY",
    "TARSADIA UNIVERSITY",
    "TARSADIA",
    "UNIVERS",
    "UKA"
]

# 🧩 Extract 15-digit enrollment number
def extract_enrollment(text):
    match = re.search(r'\b\d{15}\b', text)
    return match.group() if match else None

# 🧩 Fuzzy college name check
def check_college_name(text):
    text_upper = text.upper()
    for name in COLLEGE_NAMES:
        name_parts = name.split()
        match_count = sum(1 for part in name_parts if part in text_upper)
        if match_count / len(name_parts) >= 0.5:
            return True
    return False

# 🧩 Tamper check via ELA + blur
def is_tampered(cv_img, pil_img):
    # --- ELA ---
    ela_path = "temp_ela.jpg"
    pil_img.save(ela_path, 'JPEG', quality=90)
    diff = ImageChops.difference(pil_img, Image.open(ela_path))
    max_diff = max([ex[1] for ex in diff.getextrema()])
    ela_score = round(max_diff, 2)

    # --- Blur / noise ---
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_score = round(blur, 2)

    tampered = ela_score > 35 or noise_score < 80
    return tampered, ela_score, noise_score

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

        # --- Decode base64 ---
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        np_img = np.array(image)[:, :, ::-1].copy()  # PIL → OpenCV (BGR)

        # --- OCR with preprocessing ---
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ocr_text = pytesseract.image_to_string(thresh, config="--psm 6")
        ocr_text_clean = ocr_text.upper().strip()

        # --- Extract enrollment & college ---
        enrollment_number = extract_enrollment(ocr_text_clean)
        college_ok = check_college_name(ocr_text_clean)

        # --- Tamper check ---
        tampered, ela_score, noise_score = is_tampered(np_img, image)

        # --- Final decision ---
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
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
