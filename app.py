from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import io
import pytesseract
import re
import base64

app = Flask(__name__)

# Extract 15-digit enrollment
def extract_enrollment(text):
    match = re.search(r'\b\d{15}\b', text)
    return match.group() if match else None

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

        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Image decode/open failed: {e}"}), 400

        # OCR
        ocr_text = pytesseract.image_to_string(image)
        ocr_text_clean = ocr_text.strip().upper()

        # Enrollment
        enrollment_number = extract_enrollment(ocr_text_clean)

        return jsonify({
            "enrollment_number": enrollment_number,
            "ocr_text": ocr_text_clean
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
