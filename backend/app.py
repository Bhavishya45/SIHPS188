"""
Backend server for the AI-Based Fake Identity & Document Screening
System (SIH26188 / Team JINX).

Endpoints:
  POST /api/analyze   - multipart form: document_image, live_photo (optional), name
                         -> full JSON risk report
  GET  /api/cases      - list stored cases (for identity-reuse context)

Run:  python3 app.py   (listens on localhost:5050)
"""

import base64
import io
import json
import os
import time

from flask import Flask, request, jsonify
from PIL import Image

import mrz
import ocr
import tamper
import face_match
import risk_engine

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "cases_db.json")
REUSE_HAMMING_THRESHOLD = 8  # out of 64 bits -> "same face" for aHash


def _load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            return json.load(f)
    return []


def _save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def _pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_pipeline(document_image: Image.Image, live_photo: Image.Image, claimed_name: str) -> dict:
    t0 = time.time()

    # 1. OCR + MRZ -----------------------------------------------------
    mrz_lines = ocr.extract_mrz_lines(document_image)
    full_text = ocr.extract_full_text(document_image)
    doc_fields = ocr.extract_document_fields(full_text)

    if len(mrz_lines) == 2:
        mrz_result = mrz.parse_td3(mrz_lines[0], mrz_lines[1])
    else:
        mrz_result = mrz.MRZResult(errors=["Could not locate a 2-line MRZ band via OCR"])

    # 2. Tampering -------------------------------------------------------
    tamper_result = tamper.analyze(document_image)

    # 3. Face match + identity reuse -------------------------------------
    doc_face = face_match.detect_face(document_image, fallback_box=_default_photo_box(document_image))
    db = _load_db()

    face_score = None
    if live_photo is not None:
        live_face = face_match.detect_face(live_photo)
        if doc_face.array is not None and live_face.array is not None:
            face_score = face_match.face_similarity(doc_face.array, live_face.array)

    identity_reuse = False
    reuse_names = []
    doc_hash = face_match.perceptual_hash(doc_face.array) if doc_face.array is not None else None
    if doc_hash:
        for case in db:
            if case.get("name") == claimed_name:
                continue
            dist = face_match.hamming_distance(doc_hash, case["face_hash"])
            if dist <= REUSE_HAMMING_THRESHOLD:
                identity_reuse = True
                reuse_names.append(case["name"])

    # 4. Risk score --------------------------------------------------------
    report = risk_engine.compute_risk(mrz_result, tamper_result, face_score, identity_reuse, reuse_names)

    # Persist this case for future reuse checks
    if doc_hash:
        db.append({
            "name": claimed_name or mrz_result.surname or "UNKNOWN",
            "face_hash": doc_hash,
            "passport_number": mrz_result.passport_number,
            "timestamp": time.time(),
        })
        _save_db(db)

    elapsed_ms = round((time.time() - t0) * 1000)

    return {
        "elapsed_ms": elapsed_ms,
        "ocr": {
            "mrz_lines_detected": mrz_lines,
            "document_fields": doc_fields,
        },
        "mrz": {
            "document_type": mrz_result.document_type,
            "issuing_country": mrz_result.issuing_country,
            "surname": mrz_result.surname,
            "given_names": mrz_result.given_names,
            "passport_number": mrz_result.passport_number,
            "nationality": mrz_result.nationality,
            "date_of_birth": mrz_result.date_of_birth,
            "expiry_date": mrz_result.expiry_date,
            "overall_valid": mrz_result.overall_valid,
            "composite_valid": mrz_result.composite_valid,
            "errors": mrz_result.errors,
            "field_checks": [
                {"name": f.name, "expected": f.expected_check, "computed": f.computed_check, "valid": f.valid}
                for f in mrz_result.fields
            ],
        },
        "tamper": {
            "ela_score": tamper_result.ela_score,
            "hot_regions": tamper_result.hot_regions,
            "verdict": tamper_result.verdict,
            "metadata_flags": tamper_result.metadata_flags,
            "heatmap_data_url": _pil_to_data_url(tamper_result.heatmap) if tamper_result.heatmap else None,
        },
        "face": {
            "doc_face_found": doc_face.found,
            "match_score": round(face_score, 1) if face_score is not None else None,
            "identity_reuse": identity_reuse,
            "reuse_matched_names": reuse_names,
        },
        "risk": {
            "score": report.score,
            "level": report.level,
            "reasons": report.reasons,
        },
    }


def _default_photo_box(image: Image.Image):
    """Fallback photo-region box for stylised/synthetic demo documents
    where a Haar cascade won't fire. Standard passport bio-page layout
    places the portrait in the upper-left quadrant."""
    w, h = image.size
    return (int(w * 0.05), int(h * 0.12), int(w * 0.28), int(h * 0.45))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "document_image" not in request.files:
        return jsonify({"error": "document_image file is required"}), 400

    document_image = Image.open(request.files["document_image"].stream)
    live_photo = None
    if "live_photo" in request.files and request.files["live_photo"].filename:
        live_photo = Image.open(request.files["live_photo"].stream)

    claimed_name = request.form.get("name", "")
    result = run_pipeline(document_image, live_photo, claimed_name)
    return jsonify(result)


@app.route("/api/cases", methods=["GET"])
def list_cases():
    return jsonify(_load_db())


if __name__ == "__main__":
    app.run(port=5050, debug=False)
