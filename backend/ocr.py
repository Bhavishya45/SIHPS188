"""
OCR extraction using Tesseract (via pytesseract).
Pulls the MRZ zone (bottom strip of a passport bio page) and general
document fields.
"""

import re
import pytesseract
from PIL import Image

MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def extract_mrz_lines(image: Image.Image) -> list[str]:
    """Crop the bottom ~22% of the image (standard MRZ band location on
    a TD3 passport bio page) and OCR it with an MRZ-tuned Tesseract config."""
    w, h = image.size
    mrz_band = image.crop((0, int(h * 0.78), w, h)).convert("L")

    config = (
        f'--psm 6 -c tessedit_char_whitelist={MRZ_CHARSET}'
    )
    raw = pytesseract.image_to_string(mrz_band, config=config)

    lines = [ln.strip().upper() for ln in raw.splitlines() if ln.strip()]
    # Keep only plausible MRZ lines (mostly A-Z0-9< and reasonably long)
    lines = [ln for ln in lines if len(ln) >= 20 and re.match(r"^[A-Z0-9<]+$", ln)]
    return lines[-2:] if len(lines) >= 2 else lines


def extract_full_text(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def extract_document_fields(full_text: str) -> dict:
    """Best-effort field extraction from the non-MRZ area of the document
    using simple label-based regexes. Real deployments would use a
    layout-aware model per document type; this covers the common
    printed-field style used on passport bio pages."""
    fields = {}
    patterns = {
        "surname": r"Surname[:\s]+([A-Za-z\-'\s]+)",
        "given_names": r"Given\s*Names?[:\s]+([A-Za-z\-'\s]+)",
        "date_of_birth": r"Date of Birth[:\s]+([0-9]{1,2}[\/\-.][A-Za-z0-9]{2,4}[\/\-.][0-9]{2,4})",
        "passport_number": r"(?:Passport|Document)\s*No\.?[:\s]+([A-Z0-9]+)",
        "nationality": r"Nationality[:\s]+([A-Za-z\s]+)",
        "expiry_date": r"Date of Expiry[:\s]+([0-9]{1,2}[\/\-.][A-Za-z0-9]{2,4}[\/\-.][0-9]{2,4})",
    }
    for key, pat in patterns.items():
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            fields[key] = m.group(1).strip()
    return fields
