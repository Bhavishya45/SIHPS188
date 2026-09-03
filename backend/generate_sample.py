"""
Generates synthetic passport bio-page images so the full pipeline can
be exercised end-to-end without any real government ID data (mirrors
the MIDV-2020 approach cited in the pitch: synthetic documents only).

Produces three cases:
  1. clean.jpg       - valid MRZ, untouched image
  2. tampered.jpg     - valid-looking MRZ but the DOB text was pasted in
                         after the fact and the file re-saved (creates a
                         real, detectable ELA signature)
  3. reused_identity.jpg - a different name/passport number, but the
                            SAME face crop as clean.jpg (for identity-reuse test)
"""

import os
from PIL import Image, ImageDraw, ImageFont

import mrz

OUT_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(OUT_DIR, exist_ok=True)

FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

W, H = 900, 570


def _base_canvas():
    img = Image.new("RGB", (W, H), (235, 232, 222))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W - 1, H - 1], outline=(120, 110, 90), width=3)
    draw.rectangle([0, 0, W - 1, 60], fill=(30, 45, 90))
    title_font = ImageFont.truetype(FONT_SANS, 26)
    draw.text((20, 15), "REPUBLIC OF UTOPIA \u2014 PASSPORT", font=title_font, fill=(255, 255, 255))
    return img, draw


def _draw_photo(draw, seed_color, box):
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], fill=seed_color, outline=(50, 50, 50), width=2)
    # simple stylised "face": two eyes + mouth so Haar/contrast has *something*
    draw.ellipse([x + w * 0.25, y + h * 0.3, x + w * 0.4, y + h * 0.4], fill=(20, 20, 20))
    draw.ellipse([x + w * 0.6, y + h * 0.3, x + w * 0.75, y + h * 0.4], fill=(20, 20, 20))
    draw.arc([x + w * 0.3, y + h * 0.5, x + w * 0.7, y + h * 0.75], start=20, end=160, fill=(20, 20, 20), width=3)


def _draw_fields(draw, x, y, fields):
    label_font = ImageFont.truetype(FONT_SANS, 14)
    value_font = ImageFont.truetype(FONT_SANS, 17)
    dy = 0
    for label, value in fields:
        draw.text((x, y + dy), label, font=label_font, fill=(90, 90, 90))
        draw.text((x, y + dy + 16), value, font=value_font, fill=(20, 20, 20))
        dy += 45


def _draw_mrz(draw, line1, line2):
    mono = ImageFont.truetype(FONT_MONO, 22)
    y0 = H - 95
    draw.rectangle([0, y0 - 10, W, H], fill=(245, 245, 240))
    draw.text((30, y0), line1, font=mono, fill=(10, 10, 10))
    draw.text((30, y0 + 32), line2, font=mono, fill=(10, 10, 10))


def _build_mrz_lines(surname, given, passport_no, nationality, dob, sex, expiry, personal_no=""):
    def pad(s, n):
        return (s.replace(" ", "<").upper())[:n].ljust(n, "<")

    l1 = "P<" + pad(nationality, 3) + pad(surname, 1) + "<<" + pad(given, 1)
    l1_body = f"P<{nationality:<3}{surname}<<{given}".upper().replace(" ", "<")
    l1 = l1_body.ljust(44, "<")[:44]

    passport_no = pad(passport_no, 9)
    chk1 = mrz.check_digit(passport_no)
    dob6 = dob
    chk2 = mrz.check_digit(dob6)
    exp6 = expiry
    chk3 = mrz.check_digit(exp6)
    personal_no = pad(personal_no, 14)
    chk4 = mrz.check_digit(personal_no)
    composite_data = passport_no + str(chk1) + dob6 + str(chk2) + exp6 + str(chk3) + personal_no + str(chk4)
    chk5 = mrz.check_digit(composite_data)

    l2 = f"{passport_no}{chk1}{nationality:<3}{dob6}{chk2}{sex}{exp6}{chk3}{personal_no}{chk4}{chk5}"
    l2 = l2.upper().replace(" ", "<").ljust(44, "<")[:44]
    return l1, l2


def make_clean_document(path):
    img, draw = _base_canvas()
    _draw_photo(draw, (150, 170, 200), (30, 90, 190, 230))
    l1, l2 = _build_mrz_lines(
        surname="SHARMA", given="ROHAN", passport_no="J8291733",
        nationality="IND", dob="920514", sex="M", expiry="300822",
    )
    _draw_fields(draw, 250, 90, [
        ("Surname", "SHARMA"), ("Given Names", "ROHAN"),
        ("Passport No.", "J8291733"), ("Nationality", "INDIA"),
        ("Date of Birth", "14/MAY/1992"), ("Date of Expiry", "22/AUG/2030"),
    ])
    _draw_mrz(draw, l1, l2)
    img.save(path, "JPEG", quality=95)
    return img


def make_tampered_document(path):
    """Same as clean, but DOB text is pasted in AFTER the initial save
    and the file is re-saved once more at a different JPEG quality -
    the classic ELA-detectable edit."""
    img, draw = _base_canvas()
    _draw_photo(draw, (150, 170, 200), (30, 90, 190, 230))
    l1, l2 = _build_mrz_lines(
        surname="SHARMA", given="ROHAN", passport_no="J8291733",
        nationality="IND", dob="920514", sex="M", expiry="300822",
    )
    _draw_fields(draw, 250, 90, [
        ("Surname", "SHARMA"), ("Given Names", "ROHAN"),
        ("Passport No.", "J8291733"), ("Nationality", "INDIA"),
        ("Date of Birth", "14/MAY/1992"), ("Date of Expiry", "22/AUG/2030"),
    ])
    _draw_mrz(draw, l1, l2)
    # first save (simulates the "original" scan)
    img.save(path, "JPEG", quality=95)

    # now re-open, edit the DOB region, and re-save at a lower quality
    edited = Image.open(path).convert("RGB")
    edraw = ImageDraw.Draw(edited)
    edraw.rectangle([250, 90 + 4 * 45, 480, 90 + 4 * 45 + 40], fill=(235, 232, 222))
    value_font = ImageFont.truetype(FONT_SANS, 17)
    edraw.text((250, 90 + 4 * 45 + 16), "14/MAY/1985", font=value_font, fill=(20, 20, 20))
    edited.save(path, "JPEG", quality=70)
    return edited


def make_reused_identity_document(path):
    """Different name / passport number, SAME face crop as clean.jpg,
    to test cross-case identity-reuse detection."""
    img, draw = _base_canvas()
    _draw_photo(draw, (150, 170, 200), (30, 90, 190, 230))  # identical face params -> same aHash
    l1, l2 = _build_mrz_lines(
        surname="VERMA", given="ROHIT", passport_no="K5533210",
        nationality="IND", dob="920514", sex="M", expiry="290103",
    )
    _draw_fields(draw, 250, 90, [
        ("Surname", "VERMA"), ("Given Names", "ROHIT"),
        ("Passport No.", "K5533210"), ("Nationality", "INDIA"),
        ("Date of Birth", "14/MAY/1992"), ("Date of Expiry", "03/JAN/2029"),
    ])
    _draw_mrz(draw, l1, l2)
    img.save(path, "JPEG", quality=95)
    return img


if __name__ == "__main__":
    make_clean_document(os.path.join(OUT_DIR, "clean.jpg"))
    make_tampered_document(os.path.join(OUT_DIR, "tampered.jpg"))
    make_reused_identity_document(os.path.join(OUT_DIR, "reused_identity.jpg"))
    print("Generated sample documents in", OUT_DIR)
