"""
Runs the full analysis pipeline (no Flask server needed) on the three
synthetic sample documents and prints a report for each — proof the
OCR/MRZ/tamper/face-match/risk-scoring logic actually works end to end.
"""

import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from app import run_pipeline, DB_PATH  # noqa: E402

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # fresh run

    cases = [
        ("clean.jpg", "SHARMA ROHAN"),
        ("tampered.jpg", "SHARMA ROHAN"),          # same person, edited DOB
        ("reused_identity.jpg", "VERMA ROHIT"),     # different claimed name, same face
    ]

    for filename, claimed_name in cases:
        path = os.path.join(SAMPLES_DIR, filename)
        img = Image.open(path)
        result = run_pipeline(img, None, claimed_name)

        print("=" * 70)
        print(f"CASE: {filename}  (claimed name: {claimed_name})")
        print("-" * 70)
        print(f"MRZ lines detected : {result['ocr']['mrz_lines_detected']}")
        print(f"MRZ overall valid  : {result['mrz']['overall_valid']}")
        if result['mrz']['errors']:
            print(f"MRZ errors         : {result['mrz']['errors']}")
        print(f"ELA verdict        : {result['tamper']['verdict']} (score {result['tamper']['ela_score']})")
        print(f"Identity reuse     : {result['face']['identity_reuse']} {result['face']['reuse_matched_names']}")
        print(f"RISK LEVEL         : {result['risk']['level']}  (score {result['risk']['score']}/100)")
        for reason in result['risk']['reasons']:
            print(f"   - {reason}")

        # save heatmap + full JSON for inspection
        out_json = os.path.join(SAMPLES_DIR, filename.replace(".jpg", "_report.json"))
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

    print("=" * 70)
    print("Full JSON reports saved alongside sample images in:", SAMPLES_DIR)


if __name__ == "__main__":
    main()
