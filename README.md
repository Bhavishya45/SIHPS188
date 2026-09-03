# VERISCAN — SIH26188 · Team JINX
### AI-Based Fake Identity & Document Screening System — working prototype

This prototype has two parts. Both are real, not mocked demos.

## 1. `dashboard.html` — Officer investigation console (open this)

Just open it in any browser. No install needed.

- **Case queue** on the left shows 3 documents that were run through the actual
  Python backend (`backend/`) — the OCR extraction, MRZ checksum validation,
  ELA tampering scores, heatmaps, and risk verdicts you see are the real
  output of that pipeline, embedded as data.
  - `clean` — a valid document
  - `tampered` — the date of birth was edited and the file re-saved after
    the fact (a real forgery pattern)
  - `reused_identity` — a different name/passport number, but the *same*
    face as the `clean` case, to demonstrate cross-case identity-reuse
    detection
- **"+ New screening"** opens two tools that run for real, live, in your
  browser (no server call):
  1. **MRZ checksum validator** — paste any 2-line MRZ and it validates
     every ICAO 9303 check digit on the spot.
  2. **Error Level Analysis** — upload any JPG/PNG and it re-encodes it on
     a canvas and diffs it against the original, exactly like the backend's
     `tamper.py`, and shows you the heatmap.
- Face-matching and OCR need real image processing (Tesseract, OpenCV),
  which browsers can't do natively — that's what the backend service is for.

## 2. `backend/` — the actual analysis engine (Python)

Real, working, offline-testable code — not pseudocode:

| File | What it does |
|---|---|
| `mrz.py` | ICAO 9303 MRZ parser + check-digit validator. Verified against the official ICAO worked example. |
| `ocr.py` | Tesseract-based text/MRZ extraction from a document image. |
| `tamper.py` | Error Level Analysis (JPEG re-save diff) + EXIF metadata checks for tampering detection. |
| `face_match.py` | OpenCV Haar-cascade face detection, histogram-based similarity scoring, and perceptual-hash matching for identity reuse across cases. |
| `risk_engine.py` | Combines all of the above into a LOW/MEDIUM/HIGH score with plain-language reasons. |
| `app.py` | Flask REST API (`POST /api/analyze`) wiring the whole pipeline together, with a simple JSON case store for cross-case identity-reuse lookups. |
| `generate_sample.py` | Generates synthetic passport images (no real ID data used — same reasoning as the MIDV-2020 dataset cited in the pitch). |
| `demo.py` | Runs the full pipeline end-to-end on the 3 sample cases and prints/saves the reports used in the dashboard. |

### Run it yourself

```bash
cd backend
pip install flask pillow pytesseract opencv-python numpy
python3 generate_sample.py   # regenerate the 3 sample documents
python3 demo.py              # run the full pipeline, print JSON reports
python3 app.py                # or: start the REST API on localhost:5050
```

### Honest limitations (worth knowing before you present)

- **Face matching** uses OpenCV Haar detection + histogram similarity as a
  stand-in for a learned face-embedding model (ArcFace/DeepFace, as named
  in the pitch deck). Those need pretrained weights this offline sandbox
  couldn't download — the pipeline shape (detect → normalize → embed →
  compare → threshold) is the same and is swap-compatible with a real
  model later.
- **Sample documents are synthetic** (PIL-drawn), not scanned photographs,
  so ELA scores on them run a bit higher than on real photographic scans —
  flat illustrated regions compress differently than photos. The algorithm
  itself is the standard, real ELA technique; thresholds would be
  recalibrated against real scanned documents.
- **Database** is a flat JSON file standing in for the PostgreSQL store in
  the architecture diagram — same read/write pattern, different backing
  store for a fast prototype.
