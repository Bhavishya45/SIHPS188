"""
Face detection + similarity scoring using OpenCV (Haar cascade) and a
lightweight perceptual hash for cross-case identity-reuse matching.

Note on scope: production systems would use a learned face-embedding
model (e.g. ArcFace/DeepFace) for match accuracy. Those require
downloading pretrained weights, which this offline sandbox can't do.
This module implements the same *pipeline shape* (detect -> normalize
-> embed -> compare -> threshold) with OpenCV Haar detection + a
histogram/pHash-based similarity in place of a learned embedding, so
the architecture and API are swap-compatible with a real model later.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FACE_SIZE = (128, 128)


@dataclass
class FaceCrop:
    found: bool
    array: Optional[np.ndarray] = None
    box: Optional[Tuple[int, int, int, int]] = None


def detect_face(image: Image.Image, fallback_box: Optional[Tuple[int, int, int, int]] = None) -> FaceCrop:
    """Detect a face via Haar cascade. If none is found (e.g. a stylised
    demo image) and a fallback_box is supplied, use that region instead
    so the rest of the pipeline still has something to compare."""
    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = cv_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, FACE_SIZE)
        return FaceCrop(found=True, array=crop, box=(int(x), int(y), int(w), int(h)))

    if fallback_box:
        x, y, w, h = fallback_box
        crop = cv_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, FACE_SIZE)
        return FaceCrop(found=False, array=crop, box=fallback_box)

    return FaceCrop(found=False)


def face_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between HSV histograms of two aligned face crops,
    returned as a 0-100 match percentage."""
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, correlation)) * 100)


def perceptual_hash(a: np.ndarray, hash_size: int = 8) -> str:
    """Average hash (aHash) of a face crop, for fast cross-case lookups."""
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size))
    avg = small.mean()
    bits = (small > avg).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(hash_a, hash_b))
