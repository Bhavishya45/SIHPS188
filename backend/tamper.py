"""
Tampering detection: Error Level Analysis (ELA) + metadata checks.

ELA works by resaving the image at a known JPEG quality and diffing
against the original. Regions that were edited after the last save
compress differently than untouched regions, so they light up in the
diff. This is a real, standard forensic technique (no ML model needed).
"""

import io
from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


@dataclass
class TamperResult:
    ela_score: float = 0.0          # 0-100, higher = more suspicious
    hot_regions: int = 0            # number of high-error blocks
    metadata_flags: List[str] = field(default_factory=list)
    heatmap: Image.Image = None
    verdict: str = "CLEAN"          # CLEAN / SUSPICIOUS / LIKELY_EDITED


def error_level_analysis(image: Image.Image, quality: int = 90, scale: int = 15) -> TamperResult:
    original = image.convert("RGB")

    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    diff = ImageChops.difference(original, resaved)
    diff_arr = np.asarray(diff).astype(np.float32)

    # Per-pixel max channel error -> single-channel error map
    error_map = diff_arr.max(axis=2)

    # Block-wise scoring (16x16) to find localized hot spots rather than
    # being thrown off by uniform global noise.
    block = 16
    h, w = error_map.shape
    hot_regions = 0
    block_means = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            m = error_map[y:y + block, x:x + block].mean()
            block_means.append(m)
    block_means = np.array(block_means) if block_means else np.array([0.0])

    global_mean = float(block_means.mean())
    global_std = float(block_means.std()) + 1e-6
    threshold = global_mean + 2.2 * global_std
    hot_regions = int((block_means > threshold).sum())

    # Normalize to a 0-100 suspicion score: combines overall error energy
    # and how many localized outlier blocks exist relative to the image.
    density = hot_regions / max(len(block_means), 1)
    ela_score = float(min(100, (global_mean * 1.8) + (density * 400)))

    heatmap = ImageEnhance.Brightness(diff).enhance(scale)

    verdict = "CLEAN"
    if ela_score >= 55:
        verdict = "LIKELY_EDITED"
    elif ela_score >= 28:
        verdict = "SUSPICIOUS"

    return TamperResult(
        ela_score=round(ela_score, 1),
        hot_regions=hot_regions,
        heatmap=heatmap,
        verdict=verdict,
    )


def check_metadata(image: Image.Image, file_path: str = None) -> List[str]:
    flags = []
    exif = image.getexif()
    if not exif or len(exif) == 0:
        flags.append("No EXIF metadata present (common for screenshots/re-saved scans)")
    else:
        software_tag = exif.get(305)  # 'Software' tag
        if software_tag and any(
            tool in str(software_tag).lower()
            for tool in ["photoshop", "gimp", "paint.net", "affinity"]
        ):
            flags.append(f"Editing software signature found in metadata: {software_tag}")
    return flags


def analyze(image: Image.Image) -> TamperResult:
    result = error_level_analysis(image)
    result.metadata_flags = check_metadata(image)
    return result
