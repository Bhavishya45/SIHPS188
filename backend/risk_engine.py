"""
Combines MRZ, OCR, tamper, and face-match results into a single
LOW / MEDIUM / HIGH risk verdict with plain-language reasons —
so the officer sees a verdict, not raw module output.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RiskReport:
    score: int = 0                # 0-100
    level: str = "LOW"
    reasons: List[str] = field(default_factory=list)


def compute_risk(mrz_result, tamper_result, face_match_score: float,
                  identity_reuse: bool, reuse_names: list) -> RiskReport:
    score = 0
    reasons = []

    if not mrz_result.overall_valid:
        score += 35
        reasons.append("One or more MRZ checksums failed validation")
        for err in mrz_result.errors:
            reasons.append(f"  \u2192 {err}")
    else:
        reasons.append("All MRZ checksums valid (ICAO 9303)")

    if tamper_result.verdict == "LIKELY_EDITED":
        score += 40
        reasons.append(f"ELA flagged likely image editing (score {tamper_result.ela_score}/100, "
                        f"{tamper_result.hot_regions} localized hot region(s))")
    elif tamper_result.verdict == "SUSPICIOUS":
        score += 18
        reasons.append(f"ELA found mild irregularities (score {tamper_result.ela_score}/100) \u2014 recommend closer visual review")
    else:
        reasons.append(f"No significant ELA irregularities (score {tamper_result.ela_score}/100)")

    if tamper_result.metadata_flags:
        score += 8
        reasons.extend(tamper_result.metadata_flags)

    if face_match_score is None:
        reasons.append("Face match skipped (no live capture submitted)")
    elif face_match_score < 60:
        score += 25
        reasons.append(f"Face match low: {face_match_score:.0f}% similarity between document photo and live capture")
    else:
        reasons.append(f"Face match acceptable: {face_match_score:.0f}% similarity")

    if identity_reuse:
        score += 30
        names = ", ".join(reuse_names)
        reasons.append(f"Same face previously seen under different identity: {names}")

    score = min(100, score)
    if score >= 55:
        level = "HIGH"
    elif score >= 25:
        level = "MEDIUM"
    else:
        level = "LOW"

    return RiskReport(score=score, level=level, reasons=reasons)
