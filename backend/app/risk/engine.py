from ..config import RISK_THRESHOLDS

def calculate_risk(synthetic_score: float, windows: int, stability: float, audio_quality: str) -> tuple[int, str]:
    if audio_quality == "insufficient":
        return 0, "LOW"
    evidence_factor = min(1.0, 0.7 + (windows / 10))
    stability_factor = 0.85 + (stability / 1000)
    risk = round(max(0, min(100, synthetic_score * evidence_factor * stability_factor)))
    if risk >= RISK_THRESHOLDS["critical"]:
        level = "CRITICAL"
    elif risk >= RISK_THRESHOLDS["high"]:
        level = "HIGH"
    elif risk >= RISK_THRESHOLDS["moderate"]:
        level = "MODERATE"
    else:
        level = "LOW"
    return risk, level
