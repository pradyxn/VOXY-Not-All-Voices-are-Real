from ..config import RISK_THRESHOLDS


def calculate_risk(synthetic_score: float, windows: int, stability: float, audio_quality: str) -> tuple[int, str]:
    """Use one 0-100 signal everywhere in the UI."""
    if audio_quality == "insufficient":
        return 0, "LOW"
    risk = round(max(0.0, min(100.0, float(synthetic_score))))
    if risk >= RISK_THRESHOLDS["critical"]:
        level = "CRITICAL"
    elif risk >= RISK_THRESHOLDS["high"]:
        level = "HIGH"
    elif risk >= RISK_THRESHOLDS["moderate"]:
        level = "MODERATE"
    else:
        level = "LOW"
    return risk, level
