from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.risk.engine import calculate_risk
assert calculate_risk(10, 3, 95, "good")[1] == "LOW"
assert calculate_risk(90, 4, 95, "good")[1] in {"HIGH", "CRITICAL"}
assert calculate_risk(90, 1, 95, "insufficient")[1] == "LOW"
print("Audio quality and risk engine checks OK")
