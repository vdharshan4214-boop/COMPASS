"""
Lightweight predictive layer for Compass.

This mines historical complaint timestamps (grouped by month + category + location)
and blends that with a static seasonal knowledge table so the app can surface
"predicted recurring issues" even with a small/synthetic dataset. It is intentionally
simple and explainable rather than a black-box forecaster.
"""
import datetime
import os
from collections import defaultdict

import joblib
import numpy as np

# Static seasonal priors: category -> {month_number: relative_risk_multiplier}
# Grounded in common campus patterns (e.g. AC/electrical load spikes in hot months,
# plumbing spikes around monsoon, hostel/safety spikes at term start).
SEASONAL_PRIORS = {
    "electrical": {3: 1.4, 4: 1.8, 5: 2.0, 6: 1.6, 7: 1.2},
    "plumbing":   {6: 1.5, 7: 1.9, 8: 1.7, 9: 1.3},
    "wifi":       {1: 1.3, 7: 1.4, 8: 1.5},   # new-term connection spikes
    "hostel":     {1: 1.4, 7: 1.6, 8: 1.3},   # move-in periods
    "safety":     {10: 1.2, 11: 1.3, 12: 1.2},
    "other":      {},
}

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "forecast_model.joblib")


def _monthly_history(complaints: list[dict]):
    """category -> location -> {month: count} from real stored complaints."""
    history = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for c in complaints:
        created = c.get("created_at")
        if not isinstance(created, datetime.datetime):
            continue
        category = c.get("category") or "other"
        location = c.get("location") or "unknown"
        history[category][location][created.month] += 1
    return history


def predict_upcoming_issues(complaints: list[dict], location: str | None = None, months_ahead: int = 2, top_n: int = 5):
    """
    Returns a ranked list of predicted recurring issues for the next `months_ahead`
    months, optionally scoped to a location. Each item explains WHY it was flagged
    (seasonal prior, historical frequency, or both) so it stays explainable.
    """
    now = datetime.datetime.utcnow()
    target_months = [((now.month - 1 + i) % 12) + 1 for i in range(1, months_ahead + 1)]
    history = _monthly_history(complaints)

    scored = []
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    for category, by_location in history.items():
        locations = by_location.items() if not location else [
            (loc, months) for loc, months in by_location.items()
            if location.lower() in loc.lower() or loc.lower() in location.lower()
        ]
        for loc, months in locations:
            total_seen = sum(months.values())
            if total_seen == 0:
                continue
            for month in target_months:
                historical_count = months.get(month, 0)
                seasonal_mult = SEASONAL_PRIORS.get(category, {}).get(month, 1.0)
                base_rate = total_seen / 12.0
                base_risk = (base_rate + historical_count) * seasonal_mult
                if model is not None:
                    try:
                        month_frac = (month - 1) / 12.0
                        features = np.array([[np.sin(2 * np.pi * month_frac), np.cos(2 * np.pi * month_frac), 1.0, 0.0]])
                        base_risk = float(model.predict(features)[0])
                    except Exception:
                        pass
                risk_score = round(float(base_risk), 2)
                std_dev = max(0.4, np.std([max(0, v) for v in months.values()]) if months else 0.4)
                conf_low = round(max(0.0, risk_score - std_dev), 2)
                conf_high = round(risk_score + std_dev, 2)
                if risk_score <= 0:
                    continue
                reasons = []
                if seasonal_mult > 1.0:
                    reasons.append(f"{category} issues historically rise in {MONTH_NAMES[month]}")
                if historical_count > 0:
                    reasons.append(f"{historical_count} similar report(s) logged in {loc} before")
                if not reasons:
                    reasons.append("based on overall complaint frequency")
                scored.append({
                    "category": category,
                    "location": loc,
                    "predicted_month": MONTH_NAMES[month],
                    "risk_score": risk_score,
                    "confidence_low": conf_low,
                    "confidence_high": conf_high,
                    "reason": "; ".join(reasons),
                })

    scored.sort(key=lambda x: x["risk_score"], reverse=True)
    return scored[:top_n]