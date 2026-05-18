import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path("app")
OUTPUT_DIR = BASE_DIR / "outputs"

HEALTH_PATH = OUTPUT_DIR / "campaign_health_score.json"

DASHBOARD_TABLE = "campaign_optimization_dashboard"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _delta_pct(current: Any, predicted: Any) -> float | None:
    current_value = _safe_float(current)
    predicted_value = _safe_float(predicted)

    if current_value is None or predicted_value is None or abs(current_value) < 0.0001:
        return None

    return round(((predicted_value - current_value) / abs(current_value)) * 100, 2)


def _delta_value(current: Any, predicted: Any) -> float | None:
    current_value = _safe_float(current)
    predicted_value = _safe_float(predicted)

    if current_value is None or predicted_value is None:
        return None

    return round(predicted_value - current_value, 2)


def _build_expected_impact(current_kpis: dict[str, Any], predicted_kpis: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_roas": predicted_kpis.get("roas_h14"),
        "expected_conversions": predicted_kpis.get("conversions_h14"),
        "expected_spend": current_kpis.get("spend"),
        "delta_roas_pct": _delta_pct(current_kpis.get("roas"), predicted_kpis.get("roas_h14")),
        "delta_conversions": _delta_value(
            current_kpis.get("conversions"),
            predicted_kpis.get("conversions_h14"),
        ),
    }


def _dashboard_summary(health_score: Any, predicted_kpis: dict[str, Any]) -> str:
    predicted_roas = _safe_float(predicted_kpis.get("roas_h14"))

    if predicted_roas is not None and predicted_roas < 1:
        return (
            "Aucune anomalie prioritaire detectee, mais le ROAS predit reste faible. "
            "Continuer la surveillance et eviter toute hausse de budget avant confirmation."
        )

    return (
        "Aucune anomalie prioritaire detectee. La campagne est stable, "
        "continuer la surveillance des KPIs sans action corrective."
    )


def build_health_monitoring_rows(workflow_execution_id: str = "") -> dict[str, Any]:
    health_data = _load_json(HEALTH_PATH, {"campaigns": [], "metadata": {}})
    campaigns = health_data.get("campaigns", [])
    metadata = health_data.get("metadata", {})
    generated_at = metadata.get("generated_at") or datetime.now().isoformat()

    healthy_rows: list[dict[str, Any]] = []
    problem_count = 0

    for campaign in campaigns:
        if not isinstance(campaign, dict):
            continue

        if bool(campaign.get("trigger_causal_ai", False)):
            problem_count += 1
            continue

        details = campaign.get("details", {}) if isinstance(campaign.get("details"), dict) else {}
        prediction = details.get("prediction", {}) if isinstance(details.get("prediction"), dict) else {}
        anomaly = details.get("anomaly", {}) if isinstance(details.get("anomaly"), dict) else {}
        trend = details.get("trend", {}) if isinstance(details.get("trend"), dict) else {}
        components = campaign.get("components", {}) if isinstance(campaign.get("components"), dict) else {}

        current_kpis = campaign.get("current_kpis", {})
        if not isinstance(current_kpis, dict):
            current_kpis = {}

        predicted_kpis = prediction.get("predicted_kpis", {})
        if not isinstance(predicted_kpis, dict):
            predicted_kpis = {}

        expected_impact = _build_expected_impact(current_kpis, predicted_kpis)
        current_spend = _safe_float(current_kpis.get("spend"), 0.0) or 0.0

        row = {
            "workflow_execution_id": workflow_execution_id,
            "global_campaign_id": campaign.get("global_campaign_id"),
            "campaign_id": campaign.get("campaign_id"),
            "platform": campaign.get("platform"),
            "health_score": campaign.get("health_score"),
            "health_status": campaign.get("status") or "OK",
            "anomaly_level": anomaly.get("global_level") or "OK",
            "anomaly_score": components.get("anomaly_score"),
            "n_anomaly_days": anomaly.get("n_anomalies") or 0,
            "current_kpis": current_kpis,
            "predicted_kpis": predicted_kpis,
            "trend": trend,
            "anomaly_types": {"items": []},
            "top_anomalies": {"items": []},
            "root_cause": "none",
            "root_cause_label": "Aucune cause critique detectee",
            "causal_confidence": None,
            "recommended_action": "continue_monitoring",
            "action_label": "Continuer la surveillance",
            "priority": "OK",
            "expected_impact": expected_impact,
            "budget_adjustment": {
                "adjustment_type": "monitoring",
                "shift_pct": 0.0,
                "shift_amount": 0.0,
                "current_budget": round(current_spend, 2),
                "recommended_budget": round(current_spend, 2),
                "source_channel": None,
                "target_channel": None,
                "quantitative_explanation": "Aucun ajustement budgetaire requis.",
            },
            "dashboard_summary": _dashboard_summary(campaign.get("health_score"), predicted_kpis),
            "full_json": {
                "source": "health_monitoring_dashboard",
                "health": campaign,
                "expected_impact": expected_impact,
            },
            "source_generated_at": generated_at,
        }

        healthy_rows.append(row)

    return {
        "status": "success",
        "generated_at": datetime.now().isoformat(),
        "source_generated_at": generated_at,
        "workflow_execution_id": workflow_execution_id,
        "has_problem_campaigns": problem_count > 0,
        "problem_count": problem_count,
        "healthy_count": len(healthy_rows),
        "items": healthy_rows,
    }


def insert_health_monitoring_rows(rows: list[dict[str, Any]]) -> int:
    # n8n owns PostgreSQL insertion with the existing Postgres credential.
    # This function is kept as a stable extension point, but deliberately does
    # not open a Python DB connection.
    return 0


def run_health_monitoring_dashboard(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    workflow_execution_id = str(payload.get("workflow_execution_id") or "")

    try:
        result = build_health_monitoring_rows(workflow_execution_id=workflow_execution_id)
        result["inserted_count"] = 0
        result["message"] = (
            "Lignes de monitoring sain preparees pour insertion n8n."
            if result["healthy_count"]
            else "Aucune campagne saine a preparer."
        )
        return result
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "has_problem_campaigns": False,
            "problem_count": 0,
            "healthy_count": 0,
            "items": [],
            "inserted_count": 0,
        }
