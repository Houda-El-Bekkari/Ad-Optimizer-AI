from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path("app")
OUTPUT_DIR = BASE_DIR / "outputs"

HEALTH_PATH = OUTPUT_DIR / "campaign_health_score.json"
ANOMALY_PATH = OUTPUT_DIR / "anomaly_report.json"
CORRELATIONS_PATH = OUTPUT_DIR / "correlations.json"
CAUSAL_PATH = OUTPUT_DIR / "causal_effects.json"
OPTIMIZATION_PATH = OUTPUT_DIR / "optimization_plan.json"
XAI_PATH = OUTPUT_DIR / "xai_explanations.json"

OUT_AUDIT_JSON = OUTPUT_DIR / "audit_report.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _campaign_key(campaign_id: str | None, platform: str | None) -> str:
    return f"{campaign_id or ''}|{str(platform or '').lower()}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 2) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _nested_number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_maintain_action(action: str | None, label: str | None) -> bool:
    action_text = str(action or "").lower()
    label_text = str(label or "").lower()

    return "maintain" in action_text or label_text.startswith("maint")


def _business_action_title(
    action: str | None,
    label: str | None,
    expected_impact: dict[str, Any],
) -> str:
    expected_roas = _nested_number(expected_impact, "expected_roas")
    delta_conversions = _nested_number(expected_impact, "delta_conversions")

    if _is_maintain_action(action, label):
        if (
            expected_roas is not None
            and expected_roas < 1
        ) or (
            delta_conversions is not None
            and delta_conversions < 0
        ):
            return "Maintenir sous surveillance"

        return "Maintenir le budget"

    return label or action or "Action a confirmer"


def _business_advice(
    action: str | None,
    label: str | None,
    expected_impact: dict[str, Any],
) -> str:
    action_text = str(action or "").lower()
    expected_roas = _nested_number(expected_impact, "expected_roas")
    delta_conversions = _nested_number(expected_impact, "delta_conversions")

    if _is_maintain_action(action, label):
        if expected_roas is not None and expected_roas < 1:
            return (
                "Ne pas scaler maintenant : surveiller 48h et revoir le ciblage, "
                "les creatives ou l'offre avant toute hausse de budget."
            )

        if delta_conversions is not None and delta_conversions < 0:
            return (
                "Maintenir le budget, mais suivre les conversions de pres avant "
                "d'augmenter l'investissement."
            )

        return "Continuer la surveillance avant toute modification budgetaire."

    if "decrease" in action_text:
        return (
            "Reduire progressivement et verifier que le CPA baisse sans casser "
            "le volume de conversions."
        )

    if "increase" in action_text:
        return (
            "Augmenter par palier et controler ROAS, CPA et conversions avant "
            "de scaler davantage."
        )

    if "reallocate" in action_text:
        return (
            "Reallouer en test controle, puis comparer les performances des canaux "
            "avant de generaliser."
        )

    if "pause" in action_text:
        return (
            "Mettre en pause seulement apres validation humaine, puis relancer avec "
            "un ciblage ou une creation corrigee."
        )

    return "Appliquer l'action progressivement et suivre les KPIs prioritaires."


def _short_business_summary(
    action: str | None,
    label: str | None,
    expected_impact: dict[str, Any],
) -> str:
    action_text = str(action or "").lower()
    expected_roas = _nested_number(expected_impact, "expected_roas")
    delta_conversions = _nested_number(expected_impact, "delta_conversions")

    if _is_maintain_action(action, label):
        if expected_roas is not None and expected_roas < 1:
            return "Ne pas scaler maintenant. Revoir ciblage, creatives ou offre avant toute hausse de budget."

        if delta_conversions is not None and delta_conversions < 0:
            return "Maintenir le budget et surveiller les conversions avant toute hausse."

        return "Maintenir le budget et continuer la surveillance."

    if "decrease" in action_text:
        return "Reduire progressivement et verifier que le CPA baisse."

    if "increase" in action_text:
        return "Augmenter par palier avec controle ROAS, CPA et conversions."

    if "reallocate" in action_text:
        return "Reallouer en test controle puis comparer les canaux."

    if "pause" in action_text:
        return "Pause possible uniquement apres validation humaine."

    return "Appliquer progressivement et suivre les KPIs prioritaires."


def _business_recommendation(
    action: str | None,
    label: str | None,
    expected_impact: dict[str, Any],
    dashboard_summary: str | None,
) -> dict[str, Any]:
    title = _business_action_title(action, label, expected_impact)
    advice = _business_advice(action, label, expected_impact)
    short_summary = _short_business_summary(action, label, expected_impact)

    expected_roas = _nested_number(expected_impact, "expected_roas")
    delta_conversions = _nested_number(expected_impact, "delta_conversions")

    if _is_maintain_action(action, label):
        if expected_roas is not None and expected_roas < 1:
            summary = (
                "Il est recommande de maintenir le budget actuel sans augmentation. "
                "La campagne reste fragile : le ROAS attendu reste sous 1 et le volume "
                "de conversions peut baisser. Avant toute hausse de budget, il faut "
                "corriger le ciblage, les creatives ou l'offre."
            )
        elif delta_conversions is not None and delta_conversions < 0:
            summary = (
                "Il est recommande de maintenir le budget actuel pour eviter une decision "
                "trop agressive, car le volume de conversions risque de baisser. La campagne "
                "doit rester sous surveillance avant toute augmentation."
            )
        else:
            summary = (
                "Il est recommande de maintenir le budget actuel et de continuer la "
                "surveillance des indicateurs avant toute modification."
            )
    elif "decrease" in str(action or "").lower():
        summary = (
            "Il est recommande de reduire progressivement le budget, car la campagne "
            "presente un risque de rentabilite ou d'efficacite."
        )
    elif "increase" in str(action or "").lower():
        summary = (
            "Il est recommande d'augmenter le budget par palier, avec un controle strict "
            "du ROAS, du CPA et des conversions."
        )
    elif "reallocate" in str(action or "").lower():
        summary = (
            "Il est recommande de reallouer une partie du budget entre les canaux, puis "
            "de comparer les resultats avant de generaliser la decision."
        )
    elif "pause" in str(action or "").lower():
        summary = (
            "Il est recommande de mettre la campagne en pause uniquement apres validation "
            "humaine, car le niveau de risque est eleve."
        )
    else:
        summary = dashboard_summary or "Recommandation a valider avec les KPIs prioritaires."

    return {
        "title": title,
        "summary": summary,
        "short_summary": short_summary,
        "advice": advice,
        "technical_summary": dashboard_summary,
    }


def _clean_anomaly_text(explanation: str | None) -> str:
    if not explanation:
        return "Anomalie detectee"

    first_signal = str(explanation).split(" | ")[0]
    first_signal = re.sub(r"^\[[^\]]+\]\s*", "", first_signal).strip()

    return first_signal or "Anomalie detectee"


def _main_anomaly_label(clean_text: str) -> str:
    return clean_text.split(":")[0].strip() or clean_text


def _format_period(dates: list[str]) -> str | None:
    if not dates:
        return None

    sorted_dates = sorted(dates)
    if sorted_dates[0] == sorted_dates[-1]:
        return sorted_dates[0]

    return f"{sorted_dates[0]} et {sorted_dates[-1]}"


def _format_roas_range(values: list[float]) -> str | None:
    if not values:
        return None

    min_roas = min(values)
    max_roas = max(values)

    if round(min_roas, 2) == round(max_roas, 2):
        return f"{min_roas:.2f}x"

    return f"{min_roas:.2f}x et {max_roas:.2f}x"


def _business_anomaly_summary(anomaly: dict[str, Any]) -> dict[str, Any]:
    top_anomalies = anomaly.get("top_anomalies", [])
    if not isinstance(top_anomalies, list):
        top_anomalies = []

    dates = [
        str(item.get("date"))
        for item in top_anomalies
        if isinstance(item, dict) and item.get("date")
    ]

    roas_values = [
        float(item.get("roas"))
        for item in top_anomalies
        if isinstance(item, dict) and item.get("roas") is not None
    ]

    scores = [
        float(item.get("score"))
        for item in top_anomalies
        if isinstance(item, dict) and item.get("score") is not None
    ]

    first_explanation = None
    if top_anomalies and isinstance(top_anomalies[0], dict):
        first_explanation = top_anomalies[0].get("explanation")

    main_signal_detail = _clean_anomaly_text(first_explanation)
    main_signal = _main_anomaly_label(main_signal_detail)

    total_days = anomaly.get("n_anomaly_days") or len(set(dates))
    displayed_days = len(set(dates))
    displayed_period = _format_period(dates)
    roas_range = _format_roas_range(roas_values)
    types = anomaly.get("anomaly_types", [])
    if not isinstance(types, list):
        types = []

    return {
        "main_signal": main_signal,
        "main_signal_detail": main_signal_detail,
        "level": anomaly.get("global_level"),
        "n_days": total_days,
        "displayed_days": displayed_days,
        "period": displayed_period,
        "roas_range": roas_range,
        "max_score": round(max(scores), 4) if scores else None,
        "types": types,
        "business_summary": (
            f"{main_signal}. "
            + (
                f"{displayed_days} jour(s) anormaux prioritaires detectes entre {displayed_period}"
                if displayed_period
                else f"{total_days or 0} jour(s) anormaux detectes"
            )
            + (f", avec un ROAS observe entre {roas_range}." if roas_range else ".")
            + (
                f" Le systeme a detecte {total_days} jour(s) anormaux au total."
                if total_days and displayed_days and total_days != displayed_days
                else ""
            )
        ),
    }


def _index_health(health_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    campaigns = health_data.get("campaigns", [])
    return {
        _campaign_key(item.get("campaign_id"), item.get("platform")): item
        for item in campaigns
        if isinstance(item, dict)
    }


def _index_anomalies(anomaly_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    campaigns = anomaly_data.get("campaigns", {})

    if isinstance(campaigns, dict):
        return {
            _campaign_key(item.get("campaign_id"), item.get("platform")): item
            for item in campaigns.values()
            if isinstance(item, dict)
        }

    return {}


def _index_causal(causal_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = causal_data.get("results", [])
    return {
        _campaign_key(item.get("campaign_id"), item.get("platform")): item
        for item in results
        if isinstance(item, dict)
    }


def _index_optimization(optimization_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plan = optimization_data.get("optimization_plan", [])
    return {
        _campaign_key(item.get("campaign_id"), item.get("platform")): item
        for item in plan
        if isinstance(item, dict)
    }


def _index_xai(xai_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    explanations = xai_data.get("xai_explanations", [])
    return {
        _campaign_key(item.get("campaign_id"), item.get("platform")): item
        for item in explanations
        if isinstance(item, dict)
    }


def _source_status() -> dict[str, dict[str, Any]]:
    sources = {
        "health_score": HEALTH_PATH,
        "anomaly_report": ANOMALY_PATH,
        "correlations": CORRELATIONS_PATH,
        "causal_effects": CAUSAL_PATH,
        "optimization_plan": OPTIMIZATION_PATH,
        "xai_explanations": XAI_PATH,
    }

    return {
        name: {
            "path": str(path),
            "available": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in sources.items()
    }


def _build_campaign_section(
    key: str,
    health_index: dict[str, dict[str, Any]],
    anomaly_index: dict[str, dict[str, Any]],
    causal_index: dict[str, dict[str, Any]],
    optimization_index: dict[str, dict[str, Any]],
    xai_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    health = health_index.get(key, {})
    anomaly = anomaly_index.get(key, {})
    causal = causal_index.get(key, {})
    optimization = optimization_index.get(key, {})
    xai = xai_index.get(key, {})

    health_details = health.get("details", {}) if isinstance(health.get("details"), dict) else {}
    prediction = health_details.get("prediction", {}) if isinstance(health_details.get("prediction"), dict) else {}
    trend = health_details.get("trend", {}) if isinstance(health_details.get("trend"), dict) else {}

    causal_diag = causal.get("diagnosis", {}) if isinstance(causal.get("diagnosis"), dict) else {}
    root_cause = (
        optimization.get("root_cause")
        or causal_diag.get("root_cause")
        or xai.get("causal_explanation", {}).get("root_cause")
    )

    dashboard_summary = (
        xai.get("dashboard_summary")
        or xai.get("optimizer_explanation", {}).get("summary")
        or optimization.get("explanation")
        or xai.get("xai_summary")
    )

    recommended_action = optimization.get("recommended_action")
    action_label = optimization.get("action_label")
    expected_impact = optimization.get("expected_impact", {})
    if not isinstance(expected_impact, dict):
        expected_impact = {}

    budget_adjustment = optimization.get("budget_adjustment", {})
    if not isinstance(budget_adjustment, dict):
        budget_adjustment = {}

    anomaly_section = {
        "level": anomaly.get("global_level"),
        "score": _round(anomaly.get("anomaly_score"), 4),
        "n_anomaly_days": anomaly.get("n_anomaly_days"),
        "types": anomaly.get("anomaly_types", []),
        "top_anomalies": anomaly.get("top_anomalies", [])[:5],
    }

    return {
        "campaign_id": health.get("campaign_id") or optimization.get("campaign_id") or xai.get("campaign_id"),
        "global_campaign_id": (
            health.get("global_campaign_id")
            or optimization.get("global_campaign_id")
            or xai.get("global_campaign_id")
        ),
        "platform": health.get("platform") or optimization.get("platform") or xai.get("platform"),
        "status": health.get("status") or optimization.get("health_status"),
        "health_score": _round(health.get("health_score") or optimization.get("health_score")),
        "trigger_causal_ai": bool(health.get("trigger_causal_ai", False)),
        "current_kpis": health.get("current_kpis") or optimization.get("current_state") or {},
        "predicted_kpis": prediction.get("predicted_kpis") or {},
        "trend": {
            "roas_trend_pct": _round(trend.get("roas_trend_pct")),
            "conversions_trend_pct": _round(trend.get("conversions_trend_pct")),
            "spend_trend_pct": _round(trend.get("spend_trend_pct")),
        },
        "anomalies": anomaly_section,
        "anomaly_summary": _business_anomaly_summary(anomaly),
        "root_cause": {
            "code": root_cause,
            "label": xai.get("causal_explanation", {}).get("meaning"),
            "confidence": _round(
                optimization.get("causal_confidence")
                or causal_diag.get("confidence")
                or xai.get("causal_explanation", {}).get("confidence"),
                4,
            ),
            "evidence": optimization.get("causal_evidence") or xai.get("causal_explanation", {}).get("evidence"),
        },
        "recommended_action": {
            "action": recommended_action,
            "label": action_label,
            "priority": optimization.get("priority"),
            "expected_impact": expected_impact,
            "budget_adjustment": budget_adjustment,
            "constraints_applied": optimization.get("constraints_applied", []),
        },
        "business_recommendation": _business_recommendation(
            recommended_action,
            action_label,
            expected_impact,
            dashboard_summary,
        ),
        "explanations": {
            "xai_summary": xai.get("xai_summary"),
            "dashboard_summary": dashboard_summary,
        },
    }


def _portfolio_summary(campaigns: list[dict[str, Any]], correlations: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter(str(item.get("status") or "UNKNOWN") for item in campaigns)
    platforms = Counter(str(item.get("platform") or "unknown") for item in campaigns)
    actions = Counter(
        str(item.get("recommended_action", {}).get("action") or "no_action")
        for item in campaigns
    )

    health_scores = [
        _safe_float(item.get("health_score"), default=-1)
        for item in campaigns
        if item.get("health_score") is not None
    ]

    total_spend = sum(
        _safe_float(item.get("current_kpis", {}).get("spend"))
        for item in campaigns
    )

    return {
        "n_campaigns": len(campaigns),
        "status_counts": dict(statuses),
        "platform_counts": dict(platforms),
        "recommended_action_counts": dict(actions),
        "average_health_score": round(sum(health_scores) / len(health_scores), 2) if health_scores else None,
        "total_current_spend": round(total_spend, 2),
        "cross_channel": correlations,
    }


def _executive_summary(campaigns: list[dict[str, Any]], portfolio: dict[str, Any]) -> dict[str, Any]:
    if not campaigns:
        return {
            "headline": "Aucune campagne disponible pour l'audit.",
            "key_findings": [],
            "priority_actions": [],
        }

    sorted_campaigns = sorted(
        campaigns,
        key=lambda item: _safe_float(item.get("health_score"), default=999),
    )
    worst = sorted_campaigns[0]

    key_findings = [
        f"{portfolio['n_campaigns']} campagne(s) auditee(s).",
        f"Health score moyen: {portfolio.get('average_health_score')}.",
        f"Campagne la plus fragile: {worst.get('campaign_id')} ({worst.get('platform')}) avec un score {worst.get('health_score')}.",
    ]

    priority_actions = []
    for campaign in sorted_campaigns[:3]:
        action = campaign.get("recommended_action", {})
        business = campaign.get("business_recommendation", {})
        label = business.get("title") or action.get("label") or action.get("action") or "Action a confirmer"
        priority_actions.append({
            "campaign_id": campaign.get("campaign_id"),
            "platform": campaign.get("platform"),
            "priority": action.get("priority"),
            "action": label,
            "summary": business.get("summary") or campaign.get("explanations", {}).get("dashboard_summary"),
            "short_summary": business.get("short_summary") or business.get("advice"),
            "advice": business.get("advice"),
        })

    return {
        "headline": "Audit des campagnes existantes genere avec succes.",
        "key_findings": key_findings,
        "priority_actions": priority_actions,
    }


def build_audit_report() -> dict[str, Any]:
    health_data = _load_json(HEALTH_PATH, {"campaigns": [], "metadata": {}})
    anomaly_data = _load_json(ANOMALY_PATH, {"campaigns": {}})
    correlations = _load_json(CORRELATIONS_PATH, {})
    causal_data = _load_json(CAUSAL_PATH, {"results": [], "metadata": {}})
    optimization_data = _load_json(OPTIMIZATION_PATH, {"optimization_plan": [], "metadata": {}})
    xai_data = _load_json(XAI_PATH, {"xai_explanations": [], "metadata": {}})

    health_index = _index_health(health_data)
    anomaly_index = _index_anomalies(anomaly_data)
    causal_index = _index_causal(causal_data)
    optimization_index = _index_optimization(optimization_data)
    xai_index = _index_xai(xai_data)

    keys = sorted(
        set(health_index)
        | set(anomaly_index)
        | set(causal_index)
        | set(optimization_index)
        | set(xai_index)
    )

    campaigns = [
        _build_campaign_section(
            key,
            health_index,
            anomaly_index,
            causal_index,
            optimization_index,
            xai_index,
        )
        for key in keys
    ]

    portfolio = _portfolio_summary(campaigns, correlations)

    return {
        "title": "Rapport d'audit des campagnes existantes",
        "generated_at": datetime.now().isoformat(),
        "report_type": "campaign_optimization_audit",
        "source_status": _source_status(),
        "source_metadata": {
            "health_score": health_data.get("metadata", {}),
            "causal": causal_data.get("metadata", {}),
            "optimizer": optimization_data.get("metadata", {}),
            "xai": xai_data.get("metadata", {}),
        },
        "executive_summary": _executive_summary(campaigns, portfolio),
        "portfolio_summary": portfolio,
        "campaigns": campaigns,
        "next_steps": [
            "Valider les recommandations prioritaires avec un responsable marketing.",
            "Surveiller les campagnes dont le ROAS predit reste inferieur a 1.",
            "Generer une version business/PDF a partir de ce rapport structure.",
        ],
    }


def save_audit_report(report: dict[str, Any], output_path: Path = OUT_AUDIT_JSON) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)

    return str(output_path)


def run_audit_report() -> dict[str, Any]:
    try:
        report = build_audit_report()
        output_path = save_audit_report(report)

        return {
            "status": "success",
            "message": "Rapport d'audit genere avec succes",
            "output_file": output_path,
            "data": report,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


if __name__ == "__main__":
    print(json.dumps(run_audit_report(), ensure_ascii=True, indent=2, default=str))
