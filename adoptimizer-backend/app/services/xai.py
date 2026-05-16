# =============================================================================
# TOOL XAI GLOBAL — AdOptimizer AI
# Fusion des explications : Health Score + Causal AI + Optimizer + Feature Importance
# =============================================================================

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path("app")

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

HEALTH_PATH = OUTPUT_DIR / "campaign_health_score.json"
ANOMALY_PATH = OUTPUT_DIR / "anomaly_report.json"
CAUSAL_PATH = OUTPUT_DIR / "causal_effects.json"
OPTIMIZATION_PATH = OUTPUT_DIR / "optimization_plan.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

OUT_XAI = OUTPUT_DIR / "xai_explanations.json"
OUT_REPORT = OUTPUT_DIR / "xai_report.txt"

# =============================================================================
# HELPERS
# =============================================================================

def load_json(path, default):
    if not path.exists():
        print(f"[WARN] Fichier manquant : {path}")
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_feature_importance(path, top_n=5):
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
        cols = df.columns.tolist()

        feature_col = "feature" if "feature" in cols else cols[0]
        importance_col = "importance" if "importance" in cols else cols[-1]

        df = df.sort_values(importance_col, ascending=False).head(top_n)

        return [
            {
                "feature": str(row[feature_col]),
                "importance": round(float(row[importance_col]), 4)
            }
            for _, row in df.iterrows()
        ]

    except Exception as e:
        print(f"[WARN] Erreur feature importance : {e}")
        return []

def severity_label(score):
    if score < 30:
        return "critique"
    if score < 60:
        return "à surveiller"
    return "saine"

def explain_health(health):
    score = health.get("health_score", 50)
    status = health.get("status", "UNKNOWN")
    components = health.get("components", {})
    details = health.get("details", {})

    prediction = components.get("prediction_score", None)
    anomaly = components.get("anomaly_score", None)
    trend = components.get("trend_score", None)

    reasons = []

    if prediction is not None and prediction < 50:
        reasons.append("les prédictions futures sont défavorables")
    if anomaly is not None and anomaly < 50:
        reasons.append("des anomalies importantes ont été détectées")
    if trend is not None and trend < 50:
        reasons.append("la tendance récente est négative")

    if not reasons:
        reasons.append("les signaux globaux restent acceptables")

    return {
        "health_score": score,
        "status": status,
        "severity": severity_label(score),
        "main_reasons": reasons,
        "components": components,
        "current_kpis": health.get("current_kpis", {}),
        "predicted_kpis": details.get("prediction", {}).get("predicted_kpis", {}),
        "trend": details.get("trend", {}),
        "anomaly": details.get("anomaly", {}),
        "summary": f"La campagne est considérée comme {severity_label(score)} avec un score santé de {score}."
    }

def explain_causal(causal):
    if not causal:
        return {
            "root_cause": "non_disponible",
            "confidence": 0,
            "evidence": "Aucune analyse causale disponible.",
            "summary": "Aucune cause racine n’a été identifiée."
        }

    diagnosis = causal.get("diagnosis", {})
    root_cause = diagnosis.get("root_cause", "no_clear_cause")
    confidence = diagnosis.get("confidence", 0)
    evidence = diagnosis.get("evidence", "")

    readable = {
        "halo_effect": "effet inter-canal positif ou indirect entre Meta et Google",
        "cannibalization": "cannibalisation entre les canaux publicitaires",
        "direct_budget_impact": "impact direct du budget sur les conversions",
        "ad_saturation": "saturation publicitaire",
        "budget_inefficiency": "inefficacité budgétaire",
        "delayed_branding": "effet retardé des campagnes de branding",
        "no_clear_cause": "cause non clairement identifiée"
    }

    return {
        "root_cause": root_cause,
        "confidence": round(float(confidence), 3),
        "evidence": evidence,
        "meaning": readable.get(root_cause, root_cause),
        "summary": f"La cause principale détectée est : {readable.get(root_cause, root_cause)} avec une confiance de {confidence:.0%}."
    }

def explain_optimizer(opt):
    if not opt:
        return {
            "recommended_action": "aucune",
            "summary": "Aucune recommandation d’optimisation disponible."
        }

    action = opt.get("recommended_action")
    label = opt.get("action_label")
    impact = opt.get("expected_impact", {})
    budget = opt.get("budget_adjustment", {})
    constraints = opt.get("constraints_applied", [])

    return {
        "recommended_action": action,
        "action_label": label,
        "expected_impact": impact,
        "budget_adjustment": budget,
        "constraints_applied": constraints,
        "summary": opt.get("explanation", "Recommandation générée par l’optimizer.")
    }

def build_final_user_explanation(health_xai, causal_xai, optimizer_xai, features):
    parts = []

    # 1. Health
    parts.append(health_xai["summary"])

    # 2. Causal
    if causal_xai["root_cause"] != "non_disponible":
        parts.append(causal_xai["summary"])

    # 3. Optimizer
    if optimizer_xai.get("recommended_action") != "aucune":
        parts.append(f"Action recommandée : {optimizer_xai.get('action_label')}.")

        impact = optimizer_xai.get("expected_impact", {})
        ba = optimizer_xai.get("budget_adjustment", {})

        # Impact business clair
        if impact:
            roas_gain = impact.get("delta_roas_pct", 0)
            parts.append(f"Impact attendu : amélioration de la rentabilité (+{roas_gain:.1f}% ROAS).")

        # Quantitatif
        if ba:
            parts.append(ba.get("quantitative_explanation", ""))

        # ⚠️ Risk (IMPORTANT)
        conv_delta = impact.get("delta_conversions", 0)
        if conv_delta < 0:
            parts.append(f"⚠️ Cette optimisation peut entraîner une baisse des conversions ({conv_delta:.2f}).")

        # 🧠 Trade-off (niveau expert)
        if conv_delta < 0 and impact.get("delta_roas_pct", 0) > 0:
            parts.append("Le système privilégie une amélioration du ROAS au détriment du volume de conversions.")

    # 4. Features expliquées
    if features:
        top_features = [f["feature"] for f in features[:3]]

        mapping = {
            "budget_utilization": "l’utilisation du budget",
            "conversions": "le nombre de conversions",
            "conversions_roll7m": "la tendance récente des conversions",
            "cpa": "le coût par acquisition",
            "roas": "la rentabilité (ROAS)"
        }

        readable = [mapping.get(f, f) for f in top_features]
        parts.append(f"Les facteurs principaux influençant la performance sont : {', '.join(readable)}.")

    # 5. Confiance globale
    confidence = causal_xai.get("confidence", 0)
    if confidence >= 0.8:
        parts.append("✔ Recommandation fiable (confiance élevée).")
    elif confidence >= 0.5:
        parts.append("⚠️ Recommandation modérée (confiance moyenne).")
    else:
        parts.append("⚠️ Recommandation incertaine (faible confiance).")

    return " ".join([p for p in parts if p])

# =============================================================================
# RUN
# =============================================================================

def run():
    print("=" * 70)
    print("TOOL XAI GLOBAL — AdOptimizer AI")
    print("=" * 70)

    health_data = load_json(HEALTH_PATH, {"campaigns": []})
    anomaly_data = load_json(ANOMALY_PATH, {"campaigns": {}})
    causal_data = load_json(CAUSAL_PATH, {"results": []})
    optimization_data = load_json(OPTIMIZATION_PATH, {"optimization_plan": []})
    top_features = load_feature_importance(FEATURE_IMPORTANCE_PATH)

    health_by_campaign = {
        h["campaign_id"]: h
        for h in health_data.get("campaigns", [])
        if "campaign_id" in h
    }

    causal_by_campaign = {
        c["campaign_id"]: c
        for c in causal_data.get("results", [])
        if "campaign_id" in c
    }

    opt_by_campaign = {
        o["campaign_id"]: o
        for o in optimization_data.get("optimization_plan", [])
        if "campaign_id" in o
    }

    anomaly_by_campaign = anomaly_data.get("campaigns", {})

    all_campaign_ids = sorted(
        set(health_by_campaign.keys())
        | set(causal_by_campaign.keys())
        | set(opt_by_campaign.keys())
    )

    explanations = []

    for cid in all_campaign_ids:
        health = health_by_campaign.get(cid, {})
        causal = causal_by_campaign.get(cid, {})
        opt = opt_by_campaign.get(cid, {})
        platform = (
            health.get("platform")
            or opt.get("platform")
            or causal.get("platform")
        )
        anomaly = (
            anomaly_by_campaign.get(f"{cid}|{platform}")
            or anomaly_by_campaign.get(cid)
            or {}
        )

        health_xai = explain_health(health) if health else {
            "summary": "Health Score non disponible.",
            "health_score": None,
            "status": "UNKNOWN",
            "severity": "inconnue",
            "main_reasons": []
        }

        causal_xai = explain_causal(causal)
        optimizer_xai = explain_optimizer(opt)

        final_explanation = build_final_user_explanation(
            health_xai,
            causal_xai,
            optimizer_xai,
            top_features
        )

        explanations.append({
            "campaign_id": cid,
            "global_campaign_id": (
                health.get("global_campaign_id")
                or opt.get("global_campaign_id")
                or causal.get("global_campaign_id")
            ),
            "platform": (
                health.get("platform")
                or opt.get("platform")
                or causal.get("platform")
            ),

            "xai_summary": final_explanation,
            "anomaly_report": anomaly,

            "health_explanation": health_xai,
            "causal_explanation": causal_xai,
            "optimizer_explanation": optimizer_xai,

            "model_feature_importance": top_features,

            "xai_level": "global_fusion",
            "generated_at": datetime.now().isoformat()
        })

    output = {
        "metadata": {
            "tool": "Tool XAI Global",
            "generated_at": datetime.now().isoformat(),
            "n_campaigns_explained": len(explanations),
            "inputs": {
                "health_score": str(HEALTH_PATH),
                "anomaly": str(ANOMALY_PATH),
                "causal_ai": str(CAUSAL_PATH),
                "optimizer": str(OPTIMIZATION_PATH),
                "feature_importance": str(FEATURE_IMPORTANCE_PATH)
            }
        },
        "xai_explanations": explanations
    }

    with open(OUT_XAI, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    report_lines = [
        "=" * 70,
        "TOOL XAI GLOBAL REPORT",
        f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        f"Campagnes expliquées : {len(explanations)}",
        ""
    ]

    for x in explanations:
        report_lines.append("-" * 70)
        report_lines.append(f"[{x['campaign_id']}] {x.get('platform')}")
        report_lines.append(f"Résumé XAI : {x['xai_summary']}")
        report_lines.append("")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] XAI genere : {OUT_XAI}")
    print(f"[OK] Rapport genere : {OUT_REPORT}")
    print("=" * 70)

    return output

if __name__ == "__main__":
    print(run())

def run_xai():
    try:
        result = run()
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
