# =============================================================================
# CAMPAIGN HEALTH SCORE — AdOptimizer AI
# Agrégation :
#   • Anomalies   40%
#   • Prédictions 40%
#   • Tendance    20%
# Output :
#   outputs/campaign_health_score.json
# =============================================================================

import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

BASE_DIR = Path("app")

DATASET_PATH = BASE_DIR / "data" / "dataset_model_ready.csv"
MODEL_PATH   = BASE_DIR / "models" / "best_model.pkl"
ANOMALY_PATH = BASE_DIR / "outputs" / "anomaly_report.json"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUT_PATH = OUTPUT_DIR / "campaign_health_score.json"
CSV_PATH = OUTPUT_DIR / "campaign_health_score.csv"

RECENT_DAYS = 14
TREND_DAYS  = 7

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x) or np.isinf(x):
            return default
        return float(x)
    except Exception:
        return default


def clip_score(x):
    return float(np.clip(x, 0, 100))


def status_from_score(score):
    if score < 40:
        return "CRITICAL"
    elif score < 60:
        return "WARNING"
    return "OK"


def normalize(value, low, high, reverse=False):
    value = safe_float(value, np.nan)

    if pd.isna(value) or high <= low:
        return 50.0

    value = np.clip(value, low, high)
    score = (value - low) / (high - low) * 100

    if reverse:
        score = 100 - score

    return clip_score(score)


# -----------------------------------------------------------------------------
# 1. PREDICTION SCORE
# -----------------------------------------------------------------------------

def prepare_features(row_dict, feature_cols, imputer):
    X = pd.DataFrame([row_dict]).reindex(columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan)
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_cols)
    return X_imp


def predict_campaign(row_dict, models_by_target, feature_cols, imputer):
    X = prepare_features(row_dict, feature_cols, imputer)
    preds = {}

    for target, model in models_by_target.items():
        preds[target] = safe_float(model.predict(X)[0])

    return preds


def get_pred(preds, candidates, default=np.nan):
    for c in candidates:
        if c in preds:
            return safe_float(preds[c], default)
    return default


def compute_prediction_score(row_dict, models_by_target, feature_cols, imputer):
    preds = predict_campaign(row_dict, models_by_target, feature_cols, imputer)

    current_roas = safe_float(row_dict.get("roas", 0))
    current_conv = safe_float(row_dict.get("conversions", 0))

    pred_roas = get_pred(preds, ["target_roas_h14", "target_roas_h7", "target_roas_h3"], current_roas)
    pred_conv = get_pred(preds, ["target_conversions_h14", "target_conversions_h7", "target_conversions_h3"], current_conv)
    pred_cpa  = get_pred(preds, ["target_cpa_h14", "target_cpa_h7", "target_cpa_h3"], np.nan)
    pred_ctr  = get_pred(preds, ["target_ctr_h14", "target_ctr_h7", "target_ctr_h3"], np.nan)
    pred_cpc  = get_pred(preds, ["target_cpc_h14", "target_cpc_h7", "target_cpc_h3"], np.nan)

    roas_score = normalize(pred_roas, 0, 5)
    conv_score = normalize(pred_conv, 0, 30)
    cpa_score  = normalize(pred_cpa, 1, 120, reverse=True)
    ctr_score  = normalize(pred_ctr, 0.001, 0.10)
    cpc_score  = normalize(pred_cpc, 0.05, 5, reverse=True)

    if current_roas <= 0.05:
        roas_trend_score = 50.0
    else:
        delta_roas = (pred_roas - current_roas) / current_roas
        roas_trend_score = normalize(delta_roas, -0.50, 0.50)

    degradation_penalty = 0

    if current_roas > 0.2 and pred_roas < current_roas * 0.90:
        degradation_penalty += 10

    if current_conv > 1 and pred_conv < current_conv * 0.90:
        degradation_penalty += 8

    prediction_score = (
        0.35 * roas_score +
        0.25 * conv_score +
        0.15 * cpa_score +
        0.10 * ctr_score +
        0.10 * cpc_score +
        0.05 * roas_trend_score -
        degradation_penalty
    )

    prediction_score = clip_score(prediction_score)

    return prediction_score, {
        "roas_score": round(roas_score, 2),
        "conversions_score": round(conv_score, 2),
        "cpa_score": round(cpa_score, 2),
        "ctr_score": round(ctr_score, 2),
        "cpc_score": round(cpc_score, 2),
        "prediction_roas_trend_score": round(roas_trend_score, 2),
        "degradation_penalty": degradation_penalty,
        "predicted_kpis": {
            "roas_h14": round(safe_float(pred_roas), 4),
            "conversions_h14": round(safe_float(pred_conv), 4),
            "cpa_h14": round(safe_float(pred_cpa), 4) if not pd.isna(pred_cpa) else None,
            "ctr_h14": round(safe_float(pred_ctr), 6) if not pd.isna(pred_ctr) else None,
            "cpc_h14": round(safe_float(pred_cpc), 4) if not pd.isna(pred_cpc) else None,
        }
    }


# -----------------------------------------------------------------------------
# 2. ANOMALY SCORE
# -----------------------------------------------------------------------------

LEVEL_BASE_SCORE = {
    "OK": 95,
    "NORMAL": 95,
    "INFO": 85,
    "LOW": 80,
    "WARNING": 60,
    "MEDIUM": 55,
    "HIGH": 35,
    "CRITICAL": 15
}

SEVERITY_PENALTY = {
    "INFO": 3,
    "LOW": 5,
    "WARNING": 10,
    "MEDIUM": 12,
    "HIGH": 20,
    "CRITICAL": 35
}

KPI_IMPORTANCE = {
    "roas": 1.4,
    "conversions": 1.3,
    "cpa": 1.2,
    "ctr": 1.0,
    "cpc": 0.9,
    "spend": 0.8,
    "impressions": 0.6,
    "clicks": 0.7
}


def extract_anomalies(payload):
    for key in ["anomalies", "detected_anomalies", "details", "issues"]:
        val = payload.get(key)
        if isinstance(val, list):
            return val
    return []


def anomaly_penalty(anomaly):
    if not isinstance(anomaly, dict):
        return 5

    severity = str(
        anomaly.get("severity")
        or anomaly.get("level")
        or anomaly.get("status")
        or "LOW"
    ).upper()

    penalty = SEVERITY_PENALTY.get(severity, 5)

    metric = str(
        anomaly.get("metric")
        or anomaly.get("kpi")
        or anomaly.get("feature")
        or ""
    ).lower()

    multiplier = 1.0

    for kpi, weight in KPI_IMPORTANCE.items():
        if kpi in metric:
            multiplier = weight
            break

    return penalty * multiplier


def compute_anomaly_score(campaign_id, platform, anomaly_campaigns):
    key1 = f"{campaign_id}|{platform}"
    key2 = campaign_id

    payload = anomaly_campaigns.get(key1) or anomaly_campaigns.get(key2)

    if payload is None:
        return 75.0, {
            "global_level": "UNKNOWN",
            "n_anomalies": 0,
            "total_penalty": 20,
            "note": "campaign_not_found_in_anomaly_report"
        }

    global_level = str(payload.get("global_level", "OK")).upper()
    base_score = LEVEL_BASE_SCORE.get(global_level, 75)

    anomalies_list = extract_anomalies(payload)
    n_anomalies = len(anomalies_list)

    total_penalty = sum(anomaly_penalty(a) for a in anomalies_list)

    if n_anomalies == 0:
        if global_level == "WARNING":
            total_penalty += 10
        elif global_level == "CRITICAL":
            total_penalty += 25
        elif global_level == "HIGH":
            total_penalty += 18

    total_penalty = min(total_penalty, 70)

    anomaly_score = clip_score(base_score - total_penalty)

    return anomaly_score, {
        "global_level": global_level,
        "n_anomalies": n_anomalies,
        "total_penalty": round(total_penalty, 2)
    }


# -----------------------------------------------------------------------------
# 3. TREND SCORE
# -----------------------------------------------------------------------------

def compute_trend_score(g):
    g = g.sort_values("date").copy()

    if len(g) < TREND_DAYS * 2:
        return 50.0, {
            "note": "not_enough_history",
            "roas_trend": 0,
            "conversions_trend": 0,
            "spend_trend": 0
        }

    recent   = g.tail(TREND_DAYS)
    previous = g.iloc[-TREND_DAYS * 2:-TREND_DAYS]

    recent_roas    = safe_float(recent["roas"].mean())    if "roas"        in g.columns else 0
    previous_roas  = safe_float(previous["roas"].mean())  if "roas"        in g.columns else 0

    recent_conv    = safe_float(recent["conversions"].mean())   if "conversions" in g.columns else 0
    previous_conv  = safe_float(previous["conversions"].mean()) if "conversions" in g.columns else 0

    recent_spend   = safe_float(recent["spend"].mean())   if "spend"       in g.columns else 0
    previous_spend = safe_float(previous["spend"].mean()) if "spend"       in g.columns else 0

    roas_trend  = (recent_roas  - previous_roas)  / max(abs(previous_roas),  0.01)
    conv_trend  = (recent_conv  - previous_conv)  / max(abs(previous_conv),  0.01)
    spend_trend = (recent_spend - previous_spend) / max(abs(previous_spend), 0.01)

    roas_score = normalize(roas_trend, -0.50, 0.50)
    conv_score = normalize(conv_trend, -0.50, 0.50)

    # spend qui augmente pendant que ROAS baisse = mauvais
    spend_penalty = 0
    if spend_trend > 0.20 and roas_trend < -0.10:
        spend_penalty = 15

    trend_score = (
        0.60 * roas_score +
        0.40 * conv_score -
        spend_penalty
    )

    trend_score = clip_score(trend_score)

    return trend_score, {
        "recent_roas": round(recent_roas, 4),
        "previous_roas": round(previous_roas, 4),
        "roas_trend_pct": round(roas_trend * 100, 2),
        "recent_conversions": round(recent_conv, 4),
        "previous_conversions": round(previous_conv, 4),
        "conversions_trend_pct": round(conv_trend * 100, 2),
        "spend_trend_pct": round(spend_trend * 100, 2),
        "spend_penalty": spend_penalty
    }


# -----------------------------------------------------------------------------
# 4. RECENT SNAPSHOT
# -----------------------------------------------------------------------------

def build_recent_snapshot(df):
    cutoff    = df["date"].max() - pd.Timedelta(days=RECENT_DAYS)
    recent_df = df[df["date"] >= cutoff].copy()

    rows = []

    for campaign_id, g in recent_df.groupby("campaign_id"):
        g    = g.sort_values("date")
        last = g.iloc[-1].to_dict()

        for col in ["roas", "conversions", "spend", "clicks", "impressions"]:
            if col in g.columns:
                last[col] = safe_float(g[col].mean())

        if "platform" in g.columns:
            last["platform"] = g["platform"].dropna().iloc[-1]

        if "global_campaign_id" in g.columns:
            last["global_campaign_id"] = g["global_campaign_id"].dropna().iloc[-1]

        rows.append(last)

    return rows


# -----------------------------------------------------------------------------
# WRAPPER FASTAPI / N8N
# -----------------------------------------------------------------------------

def run_health_score():
    try:
        # ── LOAD DATA ──────────────────────────────────────────────────────
        df = pd.read_csv(DATASET_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "campaign_id"])

        bundle = joblib.load(MODEL_PATH)

        models_by_target = bundle.get("models_by_target")
        imputer          = bundle.get("imputer")
        feature_cols     = bundle.get("feature_cols") or bundle.get("feature_columns")

        if models_by_target is None:
            raise ValueError("best_model.pkl ne contient pas models_by_target")

        if imputer is None or feature_cols is None:
            raise ValueError("best_model.pkl doit contenir imputer + feature_cols")

        with open(ANOMALY_PATH, "r", encoding="utf-8") as f:
            anomaly_report = json.load(f)

        anomaly_campaigns = anomaly_report.get("campaigns", {})

        print("✅ Dataset :", df.shape)
        print("✅ Modèle chargé")
        print("✅ Anomaly report chargé")

        # ── RUN HEALTH SCORE ───────────────────────────────────────────────
        results       = []
        snapshot_rows = build_recent_snapshot(df)

        for row in snapshot_rows:
            campaign_id        = row.get("campaign_id")
            platform           = row.get("platform")
            global_campaign_id = row.get("global_campaign_id")

            try:
                full_campaign_history = df[df["campaign_id"] == campaign_id].copy()

                prediction_score, prediction_details = compute_prediction_score(
                    row, models_by_target, feature_cols, imputer
                )
                anomaly_score, anomaly_details = compute_anomaly_score(
                    campaign_id, platform, anomaly_campaigns
                )
                trend_score, trend_details = compute_trend_score(full_campaign_history)

                health_score = (
                    0.40 * prediction_score +
                    0.40 * anomaly_score +
                    0.20 * trend_score
                )

                health_score = clip_score(health_score)
                status       = status_from_score(health_score)

                results.append({
                    "campaign_id"        : campaign_id,
                    "global_campaign_id" : global_campaign_id,
                    "platform"           : platform,

                    "health_score"       : round(health_score, 2),
                    "status"             : status,
                    "trigger_causal_ai"  : health_score < 60,

                    "components": {
                        "prediction_score" : round(prediction_score, 2),
                        "anomaly_score"    : round(anomaly_score, 2),
                        "trend_score"      : round(trend_score, 2)
                    },

                    "details": {
                        "prediction" : prediction_details,
                        "anomaly"    : anomaly_details,
                        "trend"      : trend_details
                    },

                    "current_kpis": {
                        "roas"        : round(safe_float(row.get("roas", 0)), 4),
                        "conversions" : round(safe_float(row.get("conversions", 0)), 4),
                        "spend"       : round(safe_float(row.get("spend", 0)), 4)
                    }
                })

            except Exception as e:
                results.append({
                    "campaign_id"        : campaign_id,
                    "global_campaign_id" : global_campaign_id,
                    "platform"           : platform,
                    "health_score"       : 50.0,
                    "status"             : "UNKNOWN",
                    "trigger_causal_ai"  : True,
                    "error"              : str(e)
                })

        # ── SAVE JSON ──────────────────────────────────────────────────────
        output = {
            "metadata": {
                "tool"         : "campaign_health_score",
                "generated_at" : datetime.now().isoformat(),
                "n_campaigns"  : len(results),
                "weights": {
                    "prediction" : 0.40,
                    "anomaly"    : 0.40,
                    "trend"      : 0.20
                },
                "trigger_threshold" : 60,
                "score_definition"  : "100=healthy campaign, 0=critical campaign"
            },
            "campaigns": results
        }

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # ── SAVE CSV ───────────────────────────────────────────────────────
        df_result = pd.DataFrame(results)
        df_result.to_csv(CSV_PATH, index=False)

        print("✅ campaign_health_score.json généré :", OUT_PATH)
        print("\nRésumé status :")
        print(df_result["status"].value_counts())
        print("\nCampagnes déclenchées Causal AI :")
        print(df_result["trigger_causal_ai"].value_counts())

        # ── RETURN API RESPONSE ────────────────────────────────────────────
        return {
            "status"      : "success",
            "message"     : "Health score calculé",
            "outputs"     : {
                "json" : str(OUT_PATH),
                "csv"  : str(CSV_PATH)
            },
            "n_campaigns" : len(results)
        }

    except Exception as e:
        return {
            "status"  : "error",
            "message" : str(e)
        }