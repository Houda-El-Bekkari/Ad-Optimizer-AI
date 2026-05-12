#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# TOOL 4 — CAUSAL AI  (V2 corrigée finale)
# Filtre  : outputs/campaign_health_score.json  (trigger_causal_ai == True)
# Inputs  : dataset_model_ready.csv
#           outputs/campaign_health_score.json
#           outputs/correlations.json
# Outputs : outputs/causal_effects.json
#           outputs/causal_explanations.txt
#           outputs/causal.log
# =============================================================================

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path("app")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATH = BASE_DIR / "data" / "dataset_model_ready.csv"
HEALTH_PATH  = BASE_DIR / "outputs" / "campaign_health_score.json"
CORR_PATH    = BASE_DIR / "outputs" / "correlations.json"


OUT_EFFECTS = OUTPUT_DIR / "causal_effects.json"
OUT_EXPLAIN = OUTPUT_DIR / "causal_explanations.txt"
OUT_LOG     = OUTPUT_DIR / "causal.log"

DML_FOLDS = 3
DML_N_TREES = 60
RANDOM_STATE = 42
MIN_OBSERVATIONS = 30

P_VALUE_THRESHOLD = 0.05
MIN_EFFECT_EPS = 1e-6

# =============================================================================
# LOGGING
# =============================================================================

if OUT_LOG.exists():
    OUT_LOG.unlink()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUT_LOG, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger("tool4_causal_ai")

# =============================================================================
# LOADERS
# =============================================================================

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "campaign_id"])

    if "roas" not in df.columns and {"conversion_value", "spend"}.issubset(df.columns):
        df["roas"] = df["conversion_value"] / df["spend"].clip(lower=1)

    if "ctr_calc" not in df.columns and {"clicks", "impressions"}.issubset(df.columns):
        df["ctr_calc"] = df["clicks"] / df["impressions"].clip(lower=1)

    if "cpc_calc" not in df.columns and {"spend", "clicks"}.issubset(df.columns):
        df["cpc_calc"] = df["spend"] / df["clicks"].clip(lower=1)

    if "cpa" not in df.columns and {"spend", "conversions"}.issubset(df.columns):
        df["cpa"] = df["spend"] / df["conversions"].clip(lower=1)

    if "conversion_rate" not in df.columns and {"conversions", "clicks"}.issubset(df.columns):
        df["conversion_rate"] = df["conversions"] / df["clicks"].clip(lower=1)

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    if "campaign_age_days" not in df.columns:
        df["campaign_age_days"] = (
            df["date"] - df.groupby("campaign_id")["date"].transform("min")
        ).dt.days

    log.info(f"Dataset chargé : {len(df):,} lignes | {df['campaign_id'].nunique()} campagnes")
    return df


def load_json(path: Path, default):
    if not path.exists():
        log.warning(f"Fichier absent : {path.name}")
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_triggered_campaigns() -> dict:
    health_data = load_json(HEALTH_PATH, {"campaigns": []})
    triggered = {}

    for item in health_data.get("campaigns", []):
        if not isinstance(item, dict):
            continue

        cid = item.get("campaign_id")

        if cid and bool(item.get("trigger_causal_ai", False)):
            triggered[cid] = {
                "health_score": item.get("health_score"),
                "health_status": item.get("status"),
                "components": item.get("components", {}),
                "global_campaign_id": item.get("global_campaign_id"),
                "platform": item.get("platform"),
            }

    return triggered


def load_tool3_signals() -> dict:
    return load_json(CORR_PATH, {})

# =============================================================================
# CROSS PLATFORM
# =============================================================================

def get_cross_platform_pairs(df: pd.DataFrame) -> set:
    if "global_campaign_id" not in df.columns or "platform" not in df.columns:
        return set()

    pivot = (
        df.groupby(["global_campaign_id", "platform"])
        .size()
        .unstack(fill_value=0)
    )

    if "meta" not in pivot.columns or "google" not in pivot.columns:
        return set()

    return set(pivot[(pivot["meta"] > 0) & (pivot["google"] > 0)].index.tolist())

# =============================================================================
# DOUBLE MACHINE LEARNING
# =============================================================================

def double_ml(df: pd.DataFrame, treatment: str, outcome: str, controls: list) -> dict:
    needed = [treatment, outcome]

    if treatment not in df.columns or outcome not in df.columns:
        return {"error": "missing_columns", "n": 0}

    ctrl_cols = [c for c in controls if c in df.columns and c not in [treatment, outcome]]

    data = df[needed + ctrl_cols].replace([np.inf, -np.inf], np.nan)

    # supprimer seulement si treatment/outcome manquent
    data = data.dropna(subset=needed)

    if len(data) < MIN_OBSERVATIONS:
        return {"error": "insufficient_data", "n": len(data)}

    # remplir les NaN des contrôles sans supprimer les lignes
    for c in ctrl_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        median = data[c].median()
        if pd.isna(median):
            median = 0.0
        data[c] = data[c].fillna(median)

    Y = data[outcome].astype(float).values
    T = data[treatment].astype(float).values

    if np.std(T) < 1e-10:
        return {"error": "no_treatment_variation", "n": len(data)}

    if ctrl_cols:
        X = data[ctrl_cols].astype(float).values
    else:
        X = np.ones((len(data), 1))

    X = StandardScaler().fit_transform(X)

    n_splits = min(DML_FOLDS, len(data))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_res = np.zeros_like(Y, dtype=float)
    t_res = np.zeros_like(T, dtype=float)

    for train_idx, test_idx in kf.split(X):
        m_y = RandomForestRegressor(
            n_estimators=DML_N_TREES,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        m_t = RandomForestRegressor(
            n_estimators=DML_N_TREES,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        m_y.fit(X[train_idx], Y[train_idx])
        m_t.fit(X[train_idx], T[train_idx])

        y_res[test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
        t_res[test_idx] = T[test_idx] - m_t.predict(X[test_idx])

    denom = float(np.sum(t_res ** 2))

    if denom < 1e-10:
        return {"error": "no_residual_treatment_variation", "n": len(data)}

    theta = float(np.sum(t_res * y_res) / denom)

    n = len(Y)
    psi = (y_res - theta * t_res) * t_res
    var_t = float(np.var(psi)) / (denom / n) ** 2 / n
    se = float(np.sqrt(max(var_t, 1e-12)))

    ci_low = theta - 1.96 * se
    ci_high = theta + 1.96 * se

    z = theta / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "effect": round(theta, 6),
        "std_error": round(se, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "p_value": round(p_value, 6),
        "confidence": round(float(np.clip(1 - p_value, 0.01, 0.99)), 4),
        "n": int(n),
        "method": "DML",
        "controls_used": ctrl_cols,
    }

# =============================================================================
# AGGREGATION — CORRIGÉE FINALE
# =============================================================================

def aggregate_campaign_daily(df: pd.DataFrame, campaign_id: str) -> pd.DataFrame:
    sub = df[df["campaign_id"] == campaign_id].copy()

    if sub.empty:
        return pd.DataFrame()

    categorical_cols = [
        "device",
        "age_range",
        "gender",
        "audience_type",
        "campaign_objective",
        "placement",
        "match_type",
        "ad_format",
        "platform",
    ]

    for col in categorical_cols:
        if col in sub.columns:
            sub[col] = sub[col].astype(str).fillna("unknown")
            codes, _ = pd.factorize(sub[col])
            sub[f"{col}_encoded"] = codes

    agg = {}

    for col in ["spend", "impressions", "clicks", "conversions", "conversion_value"]:
        if col in sub.columns:
            agg[col] = (col, "sum")

    for col in [
        "roas",
        "cpa",
        "ctr_calc",
        "cpc_calc",
        "daily_budget",
        "frequency",
        "reach",
        "quality_score",
        "conversion_rate",
    ]:
        if col in sub.columns:
            agg[col] = (col, "mean")

    for col in ["is_weekend", "month", "campaign_age_days"]:
        if col in sub.columns:
            agg[col] = (col, "max")

    for col in categorical_cols:
        encoded_col = f"{col}_encoded"
        if encoded_col in sub.columns:
            agg[encoded_col] = (encoded_col, "mean")

    if not agg:
        return pd.DataFrame()

    daily = (
        sub.groupby("date")
        .agg(**agg)
        .reset_index()
        .sort_values("date")
    )

    if "spend" in daily.columns:
        daily["spend_lag1"] = daily["spend"].shift(1)
        daily["spend_lag3"] = daily["spend"].shift(3)
        daily["spend_squared"] = daily["spend"] ** 2

    # IMPORTANT :
    # Ne pas faire dropna() global ici.
    # Sinon une seule colonne NaN supprime toutes les lignes.
    daily = daily.replace([np.inf, -np.inf], np.nan)

    return daily

# =============================================================================
# TOOL 3 SIGNAL LOOKUP
# =============================================================================

def get_tool3_signal(corr_data: dict, global_campaign_id: str) -> dict:
    if not isinstance(corr_data, dict):
        return {}

    for key in ["results", "campaigns", "correlations"]:
        items = corr_data.get(key)

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue

                if (
                    item.get("global_campaign_id") == global_campaign_id
                    or item.get("campaign_id") == global_campaign_id
                ):
                    return item

    return corr_data.get(global_campaign_id, {}) if global_campaign_id else {}

# =============================================================================
# DIAGNOSIS
# =============================================================================

def is_valid_effect(e: dict) -> bool:
    if not isinstance(e, dict) or "effect" not in e:
        return False

    effect = float(e.get("effect", 0))
    p_value = float(e.get("p_value", 1))
    ci_low = float(e.get("ci_low", 0))
    ci_high = float(e.get("ci_high", 0))

    if abs(effect) <= MIN_EFFECT_EPS:
        return False

    stable_ci = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)

    return p_value < P_VALUE_THRESHOLD and stable_ci


def add_cause(causes: list, cause_type: str, effect_dict: dict, evidence_label: str, priority: float = 1.0):
    confidence = float(effect_dict.get("confidence", 0))
    confidence = float(np.clip(confidence, 0, 1))

    score = confidence * priority
    score = float(np.clip(score, 0, 1))

    causes.append({
        "type": cause_type,
        "score": round(score, 4),
        "evidence": (
            f"{evidence_label} = {effect_dict['effect']:+.6f}, "
            f"IC95%=[{effect_dict['ci_low']:+.6f}, {effect_dict['ci_high']:+.6f}], "
            f"p={effect_dict['p_value']:.6f}"
        ),
    })


def add_tool3_cause(causes: list, cause_type: str, confidence: float, evidence: str):
    confidence = float(np.clip(confidence, 0, 1))
    score = float(np.clip(max(confidence, 0.60), 0, 1))

    causes.append({
        "type": cause_type,
        "score": round(score, 4),
        "evidence": evidence,
    })


def diagnose_root_cause(direct: dict, saturation: dict, lag: dict, cross: dict, tool3_signal: dict) -> dict:
    causes = []

    if is_valid_effect(saturation) and saturation["effect"] < 0:
        add_cause(
            causes,
            "ad_saturation",
            saturation,
            "effet quadratique spend²",
            priority=1.20,
        )

    if is_valid_effect(cross):
        if cross["effect"] < 0:
            add_cause(
                causes,
                "cannibalization",
                cross,
                "effet inter-canal DML",
                priority=1.15,
            )
        else:
            add_cause(
                causes,
                "halo_effect",
                cross,
                "effet inter-canal DML",
                priority=1.10,
            )

    if is_valid_effect(lag):
        add_cause(
            causes,
            "delayed_branding",
            lag,
            "effet spend lag 3 jours",
            priority=1.05,
        )

    if is_valid_effect(direct):
        if direct["effect"] > 0:
            add_cause(
                causes,
                "direct_budget_impact",
                direct,
                "effet spend → conversions",
                priority=1.00,
            )
        else:
            add_cause(
                causes,
                "budget_inefficiency",
                direct,
                "effet spend négatif → conversions",
                priority=1.10,
            )

    if isinstance(tool3_signal, dict):
        effect_type = str(
            tool3_signal.get("effect_type")
            or tool3_signal.get("relation_type")
            or tool3_signal.get("type")
            or tool3_signal.get("primary_effect")
            or ""
        ).lower()

        confidence = float(tool3_signal.get("confidence") or 0.0)

        if "cannibal" in effect_type or "substitution" in effect_type:
            add_tool3_cause(
                causes,
                "cannibalization",
                confidence,
                "Tool 3 indique une cannibalisation/substitution inter-canaux",
            )

        elif "halo" in effect_type or "synergy" in effect_type:
            add_tool3_cause(
                causes,
                "halo_effect",
                confidence,
                "Tool 3 indique un effet halo inter-canaux",
            )

    if not causes:
        return {
            "root_cause": "no_clear_cause",
            "confidence": 0.0,
            "evidence": "Aucun effet causal statistiquement stable détecté.",
            "all_causes": [],
        }

    causes.sort(key=lambda x: x["score"], reverse=True)
    top = causes[0]

    return {
        "root_cause": top["type"],
        "confidence": round(float(np.clip(top["score"], 0, 1)), 3),
        "evidence": top["evidence"],
        "all_causes": causes,
    }

# =============================================================================
# CAMPAIGN ANALYSIS — CORRIGÉ FINAL
# =============================================================================

def analyze_campaign(
    df: pd.DataFrame,
    campaign_id: str,
    health_info: dict,
    corr_data: dict,
    multi_pairs: set,
) -> dict:

    daily = aggregate_campaign_daily(df, campaign_id)

    required_base = [c for c in ["spend", "conversions"] if c in daily.columns]
    valid_daily = daily.dropna(subset=required_base) if required_base else pd.DataFrame()

    if len(valid_daily) < MIN_OBSERVATIONS:
        return {
            "campaign_id": campaign_id,
            "status": "insufficient_history",
            "n_observations": int(len(valid_daily)),
            "health_context": health_info,
        }

    BASE_CONTROLS = [
        "impressions",
        "clicks",
        "ctr_calc",
        "cpc_calc",
        "cpa",
        "roas",
        "daily_budget",
        "frequency",
        "reach",
        "quality_score",
        "conversion_rate",
        "is_weekend",
        "month",
        "campaign_age_days",
        "device_encoded",
        "age_range_encoded",
        "gender_encoded",
        "audience_type_encoded",
        "campaign_objective_encoded",
        "placement_encoded",
        "match_type_encoded",
        "ad_format_encoded",
    ]

    controls = [c for c in BASE_CONTROLS if c in daily.columns]

    direct = double_ml(daily, "spend", "conversions", controls)
    saturation = double_ml(daily, "spend_squared", "conversions", controls)
    lag = double_ml(daily, "spend_lag3", "conversions", controls)

    cross = {}
    gcid = health_info.get("global_campaign_id")
    plat = health_info.get("platform")

    if gcid and gcid in multi_pairs and plat:
        partner_plat = "google" if plat == "meta" else "meta"

        partner_df = (
            df[(df["global_campaign_id"] == gcid) & (df["platform"] == partner_plat)]
            .groupby("date")["spend"]
            .sum()
            .reset_index()
            .rename(columns={"spend": "partner_spend"})
        )

        if not partner_df.empty:
            merged = daily.merge(partner_df, on="date", how="inner")
            merged = merged.replace([np.inf, -np.inf], np.nan)

            required_cross = ["partner_spend", "conversions"]
            valid_cross = merged.dropna(subset=required_cross)

            if len(valid_cross) >= MIN_OBSERVATIONS:
                cross = double_ml(
                    merged,
                    treatment="partner_spend",
                    outcome="conversions",
                    controls=controls + ["spend"],
                )

    tool3_signal = get_tool3_signal(corr_data, gcid)

    diagnosis = diagnose_root_cause(
        direct=direct,
        saturation=saturation,
        lag=lag,
        cross=cross,
        tool3_signal=tool3_signal,
    )

    return {
        "campaign_id": campaign_id,
        "global_campaign_id": gcid,
        "platform": plat,
        "status": "analyzed",
        "n_observations": int(len(valid_daily)),
        "health_context": {
            "health_score": health_info.get("health_score"),
            "health_status": health_info.get("health_status"),
            "components": health_info.get("components"),
        },
        "causal_effects": {
            "direct_spend_to_conversions": direct,
            "saturation_quadratic": saturation,
            "lagged_spend_to_conversions": lag,
            "cross_channel_effect": cross,
            "tool3_signal": tool3_signal,
        },
        "diagnosis": diagnosis,
        "dml_controls_used": controls,
        "timestamp": datetime.now().isoformat(),
    }

# =============================================================================
# REPORT
# =============================================================================

EXPLAIN_TEMPLATES = {
    "cannibalization": "Le budget partenaire réduit les conversions → cannibalisation inter-canaux.",
    "halo_effect": "Le budget partenaire améliore les conversions → effet halo positif.",
    "ad_saturation": "Rendement marginal décroissant → saturation publicitaire.",
    "delayed_branding": "Effet retardé détecté → branding différé ou délai de conversion.",
    "direct_budget_impact": "Budget a un effet causal direct et positif sur les conversions.",
    "budget_inefficiency": "Budget a un effet causal négatif → dépenses inefficaces.",
    "no_clear_cause": "Aucune cause dominante détectée avec stabilité statistique suffisante.",
}


def build_explanation(result: dict) -> str:
    cid = result.get("campaign_id")
    status = result.get("status")

    if status != "analyzed":
        return f"[{cid}] Non analysée — {status} | n={result.get('n_observations')}\n"

    diag = result.get("diagnosis", {})
    health = result.get("health_context", {})
    cause = diag.get("root_cause", "unknown")
    conf = float(diag.get("confidence", 0.0))

    txt = (
        f"[{cid}] health={health.get('health_score')} ({health.get('health_status')}) | "
        f"cause={cause.upper()} (confiance={conf:.0%}) | n={result.get('n_observations')}\n"
        f"  → {EXPLAIN_TEMPLATES.get(cause, 'Cause inconnue.')}\n"
        f"  → {diag.get('evidence', '')}\n"
    )

    eff = result.get("causal_effects", {})

    for label, key in [
        ("Direct spend→conv", "direct_spend_to_conversions"),
        ("Saturation spend²", "saturation_quadratic"),
        ("Lag 3 jours", "lagged_spend_to_conversions"),
        ("Inter-canal", "cross_channel_effect"),
    ]:
        e = eff.get(key, {})

        if isinstance(e, dict) and "effect" in e:
            txt += (
                f"  → {label} : {e['effect']:+.6f} "
                f"IC95%=[{e['ci_low']:+.6f}, {e['ci_high']:+.6f}] "
                f"p={e['p_value']:.6f}\n"
            )

    return txt + "\n"

# =============================================================================
# RUN
# =============================================================================

def run():
    log.info("=" * 70)
    log.info("TOOL 4 — CAUSAL AI  (V2 corrigée finale)")
    log.info("=" * 70)

    df = load_dataset()
    corr_data = load_tool3_signals()
    triggered_campaigns = load_triggered_campaigns()
    multi_pairs = get_cross_platform_pairs(df)

    log.info(f"Campagnes déclenchées   : {len(triggered_campaigns)}")
    log.info(f"Paires multi-plateformes: {len(multi_pairs)}")

    if not triggered_campaigns:
        log.warning("Aucune campagne déclenchée — Causal AI n'analyse rien.")

        output = {
            "metadata": {
                "tool": "Tool 4 — Causal AI (V2 corrigée finale)",
                "method": "Double Machine Learning",
                "filter_source": str(HEALTH_PATH),
                "filter_rule": "trigger_causal_ai == True",
                "n_campaigns": 0,
                "generated_at": datetime.now().isoformat(),
            },
            "results": [],
        }

        with open(OUT_EFFECTS, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        with open(OUT_EXPLAIN, "w", encoding="utf-8") as f:
            f.write("Aucune campagne déclenchée par Campaign Health Score.")

        return output

    results = []
    explanations = [
        "=" * 70,
        "TOOL 4 — RAPPORT CAUSAL  (V2 corrigée finale)",
        f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    for cid, health_info in triggered_campaigns.items():
        try:
            log.info(f"  → Analyse : {cid}")
            result = analyze_campaign(df, cid, health_info, corr_data, multi_pairs)

        except Exception as e:
            log.error(f"Erreur sur {cid} : {e}")
            result = {
                "campaign_id": cid,
                "status": "error",
                "error": str(e),
                "health_context": health_info,
            }

        results.append(result)
        explanations.append(build_explanation(result))

    n_analyzed = sum(1 for r in results if r.get("status") == "analyzed")

    n_detected = sum(
        1 for r in results
        if r.get("status") == "analyzed"
        and r.get("diagnosis", {}).get("root_cause") != "no_clear_cause"
    )

    output = {
        "metadata": {
            "tool": "Tool 4 — Causal AI (V2 corrigée finale)",
            "method": "Double Machine Learning",
            "filter_source": str(HEALTH_PATH),
            "filter_rule": "trigger_causal_ai == True",
            "cross_channel_dml": True,
            "controls_enriched": True,
            "safe_nan_handling": True,
            "n_triggered_campaigns": len(triggered_campaigns),
            "n_analyzed": n_analyzed,
            "n_root_causes_detected": n_detected,
            "generated_at": datetime.now().isoformat(),
        },
        "results": results,
    }

    with open(OUT_EFFECTS, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUT_EXPLAIN, "w", encoding="utf-8") as f:
        f.write("\n".join(explanations))

    log.info("-" * 70)
    log.info(f"Campagnes déclenchées     : {len(triggered_campaigns)}")
    log.info(f"Campagnes analysées       : {n_analyzed}")
    log.info(f"Causes racines détectées  : {n_detected}")
    log.info(f"JSON sauvegardé           : {OUT_EFFECTS.name}")
    log.info(f"Rapport sauvegardé        : {OUT_EXPLAIN.name}")
    log.info("=" * 70)

    return output


if __name__ == "__main__":
    print(run())

def run_causal():
    try:
        output = run()
        return {
            "status": "success",
            "message": "Causal AI exécuté",
            "n_analyzed": output["metadata"]["n_analyzed"],
            "outputs": {
                "effects": str(OUT_EFFECTS),
                "explanations": str(OUT_EXPLAIN),
                "log": str(OUT_LOG)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }