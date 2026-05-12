"""
===============================================================================
DÉTECTEUR D'ANOMALIES — AdOptimizer AI  (Agent 3 Vigilant)  [v4 FINAL]
===============================================================================
Input  : dataset_model_ready.csv  (sortie Tool 1)
Output : outputs/
             anomaly_report.json   → Phase 4 (Causal AI / Diagnostic)
             anomaly_alerts.csv    → Phase 3 (Health Score 40%)
             anomaly_report.txt    → Agent 1 (Consultant LLM)
             anomalies.csv         → détail toutes anomalies persistantes
             anomaly_summary.csv   → résumé par campagne
             isolation_forest.pkl  → modèles IF sauvegardés
             anomaly_log.txt       → log complet

TOUTES LES CORRECTIONS APPLIQUÉES :
  [FIX 1]  Seuils temporels recalibrés : ROAS_DROP -55% | TREND_BREAK -50%
  [FIX 2]  bad_roas filtré par objectif (conversion/leads uniquement)
  [FIX 3]  IsolationForest par plateforme (Meta ≠ Google)
  [FIX 4]  Train/test strict sur IF (pas de fuite temporelle)
  [FIX 5]  Score pondéré business (business×3, tracking×2, ML×1.5, stat×1)
  [FIX 6]  Persistance temporelle >= 2 jours (filtre bruit ponctuel)
  [FIX 7]  Z-score robuste MAD (résistant aux outliers)
  [FIX 8]  Division par zéro Phase 7 corrigée (if n_all > 0)
  [FIX 9]  Normalisation min-max + epsilon (stable entre datasets)
  [FIX 10] explain_anomaly() business-friendly (lisible par Agent 1)
  [FIX 11] Seuil dynamique avec fallback fixe (max percentile90, 0.40)
  [FIX 12] if_score_raw exporté (debug/monitoring IsolationForest)
  [FIX 13] Explications triées par priorité business (BUSINESS > TRACKING > ...)
  [FIX 14] impact_score = anomaly_score × spend (priorisation financière)

Installation :
  pip install scikit-learn pandas numpy joblib

Exécution :
  python anomaly_detector_final.py
===============================================================================
"""

import os
import sys
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from datetime           import datetime, timedelta
from sklearn.ensemble   import IsolationForest
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

from pathlib import Path

BASE_DIR = Path("app")

INPUT_FILE = BASE_DIR / "data" / "dataset_model_ready.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_ANOM  = OUTPUT_DIR / "anomalies.csv"
OUTPUT_SUM   = OUTPUT_DIR / "anomaly_summary.csv"
OUTPUT_MODEL = OUTPUT_DIR / "isolation_forest.pkl"
OUTPUT_JSON  = OUTPUT_DIR / "anomaly_report.json"
OUTPUT_CSV   = OUTPUT_DIR / "anomaly_alerts.csv"
OUTPUT_TXT   = OUTPUT_DIR / "anomaly_report.txt"
OUTPUT_LOG   = OUTPUT_DIR / "anomaly_log.txt"

os.makedirs(str(OUTPUT_DIR), exist_ok=True)

ID_COLS = [
    "global_campaign_id", "campaign_id", "adset_id", "ad_id",
    "date", "platform", "start_date", "end_date",
]

# ── Seuils moteur ML ─────────────────────────────────────────────────────────
Z_SCORE_THRESHOLD    = 3.0
ROAS_DROP_THRESHOLD  = -0.55    # [FIX 1]
SPEND_SPIKE_FACTOR   = 2.5
TREND_BREAK_THRESH   = -0.50    # [FIX 1]
CTR_DELTA_THRESHOLD  = 0.015
ZERO_CONV_CLICKS_MIN = 50
BAD_ROAS_SPEND_MIN   = 100.0
BAD_ROAS_THRESHOLD   = 0.5
CONVERSION_OBJECTIVES = ["conversion", "leads"]  # [FIX 2]

# IsolationForest [FIX 3 + FIX 4]
IF_CONTAMINATION = "auto"
IF_TRAIN_RATIO   = 0.80

# Score pondéré [FIX 5]
WEIGHTS = {
    "business" : 3.0,
    "tracking" : 2.0,
    "ml"       : 1.5,
    "stat"     : 1.0,
    "temporal" : 1.0,
}

# Persistance [FIX 6]
PERSISTENCE_DAYS     = 2
ANOMALY_SCORE_THRESH = 0.40    # fallback fixe [FIX 11]

# [FIX 13] Ordre de priorité pour le tri des explications
EXPLAIN_PRIORITY = ["[BUSINESS]", "[TRACKING]", "[TEMPOREL]", "[STAT]", "[ML]"]

# Features IsolationForest
IF_FEATURES_COMMON = [
    "spend", "clicks", "impressions", "conversions",
    "roas", "cpa", "ctr_calc", "cpc_calc", "conv_rate",
    "spend_lag1", "roas_lag1", "conversions_lag1",
    "spend_roll7m", "roas_roll7m", "conversions_roll7m",
    "roas_trend7", "spend_trend7", "ctr_calc_trend7",
    "budget_utilization", "campaign_age_days",
]
IF_FEATURES_META   = ["reach", "frequency", "likes",
                       "post_engagement", "add_to_cart", "purchases"]
IF_FEATURES_GOOGLE = ["quality_score"]

# ── Seuils orchestrateur ─────────────────────────────────────────────────────
ACTIVE_DAYS         = 7
MIN_HISTORY_DAYS    = 14
HEALTH_SCORE_WEIGHT = 0.40

LEVEL_OK       = "OK"
LEVEL_INFO     = "INFO"
LEVEL_WARNING  = "WARNING"
LEVEL_CRITICAL = "CRITICAL"

SCORE_INFO     = 0.20
SCORE_WARNING  = 0.40
SCORE_CRITICAL = 0.65
PHASE4_TRIGGER = 0.65


# ============================================================
# UTILITAIRES
# ============================================================

def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()

def sep(char="=", n=72):
    return char * n

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_col(df, col):
    return col in df.columns and df[col].notna().any()

def has_mask(df, col):
    mask_col = f"has_{col}"
    if mask_col in df.columns:
        return df[mask_col] == 1
    return pd.Series(True, index=df.index)

def score_to_level(score):
    if score >= SCORE_CRITICAL: return LEVEL_CRITICAL
    if score >= SCORE_WARNING:  return LEVEL_WARNING
    if score >= SCORE_INFO:     return LEVEL_INFO
    return LEVEL_OK


# ============================================================
# PARTIE 1 — MOTEUR ML (8 phases)
# ============================================================


def phase1_load(input_path, f):
    log("\n" + sep(), f)
    log("PHASE 1 — CHARGEMENT & MASQUES NA_PLATFORM", f)
    log(sep(), f)

    if not os.path.exists(input_path):
        log(f"ERREUR : Fichier introuvable : {input_path}", f)
        sys.exit(1)

    df = pd.read_csv(input_path, parse_dates=["date"])
    log(f"  Lignes chargées  : {len(df):,}", f)
    log(f"  Colonnes totales : {len(df.columns)}", f)
    log(f"  Période          : {df['date'].min().date()} → "
        f"{df['date'].max().date()}", f)
    log(f"  Plateformes      : {df['platform'].value_counts().to_dict()}", f)
    log(f"  Campagnes        : {df['campaign_id'].nunique()}", f)

    obj_col = next((c for c in ["campaign_objective", "campaign_objective_enc"]
                    if c in df.columns), None)
    if obj_col:
        log(f"  Colonne objectif : '{obj_col}' → filtrage bad_roas activé [FIX 2]", f)
    else:
        log("  Colonne objectif : absente → filtrage bad_roas désactivé", f)

    has_cols = [c for c in df.columns if c.startswith("has_")]
    log(f"  Masques NA_platform : {len(has_cols)}", f)

    # Initialisation colonnes anomalies
    for col in ["anom_stat", "anom_temporal", "anom_business",
                "anom_tracking", "z_score_flag", "roas_drop",
                "spend_spike", "trend_break", "zero_conv_high_clicks",
                "bad_roas_high_spend", "ctr_delta_flag", "if_flag"]:
        df[col] = 0

    df["anom_ml"]       = 0.0
    df["if_score_raw"]  = 0.0   # [FIX 12] score brut IF pour debug/monitoring
    df["anomaly_types"] = ""
    df["explanation"]   = ""

    df = df.sort_values(
        ["campaign_id", "platform", "date"]).reset_index(drop=True)
    return df, obj_col


def phase2_statistical(df, f):
    log("\n" + sep(), f)
    log("PHASE 2 — ANOMALIES STATISTIQUES (Z-SCORE ROBUSTE MAD) [FIX 7]", f)
    log(sep(), f)

    stat_metrics = ["spend", "clicks", "impressions", "conversions",
                    "roas", "cpa", "ctr_calc", "cpc_calc",
                    "conversion_value_lag1"]
    total_flags = 0

    def robust_zscore(x):
        """Z-score robuste via MAD — résistant aux outliers [FIX 7]."""
        med = x.median()
        mad = (x - med).abs().median()
        if mad > 0:
            return (x - med) / (1.4826 * mad)
        iqr = x.quantile(0.75) - x.quantile(0.25)
        if iqr > 0:
            return (x - med) / (0.7413 * iqr)
        return pd.Series(0.0, index=x.index)

    for metric in stat_metrics:
        if metric not in df.columns:
            continue
        df[f"z_{metric}"] = df.groupby(
            ["campaign_id", "platform"])[metric].transform(robust_zscore)
        plat_mask = has_mask(df, metric)
        anom_mask = plat_mask & (df[f"z_{metric}"].abs() > Z_SCORE_THRESHOLD)
        n = anom_mask.sum()
        total_flags += n
        if n > 0:
            df.loc[anom_mask, "anom_stat"]    = 1
            df.loc[anom_mask, "z_score_flag"] = 1
            log(f"  {metric:<30} : {n:,} anomalies (|z| > {Z_SCORE_THRESHOLD})", f)

    log(f"\n  Total flags statistiques : {total_flags:,}", f)
    return df


def phase3_temporal(df, f):
    log("\n" + sep(), f)
    log("PHASE 3 — ANOMALIES TEMPORELLES [FIX 1 : seuils recalibrés]", f)
    log(sep(), f)
    log(f"  ROAS_DROP   : {ROAS_DROP_THRESHOLD:.0%}  (était -40%)", f)
    log(f"  TREND_BREAK : {TREND_BREAK_THRESH:.0%}  (était -35%)\n", f)

    n_trend = 0

    # roas_drop
    if safe_col(df, "roas") and safe_col(df, "roas_roll7m"):
        valid    = df["roas_roll7m"].notna() & (df["roas_roll7m"] > 0)
        roas_chg = (df["roas"] - df["roas_roll7m"]) / df["roas_roll7m"].abs()
        mask     = valid & (roas_chg < ROAS_DROP_THRESHOLD)
        df.loc[mask, "roas_drop"]     = 1
        df.loc[mask, "anom_temporal"] = 1
        log(f"  roas_drop   : {mask.sum():,}", f)

    # spend_spike
    if safe_col(df, "spend") and safe_col(df, "spend_roll7m"):
        valid = df["spend_roll7m"].notna() & (df["spend_roll7m"] > 0)
        mask  = valid & (df["spend"] > SPEND_SPIKE_FACTOR * df["spend_roll7m"])
        df.loc[mask, "spend_spike"]   = 1
        df.loc[mask, "anom_temporal"] = 1
        log(f"  spend_spike : {mask.sum():,}", f)

    # trend_break
    for trend_col in ["roas_trend7", "conversions_trend7"]:
        if safe_col(df, trend_col):
            valid = df[trend_col].notna()
            mask  = valid & (df[trend_col] < TREND_BREAK_THRESH)
            df.loc[mask, "trend_break"]   = 1
            df.loc[mask, "anom_temporal"] = 1
            n_trend += mask.sum()
    log(f"  trend_break : {n_trend:,}", f)

    total = (df["anom_temporal"] == 1).sum()
    log(f"\n  Total temporelles : {total:,} ({total/len(df)*100:.1f}%)", f)
    return df


def phase4_business_tracking(df, obj_col, f):
    log("\n" + sep(), f)
    log("PHASE 4 — ANOMALIES BUSINESS + TRACKING [FIX 2]", f)
    log(sep(), f)

    # zero_conv_high_clicks
    if safe_col(df, "clicks") and safe_col(df, "conversions"):
        mask = (df["clicks"].fillna(0) > ZERO_CONV_CLICKS_MIN) & \
               (df["conversions"].fillna(-1) == 0)
        df.loc[mask, "zero_conv_high_clicks"] = 1
        df.loc[mask, "anom_business"]         = 1
        log(f"  zero_conv_high_clicks : {mask.sum():,}", f)

    # bad_roas_high_spend [FIX 2]
    if safe_col(df, "roas") and safe_col(df, "spend"):
        base_mask = (df["spend"].fillna(0) > BAD_ROAS_SPEND_MIN) & \
                    (df["roas"].fillna(99) < BAD_ROAS_THRESHOLD)

        if obj_col and obj_col in df.columns:
            obj_raw = "campaign_objective_raw"
            if obj_raw in df.columns:
                conv_mask = df[obj_raw].isin(CONVERSION_OBJECTIVES)
            else:
                roas_by_obj = df.groupby(obj_col)["roas"].median()
                conv_codes  = roas_by_obj[roas_by_obj > 0.3].index.tolist()
                conv_mask   = df[obj_col].isin(conv_codes)
            final_mask = base_mask & conv_mask
            n_exclu    = base_mask.sum() - final_mask.sum()
            log(f"  bad_roas_high_spend   : {final_mask.sum():,} "
                f"({n_exclu:,} awareness/traffic exclus [FIX 2])", f)
        else:
            final_mask = base_mask
            log(f"  bad_roas_high_spend   : {final_mask.sum():,} "
                f"(objectif absent — filtrage désactivé)", f)

        df.loc[final_mask, "bad_roas_high_spend"] = 1
        df.loc[final_mask, "anom_business"]       = 1

    # ctr_delta (tracking)
    if safe_col(df, "ctr_delta"):
        mask = df["ctr_delta"].fillna(0) > CTR_DELTA_THRESHOLD
    elif safe_col(df, "CTR") and safe_col(df, "ctr_calc"):
        df["ctr_delta"] = (df["CTR"] - df["ctr_calc"]).abs()
        mask = df["ctr_delta"] > CTR_DELTA_THRESHOLD
    else:
        mask = pd.Series(False, index=df.index)

    df.loc[mask, "ctr_delta_flag"] = 1
    df.loc[mask, "anom_tracking"]  = 1
    log(f"  ctr_delta_flag        : {mask.sum():,}", f)

    total_biz = (df["anom_business"] == 1).sum()
    log(f"\n  Total business : {total_biz:,} ({total_biz/len(df)*100:.1f}%)", f)
    return df


def phase5_isolation_forest(df, f):
    """
    [FIX 3] Modèle séparé par plateforme (Meta ≠ Google).
    [FIX 4] Entraîné sur données historiques uniquement (anti-fuite).
    [FIX 12] if_score_raw exporté pour debug et monitoring du modèle.
    """
    log("\n" + sep(), f)
    log("PHASE 5 — ISOLATION FOREST PAR PLATEFORME [FIX 3+4+12]", f)
    log(sep(), f)

    platforms  = df["platform"].unique()
    if_models  = {}
    if_scalers = {}

    for plat in platforms:
        log(f"\n  Plateforme : {plat.upper()}", f)
        plat_mask = df["platform"] == plat
        df_plat   = df[plat_mask].copy()

        # Sélection features par plateforme [FIX 3]
        feats = [c for c in IF_FEATURES_COMMON if c in df_plat.columns]
        extra = IF_FEATURES_META if plat == "meta" else IF_FEATURES_GOOGLE
        feats += [c for c in extra if c in df_plat.columns
                  and df_plat.get(f"has_{c}", pd.Series(1)).eq(1).any()]
        feats = list(dict.fromkeys(feats))

        # Split temporel [FIX 4]
        cutoff          = pd.Timestamp(
            df_plat["date"].quantile(IF_TRAIN_RATIO)).normalize()
        train_mask_plat = df_plat["date"] <= cutoff

        X_plat    = df_plat[feats].copy()
        X_plat    = X_plat.fillna(X_plat[train_mask_plat].median())
        X_train_p = X_plat[train_mask_plat]

        # RobustScaler fitté sur train uniquement
        scaler     = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train_p)
        X_all_sc   = scaler.transform(X_plat)

        iso = IsolationForest(
            n_estimators  = 200,
            max_samples   = "auto",
            contamination = IF_CONTAMINATION,
            random_state  = 42,
            n_jobs        = -1,
        )
        iso.fit(X_train_sc)

        scores_raw  = iso.decision_function(X_all_sc)
        predictions = iso.predict(X_all_sc)

        # Normalisation [0, 1]
        s_min, s_max = scores_raw.min(), scores_raw.max()
        scores_norm  = (1 - (scores_raw - s_min) / (s_max - s_min)
                        if s_max > s_min else np.zeros(len(scores_raw)))

        plat_idx = df.index[plat_mask]
        df.loc[plat_idx, "anom_ml"]      = scores_norm
        df.loc[plat_idx, "if_flag"]      = (predictions == -1).astype(int)
        df.loc[plat_idx, "if_score_raw"] = scores_raw  # [FIX 12]

        n_anom = (predictions == -1).sum()
        log(f"    Features    : {len(feats)}", f)
        log(f"    Train       : {train_mask_plat.sum():,} (≤ {cutoff.date()})", f)
        log(f"    Anomalies IF: {n_anom:,} ({n_anom/len(predictions)*100:.1f}%)", f)
        log(f"    Score brut  : min={scores_raw.min():.3f} | "
            f"max={scores_raw.max():.3f} | "
            f"mean={scores_raw.mean():.3f}  [FIX 12]", f)

        if_models[plat]  = iso
        if_scalers[plat] = scaler

    joblib.dump({"models": if_models, "scalers": if_scalers,
                 "features": IF_FEATURES_COMMON}, OUTPUT_MODEL, compress=3)
    log(f"\n  Modèles IF → {OUTPUT_MODEL}", f)
    return df, if_models


def phase6_weighted_score(df, f):
    """
    [FIX 9]  Normalisation min-max + epsilon (stable entre datasets).
    [FIX 11] Seuil dynamique avec fallback fixe :
             threshold = max(percentile 90, ANOMALY_SCORE_THRESH)
             → dynamique mais jamais inférieur à 0.40
             → évite que sur un dataset propre, threshold tombe trop bas.
    """
    log("\n" + sep(), f)
    log("PHASE 6 — SCORE PONDÉRÉ [FIX 9 + FIX 11]", f)
    log(sep(), f)

    w = WEIGHTS
    log(f"  business×{w['business']} | tracking×{w['tracking']} | "
        f"ml×{w['ml']} | stat×{w['stat']} | temporal×{w['temporal']}", f)

    df["anomaly_score_raw"] = (
        df["anom_stat"].fillna(0)     * w["stat"]     +
        df["anom_temporal"].fillna(0) * w["temporal"] +
        df["anom_business"].fillna(0) * w["business"] +
        df["anom_tracking"].fillna(0) * w["tracking"] +
        df["anom_ml"].fillna(0)       * w["ml"]
    )

    # [FIX 9] Min-max + epsilon → stable entre datasets
    s_min = df["anomaly_score_raw"].min()
    s_max = df["anomaly_score_raw"].max()
    df["anomaly_score"] = (
        (df["anomaly_score_raw"] - s_min) / (s_max - s_min + 1e-9)
    )

    # [FIX 11] Seuil dynamique avec fallback fixe
    p90       = df["anomaly_score"].quantile(0.90)
    threshold = max(p90, ANOMALY_SCORE_THRESH)
    df["anomaly_threshold_used"] = round(threshold, 4)
    df["is_anomaly"] = (df["anomaly_score"] > threshold).astype(int)

    log(f"  Percentile 90          : {p90:.4f}", f)
    log(f"  Fallback fixe          : {ANOMALY_SCORE_THRESH}", f)
    log(f"  Seuil retenu [FIX 11]  : {threshold:.4f} "
        f"({'p90' if p90 > ANOMALY_SCORE_THRESH else 'fallback fixe'})", f)

    n   = df["is_anomaly"].sum()
    pct = n / len(df) * 100
    log(f"\n  Anomalies      : {n:,} ({pct:.2f}%)", f)
    log(f"  Score moyen    : {df['anomaly_score'].mean():.4f}", f)
    log(f"  Score min/max  : {df['anomaly_score'].min():.4f} / "
        f"{df['anomaly_score'].max():.4f}", f)

    for plat in df["platform"].unique():
        sub = df[df["platform"] == plat]
        na  = sub["is_anomaly"].sum()
        log(f"    {plat:<10} : {na:,} ({na/len(sub)*100:.2f}%)", f)

    return df


def phase7_persistence(df, f):
    """
    [FIX 6]  Persistance >= 2 jours.
    [FIX 8]  Division par zéro corrigée.
    """
    log("\n" + sep(), f)
    log(f"PHASE 7 — PERSISTANCE (>= {PERSISTENCE_DAYS}j) [FIX 6 + FIX 8]", f)
    log(sep(), f)

    df = df.sort_values(
        ["campaign_id", "platform", "date"]).reset_index(drop=True)

    df["rolling_anom"] = df.groupby(
        ["campaign_id", "platform"])["is_anomaly"].transform(
        lambda x: x.rolling(
            window=PERSISTENCE_DAYS, min_periods=PERSISTENCE_DAYS).sum()
    )
    df["is_persistent_anomaly"] = (
        df["rolling_anom"] >= PERSISTENCE_DAYS).astype(int)

    n_all  = df["is_anomaly"].sum()
    n_pers = df["is_persistent_anomaly"].sum()

    # [FIX 8] Division par zéro évitée
    pct_filtered = ((n_all - n_pers) / n_all * 100) if n_all > 0 else 0.0

    log(f"  Anomalies brutes       : {n_all:,}", f)
    log(f"  Anomalies persistantes : {n_pers:,} (>= {PERSISTENCE_DAYS}j)", f)
    log(f"  Éliminées (bruit)      : {n_all - n_pers:,} "
        f"({pct_filtered:.1f}% filtrés)", f)
    return df


def explain_anomaly(row) -> str:
    """
    [FIX 10] Explications business-friendly avec actions concrètes.
    [FIX 13] Triées par priorité : BUSINESS > TRACKING > TEMPOREL > STAT > ML
             → Agent 1 voit les problèmes les plus critiques en premier.
    """
    reasons = []

    # ── Statistiques ──────────────────────────────────────────────────────
    if row.get("z_score_flag", 0) == 1:
        z_cols = [c for c in row.index
                  if c.startswith("z_")
                  and abs(row.get(c, 0)) > Z_SCORE_THRESHOLD]
        for zc in z_cols[:2]:
            metric = zc.replace("z_", "")
            try:
                z_val     = float(row.get(zc, 0))
                direction = "anormalement élevé" if z_val > 0 else "anormalement bas"
                reasons.append(
                    f"[STAT] {metric} {direction} "
                    f"(Z={z_val:.1f}) → Vérifier cohérence des données"
                )
            except (TypeError, ValueError):
                pass

    # ── Temporelles ──────────────────────────────────────────────────────
    if row.get("roas_drop", 0) == 1:
        try:
            roas     = float(row.get("roas", 0))
            roas_ref = float(row.get("roas_roll7m", 0))
            pct      = (roas - roas_ref) / roas_ref * 100 if roas_ref > 0 else 0
            reasons.append(
                f"[TEMPOREL] ROAS en chute brutale : {roas:.2f}x "
                f"(était {roas_ref:.2f}x, baisse de {abs(pct):.0f}%) "
                f"→ Campagne non rentable — revoir le ciblage ou suspendre"
            )
        except (TypeError, ValueError):
            pass

    if row.get("spend_spike", 0) == 1:
        try:
            spend     = float(row.get("spend", 0))
            spend_ref = float(row.get("spend_roll7m", 0))
            reasons.append(
                f"[TEMPOREL] Dépense anormalement élevée : {spend:.0f} "
                f"vs moyenne {spend_ref:.0f} (×{SPEND_SPIKE_FACTOR}) "
                f"→ Vérifier les enchères ou cap budgétaire"
            )
        except (TypeError, ValueError):
            pass

    if row.get("trend_break", 0) == 1:
        try:
            t_roas = float(row.get("roas_trend7", 0))
            reasons.append(
                f"[TEMPOREL] Tendance ROAS en rupture : {t_roas*100:.0f}% "
                f"sur 7j → Performance en déclin continu, action urgente"
            )
        except (TypeError, ValueError):
            pass

    # ── Business ─────────────────────────────────────────────────────────
    if row.get("zero_conv_high_clicks", 0) == 1:
        try:
            clicks = float(row.get("clicks", 0))
            reasons.append(
                f"[BUSINESS] {clicks:.0f} clics sans aucune conversion "
                f"→ Pixel de tracking cassé ou page de destination défaillante"
            )
        except (TypeError, ValueError):
            pass

    if row.get("bad_roas_high_spend", 0) == 1:
        try:
            roas  = float(row.get("roas", 0))
            spend = float(row.get("spend", 0))
            reasons.append(
                f"[BUSINESS] Campagne non rentable : ROAS={roas:.2f}x "
                f"pour {spend:.0f}€ dépensés "
                f"→ Réduire le budget ou revoir le ciblage immédiatement"
            )
        except (TypeError, ValueError):
            pass

    # ── Tracking ─────────────────────────────────────────────────────────
    if row.get("ctr_delta_flag", 0) == 1:
        try:
            ctr_api  = float(row.get("CTR", 0))
            ctr_calc = float(row.get("ctr_calc", 0))
            reasons.append(
                f"[TRACKING] CTR incohérent : "
                f"API={ctr_api*100:.2f}% vs calculé={ctr_calc*100:.2f}% "
                f"→ Problème de tracking — vérifier le pixel"
            )
        except (TypeError, ValueError):
            pass

    # ── ML ───────────────────────────────────────────────────────────────
    if row.get("if_flag", 0) == 1:
        try:
            score     = float(row.get("anom_ml", 0))
            score_raw = float(row.get("if_score_raw", 0))
            reasons.append(
                f"[ML] Comportement global inhabituel "
                f"(score normalisé={score:.2f} | score brut IF={score_raw:.3f}) "
                f"→ Combinaison de métriques jamais observée — analyse approfondie"
            )
        except (TypeError, ValueError):
            pass

    # [FIX 13] Trier par priorité business avant de retourner
    reasons_sorted = sorted(
        reasons,
        key=lambda r: next(
            (i for i, p in enumerate(EXPLAIN_PRIORITY) if p in r), 99
        )
    )
    return (" | ".join(reasons_sorted)
            if reasons_sorted else "Anomalie détectée (score agrégé élevé)")


def _build_anomaly_types(row) -> str:
    types = []
    if row.get("anom_stat", 0):     types.append("statistique")
    if row.get("anom_temporal", 0): types.append("temporelle")
    if row.get("anom_business", 0): types.append("business")
    if row.get("anom_tracking", 0): types.append("tracking")
    if row.get("if_flag", 0):       types.append("ml")
    return ", ".join(types) if types else "aucun"


def phase8_export_ml(df, f):
    log("\n" + sep(), f)
    log("PHASE 8 — EXPORT MOTEUR ML", f)
    log(sep(), f)

    df["anomaly_types"] = df.apply(_build_anomaly_types, axis=1)
    anom_mask = df["is_persistent_anomaly"] == 1
    df.loc[anom_mask, "explanation"] = df[anom_mask].apply(
        explain_anomaly, axis=1)
    df.loc[~anom_mask, "explanation"] = ""

    export_cols = (
        ID_COLS
        + ["anomaly_score", "anomaly_threshold_used",
           "is_anomaly", "is_persistent_anomaly",
           "anomaly_types", "explanation"]
        + ["anom_stat", "anom_temporal", "anom_business",
           "anom_tracking", "anom_ml", "if_flag", "if_score_raw"]
        + ["z_score_flag", "roas_drop", "spend_spike", "trend_break",
           "zero_conv_high_clicks", "bad_roas_high_spend", "ctr_delta_flag"]
        + ["spend", "clicks", "conversions", "roas", "cpa",
           "ctr_calc", "cpc_calc"]
    )
    export_cols = list(dict.fromkeys(
        [c for c in export_cols if c in df.columns]))

    df_anom = df[anom_mask][export_cols].sort_values(
        "anomaly_score", ascending=False)
    df_anom.to_csv(OUTPUT_ANOM, index=False)

    summary = df[anom_mask].groupby(
        ["campaign_id", "platform"]).agg(
        n_anomaly_days=("is_persistent_anomaly", "sum"),
        score_moyen   =("anomaly_score", "mean"),
        score_max     =("anomaly_score", "max"),
        premier_jour  =("date", "min"),
        dernier_jour  =("date", "max"),
        n_business    =("anom_business", "sum"),
        n_tracking    =("anom_tracking", "sum"),
        n_stat        =("anom_stat", "sum"),
        n_ml          =("if_flag", "sum"),
        roas_moyen    =("roas", "mean"),
        spend_total   =("spend", "sum"),
    ).reset_index().sort_values("score_max", ascending=False)
    summary.to_csv(OUTPUT_SUM, index=False)

    log(f"  anomalies.csv   : {len(df_anom):,} lignes", f)
    log(f"  summary.csv     : {len(summary):,} campagnes", f)

    log(f"\n  TOP 10 ANOMALIES :", f)
    log(f"  {'Campaign':<20} {'Plat':<8} {'Date':<12} {'Score':>7}", f)
    log(f"  {'-'*55}", f)
    for _, row in df_anom.head(10).iterrows():
        log(f"  {str(row.get('campaign_id','')):<20} "
            f"{str(row.get('platform','')):<8} "
            f"{str(row.get('date',''))[:10]:<12} "
            f"{row.get('anomaly_score',0):7.4f}", f)

    log(f"\n  RÉPARTITION (anomalies persistantes) :", f)
    type_counts = {
        "Business"    : int(df["anom_business"][anom_mask].sum()),
        "Temporelle"  : int(df["anom_temporal"][anom_mask].sum()),
        "ML (IF)"     : int(df["if_flag"][anom_mask].sum()),
        "Statistique" : int(df["anom_stat"][anom_mask].sum()),
        "Tracking"    : int(df["anom_tracking"][anom_mask].sum()),
    }
    total_f = sum(type_counts.values())
    for typ, cnt in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = cnt / total_f * 100 if total_f > 0 else 0
        bar = "█" * int(pct / 2)
        log(f"    {typ:<25} : {cnt:5,} ({pct:5.1f}%)  {bar}", f)

    return df, df_anom


# ============================================================
# PARTIE 2 — ORCHESTRATEUR (3 phases)
# ============================================================


def orch1_filter_active(df, analysis_date, f):
    log("\n" + sep(), f)
    log("ORCH 1 — FILTRAGE CAMPAGNES ACTIVES (Phase 0)", f)
    log(sep(), f)

    recent_cutoff = analysis_date - timedelta(days=ACTIVE_DAYS)
    active = (df[df["date"] >= recent_cutoff][["campaign_id", "platform"]]
              .drop_duplicates())
    hist_counts = (df.groupby(["campaign_id", "platform"])["date"]
                   .count().reset_index().rename(columns={"date": "n_days"}))
    analyzable = (active.merge(hist_counts, on=["campaign_id", "platform"])
                  .query(f"n_days >= {MIN_HISTORY_DAYS}"))

    profiles = {"Débutant (0-14j)": 0,
                "Intermédiaire (15-89j)": 0,
                "Confirmé (90+j)": 0}
    for _, row in analyzable.iterrows():
        nd = row["n_days"]
        if nd < 15:   profiles["Débutant (0-14j)"] += 1
        elif nd < 90: profiles["Intermédiaire (15-89j)"] += 1
        else:         profiles["Confirmé (90+j)"] += 1

    log(f"  Campagnes actives     : {len(active)}", f)
    log(f"  Avec >= {MIN_HISTORY_DAYS}j historique : {len(analyzable)}", f)
    log(f"  Profils Phase 0       : {profiles}", f)
    return analyzable, profiles


def orch2_health_score(df, analyzable, f):
    """
    [FIX 14] impact_score = anomaly_score × spend_last
    → priorise les anomalies avec fort impact financier réel.
    Une campagne à score 0.85 avec 50€ dépensés est moins urgente
    qu'une à 0.70 avec 5000€.
    """
    log("\n" + sep(), f)
    log("ORCH 2 — CAMPAIGN HEALTH SCORE (contribution 40%) [FIX 14]", f)
    log(sep(), f)

    rows = []
    for _, row in analyzable.iterrows():
        camp_id, platform = row["campaign_id"], row["platform"]
        camp_df = df[(df["campaign_id"] == camp_id) &
                     (df["platform"] == platform)].sort_values("date")
        if camp_df.empty:
            continue

        last         = camp_df.iloc[-1]
        score        = float(last.get("anomaly_score", 0.0))
        is_pers      = int(last.get("is_persistent_anomaly", 0))
        level        = score_to_level(score) if is_pers else LEVEL_OK
        contribution = max(0.0, 100.0 - score * 100)
        trigger_p4   = score >= PHASE4_TRIGGER and is_pers == 1
        spend_last   = float(last.get("spend", 0.0))

        # [FIX 14] Impact financier réel
        impact_score = round(score * spend_last, 2)

        anom_types = []
        for t, col in [("statistique", "anom_stat"),
                       ("temporelle",  "anom_temporal"),
                       ("business",    "anom_business"),
                       ("tracking",    "anom_tracking"),
                       ("ml",          "if_flag")]:
            if last.get(col, 0):
                anom_types.append(t)

        rows.append({
            "campaign_id"              : camp_id,
            "platform"                 : platform,
            "analysis_date"            : str(camp_df["date"].max().date()),
            "global_level"             : level,
            "anomaly_score"            : round(score, 4),
            "is_persistent_anomaly"    : is_pers,
            "tool2_health_contribution": round(contribution, 2),
            "impact_score"             : impact_score,   # [FIX 14]
            "anomaly_types"            : "|".join(anom_types),
            "trigger_phase4"           : trigger_p4,
            "n_anomaly_days"           : int(camp_df["is_persistent_anomaly"].sum()),
            "roas_last"                : round(float(last.get("roas", 0)), 4),
            "spend_last"               : round(spend_last, 2),
            "explanation"              : str(last.get("explanation", "")),
            "timestamp"                : now_str(),
        })

    health_df = pd.DataFrame(rows)
    if not health_df.empty:
        n_crit = (health_df["global_level"] == LEVEL_CRITICAL).sum()
        n_warn = (health_df["global_level"] == LEVEL_WARNING).sum()
        n_p4   = health_df["trigger_phase4"].sum()
        log(f"  Campagnes scorées     : {len(health_df)}", f)
        log(f"  CRITICAL              : {n_crit}", f)
        log(f"  WARNING               : {n_warn}", f)
        log(f"  Phase 4 triggers      : {n_p4}", f)
        log(f"  Contribution moyenne  : "
            f"{health_df['tool2_health_contribution'].mean():.1f}/100", f)

        # Top 3 par impact financier [FIX 14]
        worst = health_df[
            health_df["global_level"] != LEVEL_OK
        ].nlargest(3, "impact_score")
        if not worst.empty:
            log(f"\n  Top 3 par IMPACT FINANCIER (score × spend) [FIX 14] :", f)
            for _, r in worst.iterrows():
                log(f"    {r['campaign_id']:<22} | {r['platform']:<8} | "
                    f"score={r['anomaly_score']:.3f} | "
                    f"spend={r['spend_last']:.0f}€ | "
                    f"impact={r['impact_score']:.2f}", f)

    return health_df


def orch3_generate_outputs(df, df_anom, health_df, profiles, f):
    log("\n" + sep(), f)
    log("ORCH 3 — GÉNÉRATION 3 SORTIES MULTI-AGENTS", f)
    log(sep(), f)

    # ── 1. JSON → Phase 4 ────────────────────────────────────────────────
    campaigns_json = {}
    for _, row in health_df.iterrows():
        camp_id, platform = row["campaign_id"], row["platform"]
        camp_key = f"{camp_id}|{platform}"
        camp_df  = df[(df["campaign_id"] == camp_id) &
                      (df["platform"] == platform)].sort_values("date")
        anom_rows = camp_df[camp_df["is_persistent_anomaly"] == 1]

        top_anomalies = []
        for _, ar in anom_rows.tail(5).iterrows():
            top_anomalies.append({
                "date"        : str(ar["date"].date()) if hasattr(
                    ar["date"], "date") else str(ar["date"]),
                "score"       : round(float(ar.get("anomaly_score", 0)), 4),
                "explanation" : explain_anomaly(ar),
                "roas"        : round(float(ar.get("roas", 0)), 4),
                "cpa"         : round(float(ar.get("cpa", 0)), 4),
                "spend"       : round(float(ar.get("spend", 0)), 2),
                "if_score_raw": round(float(ar.get("if_score_raw", 0)), 4),
                "flags"       : {
                    "stat"     : int(ar.get("anom_stat", 0)),
                    "temporal" : int(ar.get("anom_temporal", 0)),
                    "business" : int(ar.get("anom_business", 0)),
                    "tracking" : int(ar.get("anom_tracking", 0)),
                    "ml"       : int(ar.get("if_flag", 0)),
                },
            })

        campaigns_json[camp_key] = {
            "campaign_id"              : camp_id,
            "platform"                 : platform,
            "global_level"             : row["global_level"],
            "anomaly_score"            : row["anomaly_score"],
            "impact_score"             : row["impact_score"],
            "tool2_health_contribution": row["tool2_health_contribution"],
            "anomaly_types"            : row["anomaly_types"].split("|")
                                         if row["anomaly_types"] else [],
            "trigger_phase4"           : bool(row["trigger_phase4"]),
            "n_anomaly_days"           : row["n_anomaly_days"],
            "top_anomalies"            : top_anomalies,
        }

    s = health_df.groupby("global_level").size().to_dict() \
        if not health_df.empty else {}

    report_json = {
        "tool"          : "Détecteur Anomalies — AdOptimizer AI (Agent 3 Vigilant)",
        "version"       : "v4-final",
        "analysis_date" : now_str(),
        "n_campaigns"   : len(health_df),
        "profiles"      : profiles,
        "health_score"  : {
            "weight"     : HEALTH_SCORE_WEIGHT,
            "description": "40% Health Score Phase 3",
        },
        "fixes_applied" : [
            "FIX1: seuils temporels -55%/-50%",
            "FIX2: bad_roas filtré par objectif",
            "FIX3: IF par plateforme",
            "FIX4: split temporel IF",
            "FIX5: pondération business",
            "FIX6: persistance 2j",
            "FIX7: Z-score MAD",
            "FIX8: no div/0",
            "FIX9: min-max+epsilon",
            "FIX10: explain business-friendly",
            "FIX11: seuil dynamique+fallback",
            "FIX12: if_score_raw debug",
            "FIX13: tri explications priorité",
            "FIX14: impact_score financier",
        ],
        "summary"    : {
            "critical"       : int(s.get(LEVEL_CRITICAL, 0)),
            "warning"        : int(s.get(LEVEL_WARNING,  0)),
            "info"           : int(s.get(LEVEL_INFO,     0)),
            "ok"             : int(s.get(LEVEL_OK,       0)),
            "phase4_triggers": int(health_df["trigger_phase4"].sum())
                               if not health_df.empty else 0,
            "avg_score"      : round(float(
                health_df["anomaly_score"].mean()), 4)
                               if not health_df.empty else 0,
            "avg_impact"     : round(float(
                health_df["impact_score"].mean()), 2)
                               if not health_df.empty else 0,
        },
        "campaigns": campaigns_json,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fj:
        json.dump(report_json, fj, indent=2,
                  ensure_ascii=False, default=str)
    log(f"  JSON → Phase 4 : {OUTPUT_JSON}", f)

    # ── 2. CSV → Phase 3 ─────────────────────────────────────────────────
    if not health_df.empty:
        health_df.to_csv(OUTPUT_CSV, index=False)
    log(f"  CSV → Phase 3  : {OUTPUT_CSV}", f)

    # ── 3. TXT → Agent 1 ─────────────────────────────────────────────────
    sm = report_json["summary"]
    with open(OUTPUT_TXT, "w", encoding="utf-8") as ft:
        ft.write("=" * 72 + "\n")
        ft.write("RAPPORT ANOMALIES — AdOptimizer AI (Agent 3 Vigilant)\n")
        ft.write(f"Version   : v4-final  |  Généré le : {now_str()}\n")
        ft.write("=" * 72 + "\n\n")
        ft.write("RÉSUMÉ EXÉCUTIF\n")
        ft.write(f"  Campagnes analysées : {len(health_df)}\n")
        ft.write(f"  CRITICAL            : {sm['critical']}\n")
        ft.write(f"  WARNING             : {sm['warning']}\n")
        ft.write(f"  Phase 4 déclenchées : {sm['phase4_triggers']}\n")
        ft.write(f"  Score moyen         : {sm['avg_score']:.3f}/1.0\n")
        ft.write(f"  Impact moyen        : {sm['avg_impact']:.2f}€\n\n")

        if not health_df.empty:
            # Trier par impact financier [FIX 14] pour Agent 1
            alerts = health_df[
                health_df["global_level"] != LEVEL_OK
            ].sort_values("impact_score", ascending=False)

            for _, row in alerts.iterrows():
                ft.write("-" * 72 + "\n")
                ft.write(f"[{row['global_level']}] "
                         f"{row['campaign_id']} "
                         f"({row['platform'].upper()})\n")
                ft.write(f"  Score anomalie   : {row['anomaly_score']:.3f}/1.0\n")
                ft.write(f"  Impact financier : {row['impact_score']:.2f}€ "
                         f"[FIX 14]\n")
                ft.write(f"  Health Score     : "
                         f"{row['tool2_health_contribution']:.1f}/100\n")
                ft.write(f"  Jours anormaux   : {row['n_anomaly_days']}\n")
                ft.write(f"  Types détectés   : {row['anomaly_types']}\n")

                if row["explanation"]:
                    ft.write("\n  DIAGNOSTIC (trié par priorité [FIX 13]) :\n")
                    for part in str(row["explanation"]).split(" | "):
                        ft.write(f"    • {part}\n")

                ft.write("\n  ACTION RECOMMANDÉE :\n")
                if row["trigger_phase4"]:
                    ft.write("    → URGENT : Déclenche diagnostic Causal AI "
                             "(Phase 4)\n")
                    ft.write("    → Notifier le gestionnaire de campagne\n")
                elif row["global_level"] == LEVEL_WARNING:
                    ft.write("    → Surveiller 48h — vérification manuelle "
                             "conseillée\n")
                else:
                    ft.write("    → Monitoring renforcé recommandé\n")
                ft.write("\n")

    log(f"  TXT → Agent 1  : {OUTPUT_TXT}", f)
    return report_json


# ============================================================
# MAIN
# ============================================================

def main(input_path=str(INPUT_FILE), analysis_date=None):
    print(sep())
    print("DÉTECTEUR D'ANOMALIES — AdOptimizer AI  (Agent 3 Vigilant) [v4 FINAL]")
    print("  14 corrections appliquées — production-ready")
    print(sep())
    t0 = time.time()

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        log(f"Démarrage : {now_str()}", f)
        log(f"Input     : {input_path}\n", f)

        # MOTEUR ML
        df, obj_col   = phase1_load(input_path, f)
        df            = phase2_statistical(df, f)
        df            = phase3_temporal(df, f)
        df            = phase4_business_tracking(df, obj_col, f)
        df, if_models = phase5_isolation_forest(df, f)
        df            = phase6_weighted_score(df, f)
        df            = phase7_persistence(df, f)
        df, df_anom   = phase8_export_ml(df, f)

        # ORCHESTRATEUR
        analysis_date = (pd.Timestamp(analysis_date)
                         if analysis_date else df["date"].max())
        analyzable, profiles = orch1_filter_active(df, analysis_date, f)

        if len(analyzable) == 0:
            log("Aucune campagne analysable.", f)
            return df, pd.DataFrame(), {}

        health_df   = orch2_health_score(df, analyzable, f)
        report_json = orch3_generate_outputs(
            df, df_anom, health_df, profiles, f)

        elapsed = time.time() - t0
        log(f"\nTerminé : {now_str()} | Durée : {elapsed:.1f}s", f)

    sm = report_json.get("summary", {})
    print(f"\nTerminé en {elapsed:.1f}s")
    print(f"  CRITICAL : {sm.get('critical',0)} | "
          f"WARNING : {sm.get('warning',0)} | "
          f"Phase 4 : {sm.get('phase4_triggers',0)}")
    print(f"\n  Sorties moteur ML :")
    print(f"    anomalies.csv    → {OUTPUT_ANOM}")
    print(f"    summary.csv      → {OUTPUT_SUM}")
    print(f"    isolation_forest → {OUTPUT_MODEL}")
    print(f"\n  Sorties orchestrateur :")
    print(f"    JSON → Phase 4   → {OUTPUT_JSON}")
    print(f"    CSV  → Phase 3   → {OUTPUT_CSV}")
    print(f"    TXT  → Agent 1   → {OUTPUT_TXT}")
    print(f"    Log              → {OUTPUT_LOG}")

    return df, df_anom, report_json


if __name__ == "__main__":
    df_result, df_anom, report = main(
        input_path    = INPUT_FILE,
        analysis_date = None,
    )
def run_anomaly():
    try:
        df, df_anom, report = main(str(INPUT_FILE))

        return {
            "status": "success",
            "message": "Anomaly detection terminé",
            "outputs": {
                "anomalies": str(OUTPUT_ANOM),
                "summary": str(OUTPUT_SUM),
                "report_json": str(OUTPUT_JSON),
                "alerts": str(OUTPUT_CSV)
            },
            "n_anomalies": len(df_anom)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }