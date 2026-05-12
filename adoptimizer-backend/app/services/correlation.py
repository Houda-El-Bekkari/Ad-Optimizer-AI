#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TOOL 3 — CORRÉLATIONS INTER-CANAUX  (Cross-Channel Correlations Analyzer)
AdOptimizer AI — Agent 3 Vigilant (📅 weekly)
Version : v2.2 FINAL ALIGNED  (100 % conforme à la description PFE)
===============================================================================

OBJECTIF
--------
Analyser les interactions entre Meta Ads et Google Ads afin d'identifier les
relations et influences potentielles entre leurs performances, y compris les
effets retardés. Passage d'une vision isolée à une vision systémique
multi-canaux.

CONDITION D'ACTIVATION
----------------------
Module exécuté UNIQUEMENT pour campagnes multi-plateformes :
  • même global_campaign_id présent sur Meta ET Google
  • historique commun ≥ 30 jours
  • données journalières suffisantes
Les campagnes mono-plateforme sont exclues (analyse trend simplifié).

APPROCHE STATISTIQUE HYBRIDE
----------------------------
  (i)   Corrélations synchrones    : Pearson + Spearman (jour J vs jour J)
  (ii)  Corrélations décalées      : Cross-Correlation Function lags 1..7j
  (iii) Causalité de Granger       : test prédictif bidirectionnel
  (iv)  Mutual Information         : relations non-linéaires (saturation)

EFFETS DÉTECTÉS PAR CAMPAGNE
----------------------------
  • Halo effect           : une plateforme améliore les performances de l'autre
  • Cannibalisation       : une plateforme dégrade les performances de l'autre
  • Branding différé      : impact retardé sur les conversions (lag ≥ 3j)
  • Substitution budget   : transfert de budget inefficace (spend ↔ spend < 0)
  • Saturation cross-canal: spend ↑ → performance ↘ (relation non-linéaire MI)
  • Désynchronisation     : divergence z-score Meta vs Google

RÉFÉRENCES SCIENTIFIQUES
------------------------
  • Hanssens, Parsons, Schultz (2003) — Market Response Models
  • Granger (1969) — Investigating Causal Relations (Econometrica)
  • Shao & Li (2011) — Multi-touch Attribution Models (KDD)

PIPELINE INTERNE (7 phases)
---------------------------
  Phase 1 — Chargement + validation
  Phase 2 — Classification campagnes (multi vs mono canal)
  Phase 3 — Analyse mono-canal (trend simplifié)
  Phase 4 — Analyse multi-canal (Pearson + lag + Granger + MI)
  Phase 5 — Détection effets business
  Phase 6 — Calcul trend_score + confidence
  Phase 7 — Exports JSON + CSV + TXT + cache Redis (optionnel)

INPUT  : dataset_model_ready.csv (sortie Tool 1 preprocessing)

OUTPUTS (4 fichiers, dossier outputs/)
  correlations.json        → Tool 4 (Causal AI) + Tool 6 (RL Optimizer)
  correlations.csv         → Phase 3 (Health Score) + Dashboard
  correlation_report.txt   → Agent 1 (Consultant LLM)
  correlation.log          → Log d'exécution

CACHE REDIS (optionnel, TTL 7 jours)
  Clé "tool3:correlation:{campaign_id}" → signal exploitable rapidement

INTERFACE
  get_correlation_signal(gcamp_id) → dict pour Tool 4 / Tool 6

INSTALLATION
  pip install scipy scikit-learn statsmodels pandas numpy
  pip install redis  (optionnel, pour cache)

EXÉCUTION
  python tool3_correlations.py
===============================================================================
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from datetime                  import datetime
from scipy                     import stats
from sklearn.feature_selection import mutual_info_regression

# ── Granger optionnel (statsmodels) ──────────────────────────────────────────
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    GRANGER_AVAILABLE = True
except ImportError:
    GRANGER_AVAILABLE = False
    print("INFO : statsmodels non installé → Granger désactivé "
          "(pip install statsmodels pour activer)")

# ── Cache Redis optionnel ────────────────────────────────────────────────────
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

from pathlib import Path

BASE_DIR = Path("app")

INPUT_FILE = BASE_DIR / "data" / "dataset_model_ready.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_JSON  = OUTPUT_DIR / "correlations.json"
OUTPUT_CSV   = OUTPUT_DIR / "correlations.csv"
OUTPUT_TXT   = OUTPUT_DIR / "correlation_report.txt"
OUTPUT_LOG   = OUTPUT_DIR / "correlation.log"

os.makedirs(str(OUTPUT_DIR), exist_ok=True)

# ── Cache Redis (optionnel) ──────────────────────────────────────────────────
REDIS_HOST       = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB         = int(os.environ.get("REDIS_DB", 0))
REDIS_TTL_DAYS   = 7
REDIS_TTL_SECS   = REDIS_TTL_DAYS * 24 * 3600
REDIS_KEY_PREFIX = "tool3:correlation:"

# ── Seuils statistiques ─────────────────────────────────────────────────────
CORR_STRONG     = 0.6     # |r| ≥ 0.60 → forte
CORR_MODERATE   = 0.4     # |r| ≥ 0.40 → modérée
CORR_WEAK       = 0.2     # |r| ≥ 0.20 → faible
CANNIBALIZATION = -0.4    # corr(spend_X, conv_Y) < −0.4 → cannibalisation
HALO_EFFECT     = 0.4     # corr(spend_X, conv_Y, lag>0) > 0.4 → halo
BUDGET_SUBST    = -0.4    # corr(spend_meta, spend_google) < −0.4 → substitution
SYNC_THRESHOLD  = 1.5     # |z_meta − z_google| > 1.5 → désynchro
PVALUE_THRESH   = 0.05    # significativité standard
MI_THRESHOLD    = 0.10    # information mutuelle non triviale

# ── Observations minimales (CONFORMES À LA DESCRIPTION PFE) ─────────────────
MIN_OBS_MULTI    = 30     # multi-canal : 30 jours communs (description PFE)
MIN_OBS_MONO     = 3      # mono-canal : 3 jours pour trend
MIN_OBS_GRANGER  = 30     # Granger : aligné sur MIN_OBS_MULTI
MAX_LAG_DAYS     = 7      # lags testés [0..7]
GRANGER_MAX_LAG  = 3      # lags Granger (limité car DOF)

# ── Pondération du trend_score (0-100) ──────────────────────────────────────
BASE_SCORE       = 50
STRONG_CORR_PTS  = 30
MODERATE_PTS     = 15
HALO_PTS         = 20
GRANGER_BONUS    = 10
NONLINEAR_BONUS  =  5
CANNIB_PENALTY   = -20
SUBST_PENALTY    = -15    # nouveau : substitution budgétaire
SYNC_PENALTY     = -15

# ── Estimation valeur business par effet (% budget) ─────────────────────────
BUSINESS_VALUE_PCT = {
    "halo_positive"        : (5,  15),
    "cannibalization"      : (10, 25),
    "saturation"           : (5,  10),
    "branding"             : (3,  10),
    "budget_substitution"  : (5,  15),
    "sync_anomalies"       : (5,  15),
    "none"                 : (0,   0),
}


# =============================================================================
# UTILS
# =============================================================================

def log(msg, file_handle=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if file_handle:
        file_handle.write(line + "\n")
        file_handle.flush()


def sep():
    return "=" * 72


def estimate_business_value(effect_type):
    lo, hi = BUSINESS_VALUE_PCT.get(effect_type, (0, 0))
    if hi == 0:
        return None
    return {
        "min_pct": lo,
        "max_pct": hi,
        "label"  : f"+{lo} à +{hi} % d'optimisation budget potentielle",
    }


# =============================================================================
# CACHE REDIS (optionnel)
# =============================================================================

_redis_client = None

def get_redis():
    """Connexion Redis lazy + silencieuse en cas d'échec."""
    global _redis_client
    if not REDIS_AVAILABLE:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                             db=REDIS_DB, socket_connect_timeout=1,
                             decode_responses=True)
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception:
        return None


def cache_to_redis(results, log_handle=None):
    """Stocke chaque signal de corrélation dans Redis (TTL 7 jours)."""
    client = get_redis()
    if client is None:
        log("  Redis  : non disponible (skip cache)", log_handle)
        return 0

    n_cached = 0
    for r in results:
        gcamp_id = r["global_campaign_id"]
        key = f"{REDIS_KEY_PREFIX}{gcamp_id}"
        try:
            payload = json.dumps({k: v for k, v in r.items()
                                  if k != "global_campaign_id"},
                                 ensure_ascii=False, default=str)
            client.setex(key, REDIS_TTL_SECS, payload)
            n_cached += 1
        except Exception as e:
            log(f"  ⚠️  Redis SETEX {gcamp_id} : {e}", log_handle)

    log(f"  Redis  : {n_cached} signaux cachés (TTL {REDIS_TTL_DAYS}j)",
        log_handle)
    return n_cached


# =============================================================================
# PHASE 1 — CHARGEMENT & VALIDATION
# =============================================================================

def load_and_validate(filepath, f):
    log(sep(), f)
    log("PHASE 1 — Chargement & Validation", f)
    log(sep(), f)

    df = pd.read_csv(filepath)
    log(f"  Lignes chargées : {len(df):,}", f)

    # Compatibilité campaign_id / global_campaign_id
    if "global_campaign_id" not in df.columns:
        if "campaign_id" in df.columns:
            df["global_campaign_id"] = df["campaign_id"]
            log("  campaign_id → global_campaign_id (alias)", f)
        else:
            raise ValueError("Colonne global_campaign_id ou campaign_id requise")

    required = ["global_campaign_id", "platform", "date",
                "roas", "spend", "conversions", "cpa"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df["date"] = pd.to_datetime(df["date"])

    # CTR calculé si absent
    if "ctr_calc" not in df.columns:
        if "clicks" in df.columns and "impressions" in df.columns:
            df["ctr_calc"] = df["clicks"] / df["impressions"].replace(0, np.nan)
            log("  CTR calculé à partir clicks/impressions", f)
        else:
            df["ctr_calc"] = np.nan

    df["platform"] = df["platform"].str.lower().str.strip()
    df = df[df["platform"].isin(["meta", "google"])]

    log(f"  Campagnes uniques : {df['global_campaign_id'].nunique()}", f)
    log(f"  Plateformes       : {df['platform'].unique().tolist()}", f)
    log(f"  Période           : {df['date'].min().date()} → "
        f"{df['date'].max().date()}", f)

    return df


# =============================================================================
# PHASE 2 — CLASSIFICATION CAMPAGNES
# =============================================================================

def classify_campaigns(df, f):
    log(sep(), f)
    log("PHASE 2 — Classification Campagnes (multi/mono canal)", f)
    log(sep(), f)

    campaign_platforms = df.groupby("global_campaign_id")["platform"].unique()
    multi_channel, mono_channel = [], []

    for gcamp_id, platforms in campaign_platforms.items():
        if len(platforms) > 1:
            multi_channel.append(gcamp_id)
        else:
            mono_channel.append(gcamp_id)

    log(f"  Multi-canal : {len(multi_channel)}", f)
    log(f"  Mono-canal  : {len(mono_channel)}", f)
    return multi_channel, mono_channel


# =============================================================================
# PHASE 3 — ANALYSE MONO-CANAL (trend simplifié)
# =============================================================================

def analyze_mono_channel(df, gcamp_id, f):
    camp = df[df["global_campaign_id"] == gcamp_id].sort_values("date")
    platform = camp["platform"].iloc[0]
    n_obs = len(camp)

    if n_obs < MIN_OBS_MONO:
        return _empty_result(gcamp_id, "mono_channel_insufficient",
                             [platform], n_obs)

    x = np.arange(n_obs)
    y = camp["roas"].values
    valid = ~np.isnan(y)
    slope = np.polyfit(x[valid], y[valid], 1)[0] if valid.sum() >= 3 else 0.0

    spend_mean = camp["spend"].mean()
    spend_std  = camp["spend"].std()
    cv_spend   = (spend_std / spend_mean) if spend_mean > 0 else 0.0

    score = BASE_SCORE
    if slope > 0.1:    score += 20
    elif slope < -0.1: score -= 20
    if cv_spend < 0.2:   score += 10
    elif cv_spend > 0.5: score -= 10
    score = max(0, min(100, score))

    return {
        "global_campaign_id": gcamp_id,
        "type"              : "mono_channel",
        "channels"          : [platform],
        "n_observations"    : n_obs,
        "trend_score"       : int(score),
        "confidence"        : 0.6,
        "roas_slope"        : round(float(slope), 4),
        "spend_cv"          : round(float(cv_spend), 3),
        "cross_channel_effect": None,
        "cannibalization"     : {"detected": False},
        "halo_effect"         : {"detected": False},
        "budget_substitution" : {"detected": False, "corr_spend": 0.0},
        "sync_anomalies"      : False,
        "non_linear"          : False,
        "mutual_information"  : {"mi_roas": None, "mi_spend_conv": None},
        "granger_significant" : False,
        "granger_detail"      : {"direction": "none", "any_significant": False},
        "business_value"      : None,
        "primary_effect"      : "none",
    }


def _empty_result(gcamp_id, kind, channels, n_obs):
    return {
        "global_campaign_id": gcamp_id,
        "type"              : kind,
        "channels"          : channels,
        "n_observations"    : n_obs,
        "trend_score"       : 50,
        "confidence"        : 0.3,
        "cross_channel_effect": None,
        "cannibalization"     : {"detected": False},
        "halo_effect"         : {"detected": False},
        "budget_substitution" : {"detected": False, "corr_spend": 0.0},
        "sync_anomalies"      : False,
        "non_linear"          : False,
        "mutual_information"  : {"mi_roas": None, "mi_spend_conv": None},
        "granger_significant" : False,
        "granger_detail"      : {"direction": "none", "any_significant": False},
        "business_value"      : None,
        "primary_effect"      : "none",
    }


# =============================================================================
# PHASE 4 — CALCUL CORRÉLATION AVEC LAG AUTOMATIQUE
# =============================================================================

def compute_correlation_with_lag(s1, s2, max_lag=MAX_LAG_DAYS):
    """
    Cross-Correlation Function : test des lags [0..max_lag].
    Retient le lag k* maximisant |corr(s1(t), s2(t+k))|.
    Lag positif → s1 précède s2 (effet halo / branding).
    """
    best = {"lag": 0, "pearson": 0.0, "spearman": 0.0, "p_value": 1.0}

    for lag in range(max_lag + 1):
        if lag == 0:
            x, y = s1, s2
        else:
            x = s1[:-lag]
            y = s2[lag:]

        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 3:
            continue

        x_c, y_c = x[valid], y[valid]
        if np.std(x_c) == 0 or np.std(y_c) == 0:
            continue

        try:
            r_p, p_val = stats.pearsonr(x_c, y_c)
        except Exception:
            continue

        if abs(r_p) > abs(best["pearson"]):
            try:
                r_s, _ = stats.spearmanr(x_c, y_c)
            except Exception:
                r_s = r_p
            best = {"lag": lag, "pearson": r_p,
                    "spearman": r_s, "p_value": p_val}

    return best


# =============================================================================
# PHASE 4-bis — TEST DE GRANGER (bidirectionnel)
# =============================================================================

def granger_significant(s_cause, s_effect, max_lag=GRANGER_MAX_LAG):
    """Granger F-test unidirectionnel. Retourne (sig: bool, p_min: float)."""
    if not GRANGER_AVAILABLE:
        return False, 1.0

    df = pd.DataFrame({"effect": s_effect, "cause": s_cause}).dropna()
    if len(df) < MIN_OBS_GRANGER:
        return False, 1.0

    try:
        test = grangercausalitytests(df[["effect", "cause"]],
                                     maxlag=max_lag, verbose=False)
        p_min = min(test[lag][0]["ssr_ftest"][1] for lag in test)
        return bool(p_min < PVALUE_THRESH), float(p_min)
    except Exception:
        return False, 1.0


def granger_bidirectional(s_meta, s_google, max_lag=GRANGER_MAX_LAG):
    """Test Granger dans les 2 sens, détermine la direction dominante."""
    sig_m2g, p_m2g = granger_significant(s_meta, s_google, max_lag)
    sig_g2m, p_g2m = granger_significant(s_google, s_meta, max_lag)

    if   sig_m2g and sig_g2m: direction = "bidirectional"
    elif sig_m2g:             direction = "meta_to_google"
    elif sig_g2m:             direction = "google_to_meta"
    else:                     direction = "none"

    return {
        "meta_to_google" : bool(sig_m2g),
        "p_m2g"          : round(float(p_m2g), 4),
        "google_to_meta" : bool(sig_g2m),
        "p_g2m"          : round(float(p_g2m), 4),
        "direction"      : direction,
        "any_significant": bool(sig_m2g or sig_g2m),
    }


# =============================================================================
# PHASE 4-ter — MUTUAL INFORMATION (relations non-linéaires)
# =============================================================================

def mutual_info(x, y):
    """I(X;Y) — capte les relations non-linéaires (saturation, U-shape)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < MIN_OBS_MULTI:
        return None
    try:
        mi = mutual_info_regression(x[mask].reshape(-1, 1), y[mask],
                                    random_state=42)[0]
        return float(mi)
    except Exception:
        return None


# =============================================================================
# PHASE 5 — ANALYSE MULTI-CANAL (cœur du Tool)
# =============================================================================

def analyze_multi_channel(df, gcamp_id, f):
    camp = df[df["global_campaign_id"] == gcamp_id].copy()
    meta   = (camp[camp["platform"] == "meta"]
                  .set_index("date").sort_index())
    google = (camp[camp["platform"] == "google"]
                  .set_index("date").sort_index())

    common = meta.index.intersection(google.index)
    n_obs  = len(common)

    if n_obs < MIN_OBS_MULTI:
        return _empty_result(gcamp_id, "multi_channel_insufficient_data",
                             ["meta", "google"], n_obs)

    meta_a   = meta.loc[common]
    google_a = google.loc[common]

    # ── 1) Corrélation principale ROAS ↔ ROAS avec lag ──────────────────────
    cc = compute_correlation_with_lag(
        meta_a["roas"].values,
        google_a["roas"].values,
    )
    best_lag, pearson_r = cc["lag"], cc["pearson"]
    spearman_r, p_value = cc["spearman"], cc["p_value"]

    # Direction basée UNIQUEMENT sur le lag
    if best_lag == 0:
        direction = "simultaneous"
    elif best_lag > 0:
        direction = "meta_precedes_google"
    else:
        direction = "google_precedes_meta"

    if   abs(pearson_r) >= CORR_STRONG:   strength = "forte"
    elif abs(pearson_r) >= CORR_MODERATE: strength = "modérée"
    elif abs(pearson_r) >= CORR_WEAK:     strength = "faible"
    else:                                 strength = "négligeable"

    # ── 2) Helper safe_pearson ──────────────────────────────────────────────
    def safe_pearson(a, b):
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            return 0.0
        try:
            return float(stats.pearsonr(a[m], b[m])[0])
        except Exception:
            return 0.0

    # ── 3) Cannibalisation : Spend(X) ↔ Conv(Y) ─────────────────────────────
    corr_sp_cv_mg = safe_pearson(meta_a["spend"].values,
                                 google_a["conversions"].values)
    corr_sp_cv_gm = safe_pearson(google_a["spend"].values,
                                 meta_a["conversions"].values)
    cannibalization_detected = (
        corr_sp_cv_mg < CANNIBALIZATION or
        corr_sp_cv_gm < CANNIBALIZATION
    )

    # ── 4) Halo effect : Conv(X) ↔ −CPA(Y) ──────────────────────────────────
    corr_cv_cpa_mg = safe_pearson(meta_a["conversions"].values,
                                  -google_a["cpa"].values)
    corr_cv_cpa_gm = safe_pearson(google_a["conversions"].values,
                                  -meta_a["cpa"].values)
    halo_detected = (
        corr_cv_cpa_mg > HALO_EFFECT or
        corr_cv_cpa_gm > HALO_EFFECT
    )

    # ── 5) Substitution budgétaire : Spend Meta ↔ Spend Google < 0 ─────────
    corr_spend_spend = safe_pearson(meta_a["spend"].values,
                                    google_a["spend"].values)
    budget_subst_detected = corr_spend_spend < BUDGET_SUBST

    # ── 6) Désynchronisation (z-score divergence) ──────────────────────────
    m_roas = meta_a["roas"].values
    g_roas = google_a["roas"].values
    m_z = (m_roas - np.nanmean(m_roas)) / (np.nanstd(m_roas) + 1e-6)
    g_z = (g_roas - np.nanmean(g_roas)) / (np.nanstd(g_roas) + 1e-6)
    divergence = float(np.nanmean(np.abs(m_z - g_z)))
    sync_anomalies = divergence > SYNC_THRESHOLD

    # ── 7) Granger causality bidirectionnel ─────────────────────────────────
    granger = granger_bidirectional(
        meta_a["spend"].values,
        google_a["conversions"].values,
        max_lag=GRANGER_MAX_LAG,
    )
    granger_sig = granger["any_significant"]
    granger_p   = min(granger["p_m2g"], granger["p_g2m"])

    # ── 8) Mutual Information étendue ───────────────────────────────────────
    mi_roas       = mutual_info(meta_a["roas"].values, google_a["roas"].values)
    mi_spend_conv = mutual_info(meta_a["spend"].values,
                                google_a["conversions"].values)

    is_nonlinear = (
        (mi_roas is not None and mi_roas > MI_THRESHOLD
         and abs(pearson_r) < CORR_WEAK)
        or
        (mi_spend_conv is not None and mi_spend_conv > MI_THRESHOLD
         and abs(corr_sp_cv_mg) < CORR_WEAK)
    )

    # ── 9) Trend score (0-100) ──────────────────────────────────────────────
    score = BASE_SCORE
    if pearson_r >= CORR_STRONG:    score += STRONG_CORR_PTS
    elif pearson_r >= CORR_MODERATE: score += MODERATE_PTS
    if halo_detected:                score += HALO_PTS
    if granger_sig:                  score += GRANGER_BONUS
    if is_nonlinear:                 score += NONLINEAR_BONUS
    if cannibalization_detected:     score += CANNIB_PENALTY
    if budget_subst_detected:        score += SUBST_PENALTY
    if sync_anomalies:               score += SYNC_PENALTY
    trend_score = max(0, min(100, score))

    # ── 10) Confidence score ────────────────────────────────────────────────
    conf = 0.5
    if   n_obs >= 60: conf += 0.35
    elif n_obs >= 45: conf += 0.30
    elif n_obs >= 30: conf += 0.20
    if p_value < PVALUE_THRESH:       conf += 0.15
    if abs(pearson_r) >= CORR_STRONG: conf += 0.05
    if granger_sig:                   conf += 0.05
    confidence = min(1.0, conf)

    # ── 11) Effet primaire (priorité décroissante) ──────────────────────────
    if cannibalization_detected:
        primary = "cannibalization"
    elif budget_subst_detected:
        primary = "budget_substitution"
    elif halo_detected:
        primary = "halo_positive"
    elif is_nonlinear:
        primary = "saturation"
    elif sync_anomalies:
        primary = "sync_anomalies"
    elif best_lag >= 3 and pearson_r > CORR_WEAK:
        primary = "branding"
    else:
        primary = "none"

    return {
        "global_campaign_id": gcamp_id,
        "type"              : "multi_channel",
        "channels"          : ["meta", "google"],
        "n_observations"    : n_obs,

        "cross_channel_effect": {
            "best_lag_days"   : best_lag,
            "direction"       : direction,
            "correlation_roas": round(pearson_r, 3),
            "spearman_r"      : round(spearman_r, 3),
            "p_value"         : round(p_value, 4),
            "strength"        : strength,
        },
        "cannibalization": {
            "detected"               : bool(cannibalization_detected),
            "spend_meta_conv_google" : round(corr_sp_cv_mg, 3),
            "spend_google_conv_meta" : round(corr_sp_cv_gm, 3),
        },
        "halo_effect": {
            "detected"            : bool(halo_detected),
            "conv_meta_cpa_google": round(corr_cv_cpa_mg, 3),
            "conv_google_cpa_meta": round(corr_cv_cpa_gm, 3),
        },
        "budget_substitution": {
            "detected"   : bool(budget_subst_detected),
            "corr_spend" : round(corr_spend_spend, 3),
        },
        "sync_anomalies"     : bool(sync_anomalies),
        "divergence_score"   : round(divergence, 3),
        "non_linear"         : bool(is_nonlinear),
        "mutual_information" : {
            "mi_roas"      : round(mi_roas, 4) if mi_roas is not None else None,
            "mi_spend_conv": round(mi_spend_conv, 4)
                             if mi_spend_conv is not None else None,
        },
        "granger_significant": bool(granger_sig),
        "granger_p_value"    : round(granger_p, 4),
        "granger_detail"     : granger,
        "primary_effect"     : primary,
        "business_value"     : estimate_business_value(primary),
        "trend_score"        : int(trend_score),
        "confidence"         : round(confidence, 2),
    }


# =============================================================================
# PHASE 6 — ORCHESTRATEUR
# =============================================================================

def run_correlation_analysis(df, multi_channel, mono_channel, f):
    log(sep(), f)
    log("PHASE 3 — Analyse Corrélations (par campagne)", f)
    log(sep(), f)

    results = []

    log(f"  Multi-canal  : {len(multi_channel)} campagnes...", f)
    for i, gcamp_id in enumerate(multi_channel, 1):
        if i % 10 == 0:
            log(f"    [{i}/{len(multi_channel)}]", f)
        try:
            results.append(analyze_multi_channel(df, gcamp_id, f))
        except Exception as e:
            log(f"    ⚠️  {gcamp_id} : {e}", f)

    log(f"  Mono-canal   : {len(mono_channel)} campagnes...", f)
    for i, gcamp_id in enumerate(mono_channel, 1):
        if i % 50 == 0:
            log(f"    [{i}/{len(mono_channel)}]", f)
        try:
            results.append(analyze_mono_channel(df, gcamp_id, f))
        except Exception as e:
            log(f"    ⚠️  {gcamp_id} : {e}", f)

    log(f"  Total analysé : {len(results)}", f)
    return results


# =============================================================================
# PHASE 7 — EXPORTS (JSON + CSV + TXT + Redis)
# =============================================================================

def export_results(results, f):
    log(sep(), f)
    log("PHASE 4 — Exports (JSON + CSV + TXT + Redis)", f)
    log(sep(), f)

    # ── 1) JSON → Tool 4 / Tool 6 ──────────────────────────────────────────
    json_output = {
        r["global_campaign_id"]: {k: v for k, v in r.items()
                                  if k != "global_campaign_id"}
        for r in results
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fj:
        json.dump(json_output, fj, indent=2, ensure_ascii=False, default=str)
    log(f"  JSON   → {OUTPUT_JSON}", f)

    # ── 2) CSV → Dashboard / Health Score ──────────────────────────────────
    csv_rows = []
    for r in results:
        row = {
            "global_campaign_id": r["global_campaign_id"],
            "type"              : r["type"],
            "n_observations"    : r["n_observations"],
            "trend_score"       : r["trend_score"],
            "confidence"        : r["confidence"],
            "primary_effect"    : r.get("primary_effect", "none"),
        }
        if r["type"] == "multi_channel" and r["cross_channel_effect"]:
            ce = r["cross_channel_effect"]
            mi = r.get("mutual_information", {}) or {}
            gd = r.get("granger_detail", {}) or {}
            bs = r.get("budget_substitution", {}) or {}
            row.update({
                "best_lag_days"        : ce["best_lag_days"],
                "direction"            : ce["direction"],
                "correlation_roas"     : ce["correlation_roas"],
                "spearman_r"           : ce["spearman_r"],
                "p_value"              : ce["p_value"],
                "strength"             : ce["strength"],
                "cannibalization"      : r["cannibalization"]["detected"],
                "halo_effect"          : r["halo_effect"]["detected"],
                "budget_substitution"  : bs.get("detected", False),
                "corr_spend_spend"     : bs.get("corr_spend"),
                "sync_anomalies"       : r["sync_anomalies"],
                "non_linear"           : r["non_linear"],
                "granger_significant"  : r["granger_significant"],
                "granger_direction"    : gd.get("direction"),
                "granger_meta_to_google": gd.get("meta_to_google"),
                "granger_google_to_meta": gd.get("google_to_meta"),
                "mi_roas"              : mi.get("mi_roas"),
                "mi_spend_conv"        : mi.get("mi_spend_conv"),
            })
        bv = r.get("business_value") or {}
        row["business_value_min"] = bv.get("min_pct")
        row["business_value_max"] = bv.get("max_pct")
        csv_rows.append(row)

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(OUTPUT_CSV, index=False)
    log(f"  CSV    → {OUTPUT_CSV}", f)

    # ── 3) TXT → Agent 1 (LLM) ─────────────────────────────────────────────
    with open(OUTPUT_TXT, "w", encoding="utf-8") as ft:
        ft.write("=" * 72 + "\n")
        ft.write("RAPPORT CORRÉLATIONS INTER-CANAUX — AdOptimizer AI\n")
        ft.write(f"Tool 3 v2.2 FINAL ALIGNED  |  "
                 f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        ft.write("=" * 72 + "\n\n")

        ft.write("RÉSUMÉ EXÉCUTIF\n")
        ft.write(f"  Campagnes analysées : {len(results)}\n")
        multi_results = [r for r in results if r["type"] == "multi_channel"]
        mono_results  = [r for r in results if "mono" in r["type"]]
        ft.write(f"  Multi-canal         : {len(multi_results)}\n")
        ft.write(f"  Mono-canal          : {len(mono_results)}\n")

        if multi_results:
            cannib = sum(1 for r in multi_results
                         if r["cannibalization"]["detected"])
            halo   = sum(1 for r in multi_results
                         if r["halo_effect"]["detected"])
            subst  = sum(1 for r in multi_results
                         if r.get("budget_substitution", {}).get("detected"))
            sync   = sum(1 for r in multi_results if r["sync_anomalies"])
            nonlin = sum(1 for r in multi_results if r["non_linear"])
            grang  = sum(1 for r in multi_results
                         if r["granger_significant"])

            ft.write("\n  DÉTECTIONS MULTI-CANAL :\n")
            ft.write(f"    Cannibalisation         : {cannib}\n")
            ft.write(f"    Halo Effect             : {halo}\n")
            ft.write(f"    Substitution budgétaire : {subst}\n")
            ft.write(f"    Granger significatif    : {grang}\n")
            ft.write(f"    Sync Anomalies          : {sync}\n")
            ft.write(f"    Relations non-linéaires : {nonlin}\n")

        ft.write("\n" + "-" * 72 + "\n")
        ft.write("CAMPAGNES ALERTES (trend_score < 40 ou patterns détectés)\n")
        ft.write("-" * 72 + "\n\n")

        alerts = [r for r in results
                  if r["trend_score"] < 40
                  or r.get("cannibalization", {}).get("detected")
                  or r.get("budget_substitution", {}).get("detected")
                  or r.get("sync_anomalies")]
        for r in sorted(alerts, key=lambda x: x["trend_score"]):
            ft.write(f"[{r['global_campaign_id']}]\n")
            ft.write(f"  Type             : {r['type']}\n")
            ft.write(f"  Trend Score      : {r['trend_score']}/100\n")
            ft.write(f"  Confiance        : {r['confidence']:.2f}\n")
            ft.write(f"  Effet primaire   : {r.get('primary_effect', 'none')}\n")

            if r["type"] == "multi_channel" and r["cross_channel_effect"]:
                ce = r["cross_channel_effect"]
                ft.write(f"  Corrélation ROAS : {ce['correlation_roas']:+.3f} "
                         f"({ce['strength']})\n")
                ft.write(f"  Lag optimal      : {ce['best_lag_days']} jour(s)\n")
                ft.write(f"  Direction        : {ce['direction']}\n")

                if r["granger_significant"]:
                    gd = r.get("granger_detail", {}) or {}
                    ft.write(f"  Granger          : ✓ p={r['granger_p_value']:.3f}"
                             f" | direction={gd.get('direction', 'n/a')}\n")
                if r["cannibalization"]["detected"]:
                    ft.write("  ⚠️  CANNIBALISATION DÉTECTÉE\n")
                if r.get("budget_substitution", {}).get("detected"):
                    bs = r["budget_substitution"]
                    ft.write(f"  ⚠️  SUBSTITUTION BUDGÉTAIRE "
                             f"(corr_spend={bs['corr_spend']:+.2f})\n")
                if r["halo_effect"]["detected"]:
                    ft.write("  ✅ Halo Effect (synergie)\n")
                if r["sync_anomalies"]:
                    ft.write("  ⚠️  DÉSYNCHRONISATION ANORMALE\n")
                if r["non_linear"]:
                    ft.write("  ~ Relation non-linéaire (saturation possible)\n")

            bv = r.get("business_value")
            if bv:
                ft.write(f"  Valeur business  : {bv['label']}\n")
            ft.write("\n")

    log(f"  TXT    → {OUTPUT_TXT}", f)

    # ── 4) Cache Redis (optionnel, TTL 7 jours) ────────────────────────────
    cache_to_redis(results, log_handle=f)


# =============================================================================
# INTERFACE POUR TOOL 4 / TOOL 6
# =============================================================================

def get_correlation_signal(gcamp_id: str) -> dict:
    """
    Récupère le signal de corrélation pour une campagne.
    Stratégie : Redis (rapide) → fallback JSON (toujours disponible).
    Utilisé par Tool 4 (Causal AI) et Tool 6 (RL Optimizer).
    """
    # Tentative Redis
    client = get_redis()
    if client is not None:
        try:
            cached = client.get(f"{REDIS_KEY_PREFIX}{gcamp_id}")
            if cached:
                return _format_signal(json.loads(cached))
        except Exception:
            pass

    # Fallback JSON
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as fj:
            data = json.load(fj)
        if gcamp_id not in data:
            return None
        return _format_signal(data[gcamp_id])
    except Exception as e:
        print(f"⚠️  Erreur get_correlation_signal : {e}")
        return None


def _format_signal(r):
    """Formate le signal exploitable par Tool 4 / Tool 6."""
    signal = {
        "trend_score"        : r.get("trend_score"),
        "confidence"         : r.get("confidence"),
        "primary_effect"     : r.get("primary_effect"),
        "cannibalization"    : r.get("cannibalization", {}).get("detected", False),
        "halo_effect"        : r.get("halo_effect", {}).get("detected", False),
        "budget_substitution": r.get("budget_substitution", {}).get("detected", False),
        "sync_anomalies"     : r.get("sync_anomalies", False),
        "non_linear"         : r.get("non_linear", False),
        "granger_significant": r.get("granger_significant", False),
        "granger_direction"  : (r.get("granger_detail") or {}).get("direction"),
        "mutual_information" : r.get("mutual_information"),
        "business_value"     : r.get("business_value"),
    }
    if r.get("cross_channel_effect"):
        signal.update({
            "best_lag_days"   : r["cross_channel_effect"]["best_lag_days"],
            "correlation_roas": r["cross_channel_effect"]["correlation_roas"],
            "direction"       : r["cross_channel_effect"]["direction"],
        })
    return signal


# =============================================================================
# MAIN
# =============================================================================

def main(input_path=str(INPUT_FILE)):
    print(sep())
    print("TOOL 3 — CORRÉLATIONS INTER-CANAUX")
    print("AdOptimizer AI — Agent 3 Vigilant (📅 weekly)")
    print("Version v2.2 FINAL ALIGNED  (description PFE 100 % conforme)")
    print(sep())

    t0 = time.time()

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        log(f"Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        log(f"Input     : {input_path}", f)
        log(f"Granger   : {'disponible' if GRANGER_AVAILABLE else 'désactivé'}", f)
        log(f"Redis     : {'disponible' if get_redis() else 'désactivé'}\n", f)

        df = load_and_validate(input_path, f)
        multi, mono = classify_campaigns(df, f)
        results = run_correlation_analysis(df, multi, mono, f)
        export_results(results, f)

        elapsed = time.time() - t0
        log(f"\nTerminé en {elapsed:.1f}s", f)

    print(f"\n✅ Terminé en {elapsed:.1f}s")
    print("\nSorties :")
    print(f"  JSON  → Tool 4/6  : {OUTPUT_JSON}")
    print(f"  CSV   → Dashboard : {OUTPUT_CSV}")
    print(f"  TXT   → Agent 1   : {OUTPUT_TXT}")
    print(f"  Redis → cache 7j  : {REDIS_KEY_PREFIX}* (TTL {REDIS_TTL_DAYS}j)")
    print(f"  Log               : {OUTPUT_LOG}")
    return results


if __name__ == "__main__":
    results = main()

    if results:
        first_id = results[0]["global_campaign_id"]
        signal = get_correlation_signal(first_id)
        print("\n📡 Démo interface Tool 4/6 :")
        print(f"  get_correlation_signal('{first_id}') →")
        print(f"  {json.dumps(signal, indent=4, ensure_ascii=False)}")

def run_correlation():
    try:
        results = main(str(INPUT_FILE))

        return {
            "status": "success",
            "message": "Correlation analysis terminé",
            "outputs": {
                "json": str(OUTPUT_JSON),
                "csv": str(OUTPUT_CSV),
                "report": str(OUTPUT_TXT)
            },
            "n_campaigns": len(results)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }