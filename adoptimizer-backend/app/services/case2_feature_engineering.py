"""
===============================================================================
FEATURE ENGINEERING PRO — AdOptimizer AI  (Cas 2 : Nouvelle Campagne)
===============================================================================
Version avancée compatible avec best_model.pkl (187 features exactes).

Objectif : Transformer chaque stratégie candidate en vecteur de 187 features
           identique à ce que le modèle LightGBM a vu pendant l'entraînement.

Approche :
  1. Charger best_model.pkl → lire les 187 feature_cols exactes
  2. Depuis raw_ads_dataset_strict.csv, calculer le vecteur médian
     des campagnes HIGH_PERFORMANCE, séparé par plateforme (meta / google)
  3. Pour chaque stratégie :
       vecteur = copie du vecteur base (plateforme correspondante)
               + écrasement des variables de la stratégie
               + recalcul des dérivées (lags, rolling, business vars)

Générique :
  plateforme = "meta"   → 3 lignes  (base HIGH Meta)
  plateforme = "google" → 3 lignes  (base HIGH Google)
  plateforme = "both"   → 6 lignes  (3 Meta + 3 Google)

Input  :
  agent2_outputs/strategies.json
  segmentation_outputs/cluster_profiles.csv
  segmentation_outputs/segmentation_results.csv
  raw_ads_dataset_strict.csv
  models/best_model.pkl

Output :
  feature_engineering_outputs/
    features_strategies_pro.csv   → N × 187 features (prêt pour predictor)
    features_report_pro.txt       → rapport lisible
    features_radar_pro.png        → radar comparatif

Auteur : AdOptimizer AI — PFE 2024
===============================================================================
"""

import os, json, warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CASE2_OUTPUTS_DIR = BASE_DIR / "cas2-outputs"

INPUT_STRATEGIES = CASE2_OUTPUTS_DIR / "agent2_outputs" / "strategies.json"
INPUT_PROFILES   = CASE2_OUTPUTS_DIR / "segmentation_outputs" / "cluster_profiles.csv"
INPUT_SEG        = CASE2_OUTPUTS_DIR / "segmentation_outputs" / "segmentation_results.csv"
INPUT_RAW        = DATA_DIR / "dataset_historique.csv"
MODEL_PATH       = BASE_DIR / "models" / "best_model.pkl"

OUTPUT_DIR      = CASE2_OUTPUTS_DIR / "feature_engineering_outputs"
OUTPUT_FEATURES = OUTPUT_DIR / "features_strategies_pro.csv"
OUTPUT_REPORT   = OUTPUT_DIR / "features_report_pro.txt"
OUTPUT_RADAR    = OUTPUT_DIR / "features_radar_pro.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NA_PLATFORM = "NA_platform"

# Valeur monétaire par conversion — dynamique selon le produit (cohérent avec Agent 2)
PRODUCT_VALUE_MAP = {
    "saas"      : 120.0,
    "formation" : 80.0,
    "ecommerce" : 25.0,
    "immobilier": 500.0,
    "finance"   : 200.0,
    "sante"     : 60.0,
    "autre"     : 20.0,
}

# Fix 3 — CPC aussi adapté au produit (SaaS ≠ Ecommerce)
PRODUCT_CPC_MULT = {
    "saas"      : 1.40,
    "formation" : 1.10,
    "ecommerce" : 0.90,
    "immobilier": 1.60,
    "finance"   : 1.50,
    "sante"     : 1.20,
    "autre"     : 1.00,
}

def get_value_per_conversion(produit: str) -> float:
    """Retourne la valeur monétaire par conversion selon le produit."""
    prod = produit.lower().strip()
    for key, val in PRODUCT_VALUE_MAP.items():
        if key in prod:
            return val
    return PRODUCT_VALUE_MAP["autre"]


def get_cpc_mult(produit: str) -> float:
    """Retourne le multiplicateur CPC selon le produit."""
    prod = produit.lower().strip()
    for key, mult in PRODUCT_CPC_MULT.items():
        if key in prod:
            return mult
    return PRODUCT_CPC_MULT["autre"]

# Encodages catégoriels (identiques à Tool 1)
OBJECTIF_ENC = {"conversions": 0, "leads": 1, "traffic": 2, "awareness": 3}

def sep(c="=", n=72): return c * n
def log(m):
    text = str(m)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def safe_div(a, b, fill=0.0):
    """
    Division sécurisée — fonctionne avec scalaires ET Series pandas.
    Évite ZeroDivisionError et les ambiguités pandas.
    """
    if hasattr(b, "__len__"):                          # Series / array
        return (a / b.replace(0, np.nan)).fillna(fill)
    return a / b if b and b != 0 else fill             # scalaire


# ============================================================
# ÉTAPE 1 — CHARGER LE BUNDLE (pour feature_cols exact)
# ============================================================
def load_bundle():
    log(sep()); log("ÉTAPE 1 — CHARGEMENT DU BUNDLE MODÈLE"); log(sep())
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Vérifiez que best_model.pkl est dans le dossier 'models/'."
        )
    bundle = joblib.load(MODEL_PATH)
    log(f"  Modèle      : {bundle['model_name']}")
    log(f"  Features    : {len(bundle['feature_cols'])} colonnes exactes")
    log(f"  Targets     : {bundle['target_cols']}")
    return bundle

# ============================================================
# ÉTAPE 2 — CONSTRUIRE LES VECTEURS BASE PAR PLATEFORME
# ============================================================
def build_base_vectors_from_high(feature_cols: list) -> dict:
    """
    Reconstruit les 187 features depuis raw_ads_dataset_strict.csv.
    Applique exactement le même preprocessing que Tool 1.

    Retourne UN vecteur base PAR PLATEFORME :
      {
        "meta"  : { feature: valeur, ... },  ← médianes HIGH Meta
        "google": { feature: valeur, ... },  ← médianes HIGH Google
      }

    Si une plateforme n'a pas de campagnes HIGH → fallback sur l'autre.
    """
    log(sep()); log("ÉTAPE 2 — VECTEURS BASE PAR PLATEFORME (médianes HIGH)"); log(sep())

    # ── Charger et nettoyer ──────────────────────────────────────────────────
    df = pd.read_csv(INPUT_RAW, low_memory=False)
    df = df.drop_duplicates()

    num_cols = ["spend","impressions","clicks","conversions","conversion_value",
                "CTR","CPC","CPM","daily_budget","lifetime_budget",
                "reach","frequency","link_clicks","likes","comments","shares",
                "post_engagement","video_views","add_to_cart","purchases","quality_score"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].replace(NA_PLATFORM, np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"]       = pd.to_datetime(df["date"],       errors="coerce")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    df = df.dropna(subset=["spend","clicks","impressions"])
    df = df[df["clicks"] > 0].copy()

    # ── Masques NA_platform ──────────────────────────────────────────────────
    for col in ["reach","frequency","link_clicks","likes","comments","shares",
                "post_engagement","video_views","add_to_cart","purchases",
                "network","keyword","match_type","search_term","quality_score"]:
        if col in df.columns:
            df[f"has_{col}"] = (df[col].notna()).astype(int)

    # ── Variables business ───────────────────────────────────────────────────
    df["roas"]      = (df["conversion_value"] / df["spend"].replace(0, np.nan)).fillna(0)
    df["ctr_calc"]  = (df["clicks"] / df["impressions"].replace(0, np.nan)).fillna(0)
    df["cpc_calc"]  = (df["spend"]  / df["clicks"].replace(0, np.nan)).fillna(0)
    df["conv_rate"] = (df["conversions"] / df["clicks"].replace(0, np.nan)).fillna(0)
    df["cpm_calc"]  = (df["spend"] / df["impressions"].replace(0, np.nan) * 1000).fillna(0)
    cpa_raw         = df["spend"] / df["conversions"].replace(0, np.nan)
    df["cpa"]       = cpa_raw.fillna(cpa_raw.median())
    df["budget_utilization"] = (df["spend"] / df["daily_budget"].replace(0, np.nan)).clip(0,1).fillna(0)
    df["revenue_per_click"]  = (df["conversion_value"] / df["clicks"].replace(0, np.nan)).fillna(0)
    df["ctr_delta"]          = (df["CTR"] - df["ctr_calc"]).abs().fillna(0)

    # ── Features temporelles ─────────────────────────────────────────────────
    df["day_of_week"]       = df["date"].dt.dayofweek
    df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"]      = df["date"].dt.isocalendar().week.astype(int)
    df["month"]             = df["date"].dt.month
    df["day_of_year"]       = df["date"].dt.dayofyear
    df["quarter"]           = df["date"].dt.quarter
    df["campaign_age_days"] = (df["date"] - df["start_date"]).dt.days.clip(lower=0).fillna(0)
    df["days_to_end"]       = (df["end_date"] - df["date"]).dt.days.clip(lower=0).fillna(0)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ── Lags + Rolling ───────────────────────────────────────────────────────
    df = df.sort_values(["campaign_id","platform","date"]).reset_index(drop=True)
    grp = df.groupby(["campaign_id","platform"], sort=False)
    lag_metrics = ["spend","clicks","conversions","roas","cpa",
                   "ctr_calc","cpc_calc","conv_rate","impressions"]
    for m in lag_metrics:
        if m not in df.columns: continue
        for lag in [1, 3, 7]:
            df[f"{m}_lag{lag}"] = grp[m].shift(lag)
        for w in [7, 14, 30]:
            df[f"{m}_roll{w}m"] = grp[m].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w//3)).mean())
            df[f"{m}_roll{w}s"] = grp[m].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w//3)).std().fillna(0))
        base_r = grp[m].transform(
            lambda x: x.shift(7).rolling(3, min_periods=1).mean())
        df[f"{m}_trend7"] = (
            (df[m] - base_r) / base_r.abs().replace(0, np.nan)
        ).fillna(0)
    for lag in [1, 3, 7]:
        if "conversion_value" in df.columns:
            df[f"conversion_value_lag{lag}"] = grp["conversion_value"].shift(lag)

    # ── Encodage catégoriel ──────────────────────────────────────────────────
    cat_cols = ["campaign_objective","campaign_status","budget_type","device",
                "location","ad_format","primary_text","age","gender",
                "network","keyword","match_type","search_term"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].replace(NA_PLATFORM, "__missing__").fillna("__missing__")
    all_cat = [c for c in cat_cols if c in df.columns]
    if all_cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1, encoded_missing_value=-2)
        df[all_cat] = enc.fit_transform(df[all_cat].astype(str))
        for col in ["network","keyword","match_type","search_term"]:
            if col in df.columns and f"has_{col}" in df.columns:
                df.loc[df[f"has_{col}"] == 0, col] = -1

    # ── Cross-plateforme → 0 (nouvelle campagne) ─────────────────────────────
    for c in feature_cols:
        if c.startswith("x_") and c not in df.columns:
            df[c] = 0.0

    # ── Identifier les HIGH par plateforme ──────────────────────────────────
    seg      = pd.read_csv(INPUT_SEG)
    high_ids = set(zip(
        seg[seg["performance_label"] == "HIGH_PERFORMANCE"]["global_campaign_id"],
        seg[seg["performance_label"] == "HIGH_PERFORMANCE"]["platform"]
    ))
    df["_is_high"] = df.apply(
        lambda r: (r["global_campaign_id"], r["platform"]) in high_ids, axis=1
    )
    df_high = df[df["_is_high"]].copy()
    log(f"  Lignes HIGH totales : {len(df_high):,}")

    # ── Vecteur base PAR PLATEFORME ──────────────────────────────────────────
    base_vectors = {}
    for plat in ["meta", "google"]:

        # Étape 1 — campagnes HIGH de cette plateforme (optimal)
        subset = df_high[df_high["platform"] == plat]

        # Étape 2 — pas de HIGH pour cette plateforme → toutes les lignes de cette plateforme
        if len(subset) == 0:
            log(f"  ⚠️  Pas de HIGH pour '{plat}' → fallback dataset plateforme")
            subset = df[df["platform"] == plat]

        # Étape 3 — pas de données du tout pour cette plateforme → fallback global
        if len(subset) == 0:
            log(f"  ⚠️  Pas de données '{plat}' → fallback global")
            subset = df_high

        available = [c for c in feature_cols if c in subset.columns]
        base      = subset[available].median().to_dict()
        for col in feature_cols:
            if col not in base or (isinstance(base.get(col), float) and np.isnan(base[col])):
                base[col] = 0.0

        base_vectors[plat] = base
        log(f"  Base {plat:6s} : {len(subset):,} lignes HIGH | "
            f"spend={base.get('spend',0):.0f}€ | "
            f"roas={base.get('roas',0):.3f}x | "
            f"cpc={base.get('cpc_calc',0):.3f}€ | "
            f"ctr={base.get('ctr_calc',0):.4f}")

    return base_vectors

# ============================================================
# ÉTAPE 3 — CONSTRUIRE LE VECTEUR PAR STRATÉGIE
# ============================================================
def build_strategy_vector(strategy: dict, base_vector: dict,
                           feature_cols: list, high_profile: dict,
                           hist_stats: dict) -> dict:
    """
    Copie le vecteur BASE (profil HIGH médian de la plateforme)
    et écrase toutes les variables propres à cette stratégie.

    Variables écrasées :
      A) Variables directes      : spend, CTR, CPC, clicks, impressions, conversions
      B) Variables business      : roas, cpa, ctr_calc, cpc_calc, conv_rate...
      C) Lags                    : valeurs actuelles avec légère décroissance (J0)
      D) Rolling                 : valeurs actuelles, std = 0
      E) Trends                  : 0 (nouvelle campagne)
      F) Cross-plateforme        : 0 (nouvelle campagne)
      G) Masques NA_platform     : selon plateforme réelle de la stratégie
      H) Encodages               : objectif, status, device
      I) Features temporelles    : contexte J15 de lancement

    Variables conservées du BASE :
      - Encodages catégoriels (location, ad_format...)
      - Features temporelles cycliques (dow_sin, month_cos...)
    """
    vec = base_vector.copy()

    # ── A) Variables directes ───────────────────────────────────────────────
    spend       = float(strategy["budget"])
    cpc         = float(strategy["CPC_cible"])
    ctr         = float(strategy["CTR_cible"])
    cr          = float(strategy["conversion_rate"])
    clicks      = float(strategy["clicks_est"])
    impressions = float(strategy["impressions_est"])
    conversions = float(strategy["conversions_est"])
    platform    = strategy.get("plateforme", "meta")
    objectif    = strategy.get("objectif",   "conversions")
    produit     = strategy.get("produit",    "autre")

    # ROAS dynamique — valeur par conversion cohérente avec Agent 2
    value_per_conv = float(strategy.get("val_per_conversion") or get_value_per_conversion(produit))
    cpc_mult_prod  = get_cpc_mult(produit)           # Fix 3 — CPC adapté au produit
    roas_est = (conversions * value_per_conv) / spend if spend > 0 else 0.0

    vec["spend"]           = spend
    vec["CTR"]             = ctr
    vec["CPC"]             = cpc * cpc_mult_prod     # Fix 3 — CPC × produit
    vec["clicks"]          = clicks
    vec["impressions"]     = impressions
    vec["conversions"]     = conversions
    vec["daily_budget"]    = spend / 30
    vec["lifetime_budget"] = spend * 3
    vec["CPM"]             = safe_div(spend * 1000, impressions)

    # ── B) Variables business ────────────────────────────────────────────────
    cpa_val                   = safe_div(spend, conversions)
    vec["roas"]               = roas_est
    vec["ctr_calc"]           = safe_div(clicks, impressions)
    vec["cpc_calc"]           = safe_div(spend, clicks) * cpc_mult_prod  # Fix 3
    vec["conv_rate"]          = cr
    vec["cpa"]                = cpa_val
    vec["cpm_calc"]           = safe_div(spend * 1000, impressions)
    vec["budget_utilization"] = min(1.0, safe_div(spend, vec["daily_budget"]))
    vec["revenue_per_click"]  = safe_div(roas_est * spend, clicks)
    vec["ctr_delta"]          = abs(ctr - vec["ctr_calc"])

    # Fix 6 — Feature efficiency (conv/€) — cohérent avec Agent 2 v4
    vec["efficiency"]         = safe_div(conversions, spend)

    # ── C) Lags — Fix 5 : décroissance aléatoire réaliste (≠ linéaire) ──────
    for lag in [1, 3, 7]:
        decay = float(np.random.uniform(0.85, 0.98))   # Fix 5 — réaliste
        vec[f"spend_lag{lag}"]            = spend       * decay
        vec[f"clicks_lag{lag}"]           = clicks      * decay
        vec[f"impressions_lag{lag}"]      = impressions * decay
        vec[f"conversions_lag{lag}"]      = conversions * decay
        vec[f"roas_lag{lag}"]             = roas_est
        vec[f"cpa_lag{lag}"]              = cpa_val
        vec[f"ctr_calc_lag{lag}"]         = vec["ctr_calc"]
        vec[f"cpc_calc_lag{lag}"]         = vec["cpc_calc"]
        vec[f"conv_rate_lag{lag}"]        = cr
        vec[f"conversion_value_lag{lag}"] = roas_est * spend

    # ── D) Rolling (stable au démarrage, std = 0) ────────────────────────────
    for w in [7, 14, 30]:
        for m, val in [("spend",      spend),
                       ("clicks",     clicks),
                       ("impressions",impressions),
                       ("conversions",conversions),
                       ("roas",       roas_est),
                       ("cpa",        cpa_val),
                       ("ctr_calc",   vec["ctr_calc"]),
                       ("cpc_calc",   vec["cpc_calc"]),
                       ("conv_rate",  cr)]:
            vec[f"{m}_roll{w}m"] = val
            vec[f"{m}_roll{w}s"] = 0.0

    # ── E) Trends = 0 (pas de tendance pour une nouvelle campagne) ────────────
    for m in ["spend","clicks","impressions","conversions","roas",
              "cpa","ctr_calc","cpc_calc","conv_rate"]:
        vec[f"{m}_trend7"] = 0.0

    # ── F) Cross-plateforme = 0 ───────────────────────────────────────────────
    for c in feature_cols:
        if c.startswith("x_"):
            vec[c] = 0.0

    # ── G) Masques NA_platform selon la plateforme ───────────────────────────
    is_meta   = (platform == "meta")
    is_google = (platform == "google")

    meta_cols   = ["reach","frequency","link_clicks","likes","comments",
                   "shares","post_engagement","video_views","add_to_cart","purchases"]
    google_cols = ["network","keyword","match_type","search_term","quality_score"]

    for col in meta_cols:
        vec[f"has_{col}"] = 1.0 if is_meta else 0.0
        vec[col]          = (clicks * 0.9 if col == "link_clicks" else 0.0) if is_meta else 0.0

    for col in google_cols:
        vec[f"has_{col}"] = 1.0 if is_google else 0.0
        vec[col]          = 1.0 if is_google else -1.0

    # ── H) Encodages — objectif utilisé PARTOUT dans le vecteur ML ─────────
    # AXE 1 — cohérent avec Agent 2 : objectif pilote le comportement ML
    vec["campaign_objective"] = float(OBJECTIF_ENC.get(objectif, 0))
    vec["campaign_status"]    = 0.0   # ACTIVE
    vec["budget_type"]        = 0.0   # daily
    vec["device"]             = 0.0   # mobile
    vec["age"]                = 1.0 if is_meta else -1.0
    vec["gender"]             = 1.0 if is_meta else -1.0

    # ── H2) Ajustements business selon l'objectif ────────────────────────────
    # Chaque objectif a un comportement ML différent :
    #   awareness   → impressions max, CTR boosté, conversions moins critiques
    #   traffic     → clicks max, CPC réduit, conversions secondaires
    #   leads       → conv_rate boosté, CPA critique
    #   conversions → ROAS et CR prioritaires (défaut)

    OBJECTIF_ADJUSTMENTS = {
        "awareness"  : {"ctr_mult": 1.30, "conv_weight": 0.50, "reach_focus": True},
        "traffic"    : {"ctr_mult": 1.15, "conv_weight": 0.70, "reach_focus": False},
        "leads"      : {"ctr_mult": 0.90, "conv_weight": 1.40, "reach_focus": False},
        "conversions": {"ctr_mult": 1.00, "conv_weight": 1.00, "reach_focus": False},
    }
    adj = OBJECTIF_ADJUSTMENTS.get(objectif, OBJECTIF_ADJUSTMENTS["conversions"])

    # Appliquer les ajustements dans le vecteur ML
    # → le modèle verra des valeurs cohérentes avec l'objectif réel
    vec["ctr_calc"]           = vec["ctr_calc"]   * adj["ctr_mult"]
    vec["CTR"]                = vec["CTR"]         * adj["ctr_mult"]
    vec["conv_rate"]          = cr                 * adj["conv_weight"]
    vec["conversions"]        = conversions        * adj["conv_weight"]

    # Recalculer les dérivées après ajustement objectif
    vec["roas"]               = (vec["conversions"] * value_per_conv) / spend if spend > 0 else 0.0
    vec["cpa"]                = safe_div(spend, vec["conversions"])
    vec["revenue_per_click"]  = safe_div(vec["roas"] * spend, clicks)

    # Reach score pour awareness (impressions / budget)
    if adj["reach_focus"]:
        vec["impressions"]    = impressions * 1.30   # portée augmentée
        vec["CPM"]            = safe_div(spend * 1000, vec["impressions"])

    # Fix 2 — Bruit réaliste ±5% : évite données trop parfaites → ML plus robuste
    noise            = float(np.random.normal(1.0, 0.05))
    noise            = max(0.85, min(1.15, noise))   # borné [0.85, 1.15]
    vec["ctr_calc"] *= noise
    vec["cpc_calc"] *= noise
    vec["conv_rate"] = max(0.001, vec["conv_rate"] * noise)

    # ── I) Features temporelles (contexte J15 de lancement) ──────────────────
    vec["campaign_age_days"] = 15.0
    vec["days_to_end"]       = 75.0

    return vec

# ============================================================
# ÉTAPE 4 — STATS HISTORIQUES (pour rapport comparatif)
# ============================================================
def compute_hist_stats(raw_df: pd.DataFrame) -> dict:
    for col in ["CTR","CPC","clicks","conversions"]:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].replace(NA_PLATFORM, np.nan)
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    clean = raw_df[raw_df["clicks"].fillna(0) > 0].copy()
    return {
        "hist_mean_CPC": float(raw_df["CPC"].mean()),
        "hist_mean_CTR": float(raw_df["CTR"].mean()),
        "hist_mean_CR" : float(
            (clean["conversions"] / clean["clicks"].replace(0, np.nan)).mean()
        ),
    }


def get_high_profile(profiles_df: pd.DataFrame) -> dict:
    row = profiles_df[profiles_df["label"] == "HIGH_PERFORMANCE"].iloc[0]
    return {k: float(row[k]) for k in ["CPC","CTR","conversion_rate","ROAS"]}

# ============================================================
# ÉTAPE 5 — FEATURES COMPARATIVES (pour rapport / XAI)
# ============================================================
def add_comparative_features(vec: dict, strategy: dict,
                              high: dict, hist: dict) -> dict:
    """
    Calcule ratios et deltas vs HIGH_PERFORMANCE et vs historique global.
    Non incluses dans les 187 features du modèle — utiles pour rapport XAI.
    """
    cpc = float(strategy["CPC_cible"])
    ctr = float(strategy["CTR_cible"])
    cr  = float(strategy["conversion_rate"])

    vec["ratio_CPC_vs_high"] = safe_div(cpc, high["CPC"])
    vec["ratio_CTR_vs_high"] = safe_div(ctr, high["CTR"])
    vec["ratio_CR_vs_high"]  = safe_div(cr,  high["conversion_rate"])
    vec["ratio_CPC_vs_hist"] = safe_div(cpc, hist["hist_mean_CPC"])
    vec["ratio_CTR_vs_hist"] = safe_div(ctr, hist["hist_mean_CTR"])
    vec["ratio_CR_vs_hist"]  = safe_div(cr,  hist["hist_mean_CR"])
    vec["delta_CPC_vs_high"] = cpc - high["CPC"]
    vec["delta_CTR_vs_high"] = ctr - high["CTR"]
    vec["delta_CR_vs_high"]  = cr  - high["conversion_rate"]
    vec["roas_est"]  = safe_div(
        float(strategy["conversions_est"]) * float(
            strategy.get("val_per_conversion")
            or get_value_per_conversion(strategy.get("produit", "autre"))
        ),
        float(strategy["budget"])
    )
    vec["score_agent2"]  = float(strategy.get("score_potentiel", 0))
    vec["strategy_id"]   = strategy["id"]
    vec["strategy_type"] = strategy["type"]
    vec["_plat"]         = strategy.get("plateforme", "meta")
    return vec

# ============================================================
# ÉTAPE 6 — VISUALISATION RADAR
# ============================================================
def plot_radar(rows_extras: list):
    COLORS = ["#E74C3C","#2ECC71","#3498DB","#F39C12","#9B59B6","#1ABC9C"]
    keys   = ["spend","cpc_calc","ctr_calc","conv_rate","roas_est",
              "ratio_CPC_vs_high","ratio_CR_vs_high","score_agent2"]
    labels = ["Budget","CPC","CTR","CR","ROAS","r_CPC/H","r_CR/H","Score"]

    # Filtrer les clés disponibles
    available = [k for k in keys if k in rows_extras[0]] if rows_extras else []
    if len(available) < 3:
        log("  ⚠️  Pas assez de données pour le radar")
        return

    df_plot = pd.DataFrame([{k: r.get(k, 0) for k in available + ["strategy_id","strategy_type"]}
                             for r in rows_extras])

    norm = df_plot[available].copy().astype(float)
    for col in available:
        rng = norm[col].max() - norm[col].min()
        norm[col] = (norm[col] - norm[col].min()) / rng if rng > 0 else 0.5
    for col in ["cpc_calc","ratio_CPC_vs_high"]:
        if col in norm.columns:
            norm[col] = 1 - norm[col]

    N      = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    lbls   = [labels[keys.index(k)] if k in keys else k for k in available]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_title("Profils des Stratégies — Feature Engineering PRO\n"
                 "(CPC inversé — plus haut = mieux)",
                 fontsize=11, fontweight="bold", pad=20)

    for i, row in df_plot.iterrows():
        vals  = norm.loc[i, available].tolist() + [norm.loc[i, available[0]]]
        color = COLORS[i % len(COLORS)]
        label = f"[{row['strategy_id']}] {row['strategy_type']}"
        ax.plot(angles, vals, color=color, linewidth=2.2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), lbls, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.1), fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_RADAR, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Radar : {OUTPUT_RADAR}")

# ============================================================
# ÉTAPE 7 — SAUVEGARDE
# ============================================================
def save_results(df_ml: pd.DataFrame, rows_extras: list,
                 feature_cols: list, high: dict, hist: dict):
    log(sep()); log("ÉTAPE 7 — SAUVEGARDE"); log(sep())

    df_ml.to_csv(OUTPUT_FEATURES, index=False)
    log(f"  Features CSV : {OUTPUT_FEATURES}")
    log(f"  Shape        : {df_ml.shape}  ({df_ml.shape[0]} stratégies × {df_ml.shape[1]} features) ✅")

    lines = [
        sep(), "RAPPORT FEATURE ENGINEERING PRO — AdOptimizer AI", sep(),
        f"Modèle   : {MODEL_PATH}",
        f"Features : {len(feature_cols)} colonnes exactes",
        f"Stratégies : {len(rows_extras)}",
        "",
        sep("-"), "PROFIL HIGH_PERFORMANCE (référence)", sep("-"),
        f"  CPC  : {high['CPC']:.4f}€",
        f"  CTR  : {high['CTR']:.5f}",
        f"  CR   : {high['conversion_rate']:.5f}",
        f"  ROAS : {high['ROAS']:.4f}x",
        "",
        sep("-"), "STATS HISTORIQUES", sep("-"),
        f"  CPC moyen  : {hist['hist_mean_CPC']:.4f}€",
        f"  CTR moyen  : {hist['hist_mean_CTR']:.5f}",
        f"  CR  moyen  : {hist['hist_mean_CR']:.5f}",
        "",
        sep("-"), "FEATURES COMPARATIVES PAR STRATÉGIE", sep("-"),
    ]
    for r in rows_extras:
        cpc_flag = "← mieux que HIGH" if r.get("ratio_CPC_vs_high", 1) < 1 else ""
        cr_flag  = "← mieux que HIGH" if r.get("ratio_CR_vs_high",  1) > 1 else ""
        lines += [
            f"\n[{r['strategy_id']}] {r['strategy_type'].upper()} — "
            f"Budget={r.get('spend',0):,.0f}€",
            f"  CPC      : {r.get('cpc_calc',0):.4f}€  "
            f"(ratio_vs_HIGH={r.get('ratio_CPC_vs_high',1):.3f}) {cpc_flag}",
            f"  CTR      : {r.get('ctr_calc',0):.5f}  "
            f"(ratio_vs_HIGH={r.get('ratio_CTR_vs_high',1):.3f})",
            f"  conv_rate: {r.get('conv_rate',0):.5f}  "
            f"(ratio_vs_HIGH={r.get('ratio_CR_vs_high',1):.3f}) {cr_flag}",
            f"  roas_est : {r.get('roas_est',0):.4f}x",
            f"  score_a2 : {r.get('score_agent2',0):.2f}",
        ]
    lines += ["", sep(), "FIN DU RAPPORT", sep()]
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  Rapport      : {OUTPUT_REPORT}")

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def run_feature_engineering_pro():
    log(sep("=")); log("  FEATURE ENGINEERING PRO — AdOptimizer AI (Cas 2)"); log(sep("="))
    log("  Objectif : produire N × 187 features exactes pour best_model.pkl")
    log("  Générique : meta / google / both")
    log("")

    # 1. Charger le bundle
    bundle       = load_bundle()
    feature_cols = bundle["feature_cols"]

    # 2. Vecteurs base par plateforme
    base_vectors = build_base_vectors_from_high(feature_cols)
    base_global  = next(iter(base_vectors.values()))   # fallback

    # 3. Stats historiques + profil HIGH
    log(sep()); log("ÉTAPE 3 — STATS HISTORIQUES & PROFIL HIGH"); log(sep())
    raw_df      = pd.read_csv(INPUT_RAW, low_memory=False)
    profiles_df = pd.read_csv(INPUT_PROFILES)
    hist        = compute_hist_stats(raw_df)
    high        = get_high_profile(profiles_df)
    log(f"  hist CPC : {hist['hist_mean_CPC']:.4f}  |  HIGH CPC  : {high['CPC']:.4f}")
    log(f"  hist CTR : {hist['hist_mean_CTR']:.5f}  |  HIGH CTR  : {high['CTR']:.5f}")
    log(f"  hist CR  : {hist['hist_mean_CR']:.5f}  |  HIGH CR   : {high['conversion_rate']:.5f}")

    # 4. Construire les vecteurs par stratégie
    log(sep()); log("ÉTAPE 4 — CONSTRUCTION DES VECTEURS PAR STRATÉGIE"); log(sep())
    with open(INPUT_STRATEGIES, encoding="utf-8") as f:
        strategies = json.load(f)["strategies"]
    log(f"  Stratégies à traiter : {len(strategies)}")

    rows_ml, rows_extras = [], []
    for s in strategies:
        sid  = s["id"]
        plat = s.get("plateforme", "meta")
        base = base_vectors.get(plat, base_global)

        log(f"\n  [{sid}] {s['type'].upper()} — Plateforme={plat}  "
            f"Budget={s['budget']:,.0f}€  CPC={s['CPC_cible']:.4f}  "
            f"CTR={s['CTR_cible']:.4f}  CR={s['conversion_rate']:.4f}")

        vec_ml    = build_strategy_vector(s, base, feature_cols, high, hist)
        vec_extra = add_comparative_features(dict(vec_ml), s, high, hist)
        vec_extra["_plat"] = plat   # Fix 4 — stocker la plateforme pour affichage

        rows_ml.append({col: vec_ml.get(col, 0.0) for col in feature_cols})
        rows_extras.append(vec_extra)

        log(f"    roas_est         : {vec_extra.get('roas_est',0):.4f}x")
        log(f"    ratio_CPC_vs_HIGH: {vec_extra.get('ratio_CPC_vs_high',1):.4f}"
            f"  {'← mieux que HIGH ✅' if vec_extra.get('ratio_CPC_vs_high',1) < 1 else ''}")
        log(f"    ratio_CR_vs_HIGH : {vec_extra.get('ratio_CR_vs_high',1):.4f}"
            f"  {'← mieux que HIGH ✅' if vec_extra.get('ratio_CR_vs_high',1) > 1 else ''}")

    # 5. DataFrame final
    df_ml = pd.DataFrame(rows_ml, columns=feature_cols).fillna(0.0)
    log(f"\n  Shape finale : {df_ml.shape} ✅  ({df_ml.shape[0]} × {len(feature_cols)})")

    # 6. Visualisation
    log(sep()); log("ÉTAPE 5 — VISUALISATION"); log(sep())
    plot_radar(rows_extras)

    # 7. Sauvegarde
    save_results(df_ml, rows_extras, feature_cols, high, hist)

    # Résumé
    log(sep("=")); log("  RÉSUMÉ FINAL"); log(sep("="))
    log(f"  Features ML  : {df_ml.shape[1]} colonnes ✅ (compatibles best_model.pkl)")
    log(f"  Stratégies   : {df_ml.shape[0]}")
    log(f"\n  {'ID':<6} {'Type':<12} {'Plat':<8} {'ROAS_est':>10} "
        f"{'r_CPC/H':>10} {'r_CR/H':>10}")
    log(f"  {'-'*58}")
    for r in rows_extras:
        log(f"  [{r['strategy_id']:<4}] {r['strategy_type']:<12} "
            f"{r.get('_plat','?'):<8} "
            f"{r.get('roas_est',0):>10.4f} "
            f"{r.get('ratio_CPC_vs_high',1):>10.4f} "
            f"{r.get('ratio_CR_vs_high',1):>10.4f}")
    log(f"\n  Outputs :")
    for fp in [OUTPUT_FEATURES, OUTPUT_REPORT, OUTPUT_RADAR]:
        log(f"    → {fp}")
    log(sep("=")); log("  FEATURE ENGINEERING PRO TERMINÉ"); log(sep("="))

    return df_ml, rows_extras


# ============================================================
# WRAPPER FASTAPI / N8N
# ============================================================

def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def run_case2_feature_engineering() -> dict:
    """
    Point d'entree FastAPI pour le cas 2 : feature engineering des strategies.

    Dependances :
      app/cas2-outputs/agent2_outputs/strategies.json
      app/cas2-outputs/segmentation_outputs/
      app/data/dataset_historique.csv
      app/models/best_model.pkl
    """
    try:
        df_ml, rows_extras = run_feature_engineering_pro()
        compact_rows = [
            {
                "strategy_id": r.get("strategy_id"),
                "strategy_type": r.get("strategy_type"),
                "platform": r.get("_plat"),
                "roas_est": r.get("roas_est"),
                "score_agent2": r.get("score_agent2"),
                "ratio_CPC_vs_high": r.get("ratio_CPC_vs_high"),
                "ratio_CR_vs_high": r.get("ratio_CR_vs_high"),
            }
            for r in rows_extras
        ]

        return {
            "status": "success",
            "message": "Feature engineering cas 2 termine",
            "input_files": {
                "strategies": str(INPUT_STRATEGIES),
                "profiles": str(INPUT_PROFILES),
                "segmentation": str(INPUT_SEG),
                "historical_dataset": str(INPUT_RAW),
                "model": str(MODEL_PATH),
            },
            "output_dir": str(OUTPUT_DIR),
            "output_files": {
                "features": str(OUTPUT_FEATURES),
                "report": str(OUTPUT_REPORT),
                "radar": str(OUTPUT_RADAR),
            },
            "rows": int(df_ml.shape[0]),
            "features": int(df_ml.shape[1]),
            "strategies": _json_safe(compact_rows),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input_files": {
                "strategies": str(INPUT_STRATEGIES),
                "profiles": str(INPUT_PROFILES),
                "segmentation": str(INPUT_SEG),
                "historical_dataset": str(INPUT_RAW),
                "model": str(MODEL_PATH),
            },
        }


# ============================================================
if __name__ == "__main__":
    result = run_case2_feature_engineering()
    print(result)
