"""
===============================================================================
TOOL 1 — PRÉTRAITEMENT & FEATURE ENGINEERING
AdOptimizer AI — Prédicteur de Performance (Tool 4)
===============================================================================
Input  : raw_ads_dataset_strict.csv (sortie du générateur brut)
Output : dataset_model_ready.csv    (prêt pour XGBoost / LightGBM / RF)

PIPELINE :
  Phase 0  — Chargement & audit qualité
  Phase 1  — Déduplication (doublons API simulés)
  Phase 2  — Séparation NA_platform vs NaN réels
  Phase 3  — Corrections de type (quality_score string → int)
  Phase 4  — Imputation des NaN réels (médiane par campagne/plateforme)
  Phase 5  — Calcul des variables business (CTR, CPC, CPA, ROAS…)
  Phase 6  — Feature engineering temporel (lags, rolling, cyclique)
  Phase 7  — Encodage des variables catégorielles
  Phase 8  — Construction des cibles multi-horizons (J+3, J+7, J+14)
  Phase 9  — Features cross-plateforme
  Phase 10 — Nettoyage final & export

CIBLES PRODUITES (5 métriques × 3 horizons = 15 colonnes target) :
  target_{metric}_h{horizon}
  metric  ∈ {roas, conversions, cpa, ctr, cpc}
  horizon ∈ {3, 7, 14}

CONTRAINTES RESPECTÉES :
  ✅ NA_platform traité différemment des NaN réels
  ✅ Aucune fuite temporelle (targets calculées par shift forward)
  ✅ Variables business calculées APRÈS imputation
  ✅ Masques binaires pour variables inexistantes sur une plateforme
  ✅ Encodages compatibles tree-based (OrdinalEncoder, pas OHE)
  ✅ Colonnes identifiantes conservées mais exclues du training set
  ✅ CPA(conv=0) imputé par p95 campagne (pas spend*3)
  ✅ conversion_value (J courant) exclue des features — lags uniquement
  ✅ Perte des derniers 14j par campagne documentée et acceptée

NOTE CRITIQUE — REPRODUCTIBILITÉ :
  Ce script est 100% déterministe (aucune aléatoire).
  Même raw CSV → même dataset_model_ready.csv garanti,
  indépendamment de l'environnement (Colab ou local).
  ⚠️  NE PAS regénérer le raw CSV dans des environnements différents :
      les versions NumPy/Python différentes produisent des séquences
      aléatoires différentes même avec le même seed.
      → Générer une seule fois sur Colab, copier dans app/data/, versionner.
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION — CHEMINS LOCAUX
# ============================================================
NA_PLATFORM    = "NA_platform"
HORIZONS       = [3, 7, 14]
TARGET_METRICS = ["roas", "conversions", "cpa", "ctr", "cpc"]

BASE_DIR    = Path(__file__).resolve().parent.parent  # → app/
DATA_DIR    = BASE_DIR / "data"

INPUT_FILE  = DATA_DIR / "raw_ads_dataset_strict.csv"
OUTPUT_FILE = DATA_DIR / "dataset_model_ready.csv"
AUDIT_FILE  = DATA_DIR / "preprocessing_audit.txt"

# Colonnes identifiantes
ID_COLS = [
    "global_campaign_id", "campaign_id", "adset_id", "ad_id",
    "date", "platform", "start_date", "end_date"
]

# Colonnes numériques brutes API
NUMERIC_RAW = [
    "spend", "impressions", "clicks", "conversions",
    "conversion_value", "CTR", "CPC", "CPM",
    "daily_budget", "lifetime_budget"
]

# Colonnes Meta-only
META_NUMERIC = [
    "reach", "frequency", "link_clicks", "likes",
    "comments", "shares", "post_engagement",
    "video_views", "add_to_cart", "purchases"
]

# Colonnes Google-only
GOOGLE_CAT = ["network", "keyword", "match_type", "search_term"]
GOOGLE_NUM = ["quality_score"]

# Colonnes catégorielles communes
CAT_COLS = [
    "campaign_objective", "campaign_status", "budget_type",
    "device", "location", "ad_format", "primary_text",
    "age", "gender"
]


# ============================================================
# UTILITAIRES
# ============================================================

def log(msg: str, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")


def safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    return num.div(den.replace(0, np.nan)).fillna(fill)


# ============================================================
# PHASE 0 — CHARGEMENT & AUDIT QUALITÉ INITIAL
# ============================================================

def phase0_load_and_audit(input_path: str, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 0 — CHARGEMENT & AUDIT QUALITÉ INITIAL", audit_f)
    log("=" * 72, audit_f)

    if not os.path.exists(input_path):
        log(f"❌ Fichier introuvable : {input_path}", audit_f)
        log("   → Copier raw_ads_dataset_strict.csv dans app/data/", audit_f)
        sys.exit(1)

    df = pd.read_csv(input_path, dtype=str, low_memory=False)
    log(f"Lignes chargées       : {len(df):,}", audit_f)
    log(f"Colonnes              : {len(df.columns)}", audit_f)
    log(f"Plateformes           : {df['platform'].value_counts().to_dict()}", audit_f)
    log(f"Campagnes globales    : {df['global_campaign_id'].nunique()}", audit_f)
    log(f"Période               : {df['date'].min()} → {df['date'].max()}", audit_f)

    na_plat_counts = {
        col: (df[col] == NA_PLATFORM).sum()
        for col in df.columns
        if (df[col] == NA_PLATFORM).sum() > 0
    }
    log(f"\nColonnes NA_platform  :", audit_f)
    for col, n in na_plat_counts.items():
        log(f"  {col:22s} : {n:,}", audit_f)

    real_nan = df.isnull().sum()
    real_nan = real_nan[real_nan > 0]
    log(f"\nNaN réels par colonne :", audit_f)
    for col, n in real_nan.items():
        log(f"  {col:22s} : {n:,} ({n / len(df) * 100:.2f}%)", audit_f)

    return df


# ============================================================
# PHASE 1 — DÉDUPLICATION
# ============================================================

def phase1_dedup(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 1 — DÉDUPLICATION", audit_f)
    log("=" * 72, audit_f)

    n_before = len(df)
    dedup_keys = ["campaign_id", "adset_id", "ad_id", "date", "platform"]
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    n_after = len(df)
    log(f"Doublons supprimés    : {n_before - n_after:,}", audit_f)
    log(f"Lignes restantes      : {n_after:,}", audit_f)
    return df.reset_index(drop=True)


# ============================================================
# PHASE 2 — SÉPARATION NA_PLATFORM vs NaN RÉELS
# ============================================================

def phase2_na_platform_masks(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 2 — MASQUES NA_PLATFORM", audit_f)
    log("=" * 72, audit_f)

    platform_cols = META_NUMERIC + GOOGLE_CAT + GOOGLE_NUM

    for col in platform_cols:
        if col not in df.columns:
            continue
        mask_col = f"has_{col}"
        df[mask_col] = (df[col] != NA_PLATFORM).astype(int)
        df[col] = df[col].replace(NA_PLATFORM, np.nan)
        log(f"  Masque créé : {mask_col} | 1={df[mask_col].sum():,} | "
            f"0={(df[mask_col] == 0).sum():,}", audit_f)

    return df


# ============================================================
# PHASE 3 — CORRECTIONS DE TYPE
# ============================================================

def phase3_cast_types(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 3 — CORRECTIONS DE TYPE", audit_f)
    log("=" * 72, audit_f)

    num_cols_all = NUMERIC_RAW + META_NUMERIC + GOOGLE_NUM

    for col in num_cols_all:
        if col not in df.columns:
            continue
        before_nulls = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_nulls = df[col].isnull().sum()
        new_nulls = after_nulls - before_nulls
        if new_nulls > 0:
            log(f"  {col:22s} : {new_nulls} valeurs non-parsables → NaN", audit_f)

    if "quality_score" in df.columns:
        df["quality_score"] = df["quality_score"].astype(float).round(0)
        qs_str_fixed = (df["quality_score"].notna()).sum()
        log(f"  quality_score casté : {qs_str_fixed:,} valeurs Google", audit_f)

    df["date"]       = pd.to_datetime(df["date"],       errors="coerce")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    log(f"  Dates parsées OK", audit_f)

    return df


# ============================================================
# PHASE 4 — IMPUTATION DES NaN RÉELS
# ============================================================

def phase4_impute(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 4 — IMPUTATION NaN RÉELS", audit_f)
    log("=" * 72, audit_f)

    num_to_impute = NUMERIC_RAW.copy()
    for col in META_NUMERIC + GOOGLE_NUM:
        if col in df.columns:
            num_to_impute.append(col)

    total_imputed = 0
    for col in num_to_impute:
        if col not in df.columns:
            continue

        mask_col = f"has_{col}"
        if mask_col in df.columns:
            eligible = df[mask_col] == 1
        else:
            eligible = pd.Series(True, index=df.index)

        n_missing = df.loc[eligible, col].isnull().sum()
        if n_missing == 0:
            continue

        # Niveau 1 : médiane par (campaign_id, platform)
        df.loc[eligible, col] = df.loc[eligible].groupby(
            ["campaign_id", "platform"])[col].transform(
            lambda x: x.fillna(x.median()))

        # Niveau 2 : médiane par (platform, campaign_objective)
        still_missing = eligible & df[col].isnull()
        if still_missing.sum() > 0:
            df.loc[still_missing, col] = df.loc[still_missing].groupby(
                ["platform", "campaign_objective"])[col].transform(
                lambda x: x.fillna(x.median()))

        # Niveau 3 : médiane globale par plateforme
        still_missing = eligible & df[col].isnull()
        if still_missing.sum() > 0:
            df.loc[still_missing, col] = df.loc[still_missing].groupby(
                "platform")[col].transform(lambda x: x.fillna(x.median()))

        # Niveau 4 : 0 absolu
        still_missing = eligible & df[col].isnull()
        if still_missing.sum() > 0:
            df.loc[still_missing, col] = 0.0

        n_imputed = n_missing - df.loc[eligible, col].isnull().sum()
        total_imputed += n_imputed
        log(f"  {col:22s} : {n_missing:,} NaN → {n_imputed:,} imputés", audit_f)

    log(f"\nTotal NaN imputés     : {total_imputed:,}", audit_f)
    return df


# ============================================================
# PHASE 5 — CALCUL VARIABLES BUSINESS
# ============================================================

def phase5_business_vars(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 5 — VARIABLES BUSINESS CALCULÉES", audit_f)
    log("=" * 72, audit_f)

    df["roas"] = safe_div(df["conversion_value"], df["spend"], fill=0.0)

    df["cpa"] = safe_div(df["spend"], df["conversions"], fill=np.nan)
    p95_camp   = df.groupby("campaign_id")["cpa"].transform(lambda x: x.quantile(0.95))
    p95_plat   = df.groupby(["platform", "campaign_objective"])["cpa"].transform(lambda x: x.quantile(0.95))
    p95_global = df["cpa"].quantile(0.95)
    df["cpa"]  = df["cpa"].fillna(p95_camp).fillna(p95_plat).fillna(p95_global)

    df["ctr_calc"]          = safe_div(df["clicks"], df["impressions"], fill=0.0)
    df["cpc_calc"]          = safe_div(df["spend"], df["clicks"], fill=0.0)
    df["conv_rate"]         = safe_div(df["conversions"], df["clicks"], fill=0.0)
    df["cpm_calc"]          = safe_div(df["spend"] * 1000, df["impressions"], fill=0.0)
    df["budget_utilization"]= safe_div(df["spend"], df["daily_budget"], fill=0.0).clip(0, 1)
    df["revenue_per_click"] = safe_div(df["conversion_value"], df["clicks"], fill=0.0)
    df["ctr_delta"]         = (df["CTR"] - df["ctr_calc"]).abs()

    log("  Variables créées : roas, cpa, ctr_calc, cpc_calc, conv_rate,", audit_f)
    log("                     cpm_calc, budget_utilization, revenue_per_click,", audit_f)
    log("                     ctr_delta", audit_f)
    log(f"\n  ROAS médian (global)       : {df['roas'].median():.3f}", audit_f)
    log(f"  CPA médian (global)        : {df['cpa'].median():.2f}", audit_f)
    log(f"  Conv_rate médian (global)  : {df['conv_rate'].median():.4f}", audit_f)

    return df


# ============================================================
# PHASE 6 — FEATURE ENGINEERING TEMPOREL
# ============================================================

def phase6_temporal_features(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 6 — FEATURE ENGINEERING TEMPOREL", audit_f)
    log("=" * 72, audit_f)

    df = df.sort_values(["campaign_id", "platform", "date"]).reset_index(drop=True)

    # A) Features de date
    df["day_of_week"]       = df["date"].dt.dayofweek
    df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"]      = df["date"].dt.isocalendar().week.astype(int)
    df["month"]             = df["date"].dt.month
    df["quarter"]           = df["date"].dt.quarter
    df["day_of_year"]       = df["date"].dt.dayofyear
    df["campaign_age_days"] = (df["date"] - df["start_date"]).dt.days.clip(lower=0)
    df["days_to_end"]       = (df["end_date"] - df["date"]).dt.days.clip(lower=0)

    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

    log("  Features date : day_of_week, is_weekend, week_of_year, month, quarter,", audit_f)
    log("                  campaign_age_days, days_to_end, encodages sin/cos", audit_f)

    # B) Lags & C) Rolling
    lag_metrics     = ["spend", "clicks", "conversions", "roas",
                       "cpa", "ctr_calc", "cpc_calc", "conv_rate", "impressions"]
    lag_windows     = [1, 3, 7]
    rolling_windows = [7, 14, 30]

    group = df.groupby(["campaign_id", "platform"])
    features_created = []

    for metric in lag_metrics:
        if metric not in df.columns:
            continue

        for lag in lag_windows:
            col_name = f"{metric}_lag{lag}"
            df[col_name] = group[metric].shift(lag)
            features_created.append(col_name)

        for window in rolling_windows:
            col_mean = f"{metric}_roll{window}m"
            col_std  = f"{metric}_roll{window}s"
            df[col_mean] = group[metric].transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window // 3)).mean()
            )
            df[col_std] = group[metric].transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window // 3)).std().fillna(0)
            )
            features_created.extend([col_mean, col_std])

        col_trend = f"{metric}_trend7"
        baseline  = group[metric].transform(
            lambda x: x.shift(7).rolling(3, min_periods=1).mean()
        )
        df[col_trend] = safe_div(df[metric] - baseline, baseline.abs().replace(0, np.nan), fill=0.0)
        features_created.append(col_trend)

    log(f"\n  Lags créés    : {len(lag_metrics)} métriques × {len(lag_windows)} lags", audit_f)
    log(f"  Rolling créés : {len(lag_metrics)} métriques × {len(rolling_windows)} × 2 (mean+std)", audit_f)
    log(f"  Trends créés  : {len(lag_metrics)} métriques × 1 trend7", audit_f)

    # Lags conversion_value uniquement (pas valeur J courant)
    cv_lags_created = []
    if "conversion_value" in df.columns:
        for lag in lag_windows:
            col_name = f"conversion_value_lag{lag}"
            df[col_name] = group["conversion_value"].shift(lag)
            cv_lags_created.append(col_name)
        features_created.extend(cv_lags_created)
    log(f"  Lags conversion_value : {cv_lags_created}", audit_f)
    log(f"  (rolling/trend conversion_value exclus — trop colinéaires à target_roas)", audit_f)
    log(f"\n  Total features temporels : {len(features_created)}", audit_f)

    return df


# ============================================================
# PHASE 7 — ENCODAGE CATÉGORIEL
# ============================================================

def phase7_encode(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 7 — ENCODAGE CATÉGORIEL (OrdinalEncoder)", audit_f)
    log("=" * 72, audit_f)

    cat_to_encode      = [c for c in CAT_COLS if c in df.columns]
    google_cat_present = [c for c in GOOGLE_CAT if c in df.columns]

    for col in cat_to_encode + google_cat_present:
        df[col] = df[col].fillna("__missing__")

    all_cat = cat_to_encode + google_cat_present
    if all_cat:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2      # ✅ identique à Colab
        )
        df[all_cat] = enc.fit_transform(df[all_cat].astype(str))
        for col in google_cat_present:
            mask_col = f"has_{col}"
            if mask_col in df.columns:
                df.loc[df[mask_col] == 0, col] = -1

    log(f"  Colonnes encodées : {all_cat}", audit_f)
    return df


# ============================================================
# PHASE 8 — CONSTRUCTION DES CIBLES MULTI-HORIZONS
# ============================================================

def phase8_build_targets(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 8 — CONSTRUCTION DES CIBLES", audit_f)
    log("=" * 72, audit_f)

    target_source = {
        "roas"        : "roas",
        "conversions" : "conversions",
        "cpa"         : "cpa",
        "ctr"         : "ctr_calc",
        "cpc"         : "cpc_calc",
    }

    df    = df.sort_values(["campaign_id", "platform", "date"]).reset_index(drop=True)
    group = df.groupby(["campaign_id", "platform"])

    for metric, source_col in target_source.items():
        if source_col not in df.columns:
            log(f"  ⚠️  Colonne source manquante : {source_col}", audit_f)
            continue
        for h in HORIZONS:
            target_col = f"target_{metric}_h{h}"
            df[target_col] = group[source_col].shift(-h)
            n_valid = df[target_col].notna().sum()
            log(f"  {target_col:28s} : {n_valid:,} lignes valides", audit_f)

    return df


# ============================================================
# PHASE 9 — FEATURES CROSS-PLATEFORME
# ============================================================

def phase9_cross_platform(df: pd.DataFrame, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 9 — FEATURES CROSS-PLATEFORME", audit_f)
    log("=" * 72, audit_f)

    cross_metrics = ["roas", "spend", "conversions", "cpa"]
    cross_lags    = [1, 3, 7]

    for metric in cross_metrics:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(
            index=["global_campaign_id", "date"],
            columns="platform",
            values=metric,
            aggfunc="mean"
        ).reset_index()
        pivot.columns = ["global_campaign_id", "date"] + [
            f"x_{metric}_{c}" for c in pivot.columns[2:]
        ]

        for lag in cross_lags:
            for plat in ["meta", "google"]:
                src_col = f"x_{metric}_{plat}"
                if src_col not in pivot.columns:
                    continue
                lag_col = f"x_{metric}_{plat}_lag{lag}"
                pivot   = pivot.sort_values(["global_campaign_id", "date"])
                pivot[lag_col] = pivot.groupby("global_campaign_id")[src_col].shift(lag)

        merge_cols = ["global_campaign_id", "date"] + [
            c for c in pivot.columns if c.startswith("x_") and "lag" in c
        ]
        df = df.merge(pivot[merge_cols], on=["global_campaign_id", "date"], how="left")

    created = [c for c in df.columns if c.startswith("x_") and "lag" in c]
    log(f"  Features cross-plateforme créées : {len(created)}", audit_f)
    if created:
        log(f"  Exemple : {created[:4]}", audit_f)

    return df


# ============================================================
# PHASE 10 — NETTOYAGE FINAL & EXPORT
# ============================================================

def phase10_final_clean_export(df: pd.DataFrame, output_path: str, audit_f) -> pd.DataFrame:
    log("\n" + "=" * 72, audit_f)
    log("PHASE 10 — NETTOYAGE FINAL & EXPORT", audit_f)
    log("=" * 72, audit_f)

    target_cols    = [c for c in df.columns if c.startswith("target_")]
    has_any_target = df[target_cols].notna().any(axis=1)
    n_before       = len(df)
    df             = df[has_any_target].reset_index(drop=True)
    log(f"Lignes supprimées (0 target) : {n_before - len(df):,}", audit_f)
    log(f"Lignes finales               : {len(df):,}", audit_f)

    log(f"\nLignes valides par horizon (perte en fin de série = NORMALE) :", audit_f)
    n_total = len(df)
    for h in HORIZONS:
        h_cols    = [c for c in target_cols if c.endswith(f"_h{h}")]
        n_complet = df[h_cols].notna().all(axis=1).sum()
        n_perdu   = n_total - n_complet
        pct_perdu = n_perdu / n_total * 100
        log(f"  Horizon J+{h:<2d} ({len(h_cols)} targets) : "
            f"{n_complet:,} lignes complètes | "
            f"{n_perdu:,} sans target ({pct_perdu:.1f}%) ← derniers {h}j/campagne", audit_f)
    log(f"\n  ℹ️  La perte J+14 (~15% max) est attendue et correcte.", audit_f)
    log(f"     À l'entraînement : filtrer df[target_X_hY.notna()] par horizon.", audit_f)

    EXCLUDED_FROM_FEATURES = ID_COLS + target_cols + [
        "start_date", "end_date", "conversion_value"
    ]
    feature_cols = [c for c in df.columns if c not in EXCLUDED_FROM_FEATURES]

    log(f"\nFeatures totales : {len(feature_cols)}", audit_f)
    log(f"  (conversion_value J courant exclue — lags conservés)", audit_f)
    log(f"Targets totales  : {len(target_cols)}", audit_f)
    log(f"Colonnes ID      : {len(ID_COLS)}", audit_f)

    for col in feature_cols:
        if df[col].dtype == object:
            log(f"  ⚠️  Colonne feature encore object : {col}", audit_f)

    # Export CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"\n✅ Dataset exporté : {output_path}", audit_f)
    log(f"   {len(df):,} lignes × {len(df.columns)} colonnes", audit_f)

    # Dictionnaire des features
    feature_dict_path = str(Path(output_path).with_suffix("")) + "_features.txt"
    with open(feature_dict_path, "w", encoding="utf-8") as f:
        f.write("# COLONNES TRAINING (features uniquement, sans ID ni targets)\n\n")
        f.write("FEATURE_COLS = [\n")
        for c in feature_cols:
            f.write(f'    "{c}",\n')
        f.write("]\n\n")
        f.write("TARGET_COLS = [\n")
        for c in target_cols:
            f.write(f'    "{c}",\n')
        f.write("]\n\n")
        f.write("ID_COLS = [\n")
        for c in ID_COLS + ["start_date", "end_date"]:
            f.write(f'    "{c}",\n')
        f.write("]\n")
    log(f"   Dictionnaire features : {feature_dict_path}", audit_f)

    return df


# ============================================================
# PIPELINE COMPLET
# ============================================================

def run_pipeline(
    input_path: str  = str(INPUT_FILE),
    output_path: str = str(OUTPUT_FILE),
    audit_path: str  = str(AUDIT_FILE)
) -> pd.DataFrame:

    print("=" * 72)
    print("TOOL 1 — PIPELINE PRÉTRAITEMENT AdOptimizer AI")
    print("=" * 72)

    os.makedirs(os.path.dirname(audit_path), exist_ok=True)

    with open(audit_path, "w", encoding="utf-8") as audit_f:
        log(f"Pipeline démarré  : {pd.Timestamp.now()}", audit_f)
        log(f"Input             : {input_path}", audit_f)
        log(f"Output            : {output_path}", audit_f)

        df = phase0_load_and_audit(input_path, audit_f)
        df = phase1_dedup(df, audit_f)
        df = phase2_na_platform_masks(df, audit_f)
        df = phase3_cast_types(df, audit_f)
        df = phase4_impute(df, audit_f)
        df = phase5_business_vars(df, audit_f)
        df = phase6_temporal_features(df, audit_f)
        df = phase7_encode(df, audit_f)
        df = phase8_build_targets(df, audit_f)
        df = phase9_cross_platform(df, audit_f)
        df = phase10_final_clean_export(df, output_path, audit_f)

        log(f"\nPipeline terminé  : {pd.Timestamp.now()}", audit_f)
        log("=" * 72, audit_f)

    print(f"\n✅ Prétraitement terminé.")
    print(f"   Dataset prêt    : {output_path}")
    print(f"   Rapport audit   : {audit_path}")
    return df


# ============================================================
# WRAPPER FASTAPI / N8N
# ============================================================

def run_preprocessing(
    input_path: str  = str(INPUT_FILE),
    output_path: str = str(OUTPUT_FILE),
    audit_path: str  = str(AUDIT_FILE)
) -> dict:
    """
    Point d'entrée pour FastAPI et n8n.
    Accepte des chemins personnalisés pour les tests ou multi-datasets.
    """
    try:
        df = run_pipeline(input_path, output_path, audit_path)
        target_cols  = [c for c in df.columns if c.startswith("target_")]
        feature_cols = [
            c for c in df.columns
            if not c.startswith("target_") and c not in ID_COLS
               and c not in ["start_date", "end_date", "conversion_value"]
        ]
        return {
            "status"       : "success",
            "message"      : "Prétraitement Tool 1 terminé",
            "output_file"  : output_path,
            "rows"         : len(df),
            "columns"      : len(df.columns),
            "features"     : len(feature_cols),
            "targets"      : len(target_cols),
        }
    except Exception as e:
        return {
            "status"  : "error",
            "message" : str(e)
        }


# ============================================================
# POINT D'ENTRÉE SCRIPT DIRECT
# ============================================================

if __name__ == "__main__":
    df_ready = run_pipeline()
    print(f"\nDataset final : {len(df_ready):,} lignes × {len(df_ready.columns)} colonnes")
    print("\nAperçu colonnes targets :")
    target_cols = [c for c in df_ready.columns if c.startswith("target_")]
    print(df_ready[target_cols].describe().round(4).to_string())