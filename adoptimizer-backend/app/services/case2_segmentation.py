"""
===============================================================================
SEGMENTATION DES CAMPAGNES PUBLICITAIRES — AdOptimizer AI  [VERSION PRO]
===============================================================================
Améliorations niveau industrie vs version standard :

  ① RobustScaler      → résistant aux outliers marketing (médiane + IQR)
  ② Log Transform     → stabilise les distributions fortement skewées
                        (spend, impressions, clicks, conversions)
  ③ conversion_rate   → feature enrichie (qualité post-clic)
  ④ K auto + override → silhouette automatique + business override si k<3
  ⑤ PCA débruitage    → réduit le bruit, garde 95% variance
  ⑥ GMM (option)      → clusters non sphériques, plus réalistes

Pipeline :
  Charger → Nettoyer → Agréger → Sélectionner → Log-Transform
  → RobustScaler → PCA → Clustering (KMeans ou GMM) → Labéliser → Sauvegarder

Input  : app/data/dataset_historique.csv
Outputs: app/cas2-outputs/segmentation_outputs/
  segmentation_results.csv     — campagnes + cluster + label
  cluster_profiles.csv         — profil moyen de chaque cluster
  segmentation_report.txt      — rapport complet
  segmentation_best_k.png      — courbe silhouette + inertie
  segmentation_cluster_viz.png — scatter PCA 2D
  segmentation_radar.png       — radar chart profils

Auteur : AdOptimizer AI — PFE 2024
===============================================================================
"""

import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.preprocessing import RobustScaler          # ① PRO
from sklearn.cluster       import KMeans
from sklearn.mixture       import GaussianMixture        # ⑥ PRO
from sklearn.metrics       import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CASE2_OUTPUTS_DIR = BASE_DIR / "cas2-outputs"

INPUT_FILE = DATA_DIR / "dataset_historique.csv"
OUTPUT_DIR = CASE2_OUTPUTS_DIR / "segmentation_outputs"

OUTPUT_RESULTS  = OUTPUT_DIR / "segmentation_results.csv"
OUTPUT_PROFILES = OUTPUT_DIR / "cluster_profiles.csv"
OUTPUT_REPORT   = OUTPUT_DIR / "segmentation_report.txt"
OUTPUT_SILH     = OUTPUT_DIR / "segmentation_best_k.png"
OUTPUT_PCA      = OUTPUT_DIR / "segmentation_cluster_viz.png"
OUTPUT_RADAR    = OUTPUT_DIR / "segmentation_radar.png"

# ③ PRO — conversion_rate ajouté
FEATURE_COLS = [
    "spend", "impressions", "clicks",
    "conversions", "CTR", "CPC",
    "conversion_rate",   # qualite post-clic
]

# ② Colonnes a transformer en log1p avant le scaling
LOG_COLS = ["spend", "impressions", "clicks", "conversions"]

# Plage K pour silhouette
K_RANGE = range(2, 8)

# ④ Override business : si silhouette prefere k<3, on force k=3
BUSINESS_MIN_K = 3

# ⑥ USE_GMM = True  → GaussianMixture (clusters non spheriques)
#   USE_GMM = False → KMeans classique
USE_GMM = True

RANDOM_STATE = 42
NA_PLATFORM  = "NA_platform"

# ============================================================
# UTILITAIRES
# ============================================================

def sep(c="=", n=72): return c * n
def log(msg): print(msg)

def save_report(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(l) for l in lines))

# ============================================================
# ETAPE 1 — CHARGEMENT
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:
    log(sep()); log("ETAPE 1 — CHARGEMENT DU DATASET"); log(sep())
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset introuvable : '{path}'\n"
            "Lancez d'abord : python generate_raw_dataset_strict.py"
        )
    df = pd.read_csv(path, low_memory=False)
    log(f"  Fichier        : {path}")
    log(f"  Lignes brutes  : {len(df):,}")
    log(f"  Colonnes       : {len(df.columns)}")
    log(f"  Plateformes    : {df['platform'].value_counts().to_dict()}")
    return df

# ============================================================
# ETAPE 2 — NETTOYAGE
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    log(sep()); log("ETAPE 2 — NETTOYAGE DES DONNEES"); log(sep())
    n0 = len(df)
    df = df.drop_duplicates()
    log(f"  Doublons supprimes           : {n0 - len(df):,}")

    all_num_cols = FEATURE_COLS + ["conversion_value"]
    for col in all_num_cols:
        if col in df.columns:
            df[col] = df[col].replace(NA_PLATFORM, np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["spend", "impressions"])
    log(f"  Lignes apres dedup           : {before:,}")
    log(f"  Lignes critiques supprimees  : {before - len(df):,}")
    log(f"  Lignes propres               : {len(df):,}")

    for col in ["impressions", "clicks", "conversions"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
    return df

# ============================================================
# ETAPE 3 — AGREGATION PAR CAMPAGNE
# ============================================================

def aggregate_by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    log(sep()); log("ETAPE 3 — AGREGATION (global_campaign_id x platform)"); log(sep())

    agg_dict = {
        "spend"           : "sum",
        "impressions"     : "sum",
        "clicks"          : "sum",
        "conversions"     : "sum",
        "CTR"             : "mean",
        "CPC"             : "mean",
        "conversion_value": "sum",
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    agg = (
        df
        .groupby(["global_campaign_id", "platform",
                  "campaign_objective", "campaign_status", "location"])
        .agg(agg_dict)
        .reset_index()
    )

    agg["ROAS"] = np.where(
        agg["spend"] > 0, agg["conversion_value"] / agg["spend"], 0.0
    ).round(4)

    # ③ conversion_rate — taux de conversion post-clic
    agg["conversion_rate"] = np.where(
        agg["clicks"] > 0, agg["conversions"] / agg["clicks"], 0.0
    ).round(6)

    log(f"  Campagnes agreges  : {len(agg):,}")
    log(f"  Colonnes           : {list(agg.columns)}")
    return agg

# ============================================================
# ETAPE 4 — SELECTION + LOG TRANSFORM
# ============================================================

def select_and_transform_features(agg: pd.DataFrame) -> pd.DataFrame:
    log(sep()); log("ETAPE 4 — SELECTION + LOG TRANSFORM"); log(sep())

    available = [c for c in FEATURE_COLS if c in agg.columns]
    log(f"  Features retenues  : {available}")
    X = agg[available].copy()

    # Imputation mediane
    for col in X.columns:
        n_nan = X[col].isna().sum()
        if n_nan > 0:
            med = X[col].median()
            X[col] = X[col].fillna(med)
            log(f"  {col:20s} : {n_nan} NaN imputes -> mediane {med:.5f}")

    # Cap outliers P99.5
    for col in X.columns:
        cap = X[col].quantile(0.995)
        n   = (X[col] > cap).sum()
        if n > 0:
            X[col] = X[col].clip(upper=cap)
            log(f"  {col:20s} : {n} outliers cappes a {cap:.2f}")

    # ② Log1p
    log_applied = [c for c in LOG_COLS if c in X.columns]
    log(f"\n  Log1p applique sur : {log_applied}")
    for col in log_applied:
        X[col] = np.log1p(X[col])
        log(f"  {col:20s} -> log1p | mean={X[col].mean():.3f} std={X[col].std():.3f}")

    return X

# ============================================================
# ETAPE 5 — ROBUST SCALING + PCA DEBRUITAGE
# ============================================================

def scale_and_reduce(X: pd.DataFrame):
    log(sep()); log("ETAPE 5 — ROBUST SCALING + PCA DEBRUITAGE"); log(sep())

    # ① RobustScaler
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    log(f"  RobustScaler : shape={X_scaled.shape}")
    log(f"  Mediane post-scale : {np.median(X_scaled, axis=0).round(4)}")

    # ⑤ PCA debruitage — garde 95% variance
    pca    = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X_scaled)
    var    = pca.explained_variance_ratio_.cumsum()[-1] * 100
    log(f"  PCA : {X.shape[1]} features -> {pca.n_components_} composantes")
    log(f"  Variance expliquee : {var:.2f}%")

    return X_pca, scaler, pca

# ============================================================
# ETAPE 6A — RECHERCHE DU MEILLEUR K
# ============================================================

def find_best_k(X_pca: np.ndarray):
    log(sep()); log("ETAPE 6A — RECHERCHE DU MEILLEUR K (Silhouette)"); log(sep())

    sil_scores = {}
    inertias   = {}

    for k in K_RANGE:
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
        labels = km.fit_predict(X_pca)
        sil    = silhouette_score(X_pca, labels)
        sil_scores[k] = sil
        inertias[k]   = km.inertia_
        log(f"  k={k} | silhouette={sil:.4f} | inertia={km.inertia_:,.0f}")

    best_k_auto = max(sil_scores, key=sil_scores.get)
    log(f"\n  k auto (silhouette) : {best_k_auto}")

    # ④ Override
    if best_k_auto < BUSINESS_MIN_K:
        best_k = BUSINESS_MIN_K
        log(f"  Override business   : k {best_k_auto} -> {best_k} (LOW/MEDIUM/HIGH requis)")
    else:
        best_k = best_k_auto
        log(f"  k retenu            : {best_k}")

    return best_k, best_k_auto, sil_scores, inertias

def plot_silhouette_curve(sil_scores, inertias, best_k, best_k_auto):
    ks = list(sil_scores.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Selection du nombre optimal de clusters — Version PRO",
                 fontsize=13, fontweight="bold")

    ax1.plot(ks, list(sil_scores.values()), marker="o", color="#4C72B0", linewidth=2)
    if best_k_auto != best_k:
        ax1.axvline(best_k_auto, color="#F39C12", linestyle=":", linewidth=1.5,
                    label=f"Auto k={best_k_auto}")
    ax1.axvline(best_k, color="#DD4949", linestyle="--", linewidth=2,
                label=f"Retenu k={best_k}")
    ax1.set_xlabel("k"); ax1.set_ylabel("Silhouette")
    ax1.set_title("Score de Silhouette"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ks, list(inertias.values()), marker="s", color="#55A868", linewidth=2)
    ax2.axvline(best_k, color="#DD4949", linestyle="--", linewidth=2, label=f"k={best_k}")
    ax2.set_xlabel("k"); ax2.set_ylabel("Inertie (SSE)")
    ax2.set_title("Methode du Coude"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_SILH, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Graphique : {OUTPUT_SILH}")

# ============================================================
# ETAPE 6B — CLUSTERING (KMeans ou GMM)
# ============================================================

def run_clustering(X_pca: np.ndarray, best_k: int):
    log(sep())
    algo = "GaussianMixture (GMM)" if USE_GMM else "KMeans"
    log(f"ETAPE 6B — CLUSTERING FINAL : {algo}  (k={best_k})")
    log(sep())

    if USE_GMM:
        model  = GaussianMixture(
            n_components=best_k, covariance_type="full",
            n_init=10, random_state=RANDOM_STATE, max_iter=300,
        )
        labels = model.fit_predict(X_pca)
        proba  = model.predict_proba(X_pca)
        log(f"  BIC          : {model.bic(X_pca):,.2f}")
        log(f"  Certitude moy: {proba.max(axis=1).mean() * 100:.1f}%")
    else:
        model  = KMeans(n_clusters=best_k, random_state=RANDOM_STATE,
                        n_init=30, max_iter=1000)
        labels = model.fit_predict(X_pca)
        log(f"  Inertie      : {model.inertia_:,.2f}")

    sil = silhouette_score(X_pca, labels)
    log(f"  Silhouette   : {sil:.4f}")
    for c in range(best_k):
        n = (labels == c).sum()
        log(f"  Cluster {c}     : {n} campagnes ({n / len(labels) * 100:.1f}%)")

    return model, labels

# ============================================================
# ETAPE 7 — ANALYSE DES CLUSTERS
# ============================================================

def analyze_clusters(agg: pd.DataFrame, labels: np.ndarray):
    log(sep()); log("ETAPE 7 — ANALYSE DES CLUSTERS"); log(sep())

    agg = agg.copy()
    agg["cluster"] = labels

    perf_cols = [c for c in
                 ["spend", "impressions", "clicks", "conversions",
                  "CTR", "CPC", "ROAS", "conversion_rate", "conversion_value"]
                 if c in agg.columns]

    profiles = agg.groupby("cluster")[perf_cols].mean().round(4)
    counts   = agg.groupby("cluster").size().rename("n_campaigns")
    profiles = profiles.join(counts)

    log(f"\n  Profil moyen par cluster :")
    log(profiles.to_string())
    return profiles, agg

# ============================================================
# ETAPE 8 — LABELISATION
# ============================================================

def label_clusters(profiles: pd.DataFrame, best_k: int):
    """
    Score composite pondere (valeurs normalisees [0,1]) :
      ROAS            x 3.0
      conversions     x 2.5
      conversion_rate x 2.0  (qualite post-clic)
      CTR             x 1.5
      CPC             x -1.5 (plus bas = mieux)
      spend           x 1.0
    """
    log(sep()); log("ETAPE 8 — LABELISATION DES CLUSTERS"); log(sep())

    weights = {
        "ROAS"           : 3.0,
        "conversions"    : 2.5,
        "conversion_rate": 2.0,
        "CTR"            : 1.5,
        "CPC"            : -1.5,
        "spend"          : 1.0,
    }

    score_df = profiles.copy()
    for col, w in weights.items():
        if col not in score_df.columns:
            continue
        rng = score_df[col].max() - score_df[col].min()
        score_df[col] = (score_df[col] - score_df[col].min()) / rng if rng > 0 else 0.5

    composite = sum(
        score_df[col] * w
        for col, w in weights.items()
        if col in score_df.columns
    )
    ranking = composite.sort_values(ascending=False)

    label_map = {}
    for rank, cid in enumerate(ranking.index):
        if rank == 0:
            label_map[cid] = "HIGH_PERFORMANCE"
        elif rank == len(ranking) - 1:
            label_map[cid] = "LOW_PERFORMANCE"
        else:
            label_map[cid] = "MEDIUM_PERFORMANCE"

    log(f"\n  Score composite pondere :")
    for cid, score in composite.items():
        lbl = label_map[cid]
        log(f"  Cluster {cid} -> score={score:.3f}  ->  {lbl}")

    return label_map, composite

# ============================================================
# ETAPE 9 — VISUALISATIONS
# ============================================================

COLOR_MAP = {
    "HIGH_PERFORMANCE"  : "#2ECC71",
    "MEDIUM_PERFORMANCE": "#F39C12",
    "LOW_PERFORMANCE"   : "#E74C3C",
}

def plot_pca_scatter(X_pca: np.ndarray, labels: np.ndarray, label_map: dict):
    pca2d  = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca2d.fit_transform(X_pca)
    v1, v2 = pca2d.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Segmentation — Vue PCA 2D  [Version PRO]", fontsize=14, fontweight="bold")
    patches = []
    for c in sorted(set(labels)):
        lbl   = label_map[c]
        color = COLOR_MAP[lbl]
        mask  = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, alpha=0.72, edgecolors="white",
                   linewidths=0.4, s=65, zorder=3)
        patches.append(mpatches.Patch(color=color, label=f"Cluster {c} — {lbl}"))

    ax.set_xlabel(f"PC1 ({v1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({v2:.1f}% var)", fontsize=11)
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_PCA, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Scatter PCA : {OUTPUT_PCA}")


def plot_radar(profiles: pd.DataFrame, label_map: dict):
    radar_cols = [c for c in
                  ["ROAS", "CTR", "conversion_rate", "conversions", "CPC", "spend"]
                  if c in profiles.columns]
    if len(radar_cols) < 3:
        return

    norm = profiles[radar_cols].copy()
    for col in radar_cols:
        rng = norm[col].max() - norm[col].min()
        norm[col] = (norm[col] - norm[col].min()) / rng if rng > 0 else 0.5
    if "CPC" in norm.columns:
        norm["CPC"] = 1 - norm["CPC"]

    N      = len(radar_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title("Profils des Clusters (normalises) — Version PRO",
                 fontsize=12, fontweight="bold", pad=20)
    for c in norm.index:
        lbl   = label_map[c]
        color = COLOR_MAP[lbl]
        vals  = norm.loc[c, radar_cols].tolist() + [norm.loc[c, radar_cols[0]]]
        ax.plot(angles, vals, color=color, linewidth=2.5, label=f"Cluster {c} — {lbl}")
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_cols, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.1), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_RADAR, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Radar chart : {OUTPUT_RADAR}")

# ============================================================
# ETAPE 10 — SAUVEGARDE
# ============================================================

def save_results(agg_labeled, profiles, label_map, composite,
                 sil_scores, best_k, best_k_auto, X):
    log(sep()); log("ETAPE 10 — SAUVEGARDE DES RESULTATS"); log(sep())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    agg_labeled.to_csv(OUTPUT_RESULTS, index=False)
    log(f"  Resultats  : {OUTPUT_RESULTS}")

    profiles_exp = profiles.copy()
    profiles_exp["label"] = profiles_exp.index.map(label_map)
    profiles_exp["score"] = composite
    profiles_exp.sort_values("score", ascending=False).to_csv(OUTPUT_PROFILES)
    log(f"  Profils    : {OUTPUT_PROFILES}")

    algo = "GaussianMixture (GMM)" if USE_GMM else "KMeans"
    lines = [
        sep(), "RAPPORT SEGMENTATION PRO — AdOptimizer AI", sep(),
        f"Dataset        : {INPUT_FILE}",
        f"Algorithme     : {algo}",
        f"Features       : {list(X.columns)}",
        f"Log1p applique : {LOG_COLS}",
        f"Scaler         : RobustScaler",
        f"PCA            : 95% variance",
        f"k auto         : {best_k_auto}",
        f"k retenu       : {best_k}",
        f"Silhouette(k)  : {sil_scores[best_k]:.4f}",
        "",
        sep("-"), "SCORES SILHOUETTE PAR K", sep("-"),
        *[f"  k={k} : {s:.4f}" + (" <- RETENU" if k == best_k else "")
          for k, s in sil_scores.items()],
        "",
        sep("-"), "PROFIL DES CLUSTERS", sep("-"),
        profiles_exp.sort_values("score", ascending=False).to_string(),
        "",
        sep("-"), "DISTRIBUTION PAR LABEL", sep("-"),
        *[f"  {lbl:22s} : {n:4d} campagnes ({n / len(agg_labeled) * 100:.1f}%)"
          for lbl, n in agg_labeled["performance_label"].value_counts().items()],
        "",
        sep("-"), "TOP 10 HIGH_PERFORMANCE (par ROAS)", sep("-"),
    ]
    top = agg_labeled[agg_labeled["performance_label"] == "HIGH_PERFORMANCE"]
    if len(top) > 0 and "ROAS" in top.columns:
        lines.append(
            top.nlargest(10, "ROAS")[
                ["global_campaign_id", "platform", "ROAS",
                 "conversions", "conversion_rate", "CTR", "spend"]
            ].to_string(index=False)
        )
    lines += ["", sep(), "FIN DU RAPPORT", sep()]
    save_report(lines, OUTPUT_REPORT)
    log(f"  Rapport    : {OUTPUT_REPORT}")

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_segmentation_pipeline():
    log(sep("=")); log("  PIPELINE SEGMENTATION — AdOptimizer AI  [VERSION PRO]"); log(sep("="))
    log(f"  Ameliorations actives :")
    log(f"    RobustScaler     OK")
    log(f"    Log1p Transform  OK  ({LOG_COLS})")
    log(f"    conversion_rate  OK  (feature enrichie)")
    log(f"    K auto + override OK  (BUSINESS_MIN_K={BUSINESS_MIN_K})")
    log(f"    PCA debruitage   OK  (95% variance)")
    log(f"    GMM              {'OK' if USE_GMM else 'OFF (KMeans actif)'}")
    log("")

    df  = load_dataset(INPUT_FILE)
    df  = clean_dataset(df)
    agg = aggregate_by_campaign(df)
    X   = select_and_transform_features(agg)
    X_pca, scaler, pca = scale_and_reduce(X)

    best_k, best_k_auto, sil_scores, inertias = find_best_k(X_pca)
    plot_silhouette_curve(sil_scores, inertias, best_k, best_k_auto)

    model, labels = run_clustering(X_pca, best_k)
    profiles, agg_labeled = analyze_clusters(agg, labels)
    label_map, composite  = label_clusters(profiles, best_k)
    agg_labeled["performance_label"] = agg_labeled["cluster"].map(label_map)

    plot_pca_scatter(X_pca, labels, label_map)
    plot_radar(profiles, label_map)

    save_results(agg_labeled, profiles, label_map, composite,
                 sil_scores, best_k, best_k_auto, X)

    log(sep("=")); log("  RESUME FINAL"); log(sep("="))
    dist  = agg_labeled["performance_label"].value_counts()
    for lbl in ["HIGH_PERFORMANCE", "MEDIUM_PERFORMANCE", "LOW_PERFORMANCE"]:
        if lbl in dist:
            n = dist[lbl]
            log(f"  {lbl:22s} : {n:4d} campagnes ({n / len(agg_labeled) * 100:.1f}%)")
    log("")
    for f in [OUTPUT_RESULTS, OUTPUT_PROFILES, OUTPUT_REPORT, OUTPUT_SILH, OUTPUT_PCA, OUTPUT_RADAR]:
        log(f"    -> {f}")
    log(sep("=")); log("  SEGMENTATION PRO TERMINEE"); log(sep("="))
    return agg_labeled, profiles, label_map


# ============================================================
# WRAPPER FASTAPI / N8N
# ============================================================

def run_case2_segmentation() -> dict:
    """
    Point d'entree FastAPI pour le cas 2 : segmentation de l'historique.

    Input par defaut :
      app/data/dataset_historique.csv

    Outputs par defaut :
      app/cas2-outputs/segmentation_outputs/
    """
    try:
        agg_labeled, profiles, label_map = run_segmentation_pipeline()
        label_distribution = agg_labeled["performance_label"].value_counts()

        return {
            "status": "success",
            "message": "Segmentation cas 2 terminee",
            "input_file": str(INPUT_FILE),
            "output_dir": str(OUTPUT_DIR),
            "output_files": {
                "results": str(OUTPUT_RESULTS),
                "profiles": str(OUTPUT_PROFILES),
                "report": str(OUTPUT_REPORT),
                "best_k_chart": str(OUTPUT_SILH),
                "cluster_viz": str(OUTPUT_PCA),
                "radar": str(OUTPUT_RADAR),
            },
            "campaigns": int(len(agg_labeled)),
            "clusters": int(len(profiles)),
            "label_distribution": {
                str(label): int(count)
                for label, count in label_distribution.items()
            },
            "label_map": {
                str(cluster): label
                for cluster, label in label_map.items()
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input_file": str(INPUT_FILE),
        }


# ============================================================
if __name__ == "__main__":
    result = run_case2_segmentation()
    print(result)
