"""
===============================================================================
CORRÉLATION INTRA-VARIABLES — AdOptimizer AI  (Cas 2 : Nouvelle Campagne)
===============================================================================
Objectif : Identifier les relations entre variables marketing afin de
           comprendre les facteurs de performance et guider l'Agent 2
           (Décision) dans la génération de nouvelles stratégies.

Approche  : Corrélation de Pearson simplifiée
            But = produire des règles exploitables, pas de causalité complexe

Pipeline :
  Charger → Nettoyer → Agréger → Matrice Pearson → Filtrer (|r| ≥ 0.30)
  → Importance variables → Règles métier → Exports

Input  : app/data/dataset_historique.csv  (dataset historique cas 2)
Outputs: app/cas2-outputs/correlation_outputs/
  correlation_matrix.csv       — matrice complète des corrélations
  correlation_rules.json       — règles exploitables par Agent 2
  correlation_report.txt       — rapport lisible complet
  heatmap.png                  — visualisation matrice
  barplot_importance.png       — importance des variables
  scatter_top_relations.png    — scatter plots des relations clés

Auteur : AdOptimizer AI — PFE 2024
===============================================================================
"""

import os
import json
import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CASE2_OUTPUTS_DIR = BASE_DIR / "cas2-outputs"

INPUT_FILE = DATA_DIR / "dataset_historique.csv"
OUTPUT_DIR = CASE2_OUTPUTS_DIR / "correlation_outputs"

OUTPUT_MATRIX   = OUTPUT_DIR / "correlation_matrix.csv"
OUTPUT_RULES    = OUTPUT_DIR / "correlation_rules.json"
OUTPUT_REPORT   = OUTPUT_DIR / "correlation_report.txt"
OUTPUT_HEATMAP  = OUTPUT_DIR / "heatmap.png"
OUTPUT_BARPLOT  = OUTPUT_DIR / "barplot_importance.png"
OUTPUT_SCATTER  = OUTPUT_DIR / "scatter_top_relations.png"

# Variables quantitatives à analyser
FEATURE_COLS = [
    "spend", "impressions", "clicks",
    "CTR", "CPC", "conversions", "conversion_rate",
]

# Variable cible principale pour l'importance
TARGET_VAR = "conversions"

# Seuil de corrélation significative
CORR_THRESHOLD = 0.30   # |r| >= 0.30

# Seuil p-value
PVALUE_THRESH = 0.05

# Marqueur NA_platform
NA_PLATFORM = "NA_platform"

# ============================================================
# UTILITAIRES
# ============================================================

def sep(c="=", n=72): return c * n
def log(msg):
    text = str(msg)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

def corr_strength(r):
    r = abs(r)
    if r >= 0.70: return "forte"
    if r >= 0.50: return "modérée"
    if r >= 0.30: return "faible"
    return "négligeable"

def corr_arrow(r):
    return "↑" if r > 0 else "↓"

def corr_sign(r):
    return "positive" if r > 0 else "négative"

def business_rule(var_x, var_y, r):
    """Traduit une corrélation en règle métier exploitable."""
    arrow_x = "↑" if r > 0 else "↓"
    arrow_y = "↑"
    strength = corr_strength(r)

    rules = {
        ("spend",           "conversions")      : f"Augmenter le budget {arrow_x} augmente les conversions {arrow_y} (corrélation {strength})",
        ("clicks",          "conversions")      : f"Plus de clics {arrow_x} génère plus de conversions {arrow_y} (corrélation {strength})",
        ("CTR",             "conversions")      : f"Un CTR élevé {arrow_x} améliore les conversions {arrow_y} (corrélation {strength})",
        ("CPC",             "conversions")      : f"Un CPC élevé {arrow_x} réduit les conversions {arrow_y} (corrélation {strength})" if r < 0 else f"CPC et conversions corrélés positivement (corrélation {strength})",
        ("conversion_rate", "conversions")      : f"Améliorer le taux de conversion {arrow_x} booste directement les conversions {arrow_y}",
        ("impressions",     "conversions")      : f"Plus d'impressions {arrow_x} génère plus de conversions {arrow_y} (volume)",
        ("spend",           "CTR")              : f"Augmenter le budget influence le CTR ({corr_sign(r)}, {strength})",
        ("CTR",             "CPC")              : f"CTR et CPC sont liés ({corr_sign(r)}) — optimiser le CTR impacte le coût",
        ("spend",           "clicks")           : f"Budget et clics corrélés ({strength}) — dépenser plus génère plus de clics",
        ("impressions",     "clicks")           : f"Impressions et clics liés ({strength}) — portée = volume clics",
        ("spend",           "impressions")      : f"Budget et impressions corrélés ({strength}) — budget = visibilité",
        ("CPC",             "CTR")              : f"CPC et CTR liés ({corr_sign(r)}, {strength})",
        ("conversion_rate", "CTR")              : f"CTR et taux de conversion liés ({corr_sign(r)}, {strength})",
        ("clicks",          "CTR")              : f"Clics et CTR corrélés ({corr_sign(r)}, {strength})",
        ("impressions",     "CTR")              : f"Impressions et CTR liés ({corr_sign(r)}, {strength})",
        ("spend",           "CPC")              : f"Budget et CPC liés ({corr_sign(r)}, {strength})",
        ("clicks",          "CPC")              : f"Clics et CPC corrélés ({corr_sign(r)}, {strength})",
        ("conversion_rate", "CPC")              : f"Taux de conversion et CPC liés ({corr_sign(r)}, {strength})",
        ("impressions",     "spend")            : f"Impressions et budget fortement liés — budget = visibilité",
        ("clicks",          "impressions")      : f"Clics et impressions corrélés ({strength})",
        ("conversion_rate", "spend")            : f"Budget et taux de conversion liés ({corr_sign(r)}, {strength})",
        ("conversion_rate", "clicks")           : f"Taux de conversion et clics liés ({corr_sign(r)}, {strength})",
        ("conversion_rate", "impressions")      : f"Taux de conversion et impressions liés ({corr_sign(r)}, {strength})",
    }

    key  = (var_x, var_y)
    key2 = (var_y, var_x)
    return rules.get(key, rules.get(key2,
        f"{var_x} {arrow_x} → {var_y} {arrow_y} (corrélation {corr_sign(r)}, {strength})"
    ))

# ============================================================
# ÉTAPE 1 — CHARGEMENT
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:
    log(sep()); log("ÉTAPE 1 — CHARGEMENT DU DATASET BRUT"); log(sep())

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
# ÉTAPE 2 — NETTOYAGE
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    log(sep()); log("ÉTAPE 2 — NETTOYAGE DES DONNÉES"); log(sep())

    n0 = len(df)
    df = df.drop_duplicates()
    log(f"  Doublons supprimés  : {n0 - len(df):,}")

    # Remplacer NA_platform par NaN pour les colonnes numériques
    num_cols = ["spend", "impressions", "clicks", "CTR", "CPC",
                "conversions", "conversion_value"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].replace(NA_PLATFORM, np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprimer lignes sans données critiques
    before = len(df)
    df = df.dropna(subset=["spend", "clicks"])
    log(f"  Lignes sans données critiques supprimées : {before - len(df):,}")
    log(f"  Lignes propres      : {len(df):,}")

    return df

# ============================================================
# ÉTAPE 3 — AGRÉGATION PAR CAMPAGNE
# ============================================================

def aggregate_by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    log(sep()); log("ÉTAPE 3 — AGRÉGATION PAR CAMPAGNE"); log(sep())

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
        .groupby(["global_campaign_id", "platform", "campaign_objective"])
        .agg(agg_dict)
        .reset_index()
    )

    # Calcul conversion_rate après agrégation
    agg["conversion_rate"] = np.where(
        agg["clicks"] > 0,
        agg["conversions"] / agg["clicks"],
        0.0
    ).round(6)

    # ROAS (pour le rapport)
    agg["ROAS"] = np.where(
        agg["spend"] > 0,
        agg["conversion_value"] / agg["spend"],
        0.0
    ).round(4)

    log(f"  Campagnes agrégées  : {len(agg):,}")
    log(f"  Variables produites : {list(agg.columns)}")
    return agg

# ============================================================
# ÉTAPE 4 — CALCUL MATRICE DE CORRÉLATION PEARSON
# ============================================================

def compute_correlation_matrix(agg: pd.DataFrame):
    log(sep()); log("ÉTAPE 4 — CALCUL MATRICE DE CORRÉLATION (Pearson)"); log(sep())

    available = [c for c in FEATURE_COLS if c in agg.columns]
    log(f"  Variables analysées : {available}")

    X = agg[available].copy()

    # Imputation médiane pour NaN résiduels
    for col in X.columns:
        n_nan = X[col].isna().sum()
        if n_nan > 0:
            X[col] = X[col].fillna(X[col].median())
            log(f"  {col:20s} : {n_nan} NaN imputés")

    # Matrice de corrélation Pearson
    corr_matrix = X.corr(method="pearson")

    # Matrice des p-values (significativité statistique)
    n = len(X)
    pvalue_matrix = pd.DataFrame(np.ones((len(available), len(available))),
                                  index=available, columns=available)
    for i, col1 in enumerate(available):
        for j, col2 in enumerate(available):
            if i != j:
                r, p = stats.pearsonr(X[col1].values, X[col2].values)
                pvalue_matrix.loc[col1, col2] = p

    log(f"\n  Matrice calculée : {len(available)} × {len(available)}")
    log(f"  Observations     : {n}")
    log(f"\n  Matrice complète :")
    log(corr_matrix.round(3).to_string())

    return corr_matrix, pvalue_matrix, X

# ============================================================
# ÉTAPE 5 — FILTRAGE DES RELATIONS SIGNIFICATIVES
# ============================================================

def filter_significant_relations(corr_matrix: pd.DataFrame,
                                  pvalue_matrix: pd.DataFrame):
    log(sep()); log(f"ÉTAPE 5 — FILTRAGE (|r| ≥ {CORR_THRESHOLD})"); log(sep())

    relations = []
    cols = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            var_x = cols[i]
            var_y = cols[j]
            r     = corr_matrix.loc[var_x, var_y]
            p     = pvalue_matrix.loc[var_x, var_y]

            if abs(r) >= CORR_THRESHOLD:
                relations.append({
                    "var_x"     : var_x,
                    "var_y"     : var_y,
                    "pearson_r" : round(float(r), 4),
                    "p_value"   : round(float(p), 4),
                    "significant": bool(p < PVALUE_THRESH),
                    "strength"  : corr_strength(r),
                    "direction" : corr_sign(r),
                    "abs_r"     : abs(r),
                })

    # Trier par force décroissante
    relations = sorted(relations, key=lambda x: x["abs_r"], reverse=True)

    log(f"  Relations trouvées (|r| ≥ {CORR_THRESHOLD}) : {len(relations)}")
    log(f"\n  {'Variable X':20s} {'Variable Y':20s} {'r':>8} {'p':>8} {'Force':12} {'Direction'}")
    log(f"  {'-'*20} {'-'*20} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
    for rel in relations:
        sig = "✓" if rel["significant"] else "~"
        log(f"  {rel['var_x']:20s} {rel['var_y']:20s} "
            f"{rel['pearson_r']:>8.4f} {rel['p_value']:>8.4f} "
            f"{rel['strength']:12s} {rel['direction']} {sig}")

    return relations

# ============================================================
# ÉTAPE 6 — IMPORTANCE DES VARIABLES
# ============================================================

def compute_variable_importance(corr_matrix: pd.DataFrame, relations: list):
    log(sep()); log("ÉTAPE 6 — IMPORTANCE DES VARIABLES"); log(sep())

    available = corr_matrix.columns.tolist()

    # Score d'importance = moyenne des |r| avec toutes les autres variables
    importance = {}
    for var in available:
        others = [c for c in available if c != var]
        mean_abs_r = corr_matrix.loc[var, others].abs().mean()
        importance[var] = round(float(mean_abs_r), 4)

    # Importance spécifique vs TARGET_VAR
    importance_vs_target = {}
    if TARGET_VAR in corr_matrix.columns:
        for var in available:
            if var != TARGET_VAR:
                r = corr_matrix.loc[var, TARGET_VAR]
                importance_vs_target[var] = round(float(r), 4)

    # Trier par importance décroissante
    importance_sorted = dict(sorted(importance.items(),
                                     key=lambda x: x[1], reverse=True))

    log(f"\n  Importance globale (moyenne |r| vs toutes variables) :")
    for var, score in importance_sorted.items():
        bar = "█" * int(score * 20)
        log(f"  {var:20s} : {score:.4f}  {bar}")

    if importance_vs_target:
        log(f"\n  Importance vs {TARGET_VAR} (corrélation directe) :")
        target_sorted = dict(sorted(importance_vs_target.items(),
                                     key=lambda x: abs(x[1]), reverse=True))
        for var, r in target_sorted.items():
            arrow = corr_arrow(r)
            log(f"  {var:20s} : {r:+.4f}  {arrow}")

    return importance_sorted, importance_vs_target

# ============================================================
# ÉTAPE 7 — GÉNÉRATION DES RÈGLES MÉTIER
# ============================================================

def generate_business_rules(relations: list, importance_vs_target: dict):
    log(sep()); log("ÉTAPE 7 — RÈGLES MÉTIER EXPLOITABLES"); log(sep())

    rules = []
    for rel in relations:
        if rel["significant"]:
            rule_text = business_rule(rel["var_x"], rel["var_y"], rel["pearson_r"])
            rules.append({
                "var_x"      : rel["var_x"],
                "var_y"      : rel["var_y"],
                "pearson_r"  : rel["pearson_r"],
                "strength"   : rel["strength"],
                "direction"  : rel["direction"],
                "rule"       : rule_text,
                "priority"   : "haute" if abs(rel["pearson_r"]) >= 0.5 else "normale",
            })

    # Règles spécifiques pour l'Agent 2
    agent2_rules = {
        "variables_cles": [],
        "regles_action" : [],
        "variables_coût": [],
        "variables_volume": [],
    }

    # Classer les variables
    for var, r in sorted(importance_vs_target.items(),
                          key=lambda x: abs(x[1]), reverse=True):
        if abs(r) >= 0.5:
            agent2_rules["variables_cles"].append({
                "variable": var, "correlation_conversions": r,
                "impact": "fort", "action": corr_arrow(r)
            })
        if var in ["CPC", "spend"]:
            agent2_rules["variables_coût"].append(var)
        if var in ["impressions", "clicks", "spend"]:
            agent2_rules["variables_volume"].append(var)

    # Règles d'action pour l'Agent 2
    for var, r in importance_vs_target.items():
        if abs(r) >= CORR_THRESHOLD:
            if r > 0:
                action = f"Maximiser {var} pour augmenter les conversions"
            else:
                action = f"Réduire {var} pour améliorer les conversions"
            agent2_rules["regles_action"].append({
                "variable": var, "correlation": r,
                "action": action, "priorite": "haute" if abs(r) >= 0.5 else "normale"
            })

    log(f"\n  Règles générées : {len(rules)}")
    log(f"\n  Règles pour Agent 2 :")
    for rule in rules[:8]:
        prio = "🔴" if rule["priority"] == "haute" else "🟡"
        log(f"  {prio} {rule['rule']}")

    log(f"\n  Variables clés pour Agent 2 :")
    for v in agent2_rules["variables_cles"]:
        log(f"    → {v['variable']:20s} r={v['correlation_conversions']:+.4f}  impact={v['impact']}")

    return rules, agent2_rules

# ============================================================
# ÉTAPE 8 — VISUALISATIONS
# ============================================================

def plot_heatmap(corr_matrix: pd.DataFrame):
    """Heatmap de la matrice de corrélation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 10, "weight": "bold"},
            ax=ax
        )
    else:
        im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns)
        ax.set_yticklabels(corr_matrix.index)
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax.text(
                    j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                )
    ax.set_title("Matrice de Corrélation — Variables Marketing\n(Pearson)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_HEATMAP, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Heatmap sauvegardée    : {OUTPUT_HEATMAP}")


def plot_importance_barplot(importance_vs_target: dict):
    """Barplot de l'importance des variables vs conversions."""
    if not importance_vs_target:
        return

    vars_list = list(importance_vs_target.keys())
    vals      = list(importance_vs_target.values())
    colors    = ["#2ECC71" if v > 0 else "#E74C3C" for v in vals]

    # Trier par valeur absolue
    sorted_pairs = sorted(zip(vars_list, vals, colors),
                           key=lambda x: abs(x[1]), reverse=True)
    vars_s, vals_s, colors_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(vars_s, vals_s, color=colors_s, edgecolor="white",
                   linewidth=0.8, height=0.6)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(CORR_THRESHOLD, color="#F39C12", linewidth=1.2,
               linestyle=":", label=f"Seuil |r|={CORR_THRESHOLD}")
    ax.axvline(-CORR_THRESHOLD, color="#F39C12", linewidth=1.2, linestyle=":")

    for bar, val in zip(bars, vals_s):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=9, fontweight="bold")

    ax.set_xlabel("Corrélation de Pearson (r)", fontsize=11)
    ax.set_title(f"Importance des Variables vs {TARGET_VAR}\n"
                 f"(Vert = impact positif | Rouge = impact négatif)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_BARPLOT, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Barplot sauvegardé     : {OUTPUT_BARPLOT}")


def plot_scatter_top(X: pd.DataFrame, relations: list):
    """Scatter plots des 6 relations les plus fortes."""
    top = [r for r in relations if r["significant"]][:6]
    if not top:
        return

    ncols = 3
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, rel in enumerate(top):
        ax  = axes[i]
        x   = X[rel["var_x"]].values
        y   = X[rel["var_y"]].values
        r   = rel["pearson_r"]
        col = "#2ECC71" if r > 0 else "#E74C3C"

        ax.scatter(x, y, alpha=0.5, s=25, color=col, edgecolors="white",
                   linewidths=0.3)
        # Droite de régression
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
                linestyle="--", alpha=0.7)

        ax.set_xlabel(rel["var_x"], fontsize=9)
        ax.set_ylabel(rel["var_y"], fontsize=9)
        ax.set_title(f"r = {r:+.3f}  ({rel['strength']})", fontsize=10,
                     fontweight="bold", color=col)
        ax.grid(alpha=0.2)

    # Masquer les axes vides
    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Relations — Scatter Plots", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_SCATTER, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Scatter plots sauvegardés : {OUTPUT_SCATTER}")

# ============================================================
# ÉTAPE 9 — SAUVEGARDE
# ============================================================

def save_results(corr_matrix, relations, importance_sorted,
                 importance_vs_target, rules, agent2_rules):
    log(sep()); log("ÉTAPE 9 — SAUVEGARDE DES RÉSULTATS"); log(sep())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ① Matrice CSV
    corr_matrix.round(4).to_csv(OUTPUT_MATRIX)
    log(f"  Matrice CSV    : {OUTPUT_MATRIX}")

    # ② Règles JSON → Agent 2
    json_output = {
        "metadata": {
            "outil"      : "Corrélation Intra-Variables",
            "cas"        : "Cas 2 — Nouvelle Campagne",
            "methode"    : "Pearson",
            "seuil"      : CORR_THRESHOLD,
            "target_var" : TARGET_VAR,
            "n_relations": len(relations),
        },
        "relations_significatives": relations,
        "importance_vs_conversions": importance_vs_target,
        "importance_globale"       : importance_sorted,
        "regles_agent2"            : agent2_rules,
        "regles_metier"            : [r["rule"] for r in rules],
    }
    with open(OUTPUT_RULES, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False, default=str)
    log(f"  Règles JSON    : {OUTPUT_RULES}")

    # ③ Rapport TXT → lisible
    report_lines = [
        sep(), "RAPPORT CORRÉLATION — AdOptimizer AI (Cas 2 : Nouvelle Campagne)", sep(),
        f"Méthode        : Pearson (|r| ≥ {CORR_THRESHOLD})",
        f"Variables      : {list(corr_matrix.columns)}",
        f"Relations sig. : {len(relations)}",
        "",
        sep("-"), "MATRICE DE CORRÉLATION", sep("-"),
        corr_matrix.round(3).to_string(),
        "",
        sep("-"), "RELATIONS SIGNIFICATIVES (|r| ≥ 0.30)", sep("-"),
    ]
    for rel in relations:
        sig = "✓ sig." if rel["significant"] else "~ non sig."
        report_lines.append(
            f"  {rel['var_x']:20s} ↔ {rel['var_y']:20s} "
            f"r={rel['pearson_r']:+.4f}  {rel['strength']:10s}  {rel['direction']:10s}  {sig}"
        )

    report_lines += [
        "",
        sep("-"), f"IMPORTANCE DES VARIABLES vs {TARGET_VAR}", sep("-"),
    ]
    for var, r in sorted(importance_vs_target.items(),
                          key=lambda x: abs(x[1]), reverse=True):
        arrow = "↑" if r > 0 else "↓"
        report_lines.append(f"  {var:20s} : r={r:+.4f}  {arrow}")

    report_lines += [
        "",
        sep("-"), "RÈGLES MÉTIER EXPLOITABLES (Agent 2)", sep("-"),
    ]
    for rule in rules:
        prio = "[HAUTE]" if rule["priority"] == "haute" else "[NORM.]"
        report_lines.append(f"  {prio}  {rule['rule']}")

    report_lines += [
        "",
        sep("-"), "RÉSUMÉ VARIABLES CLÉS POUR AGENT 2", sep("-"),
        "  Augmenter pour plus de conversions :",
    ]
    pos = [(v, r) for v, r in importance_vs_target.items() if r >= CORR_THRESHOLD]
    neg = [(v, r) for v, r in importance_vs_target.items() if r <= -CORR_THRESHOLD]
    for v, r in sorted(pos, key=lambda x: x[1], reverse=True):
        report_lines.append(f"    → {v:20s} (r={r:+.4f})")
    report_lines.append("  Réduire pour plus de conversions :")
    for v, r in sorted(neg, key=lambda x: x[1]):
        report_lines.append(f"    → {v:20s} (r={r:+.4f})")

    report_lines += ["", sep(), "FIN DU RAPPORT", sep()]

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"  Rapport TXT    : {OUTPUT_REPORT}")

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_correlation_pipeline():
    log(sep("=")); log("  CORRÉLATION INTRA-VARIABLES — AdOptimizer AI (Cas 2)"); log(sep("="))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Charger
    df = load_dataset(INPUT_FILE)

    # 2. Nettoyer
    df = clean_dataset(df)

    # 3. Agréger par campagne
    agg = aggregate_by_campaign(df)

    # 4. Matrice Pearson
    corr_matrix, pvalue_matrix, X = compute_correlation_matrix(agg)

    # 5. Filtrer relations significatives
    relations = filter_significant_relations(corr_matrix, pvalue_matrix)

    # 6. Importance variables
    importance_sorted, importance_vs_target = compute_variable_importance(
        corr_matrix, relations
    )

    # 7. Règles métier
    rules, agent2_rules = generate_business_rules(relations, importance_vs_target)

    # 8. Visualisations
    log(sep()); log("ÉTAPE 8 — VISUALISATIONS"); log(sep())
    plot_heatmap(corr_matrix)
    plot_importance_barplot(importance_vs_target)
    plot_scatter_top(X, relations)

    # 9. Sauvegarder
    save_results(corr_matrix, relations, importance_sorted,
                 importance_vs_target, rules, agent2_rules)

    # Résumé final
    log(sep("=")); log("  RÉSUMÉ FINAL"); log(sep("="))
    log(f"  Relations significatives : {len(relations)}")
    log(f"  Variables analysées      : {len(corr_matrix.columns)}")
    log(f"\n  Top 3 facteurs → {TARGET_VAR} :")
    top3 = sorted(importance_vs_target.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for var, r in top3:
        arrow = "↑" if r > 0 else "↓"
        log(f"    {arrow} {var:20s}  r = {r:+.4f}  ({corr_strength(r)})")
    log(f"\n  Outputs : {OUTPUT_DIR}/")
    for f in [OUTPUT_MATRIX, OUTPUT_RULES, OUTPUT_REPORT,
              OUTPUT_HEATMAP, OUTPUT_BARPLOT, OUTPUT_SCATTER]:
        log(f"    → {f}")
    log(sep("=")); log("  CORRÉLATION TERMINÉE"); log(sep("="))

    return corr_matrix, relations, rules, agent2_rules


# ============================================================
# WRAPPER FASTAPI / N8N
# ============================================================

def run_case2_correlation() -> dict:
    """
    Point d'entree FastAPI pour le cas 2 : correlations de l'historique.

    Input par defaut :
      app/data/dataset_historique.csv

    Outputs par defaut :
      app/cas2-outputs/correlation_outputs/
    """
    try:
        corr_matrix, relations, rules, agent2_rules = run_correlation_pipeline()
        importance_vs_target = agent2_rules.get("regles_action", [])
        top_factors = [
            {
                "variable": item["variable"],
                "correlation": item["correlation"],
                "priorite": item["priorite"],
            }
            for item in sorted(
                importance_vs_target,
                key=lambda x: abs(x["correlation"]),
                reverse=True,
            )[:3]
        ]

        return {
            "status": "success",
            "message": "Correlation cas 2 terminee",
            "input_file": str(INPUT_FILE),
            "output_dir": str(OUTPUT_DIR),
            "output_files": {
                "matrix": str(OUTPUT_MATRIX),
                "rules": str(OUTPUT_RULES),
                "report": str(OUTPUT_REPORT),
                "heatmap": str(OUTPUT_HEATMAP),
                "importance_chart": str(OUTPUT_BARPLOT),
                "scatter": str(OUTPUT_SCATTER),
            },
            "variables": list(corr_matrix.columns),
            "relations": len(relations),
            "business_rules": len(rules),
            "top_factors": top_factors,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input_file": str(INPUT_FILE),
        }


# ============================================================
if __name__ == "__main__":
    result = run_case2_correlation()
    print(result)
