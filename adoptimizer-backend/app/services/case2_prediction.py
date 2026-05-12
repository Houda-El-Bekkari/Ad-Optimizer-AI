"""
===============================================================================
PRÉDICTEUR ML — AdOptimizer AI  (Cas 2 : Nouvelle Campagne)
===============================================================================
Objectif : Prédire les KPIs futurs (ROAS, conversions, CPA, CTR, CPC)
           sur J+3 / J+7 / J+14 pour chaque stratégie candidate.

Améliorations vs version de base :
  ✅ MODEL_PATH corrigé → models/best_model.pkl
  ✅ Score multi-critères adapté à l'OBJECTIF utilisateur
       conversions → ROAS×0.5 + Conv×0.3 − CPA×0.2
       leads       → Conv×0.4 + CR×0.35  − CPA×0.25
       awareness   → CTR×0.5  + Conv×0.3 − CPC×0.2
       traffic     → Conv×0.4 + CTR×0.35 − CPC×0.25
  ✅ Générique N stratégies (3, 6 ou autre)

Input  :
  models/best_model.pkl
  feature_engineering_outputs/features_strategies_pro.csv
  agent2_outputs/strategies.json

Output :
  predictor_outputs/
    predictions.json
    predictions.csv
    predictor_report.txt
    predictions_chart.png

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

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CASE2_OUTPUTS_DIR = BASE_DIR / "cas2-outputs"

MODEL_PATH       = BASE_DIR / "models" / "best_model.pkl"
INPUT_FEATURES   = CASE2_OUTPUTS_DIR / "feature_engineering_outputs" / "features_strategies_pro.csv"
INPUT_STRATEGIES = CASE2_OUTPUTS_DIR / "agent2_outputs" / "strategies.json"
OUTPUT_DIR       = CASE2_OUTPUTS_DIR / "predictor_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_JSON   = OUTPUT_DIR / "predictions.json"
OUTPUT_CSV    = OUTPUT_DIR / "predictions.csv"
OUTPUT_REPORT = OUTPUT_DIR / "predictor_report.txt"
OUTPUT_CHART  = OUTPUT_DIR / "predictions_chart.png"

# ============================================================
# SCORE MULTI-CRITÈRES — ADAPTÉ À L'OBJECTIF
# ============================================================
# Chaque objectif a une logique business différente :
#   conversions → ROAS et volume sont prioritaires, CPA pénalisé
#   leads       → CR et volume qualifié, CPA très pénalisé
#   awareness   → CTR et volume, CPC pénalisé (coût par impression)
#   traffic     → clicks et CTR, CPC pénalisé (coût par clic)

SCORING_BY_OBJECTIF = {
    "conversions": {
        "label"   : "ROAS×0.5 + Conv×0.3 − CPA×0.2",
        "function": lambda p: (
            p["roas"]["J+14"]        * 0.5
            + p["conversions"]["J+14"] * 0.3
            - p["cpa"]["J+14"]         * 0.2
        ),
    },
    "leads": {
        "label"   : "Conv×0.4 + CTR×300×0.35 − CPA×0.25",
        "function": lambda p: (
            p["conversions"]["J+14"]   * 0.40
            + p["ctr"]["J+14"] * 300   * 0.35   # CTR normalisé × 300
            - p["cpa"]["J+14"]         * 0.25
        ),
    },
    "awareness": {
        "label"   : "CTR×300×0.5 + Conv×0.3 − CPC×0.2",
        "function": lambda p: (
            p["ctr"]["J+14"] * 300     * 0.50   # CTR normalisé × 300
            + p["conversions"]["J+14"] * 0.30
            - p["cpc"]["J+14"]         * 0.20
        ),
    },
    "traffic": {
        "label"   : "Conv×0.4 + CTR×300×0.35 − CPC×0.25",
        "function": lambda p: (
            p["conversions"]["J+14"]   * 0.40
            + p["ctr"]["J+14"] * 300   * 0.35
            - p["cpc"]["J+14"]         * 0.25
        ),
    },
}

def sep(c="=", n=72): return c * n
def log(m):
    text = str(m)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def compute_score(r: dict, objectif: str = "conversions") -> float:
    """
    Score multi-critères adapté à l'objectif utilisateur.

    Ne repose pas uniquement sur le ROAS — intègre conversions et coût.
    La formule change selon l'objectif pour refléter la logique business :
      conversions → ROAS × 0.5 + Conv × 0.3 − CPA × 0.2
      leads       → Conv × 0.4 + CTR × 0.35 − CPA × 0.25
      awareness   → CTR × 0.5  + Conv × 0.3 − CPC × 0.2
      traffic     → Conv × 0.4 + CTR × 0.35 − CPC × 0.25

    Phrase PFE :
    La sélection de la meilleure stratégie ne repose pas uniquement
    sur le ROAS, mais sur un score multi-critères adapté à l'objectif
    de la campagne et intégrant conversions, coût et engagement.
    """
    formula = SCORING_BY_OBJECTIF.get(objectif, SCORING_BY_OBJECTIF["conversions"])
    return formula["function"](r["predictions"])

# ============================================================
# ÉTAPE 1 — CHARGER LE MODÈLE
# ============================================================
def load_bundle():
    log(sep()); log("ÉTAPE 1 — CHARGEMENT DU MODÈLE"); log(sep())
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Vérifiez que best_model.pkl est dans le dossier 'models/'."
        )
    bundle = joblib.load(MODEL_PATH)
    log(f"  Modèle      : {bundle['model_name']}")
    log(f"  Features    : {len(bundle['feature_cols'])}")
    log(f"  Targets     : {bundle['target_cols']}")
    return bundle

# ============================================================
# ÉTAPE 2 — CHARGER LES FEATURES PRO
# ============================================================
def load_features(bundle: dict) -> pd.DataFrame:
    log(sep()); log("ÉTAPE 2 — CHARGEMENT DES FEATURES PRO"); log(sep())

    if not os.path.exists(INPUT_FEATURES):
        raise FileNotFoundError(
            f"Features introuvables : {INPUT_FEATURES}\n"
            "Lancez feature_engineering_pro.py d'abord."
        )

    df           = pd.read_csv(INPUT_FEATURES)
    feature_cols = bundle["feature_cols"]

    log(f"  Fichier chargé  : {INPUT_FEATURES}")
    log(f"  Shape           : {df.shape}  ({df.shape[0]} stratégies × {df.shape[1]} features)")

    # Colonnes manquantes → 0
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        log(f"  ⚠️  {len(missing)} colonnes manquantes → ajoutées à 0")
        for col in missing:
            df[col] = 0.0

    extra = [c for c in df.columns if c not in feature_cols]
    if extra:
        log(f"  ℹ️  {len(extra)} colonnes en plus ignorées")

    df = df.reindex(columns=feature_cols, fill_value=0.0)
    log(f"  Shape alignée   : {df.shape} ✅")
    return df

# ============================================================
# ÉTAPE 3 — PRÉDICTION
# ============================================================
def predict(df_features: pd.DataFrame, bundle: dict) -> list:
    """
    Prédit les 15 targets pour chaque ligne de df_features.
    Générique : fonctionne avec N stratégies (3, 6 ou autre).

    Alignement sur min(n_features_rows, n_strategies).
    """
    log(sep()); log("ÉTAPE 3 — PRÉDICTIONS ML"); log(sep())

    models       = bundle["models_by_target"]
    imputer      = bundle["imputer"]
    feature_cols = bundle["feature_cols"]

    X_imp = pd.DataFrame(
        imputer.transform(df_features),
        columns=feature_cols
    )
    log(f"  Imputation OK  : shape={X_imp.shape}")

    with open(INPUT_STRATEGIES, encoding="utf-8") as f:
        strategies = json.load(f)["strategies"]

    n_rows    = len(X_imp)
    n_strat   = len(strategies)
    n_predict = min(n_rows, n_strat)

    if n_rows != n_strat:
        log(f"  ℹ️  Alignement : {n_rows} features vs {n_strat} stratégies "
            f"→ {n_predict} prédictions")

    results = []
    for i in range(n_predict):
        s   = strategies[i]
        sid = s["id"]
        row = X_imp.iloc[i:i+1]

        if row.empty:
            log(f"  ⚠️  [{sid}] ligne {i} vide — ignorée")
            continue

        log(f"\n  [{sid}] {s['type'].upper()} ({s.get('plateforme','?')}) "
            f"— Budget={s['budget']:,.0f}€  CPC={s['CPC_cible']:.4f}  "
            f"CTR={s['CTR_cible']:.4f}")

        raw  = {t: float(m.predict(row)[0]) for t, m in models.items()}
        pred = {
            metric: {
                "J+3" : raw.get(f"target_{metric}_h3",  0.0),
                "J+7" : raw.get(f"target_{metric}_h7",  0.0),
                "J+14": raw.get(f"target_{metric}_h14", 0.0),
            }
            for metric in ["roas", "conversions", "cpa", "ctr", "cpc"]
        }

        results.append({
            "strategy_id"  : sid,
            "strategy_type": s["type"],
            "plateforme"   : s.get("plateforme", "meta"),
            "objectif"     : s.get("objectif",   "conversions"),
            "budget"       : float(s["budget"]),
            "CPC_cible"    : float(s["CPC_cible"]),
            "CTR_cible"    : float(s["CTR_cible"]),
            "conv_rate"    : float(s["conversion_rate"]),
            "predictions"  : pred,
        })

        log(f"    ROAS        : J+3={pred['roas']['J+3']:.3f}  "
            f"J+7={pred['roas']['J+7']:.3f}  J+14={pred['roas']['J+14']:.3f}")
        log(f"    Conversions : J+3={pred['conversions']['J+3']:.0f}  "
            f"J+7={pred['conversions']['J+7']:.0f}  J+14={pred['conversions']['J+14']:.0f}")
        log(f"    CPA         : J+3={pred['cpa']['J+3']:.2f}€  "
            f"J+7={pred['cpa']['J+7']:.2f}€  J+14={pred['cpa']['J+14']:.2f}€")
        log(f"    CTR         : J+3={pred['ctr']['J+3']*100:.3f}%  "
            f"J+7={pred['ctr']['J+7']*100:.3f}%  J+14={pred['ctr']['J+14']*100:.3f}%")
        log(f"    CPC         : J+3={pred['cpc']['J+3']:.3f}€  "
            f"J+7={pred['cpc']['J+7']:.3f}€  J+14={pred['cpc']['J+14']:.3f}€")

    log(f"\n  Total prédictions : {len(results)}")
    return results

# ============================================================
# ÉTAPE 4 — VISUALISATION
# ============================================================
def plot_predictions(results: list):
    """5 KPIs × 3 horizons — générique N stratégies."""
    metrics_cfg = [
        ("roas",        "ROAS prédit",  "x" ),
        ("conversions", "Conversions",  ""  ),
        ("cpa",         "CPA (€)",      "€" ),
        ("ctr",         "CTR (%)",      ""  ),
        ("cpc",         "CPC (€)",      "€" ),
    ]
    horizons = ["J+3", "J+7", "J+14"]
    COLORS   = ["#E74C3C","#2ECC71","#3498DB","#F39C12","#9B59B6","#1ABC9C"]
    n        = len(results)
    colors   = [COLORS[i % len(COLORS)] for i in range(n)]

    fig, axes = plt.subplots(len(metrics_cfg), 3, figsize=(14, 14))
    fig.suptitle(
        "Prédictions ML — 5 KPIs × 3 Horizons\n"
        "(LightGBM  ·  best_model.pkl  ·  score adapté à l'objectif)",
        fontsize=12, fontweight="bold"
    )

    for ri, (metric, title, unit) in enumerate(metrics_cfg):
        for ci, horizon in enumerate(horizons):
            ax   = axes[ri, ci]
            vals = [r["predictions"][metric][horizon] * (100 if metric == "ctr" else 1)
                    for r in results]
            bars = ax.bar(range(n), vals, color=colors,
                          edgecolor="white", linewidth=0.8, width=0.55)
            ax.set_title(f"{title}\n{horizon}", fontsize=8, fontweight="bold")
            ax.set_xticks(range(n))
            ax.set_xticklabels([f"[{r['strategy_id']}]" for r in results],
                               fontsize=7, rotation=30 if n > 4 else 0)
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                fmt = (f"{int(val)}"      if metric == "conversions" else
                       f"{val:.3f}%"      if metric == "ctr" else
                       f"{val:.3f}{unit}")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals, default=1) * 0.03,
                        fmt, ha="center", va="bottom",
                        fontsize=6, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Graphique : {OUTPUT_CHART}")

# ============================================================
# ÉTAPE 5 — SAUVEGARDE
# ============================================================
def save_results(results: list, objectif: str):
    log(sep()); log("ÉTAPE 5 — SAUVEGARDE"); log(sep())

    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  JSON    : {OUTPUT_JSON}")

    # CSV
    rows = []
    for r in results:
        row = {k: r[k] for k in
               ["strategy_id","strategy_type","plateforme","objectif",
                "budget","CPC_cible","CTR_cible","conv_rate"]}
        for metric in ["roas", "conversions", "cpa", "ctr", "cpc"]:
            for h in ["J+3", "J+7", "J+14"]:
                v = r["predictions"][metric][h]
                row[f"{metric}_{h.replace('+','')}"] = round(
                    v * 100 if metric == "ctr" else v, 4
                )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    log(f"  CSV     : {OUTPUT_CSV}")

    # Rapport TXT
    fmts = {
        "roas"       : lambda v: f"{v:.3f}x",
        "conversions": lambda v: f"{int(v)}",
        "cpa"        : lambda v: f"{v:.2f}€",
        "ctr"        : lambda v: f"{v*100:.3f}%",
        "cpc"        : lambda v: f"{v:.3f}€",
    }
    kpi_labels = {
        "roas":"ROAS","conversions":"Conversions",
        "cpa":"CPA","ctr":"CTR","cpc":"CPC"
    }
    formula = SCORING_BY_OBJECTIF.get(objectif, SCORING_BY_OBJECTIF["conversions"])
    best    = max(results, key=lambda r: compute_score(r, objectif))

    lines = [
        sep(), "RAPPORT PRÉDICTIONS ML — AdOptimizer AI (Cas 2)", sep(),
        f"Modèle    : LightGBM (best_model.pkl)",
        f"Features  : features_strategies_pro.csv (187 features exactes)",
        f"Targets   : 5 KPIs × 3 horizons (J+3, J+7, J+14)",
        f"Objectif  : {objectif}",
        f"Scoring   : {formula['label']}", "",
    ]
    for r in results:
        lines += [
            f"\n[{r['strategy_id']}] {r['strategy_type'].upper()} "
            f"({r['plateforme']}) — Budget={r['budget']:,.0f}€",
            f"  {'KPI':<15} {'J+3':>10} {'J+7':>10} {'J+14':>10}",
            f"  {'-'*47}",
        ]
        for m, fmt in fmts.items():
            p = r["predictions"][m]
            lines.append(
                f"  {kpi_labels[m]:<15} "
                f"{fmt(p['J+3']):>10} "
                f"{fmt(p['J+7']):>10} "
                f"{fmt(p['J+14']):>10}"
            )
        sc = compute_score(r, objectif)
        lines.append(f"  {'Score':<15} {sc:>10.4f}")

    sc_best = compute_score(best, objectif)
    lines += [
        "", sep("-"),
        f"MEILLEURE STRATÉGIE — Score adapté à l'objectif '{objectif}'",
        sep("-"),
        f"  [{best['strategy_id']}] {best['strategy_type'].upper()} ({best['plateforme']})",
        f"  Formule          : {formula['label']}",
        f"  Score J+14       : {sc_best:.4f}",
        f"  ROAS J+14        : {best['predictions']['roas']['J+14']:.3f}x",
        f"  Conversions J+14 : {int(best['predictions']['conversions']['J+14'])}",
        f"  CPA J+14         : {best['predictions']['cpa']['J+14']:.2f}€",
        f"  CTR J+14         : {best['predictions']['ctr']['J+14']*100:.3f}%",
        f"  CPC J+14         : {best['predictions']['cpc']['J+14']:.3f}€",
        "", sep(), "FIN DU RAPPORT", sep(),
    ]
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  Rapport : {OUTPUT_REPORT}")

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def run_predictor_pipeline():
    log(sep("=")); log("  PRÉDICTEUR ML — AdOptimizer AI (Cas 2)"); log(sep("="))
    log("  Input  : features_strategies_pro.csv → best_model.pkl")
    log("  Output : predictions J+3 / J+7 / J+14")
    log("")

    bundle      = load_bundle()
    df_features = load_features(bundle)
    results     = predict(df_features, bundle)

    # Récupérer l'objectif depuis les stratégies
    objectif = results[0]["objectif"] if results else "conversions"
    formula  = SCORING_BY_OBJECTIF.get(objectif, SCORING_BY_OBJECTIF["conversions"])
    log(f"\n  Objectif détecté : {objectif}")
    log(f"  Formule score    : {formula['label']}")

    log(sep()); log("ÉTAPE 4 — VISUALISATION"); log(sep())
    plot_predictions(results)

    save_results(results, objectif)

    # Résumé
    best = max(results, key=lambda r: compute_score(r, objectif))
    log(sep("=")); log("  RÉSUMÉ FINAL"); log(sep("="))
    log(f"\n  Objectif : {objectif}  |  Score : {formula['label']}")
    log(f"\n  {'ID':<6} {'Type':<12} {'Plat':<8} {'ROAS J+14':>10} "
        f"{'Conv':>6} {'CPA':>8} {'Score':>8}")
    log(f"  {'-'*62}")
    for r in results:
        p  = r["predictions"]
        sc = compute_score(r, objectif)
        flag = " ←" if r == best else ""
        log(f"  [{r['strategy_id']:<4}] {r['strategy_type']:<12} "
            f"{r['plateforme']:<8} {p['roas']['J+14']:>10.3f}x "
            f"{int(p['conversions']['J+14']):>6} "
            f"{p['cpa']['J+14']:>7.2f}€ {sc:>8.3f}{flag}")

    sc_best = compute_score(best, objectif)
    log(f"\n  ✅ Meilleure : [{best['strategy_id']}] "
        f"{best['strategy_type'].upper()} ({best['plateforme']}) "
        f"— score={sc_best:.4f}")
    log(f"     Formule  : {formula['label']}")
    for fp in [OUTPUT_JSON, OUTPUT_CSV, OUTPUT_REPORT, OUTPUT_CHART]:
        log(f"    → {fp}")
    log(sep("=")); log("  PRÉDICTEUR ML TERMINÉ — Prêt pour Comparaison"); log(sep("="))

    return results


# ============================================================
# WRAPPER FASTAPI / N8N
# ============================================================

def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def run_case2_prediction() -> dict:
    """
    Point d'entree FastAPI pour le cas 2 : prediction ML des strategies.

    Dependances :
      app/models/best_model.pkl
      app/cas2-outputs/feature_engineering_outputs/features_strategies_pro.csv
      app/cas2-outputs/agent2_outputs/strategies.json
    """
    try:
        results = run_predictor_pipeline()
        objectif = results[0]["objectif"] if results else "conversions"
        best = max(results, key=lambda r: compute_score(r, objectif)) if results else None

        return {
            "status": "success",
            "message": "Predictions cas 2 terminees",
            "input_files": {
                "model": str(MODEL_PATH),
                "features": str(INPUT_FEATURES),
                "strategies": str(INPUT_STRATEGIES),
            },
            "output_dir": str(OUTPUT_DIR),
            "output_files": {
                "json": str(OUTPUT_JSON),
                "csv": str(OUTPUT_CSV),
                "report": str(OUTPUT_REPORT),
                "chart": str(OUTPUT_CHART),
            },
            "objective": objectif,
            "predictions": len(results),
            "best_strategy": _json_safe(best),
            "results": _json_safe(results),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input_files": {
                "model": str(MODEL_PATH),
                "features": str(INPUT_FEATURES),
                "strategies": str(INPUT_STRATEGIES),
            },
        }


# ============================================================
if __name__ == "__main__":
    result = run_case2_prediction()
    print(result)
