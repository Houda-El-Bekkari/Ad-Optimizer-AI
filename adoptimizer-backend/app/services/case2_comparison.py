"""
===============================================================================
COMPARAISON DES STRATÉGIES — AdOptimizer AI  (Cas 2 : Nouvelle Campagne)
===============================================================================
Objectif : Comparer N stratégies candidates sur l'ensemble des KPIs prédits
           et sélectionner la meilleure de façon argumentée.

Améliorations vs version de base :
  ✅ WEIGHTS adaptés à l'objectif utilisateur
       conversions → ROAS×0.35 + Conv×0.25 − CPA×0.20 + CTR×0.10 − CPC×0.10
       leads       → Conv×0.35 + CPA×0.30  + CTR×0.20 − CPC×0.15
       awareness   → CTR×0.40  + Conv×0.25 − CPC×0.20 + ROAS×0.15
       traffic     → Conv×0.35 + CTR×0.30  − CPC×0.25 + ROAS×0.10
  ✅ Palette de couleurs générique (N stratégies quelconques)
  ✅ Générique : meta / google / both (3 ou 6 stratégies)
  ✅ Plateforme affichée dans le rapport et les graphiques

Input  :
  predictor_outputs/predictions.json
  agent2_outputs/strategies.json

Output :
  comparison_outputs/
    comparison_report.txt
    comparison_scores.csv
    comparison_chart.png
    comparison_radar.png

Auteur : AdOptimizer AI — PFE 2024
===============================================================================
"""

import os, json, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CASE2_OUTPUTS_DIR = BASE_DIR / "cas2-outputs"

INPUT_PREDICTIONS = CASE2_OUTPUTS_DIR / "predictor_outputs" / "predictions.json"
INPUT_STRATEGIES  = CASE2_OUTPUTS_DIR / "agent2_outputs" / "strategies.json"
OUTPUT_DIR        = CASE2_OUTPUTS_DIR / "comparison_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_REPORT = OUTPUT_DIR / "comparison_report.txt"
OUTPUT_CSV    = OUTPUT_DIR / "comparison_scores.csv"
OUTPUT_CHART  = OUTPUT_DIR / "comparison_chart.png"
OUTPUT_RADAR  = OUTPUT_DIR / "comparison_radar.png"
OUTPUT_PLAN_JSON = OUTPUT_DIR / "comparison_plan.json"

HORIZON = "J+14"

# ✅ Pondérations adaptées à l'objectif
# Chaque objectif a une logique business différente
WEIGHTS_BY_OBJECTIF = {
    "conversions": {
        "roas": 0.35, "conversions": 0.25,
        "cpa" : 0.20, "ctr": 0.10, "cpc": 0.10,
        "label": "ROAS×0.35 + Conv×0.25 − CPA×0.20 + CTR×0.10 − CPC×0.10",
    },
    "leads": {
        "roas": 0.15, "conversions": 0.35,
        "cpa" : 0.30, "ctr": 0.20, "cpc": 0.00,
        "label": "Conv×0.35 − CPA×0.30 + CTR×0.20 + ROAS×0.15",
    },
    "awareness": {
        "roas": 0.15, "conversions": 0.25,
        "cpa" : 0.00, "ctr": 0.40, "cpc": 0.20,
        "label": "CTR×0.40 + Conv×0.25 − CPC×0.20 + ROAS×0.15",
    },
    "traffic": {
        "roas": 0.10, "conversions": 0.35,
        "cpa" : 0.00, "ctr": 0.30, "cpc": 0.25,
        "label": "Conv×0.35 + CTR×0.30 − CPC×0.25 + ROAS×0.10",
    },
}

# ✅ Palette générique — jusqu'à 10 stratégies
COLORS = [
    "#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
    "#E91E63", "#00BCD4",
]

def sep(c="=", n=72): return c * n

def log(m):
    text = str(m)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

def get_color(i: int) -> str:
    return COLORS[i % len(COLORS)]

# ============================================================
# ÉTAPE 1 — CHARGEMENT
# ============================================================
def load_inputs() -> tuple:
    log(sep()); log("ÉTAPE 1 — CHARGEMENT DES PRÉDICTIONS"); log(sep())

    if not os.path.exists(INPUT_PREDICTIONS):
        raise FileNotFoundError(
            f"Prédictions introuvables : {INPUT_PREDICTIONS}\n"
            "Lancez predictor_ml_pro.py d'abord."
        )

    if not os.path.exists(INPUT_STRATEGIES):
        raise FileNotFoundError(
            f"Strategies introuvables : {INPUT_STRATEGIES}\n"
            "Lancez case2_strategy d'abord."
        )

    with open(INPUT_PREDICTIONS, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(INPUT_STRATEGIES, encoding="utf-8") as f:
        strategies_payload = json.load(f)
        strategies = strategies_payload["strategies"]

    metadata = strategies_payload.get("metadata", {})
    user_input = metadata.get("user_input", {})

    strat_map = {s["id"]: s for s in strategies}
    results   = []
    for p in predictions:
        sid = p["strategy_id"]
        results.append({**p, "params": strat_map.get(sid, {})})

    # Détecter l'objectif depuis les stratégies
    objectif = strategies[0].get("objectif", "conversions") if strategies else "conversions"
    weights  = WEIGHTS_BY_OBJECTIF.get(objectif, WEIGHTS_BY_OBJECTIF["conversions"])

    log(f"  Stratégies  : {[r['strategy_id'] for r in results]}")
    log(f"  Plateformes : {list(set(r.get('plateforme','?') for r in results))}")
    log(f"  Objectif    : {objectif}")
    log(f"  Pondération : {weights['label']}")
    log(f"  Horizon     : {HORIZON}")

    return results, objectif, weights, user_input

# ============================================================
# ÉTAPE 2 — SCORE MULTI-CRITÈRES
# ============================================================
def compute_scores(results: list, weights: dict) -> list:
    """
    Score composite normalisé sur [0, 10].

    Méthode :
      1. Extraire les valeurs J+14
      2. Normaliser chaque KPI sur [0, 1]
      3. Inverser CPA et CPC (plus bas = mieux)
      4. Pondérer selon WEIGHTS de l'objectif
      5. × 10 → score [0, 10]
    """
    log(sep()); log("ÉTAPE 2 — CALCUL DES SCORES MULTI-CRITÈRES"); log(sep())

    metrics = ["roas", "conversions", "cpa", "ctr", "cpc"]

    # Extraire KPIs à HORIZON
    for r in results:
        r["kpis"] = {
            m: r["predictions"][m][HORIZON] for m in metrics
        }

    # Normaliser
    for metric in metrics:
        vals = [r["kpis"][metric] for r in results]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1.0
        for r in results:
            norm = (r["kpis"][metric] - vmin) / rng
            if metric in ["cpa", "cpc"]:
                norm = 1 - norm
            r.setdefault("norm_kpis", {})[metric] = round(norm, 4)

    # Score pondéré × 10
    for r in results:
        score = sum(r["norm_kpis"][m] * weights[m] for m in metrics) * 10
        r["score_final"]  = round(score, 4)
        r["score_detail"] = {
            m: round(r["norm_kpis"][m] * weights[m] * 10, 4)
            for m in metrics
        }

    results = sorted(results, key=lambda x: x["score_final"], reverse=True)

    log(f"\n  {'ID':<6} {'Type':<12} {'Plat':<8} {'ROAS':>8} {'Conv':>6} "
        f"{'CPA':>8} {'CTR':>8} {'CPC':>7} {'Score':>8}")
    log(f"  {'-'*74}")
    for r in results:
        p    = r["kpis"]
        plat = r.get("plateforme", r.get("params", {}).get("plateforme", "?"))
        log(f"  [{r['strategy_id']:<4}] {r['strategy_type']:<12} {plat:<8} "
            f"{p['roas']:>8.3f} {p['conversions']:>6.0f} "
            f"{p['cpa']:>8.2f} {p['ctr']*100:>7.3f}% "
            f"{p['cpc']:>7.3f} {r['score_final']:>8.4f}")

    log(f"\n  Pondération : {weights['label']}")
    return results

# ============================================================
# ÉTAPE 3 — CLASSEMENT ET JUSTIFICATION
# ============================================================
def build_platform_plan(ranking: list, user_input: dict) -> dict:
    """
    Conserve le gagnant global historique, et ajoute les gagnants par canal.

    Pour plateforme="both", on obtient un plan clair:
      - meilleure strategie Meta
      - meilleure strategie Google
      - budget alloue
      - budget restant en reserve
    """
    best_by_platform = {}

    for entry in ranking:
        platform = entry.get("plateforme")
        if platform in ("meta", "google") and platform not in best_by_platform:
            best_by_platform[platform] = entry

    requested_platform = str(user_input.get("plateforme", "")).lower().strip()
    total_budget = float(user_input.get("budget", 0) or 0)

    multi_platform_plan = None
    if requested_platform == "both" and best_by_platform.get("meta") and best_by_platform.get("google"):
        channels = []
        for platform in ("meta", "google"):
            entry = best_by_platform[platform]
            channels.append({
                "platform": platform,
                "strategy_id": entry.get("strategy_id"),
                "strategy_type": entry.get("strategy_type"),
                "budget": entry.get("budget", 0),
                "score_final": entry.get("score_final"),
                "kpis_h14": entry.get("kpis_h14", {}),
            })

        allocated_budget = sum(float(channel.get("budget", 0) or 0) for channel in channels)
        multi_platform_plan = {
            "mode": "both",
            "total_budget": total_budget,
            "allocated_budget": round(allocated_budget, 2),
            "reserve_budget": round(total_budget - allocated_budget, 2),
            "channels": channels,
        }

    return {
        "best_by_platform": best_by_platform,
        "multi_platform_plan": multi_platform_plan,
    }


def generate_ranking(results: list, objectif: str, user_input: dict | None = None) -> dict:
    log(sep()); log("ÉTAPE 3 — CLASSEMENT FINAL"); log(sep())

    medals  = ["🥇", "🥈", "🥉"]
    ranking = []

    for i, r in enumerate(results):
        sid    = r["strategy_id"]
        stype  = r["strategy_type"]
        score  = r["score_final"]
        kpis   = r["kpis"]
        params = r.get("params", {})
        plat   = r.get("plateforme", params.get("plateforme", "?"))
        budget = params.get("budget", r.get("budget", 0))

        # Justification adaptée à l'objectif
        if i == 0:
            justification = [
                f"Meilleur score global : {score:.4f}/10",
                f"ROAS J+14 = {kpis['roas']:.3f}x  |  "
                f"Conv = {int(kpis['conversions'])}  |  "
                f"CPA = {kpis['cpa']:.2f}€",
                f"Formule adaptée à l'objectif '{objectif}'",
            ]
        elif i == 1:
            justification = [
                f"Score intermédiaire : {score:.4f}/10",
                f"ROAS = {kpis['roas']:.3f}x  |  CPA = {kpis['cpa']:.2f}€",
            ]
        else:
            justification = [
                f"Score : {score:.4f}/10",
                f"CPA élevé : {kpis['cpa']:.2f}€",
                f"Budget {budget:,.0f}€ sans résultats supérieurs",
            ]

        entry = {
            "rank"         : i + 1,
            "medal"        : medals[i] if i < len(medals) else "  ",
            "strategy_id"  : sid,
            "strategy_type": stype,
            "plateforme"   : plat,
            "budget"       : budget,
            "score_final"  : score,
            "kpis_h14"     : kpis,
            "justification": justification,
        }
        ranking.append(entry)

        log(f"\n  {medals[i] if i < len(medals) else '  '} Rang {i+1} — "
            f"[{sid}] {stype.upper()} ({plat}) — Score={score:.4f}/10")
        for j in justification:
            log(f"       → {j}")

    best = ranking[0]
    platform_plan = build_platform_plan(ranking, user_input or {})
    log(f"\n  ✅ DÉCISION → [{best['strategy_id']}] "
        f"{best['strategy_type'].upper()} ({best['plateforme']})")
    log(f"     Budget : {best['budget']:,.0f}€  |  "
        f"ROAS : {best['kpis_h14']['roas']:.3f}x  |  "
        f"CPA : {best['kpis_h14']['cpa']:.2f}€")

    if platform_plan["multi_platform_plan"]:
        log("\n  PLAN BOTH - meilleur choix par plateforme")
        for channel in platform_plan["multi_platform_plan"]["channels"]:
            log(
                f"     {channel['platform']:<6} -> "
                f"[{channel['strategy_id']}] {channel['strategy_type'].upper()} | "
                f"Budget : {channel['budget']:,.0f} EUR | "
                f"Score : {channel['score_final']:.4f}/10"
            )
        log(
            f"     Reserve : {platform_plan['multi_platform_plan']['reserve_budget']:,.0f} EUR"
        )

    return {
        "ranking": ranking,
        "best": best,
        "best_by_platform": platform_plan["best_by_platform"],
        "multi_platform_plan": platform_plan["multi_platform_plan"],
    }

# ============================================================
# ÉTAPE 4 — VISUALISATION
# ============================================================
def plot_comparison_chart(results: list, objectif: str, weights: dict):
    """Barplot comparatif — 5 KPIs à J+14, générique N stratégies."""
    kpi_cfg = [
        ("roas",        "ROAS J+14",       "x",  True ),
        ("conversions", "Conversions J+14", "",   True ),
        ("cpa",         "CPA J+14 (€)",     "€",  False),
        ("ctr",         "CTR J+14 (%)",     "%",  True ),
        ("cpc",         "CPC J+14 (€)",     "€",  False),
    ]
    n      = len(results)
    colors = [get_color(i) for i in range(n)]
    xlbls  = [f"[{r['strategy_id']}]" for r in results]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle(
        f"Comparaison des Stratégies — KPIs à {HORIZON}  |  Objectif : {objectif}\n"
        f"Score : {weights['label']}",
        fontsize=10, fontweight="bold"
    )

    for ax, (metric, title, unit, higher_better) in zip(axes, kpi_cfg):
        vals = [r["kpis"][metric] * (100 if metric == "ctr" else 1)
                for r in results]

        bars = ax.bar(range(n), vals, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.55)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(xlbls, fontsize=8, rotation=30 if n > 4 else 0)
        ax.grid(axis="y", alpha=0.3)

        arrow = "↑ mieux" if higher_better else "↓ mieux"
        ax.text(0.98, 0.98, arrow, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray", style="italic")

        for bar, val in zip(bars, vals):
            fmt = (f"{int(val)}"      if metric == "conversions" else
                   f"{val:.3f}%"      if metric == "ctr" else
                   f"{val:.3f}{unit}")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.03,
                    fmt, ha="center", va="bottom", fontsize=7, fontweight="bold")

    patches = [
        mpatches.Patch(
            color=get_color(i),
            label=f"[{r['strategy_id']}] {r['strategy_type']} "
                  f"({r.get('plateforme','?')}) "
                  f"score={r['score_final']:.2f}/10"
        )
        for i, r in enumerate(results)
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=min(n, 4), fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Barplot : {OUTPUT_CHART}")


def plot_radar_comparison(results: list):
    """Radar chart — profils normalisés, générique N stratégies."""
    radar_metrics = ["roas", "conversions", "cpa", "ctr", "cpc"]
    labels_radar  = ["ROAS", "Conv.", "CPA(inv)", "CTR", "CPC(inv)"]

    N      = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_title(
        f"Profils Normalisés à {HORIZON}\n(CPA et CPC inversés — plus haut = mieux)",
        fontsize=11, fontweight="bold", pad=20
    )

    for i, r in enumerate(results):
        color = get_color(i)
        plat  = r.get("plateforme", r.get("params", {}).get("plateforme", "?"))
        vals  = [r["norm_kpis"][m] for m in radar_metrics] + [r["norm_kpis"][radar_metrics[0]]]
        ax.plot(angles, vals, color=color, linewidth=2.2,
                label=f"[{r['strategy_id']}] {r['strategy_type']} "
                      f"({plat}) {r['score_final']:.2f}/10")
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.15), fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_RADAR, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Radar   : {OUTPUT_RADAR}")

# ============================================================
# ÉTAPE 5 — SAUVEGARDE
# ============================================================
def save_results(results: list, ranking_data: dict,
                 objectif: str, weights: dict):
    log(sep()); log("ÉTAPE 5 — SAUVEGARDE"); log(sep())

    ranking = ranking_data["ranking"]
    best    = ranking_data["best"]
    best_by_platform = ranking_data.get("best_by_platform", {})
    multi_platform_plan = ranking_data.get("multi_platform_plan")
    metrics = ["roas", "conversions", "cpa", "ctr", "cpc"]

    # CSV
    rows = []
    for r in results:
        plat = r.get("plateforme", r.get("params", {}).get("plateforme", "?"))
        row  = {
            "strategy_id"  : r["strategy_id"],
            "strategy_type": r["strategy_type"],
            "plateforme"   : plat,
            "budget"       : r.get("params", {}).get("budget", r.get("budget", 0)),
            "score_final"  : r["score_final"],
        }
        for m in metrics:
            row[f"kpi_{m}_h14"] = round(r["kpis"][m], 4)
            row[f"norm_{m}"]    = r["norm_kpis"][m]
            row[f"score_{m}"]   = r["score_detail"][m]
        rows.append(row)
    pd.DataFrame(rows).sort_values("score_final", ascending=False).to_csv(
        OUTPUT_CSV, index=False
    )
    log(f"  CSV     : {OUTPUT_CSV}")

    with open(OUTPUT_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_strategy": best,
                "best_by_platform": best_by_platform,
                "multi_platform_plan": multi_platform_plan,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log(f"  Plan    : {OUTPUT_PLAN_JSON}")

    # Rapport TXT
    lines = [
        sep(), "RAPPORT COMPARAISON — AdOptimizer AI (Cas 2)", sep(),
        f"Objectif    : {objectif}",
        f"Horizon     : {HORIZON}",
        f"Stratégies  : {len(results)}",
        f"Pondération : {weights['label']}", "",
        sep("-"), "CLASSEMENT FINAL", sep("-"),
    ]
    for entry in ranking:
        lines += [
            f"\n{entry['medal']} Rang {entry['rank']} — "
            f"[{entry['strategy_id']}] {entry['strategy_type'].upper()} "
            f"({entry['plateforme']})",
            f"  Score global  : {entry['score_final']:.4f} / 10",
            f"  Budget        : {entry['budget']:,.0f}€",
            f"  ROAS J+14     : {entry['kpis_h14']['roas']:.3f}x",
            f"  Conversions   : {int(entry['kpis_h14']['conversions'])}",
            f"  CPA J+14      : {entry['kpis_h14']['cpa']:.2f}€",
            f"  CTR J+14      : {entry['kpis_h14']['ctr']*100:.3f}%",
            f"  CPC J+14      : {entry['kpis_h14']['cpc']:.3f}€",
            "  Justification :",
        ]
        for j in entry["justification"]:
            lines.append(f"    → {j}")

    lines += [
        "",
        sep("-"), "TABLEAU COMPARATIF COMPLET", sep("-"),
        f"  {'ID':<6} {'Type':<12} {'Plat':<8} "
        f"{'ROAS':>8} {'Conv':>6} {'CPA':>8} {'CTR':>8} {'Score':>8}",
        f"  {'-'*66}",
    ]
    for r in results:
        p    = r["kpis"]
        plat = r.get("plateforme", r.get("params", {}).get("plateforme", "?"))
        lines.append(
            f"  [{r['strategy_id']:<4}] {r['strategy_type']:<12} {plat:<8} "
            f"{p['roas']:>8.3f}x {int(p['conversions']):>6} "
            f"{p['cpa']:>7.2f}€ {p['ctr']*100:>7.3f}% {r['score_final']:>8.4f}"
        )

    lines += [
        "",
        sep("-"), "DÉCISION FINALE", sep("-"),
        f"  Stratégie     : [{best['strategy_id']}] "
        f"{best['strategy_type'].upper()} ({best['plateforme']})",
        f"  Budget        : {best['budget']:,.0f}€",
        f"  ROAS J+14     : {best['kpis_h14']['roas']:.3f}x",
        f"  Conversions   : {int(best['kpis_h14']['conversions'])}",
        f"  CPA J+14      : {best['kpis_h14']['cpa']:.2f}€",
        "",
        sep("-"), "PROCHAINE ÉTAPE : XAI + LLM", sep("-"),
        "  Les résultats sont transmis au module XAI pour expliquer",
        "  les décisions, puis au LLM pour la réponse finale à l'utilisateur.",
        "", sep(), "FIN DU RAPPORT", sep(),
    ]
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  Rapport : {OUTPUT_REPORT}")

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def run_comparison_pipeline() -> dict:
    log(sep("=")); log("  COMPARAISON DES STRATÉGIES — AdOptimizer AI (Cas 2)"); log(sep("="))

    results, objectif, weights, user_input = load_inputs()
    results                    = compute_scores(results, weights)
    ranking_data               = generate_ranking(results, objectif, user_input)

    log(sep()); log("ÉTAPE 4 — VISUALISATIONS"); log(sep())
    plot_comparison_chart(results, objectif, weights)
    plot_radar_comparison(results)

    save_results(results, ranking_data, objectif, weights)

    # Résumé
    best = ranking_data["best"]
    log(sep("=")); log("  RÉSUMÉ FINAL"); log(sep("="))
    log(f"\n  Objectif : {objectif}  |  {weights['label']}")
    log(f"\n  {'Rang':<5} {'ID':<6} {'Type':<12} {'Plat':<8} "
        f"{'Score':>8} {'ROAS':>8} {'Conv':>6} {'CPA':>8}")
    log(f"  {'-'*66}")
    medals = ["🥇","🥈","🥉"]
    for i, r in enumerate(results):
        plat = r.get("plateforme", r.get("params", {}).get("plateforme", "?"))
        log(f"  {medals[i] if i<3 else '  ':<5} "
            f"[{r['strategy_id']:<4}] {r['strategy_type']:<12} {plat:<8} "
            f"{r['score_final']:>8.4f} "
            f"{r['kpis']['roas']:>8.3f}x "
            f"{int(r['kpis']['conversions']):>6} "
            f"{r['kpis']['cpa']:>7.2f}€")

    log(f"\n  ✅ DÉCISION → [{best['strategy_id']}] "
        f"{best['strategy_type'].upper()} ({best['plateforme']})")
    log(f"     Budget : {best['budget']:,.0f}€  |  "
        f"ROAS : {best['kpis_h14']['roas']:.3f}x  |  "
        f"CPA : {best['kpis_h14']['cpa']:.2f}€")
    for fp in [OUTPUT_CSV, OUTPUT_REPORT, OUTPUT_CHART, OUTPUT_RADAR, OUTPUT_PLAN_JSON]:
        log(f"    → {fp}")
    log(sep("=")); log("  COMPARAISON TERMINÉE — Prêt pour XAI + LLM"); log(sep("="))

    ranking_data["objectif"] = objectif
    ranking_data["weights_label"] = weights["label"]
    ranking_data["horizon"] = HORIZON
    ranking_data["user_input"] = user_input

    return ranking_data


# ============================================================
def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def run_case2_comparison() -> dict:
    try:
        ranking_data = run_comparison_pipeline()
        return {
            "status": "success",
            "message": "Comparaison cas 2 terminee",
            "objectif": ranking_data.get("objectif"),
            "horizon": ranking_data.get("horizon"),
            "weights": ranking_data.get("weights_label"),
            "input_files": {
                "predictions": str(INPUT_PREDICTIONS),
                "strategies": str(INPUT_STRATEGIES),
            },
            "output_dir": str(OUTPUT_DIR),
            "output_files": {
                "report": str(OUTPUT_REPORT),
                "scores": str(OUTPUT_CSV),
                "chart": str(OUTPUT_CHART),
                "radar": str(OUTPUT_RADAR),
                "plan": str(OUTPUT_PLAN_JSON),
            },
            "best_strategy": _json_safe(ranking_data.get("best")),
            "best_by_platform": _json_safe(ranking_data.get("best_by_platform", {})),
            "multi_platform_plan": _json_safe(ranking_data.get("multi_platform_plan")),
            "ranking": _json_safe(ranking_data.get("ranking", [])),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input_files": {
                "predictions": str(INPUT_PREDICTIONS),
                "strategies": str(INPUT_STRATEGIES),
            },
        }


# ============================================================
if __name__ == "__main__":
    result = run_case2_comparison()
    print(json.dumps(result, ensure_ascii=True, default=str))
