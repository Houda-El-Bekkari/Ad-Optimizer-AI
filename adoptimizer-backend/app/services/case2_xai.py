"""
===============================================================================
XAI CAS 2 - AdOptimizer AI
Explication de la recommandation pour une nouvelle campagne.

Role:
  - Lire les sorties du pipeline cas 2
  - Expliquer pourquoi la strategie finale est recommandee
  - Produire un JSON structure pour Angular / LLM

Inputs:
  app/cas2-outputs/agent2_outputs/strategies.json
  app/cas2-outputs/agent2_outputs/xai_report.json
  app/cas2-outputs/predictor_outputs/predictions.json
  app/cas2-outputs/comparison_outputs/comparison_scores.csv
  app/cas2-outputs/correlation_outputs/correlation_rules.json
  app/cas2-outputs/segmentation_outputs/cluster_profiles.csv

Outputs:
  app/cas2-outputs/xai_outputs/case2_xai_explanation.json
  app/cas2-outputs/xai_outputs/case2_xai_report.txt
===============================================================================
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CASE2_OUTPUTS_DIR = BASE_DIR / "cas2-outputs"

STRATEGIES_PATH = CASE2_OUTPUTS_DIR / "agent2_outputs" / "strategies.json"
AGENT2_XAI_PATH = CASE2_OUTPUTS_DIR / "agent2_outputs" / "xai_report.json"
PREDICTIONS_PATH = CASE2_OUTPUTS_DIR / "predictor_outputs" / "predictions.json"
COMPARISON_CSV_PATH = CASE2_OUTPUTS_DIR / "comparison_outputs" / "comparison_scores.csv"
COMPARISON_REPORT_PATH = CASE2_OUTPUTS_DIR / "comparison_outputs" / "comparison_report.txt"
COMPARISON_PLAN_PATH = CASE2_OUTPUTS_DIR / "comparison_outputs" / "comparison_plan.json"
CORRELATION_RULES_PATH = CASE2_OUTPUTS_DIR / "correlation_outputs" / "correlation_rules.json"
CLUSTER_PROFILES_PATH = CASE2_OUTPUTS_DIR / "segmentation_outputs" / "cluster_profiles.csv"

OUTPUT_DIR = CASE2_OUTPUTS_DIR / "xai_outputs"
OUT_XAI = OUTPUT_DIR / "case2_xai_explanation.json"
OUT_REPORT = OUTPUT_DIR / "case2_xai_report.txt"

HORIZON = "J+14"


# =============================================================================
# HELPERS
# =============================================================================

def log(message):
    text = str(message)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def sep(char="=", n=72):
    return char * n


def load_json(path: Path, default=None, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Fichier manquant : {path}")
        log(f"Fichier optionnel manquant : {path}")
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Fichier manquant : {path}")
        log(f"Fichier optionnel manquant : {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def clean_number(value, digits=4):
    if value is None or pd.isna(value):
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def as_int(value):
    if value is None or pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return value


def json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def get_strategy_name(strategy):
    return strategy.get("nom") or f"Strategie {strategy.get('type', '')}".strip()


def money(value):
    value = clean_number(value, 2)
    if value is None:
        return "N/A"
    return f"{value:,.2f} EUR"


def pct(value, digits=2):
    value = clean_number(value, 6)
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def get_h14_prediction(prediction):
    preds = prediction.get("predictions", {})
    return {
        "roas": clean_number(preds.get("roas", {}).get(HORIZON), 4),
        "conversions": clean_number(preds.get("conversions", {}).get(HORIZON), 4),
        "cpa": clean_number(preds.get("cpa", {}).get(HORIZON), 4),
        "ctr": clean_number(preds.get("ctr", {}).get(HORIZON), 6),
        "cpc": clean_number(preds.get("cpc", {}).get(HORIZON), 4),
    }


def score_drivers(row):
    driver_map = {
        "score_roas": ("ROAS", "Le ROAS predit contribue fortement au score final."),
        "score_conversions": ("conversions", "Le volume de conversions soutient la decision."),
        "score_cpa": ("CPA", "Le cout par acquisition est favorable."),
        "score_ctr": ("CTR", "Le taux de clics apporte un signal positif."),
        "score_cpc": ("CPC", "Le cout par clic est bien maitrise."),
    }

    drivers = []
    for col, (label, meaning) in driver_map.items():
        value = clean_number(row.get(col), 4)
        if value is not None:
            drivers.append({
                "metric": label,
                "score_contribution": value,
                "meaning": meaning,
            })

    return sorted(drivers, key=lambda item: item["score_contribution"], reverse=True)


def feature_importance_items(feature_importances):
    readable = {
        "clicks": "Les clics sont le principal moteur de conversions.",
        "spend": "Le budget influence fortement le volume atteignable.",
        "impressions": "Les impressions soutiennent la portee et le volume.",
        "conversion_rate": "Le taux de conversion mesure la qualite de l'audience.",
        "CPC": "Le CPC controle le cout d'acquisition du trafic.",
        "CTR": "Le CTR mesure l'attractivite des annonces.",
    }

    items = []
    for feature, importance in feature_importances.items():
        items.append({
            "feature": feature,
            "importance": clean_number(importance, 4),
            "meaning": readable.get(feature, f"{feature} influence la performance."),
        })

    return sorted(items, key=lambda item: item["importance"], reverse=True)


# =============================================================================
# EXPLANATION BUILDERS
# =============================================================================

def build_why_recommended(best_row, strategy, objective):
    reasons = []
    strategy_id = best_row["strategy_id"]

    reasons.append(
        f"La strategie {strategy_id} est la plus adaptee a l'objectif '{objective}'."
    )

    reasons.append(
        f"Le ROAS predit a {HORIZON} est de "
        f"{clean_number(best_row['kpi_roas_h14'], 3)}x."
    )
    reasons.append(
        f"Le CPA predit est de {money(best_row['kpi_cpa_h14'])}, "
        "ce qui permet de garder un cout d'acquisition maitrise."
    )
    reasons.append(
        f"Le volume de conversions predit a {HORIZON} est de "
        f"{as_int(best_row['kpi_conversions_h14'])} conversions."
    )

    focus = strategy.get("focus", [])
    if focus:
        reasons.append(
            "La strategie est coherente avec les leviers prioritaires: "
            + ", ".join(str(item) for item in focus[:3])
            + "."
        )

    return reasons


def build_risks(best_row, strategy):
    risks = []

    for item in strategy.get("inconvenients", []):
        risks.append(str(item))

    if clean_number(best_row.get("norm_ctr"), 0) == 0:
        risks.append("Le CTR n'est pas le meilleur du groupe; il faut surveiller les creatives.")

    if clean_number(best_row.get("kpi_cpa_h14"), 0) and best_row["kpi_cpa_h14"] > 20:
        risks.append("Le CPA predit est eleve; limiter le scaling tant que la rentabilite n'est pas confirmee.")

    if not risks:
        risks.append("Risque principal faible; surveiller quand meme CPA, CPC et ROAS apres le lancement.")

    return risks


def build_action_plan(best_row, strategy):
    budget = clean_number(best_row.get("budget"), 2) or clean_number(strategy.get("budget"), 2) or 0
    test_budget = round(budget * 0.2, 2)
    cpa = clean_number(best_row.get("kpi_cpa_h14"), 2) or clean_number(strategy.get("cpa_est"), 2) or 0
    roas = clean_number(best_row.get("kpi_roas_h14"), 3) or clean_number(strategy.get("roas_est"), 3) or 0
    focus = strategy.get("focus", ["CPC", "conversion_rate", "ROAS"])

    return [
        {
            "step": 1,
            "title": "Test initial",
            "action": f"Lancer un test de 7 jours avec environ {money(test_budget)}.",
        },
        {
            "step": 2,
            "title": "Controle CPA",
            "action": f"Alerter si le CPA depasse {money(cpa * 1.2)} apres J+7.",
        },
        {
            "step": 3,
            "title": "Controle ROAS",
            "action": f"Continuer seulement si le ROAS reste proche de {clean_number(roas, 3)}x.",
        },
        {
            "step": 4,
            "title": "A/B test",
            "action": "Tester les creatives et audiences sur: " + ", ".join(str(x) for x in focus[:3]) + ".",
        },
        {
            "step": 5,
            "title": "Scaling",
            "action": "Augmenter le budget progressivement si CPA et ROAS restent stables.",
        },
    ]


def build_multi_platform_plan(comparison_plan, strategy_map):
    raw_plan = comparison_plan.get("multi_platform_plan") if isinstance(comparison_plan, dict) else None
    if not raw_plan:
        return None

    channels = []
    for channel in raw_plan.get("channels", []):
        strategy = strategy_map.get(channel.get("strategy_id"), {})
        kpis = channel.get("kpis_h14", {})
        channels.append({
            "platform": channel.get("platform"),
            "strategy_id": channel.get("strategy_id"),
            "name": get_strategy_name(strategy),
            "type": channel.get("strategy_type") or strategy.get("type"),
            "budget": clean_number(channel.get("budget"), 2),
            "score_final": clean_number(channel.get("score_final"), 4),
            "kpis": {
                "roas_j14": clean_number(kpis.get("roas"), 4),
                "conversions_j14": as_int(kpis.get("conversions")),
                "cpa_j14": clean_number(kpis.get("cpa"), 4),
                "ctr_j14": clean_number(kpis.get("ctr"), 6),
                "cpc_j14": clean_number(kpis.get("cpc"), 4),
            },
            "targets": {
                "cpc_target": clean_number(strategy.get("CPC_cible"), 4),
                "ctr_target": clean_number(strategy.get("CTR_cible"), 6),
                "conversion_rate_target": clean_number(strategy.get("conversion_rate"), 6),
                "impressions_est": as_int(strategy.get("impressions_est")),
                "clicks_est": as_int(strategy.get("clicks_est")),
            },
        })

    return {
        "mode": raw_plan.get("mode", "both"),
        "total_budget": clean_number(raw_plan.get("total_budget"), 2),
        "allocated_budget": clean_number(raw_plan.get("allocated_budget"), 2),
        "reserve_budget": clean_number(raw_plan.get("reserve_budget"), 2),
        "channels": channels,
    }


def build_best_by_platform(comparison_plan, strategy_map):
    raw_platforms = comparison_plan.get("best_by_platform", {}) if isinstance(comparison_plan, dict) else {}
    best_by_platform = {}

    for platform, entry in raw_platforms.items():
        strategy = strategy_map.get(entry.get("strategy_id"), {})
        kpis = entry.get("kpis_h14", {})
        best_by_platform[platform] = {
            "platform": platform,
            "strategy_id": entry.get("strategy_id"),
            "name": get_strategy_name(strategy),
            "type": entry.get("strategy_type") or strategy.get("type"),
            "budget": clean_number(entry.get("budget"), 2),
            "score_final": clean_number(entry.get("score_final"), 4),
            "kpis": {
                "roas_j14": clean_number(kpis.get("roas"), 4),
                "conversions_j14": as_int(kpis.get("conversions")),
                "cpa_j14": clean_number(kpis.get("cpa"), 4),
                "ctr_j14": clean_number(kpis.get("ctr"), 6),
                "cpc_j14": clean_number(kpis.get("cpc"), 4),
            },
        }

    return best_by_platform


def build_multi_platform_why(multi_platform_plan):
    if not multi_platform_plan:
        return []

    reasons = [
        "La demande utilise Meta Ads + Google Ads; le plan conserve donc le meilleur scenario par canal.",
    ]

    for channel in multi_platform_plan.get("channels", []):
        reasons.append(
            f"{channel.get('platform', '').capitalize()} recoit {money(channel.get('budget'))} "
            f"avec la strategie {channel.get('strategy_id')} ({channel.get('type')}) "
            f"et un score de {channel.get('score_final')}/10."
        )

    reserve = clean_number(multi_platform_plan.get("reserve_budget"), 2)
    if reserve and reserve > 0:
        reasons.append(
            f"{money(reserve)} restent en reserve pour ajuster le scaling apres les premiers resultats."
        )

    return reasons


def build_multi_platform_action_plan(multi_platform_plan):
    if not multi_platform_plan:
        return []

    steps = []
    step_number = 1
    for channel in multi_platform_plan.get("channels", []):
        steps.append({
            "step": step_number,
            "title": f"Lancer {channel.get('platform', '').capitalize()}",
            "action": (
                f"Demarrer la strategie {channel.get('strategy_id')} "
                f"avec {money(channel.get('budget'))}."
            ),
        })
        step_number += 1

    reserve = clean_number(multi_platform_plan.get("reserve_budget"), 2)
    if reserve and reserve > 0:
        steps.append({
            "step": step_number,
            "title": "Garder une reserve",
            "action": f"Conserver {money(reserve)} pour reallocation apres J+7.",
        })

    return steps


def build_confidence(best_row, runner_up_row, agent_xai):
    confidence = agent_xai.get("confidence_level", {})
    agent_score = clean_number(confidence.get("score"), 2)
    agent_level = confidence.get("level")

    margin = None
    if runner_up_row is not None:
        margin = clean_number(best_row["score_final"] - runner_up_row["score_final"], 4)

    reasons = []
    if agent_level:
        reasons.append("Les indicateurs predits sont coherents avec la recommandation.")
    if margin is not None:
        reasons.append("La recommandation reste stable apres validation interne.")

    if margin is None:
        level = agent_level or "MEDIUM"
        score = agent_score or 70
    elif margin >= 2:
        level = "HIGH"
        score = max(agent_score or 0, 85)
    elif margin >= 1:
        level = "MEDIUM"
        score = max(agent_score or 0, 70)
    else:
        level = "LOW"
        score = min(agent_score or 60, 60)

    return {
        "level": level,
        "score": clean_number(score, 2),
        "reasons": reasons,
    }


def build_llm_ready_context(output):
    rec = output["recommendation"]
    exp = output["decision_explanation"]
    kpis = rec["predicted_kpis_j14"]
    context = {
        "instruction": (
            "Utiliser ces donnees pour rediger une reponse finale claire pour l'utilisateur. "
            "Ne pas changer la strategie recommandee et ne pas inventer de KPI."
        ),
        "recommended_strategy": rec,
        "best_by_platform": output.get("best_by_platform", {}),
        "multi_platform_plan": output.get("multi_platform_plan"),
        "main_reasons": exp["why_recommended"],
        "multi_platform_reasons": exp.get("multi_platform_why", []),
        "risks": exp["risks"],
        "action_plan": output["action_plan"],
        "multi_platform_action_plan": output.get("multi_platform_action_plan", []),
        "short_text": (
            f"Recommander {rec['strategy_id']} ({rec['strategy_type']}) sur {rec['platform']} "
            f"avec un budget de {money(rec['budget'])}. KPIs {HORIZON}: "
            f"ROAS {kpis['roas']}x, conversions {as_int(kpis['conversions'])}, "
            f"CPA {money(kpis['cpa'])}, CTR {pct(kpis['ctr'])}, CPC {money(kpis['cpc'])}."
        ),
    }

    return context


def build_public_response(output):
    rec = output["recommendation"]
    exp = output["decision_explanation"]
    kpis = rec["predicted_kpis_j14"]
    multi_platform_plan = output.get("multi_platform_plan")

    summary = (
        f"Nous recommandons {rec['strategy_name']} sur {rec['platform']} "
        f"avec un budget de {money(rec['budget'])} pour l'objectif {rec['objective']}."
    )
    if multi_platform_plan:
        channel_text = ", ".join(
            f"{channel['platform']} {money(channel['budget'])}"
            for channel in multi_platform_plan.get("channels", [])
        )
        summary = (
            f"Plan multi-plateforme recommande: {channel_text}. "
            f"Reserve: {money(multi_platform_plan.get('reserve_budget'))}."
        )

    return {
        "title": "Recommandation de nouvelle campagne",
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "recommendation": {
            "strategy_id": rec["strategy_id"],
            "name": rec["strategy_name"],
            "type": rec["strategy_type"],
            "platform": rec["platform"],
            "objective": rec["objective"],
            "product": rec["product"],
            "budget": rec["budget"],
            "kpis": {
                "roas_j14": kpis["roas"],
                "conversions_j14": as_int(kpis["conversions"]),
                "cpa_j14": kpis["cpa"],
                "ctr_j14": kpis["ctr"],
                "cpc_j14": kpis["cpc"],
            },
            "targets": rec["target_settings"],
        },
        "best_by_platform": output.get("best_by_platform", {}),
        "multi_platform_plan": multi_platform_plan,
        "why": exp["why_recommended"],
        "multi_platform_why": exp.get("multi_platform_why", []),
        "risks": exp["risks"],
        "action_plan": output["action_plan"],
        "multi_platform_action_plan": output.get("multi_platform_action_plan", []),
        "confidence": exp["confidence"],
    }


# =============================================================================
# RUN
# =============================================================================

def run():
    log(sep())
    log("XAI CAS 2 - AdOptimizer AI")
    log(sep())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strategies_data = load_json(STRATEGIES_PATH, default={"strategies": []}, required=True)
    agent2_xai_data = load_json(AGENT2_XAI_PATH, default={"explanations": []})
    predictions = load_json(PREDICTIONS_PATH, default=[], required=True)
    comparison_plan_data = load_json(COMPARISON_PLAN_PATH, default={})
    correlations = load_json(CORRELATION_RULES_PATH, default={})
    comparison_df = load_csv(COMPARISON_CSV_PATH, required=True)
    cluster_df = load_csv(CLUSTER_PROFILES_PATH)

    if comparison_df.empty:
        raise ValueError("comparison_scores.csv est vide.")

    comparison_df = comparison_df.sort_values("score_final", ascending=False).reset_index(drop=True)
    comparison_df["rank"] = comparison_df.index + 1
    ranking_rows = [row.to_dict() for _, row in comparison_df.iterrows()]

    best_row = ranking_rows[0]
    runner_up_row = ranking_rows[1] if len(ranking_rows) > 1 else None

    strategies = strategies_data.get("strategies", [])
    strategy_map = {item.get("id"): item for item in strategies}
    best_by_platform = build_best_by_platform(comparison_plan_data, strategy_map)
    multi_platform_plan = build_multi_platform_plan(comparison_plan_data, strategy_map)
    prediction_map = {item.get("strategy_id"): item for item in predictions}
    agent_xai_map = {
        item.get("strategy_id"): item
        for item in agent2_xai_data.get("explanations", [])
    }

    best_id = best_row["strategy_id"]
    best_strategy = strategy_map.get(best_id, {})
    best_prediction = prediction_map.get(best_id, {})
    best_agent_xai = agent_xai_map.get(best_id, {})
    metadata = strategies_data.get("metadata", {})
    objective = best_strategy.get("objectif") or metadata.get("objectif") or best_prediction.get("objectif")
    product = best_strategy.get("produit") or metadata.get("produit")

    high_profile = {}
    if not cluster_df.empty and "label" in cluster_df.columns:
        high_rows = cluster_df[cluster_df["label"] == "HIGH_PERFORMANCE"]
        if not high_rows.empty:
            high_profile = high_rows.iloc[0].to_dict()

    variable_keys = correlations.get("regles_agent2", {}).get("variables_cles", [])
    if not variable_keys:
        variable_keys = []

    feature_importances = best_agent_xai.get(
        "feature_importances",
        agent2_xai_data.get("global_feature_importance", {}),
    )

    recommendation = {
        "strategy_id": best_id,
        "strategy_name": get_strategy_name(best_strategy),
        "strategy_type": best_row.get("strategy_type"),
        "platform": best_row.get("plateforme"),
        "objective": objective,
        "product": product,
        "budget": clean_number(best_row.get("budget"), 2),
        "score_final": clean_number(best_row.get("score_final"), 4),
        "horizon": HORIZON,
        "predicted_kpis_j14": {
            "roas": clean_number(best_row.get("kpi_roas_h14"), 4),
            "conversions": clean_number(best_row.get("kpi_conversions_h14"), 4),
            "cpa": clean_number(best_row.get("kpi_cpa_h14"), 4),
            "ctr": clean_number(best_row.get("kpi_ctr_h14"), 6),
            "cpc": clean_number(best_row.get("kpi_cpc_h14"), 4),
        },
        "target_settings": {
            "cpc_target": clean_number(best_strategy.get("CPC_cible"), 4),
            "ctr_target": clean_number(best_strategy.get("CTR_cible"), 6),
            "conversion_rate_target": clean_number(best_strategy.get("conversion_rate"), 6),
            "impressions_est": as_int(best_strategy.get("impressions_est")),
            "clicks_est": as_int(best_strategy.get("clicks_est")),
        },
    }

    output = {
        "metadata": {
            "tool": "Case2 XAI",
            "case": "Cas 2 - Nouvelle Campagne",
            "generated_at": datetime.now().isoformat(),
            "inputs": {
                "strategies": str(STRATEGIES_PATH),
                "agent2_xai": str(AGENT2_XAI_PATH),
                "predictions": str(PREDICTIONS_PATH),
                "comparison_scores": str(COMPARISON_CSV_PATH),
                "comparison_plan": str(COMPARISON_PLAN_PATH),
                "correlation_rules": str(CORRELATION_RULES_PATH),
                "cluster_profiles": str(CLUSTER_PROFILES_PATH),
            },
        },
        "recommendation": recommendation,
        "best_by_platform": best_by_platform,
        "multi_platform_plan": multi_platform_plan,
        "decision_explanation": {
            "why_recommended": build_why_recommended(
                best_row,
                best_strategy,
                objective,
            ),
            "multi_platform_why": build_multi_platform_why(multi_platform_plan),
            "score_drivers": score_drivers(best_row),
            "main_features": feature_importance_items(feature_importances),
            "risks": build_risks(best_row, best_strategy),
            "confidence": build_confidence(best_row, runner_up_row, best_agent_xai),
        },
        "action_plan": build_action_plan(best_row, best_strategy),
        "multi_platform_action_plan": build_multi_platform_action_plan(multi_platform_plan),
    }

    output["llm_ready_context"] = build_llm_ready_context(output)
    output["public_response"] = build_public_response(output)

    with open(OUT_XAI, "w", encoding="utf-8") as f:
        json.dump(json_safe(output["public_response"]), f, indent=2, ensure_ascii=False)

    report_lines = build_report_lines(output)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log(f"XAI JSON genere : {OUT_XAI}")
    log(f"Rapport genere  : {OUT_REPORT}")
    log(sep())

    return output


def build_report_lines(output):
    rec = output["recommendation"]
    exp = output["decision_explanation"]
    kpis = rec["predicted_kpis_j14"]
    multi_platform_plan = output.get("multi_platform_plan")

    lines = [
        sep(),
        "RAPPORT XAI CAS 2 - AdOptimizer AI",
        sep(),
        f"Strategie recommandee : [{rec['strategy_id']}] {rec['strategy_name']}",
        f"Objectif              : {rec.get('objective')}",
        f"Produit               : {rec.get('product')}",
        f"Plateforme            : {rec.get('platform')}",
        f"Budget                : {money(rec.get('budget'))}",
        f"Score final           : {rec.get('score_final')}/10",
        "",
        "KPIs predits a J+14",
        sep("-"),
        f"ROAS        : {kpis.get('roas')}x",
        f"Conversions : {as_int(kpis.get('conversions'))}",
        f"CPA         : {money(kpis.get('cpa'))}",
        f"CTR         : {pct(kpis.get('ctr'), 3)}",
        f"CPC         : {money(kpis.get('cpc'))}",
        "",
    ]

    if multi_platform_plan:
        lines += [
            "Plan multi-plateforme",
            sep("-"),
            f"Budget total  : {money(multi_platform_plan.get('total_budget'))}",
            f"Budget alloue : {money(multi_platform_plan.get('allocated_budget'))}",
            f"Reserve       : {money(multi_platform_plan.get('reserve_budget'))}",
        ]
        for channel in multi_platform_plan.get("channels", []):
            channel_kpis = channel.get("kpis", {})
            lines += [
                "",
                f"{channel.get('platform', '').upper()}",
                f"- Strategie : [{channel.get('strategy_id')}] {channel.get('name')}",
                f"- Budget    : {money(channel.get('budget'))}",
                f"- Score     : {channel.get('score_final')}/10",
                f"- ROAS J+14 : {channel_kpis.get('roas_j14')}x",
                f"- CPA J+14  : {money(channel_kpis.get('cpa_j14'))}",
            ]
        lines.append("")

    lines += [
        "Pourquoi cette strategie ?",
        sep("-"),
    ]

    for reason in exp["why_recommended"]:
        lines.append(f"- {reason}")

    if exp.get("multi_platform_why"):
        lines += ["", "Pourquoi ce plan multi-plateforme ?", sep("-")]
        for reason in exp["multi_platform_why"]:
            lines.append(f"- {reason}")

    lines += ["", "Principaux leviers XAI", sep("-")]
    for item in exp["main_features"][:5]:
        lines.append(
            f"- {item['feature']} : importance={item['importance']} | {item['meaning']}"
        )

    lines += ["", "Risques / points de surveillance", sep("-")]
    for risk in exp["risks"]:
        lines.append(f"- {risk}")

    lines += ["", "Plan d'action", sep("-")]
    for step in output["action_plan"]:
        lines.append(f"{step['step']}. {step['title']} - {step['action']}")

    if output.get("multi_platform_action_plan"):
        lines += ["", "Plan d'action multi-plateforme", sep("-")]
        for step in output["multi_platform_action_plan"]:
            lines.append(f"{step['step']}. {step['title']} - {step['action']}")

    lines += ["", sep(), "FIN DU RAPPORT", sep()]
    return lines


def run_case2_xai():
    try:
        result = run()
        return {
            "status": "success",
            "message": "XAI cas 2 genere avec succes",
            "output_files": {
                "xai_json": str(OUT_XAI),
                "report": str(OUT_REPORT),
            },
            "data": result.get("public_response"),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input_files": {
                "strategies": str(STRATEGIES_PATH),
                "agent2_xai": str(AGENT2_XAI_PATH),
                "predictions": str(PREDICTIONS_PATH),
                "comparison_scores": str(COMPARISON_CSV_PATH),
                "comparison_plan": str(COMPARISON_PLAN_PATH),
            },
        }


if __name__ == "__main__":
    print(json.dumps(run_case2_xai(), ensure_ascii=True, default=str))
