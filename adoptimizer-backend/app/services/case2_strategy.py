"""
===============================================================================
AGENT 2 — MODULE DE DÉCISION STRATÉGIQUE  [VERSION 4 — SAAS DÉCIDEUR]
AdOptimizer AI — Cas 2 : Nouvelle Campagne Publicitaire
===============================================================================
Améliorations v4 (moteur décision SaaS) :

  ✅ AXE 1 — OBJECTIF UTILISÉ PARTOUT
     → scoring adapté par objectif (awareness ≠ conversions ≠ traffic ≠ leads)
     → CPC / CTR / CR forcés selon objectif dans la génération
     → logique business : awareness→volume+CTR | conversions→ROAS+CR | traffic→clicks

  ✅ AXE 2 — PRODUIT UTILISÉ (NOUVEAU)
     → PRODUCT_PROFILE_MAP : CPC_mult, CR_mult, CTR_mult par produit
     → SaaS ≠ Ecommerce ≠ Finance ≠ Immobilier
     → impact direct sur chaque KPI généré

  ✅ AXE 3 — TOUTES LES CORRÉLATIONS EXPLOITÉES
     → r_clicks, r_ctr, r_spend, r_cr, r_cpc
     → chaque corrélation pilote son KPI correspondant
     → budget_usage dynamique via r_spend

  ✅ AXE 4 — BUDGET INTELLIGENT
     → budget_usage ajusté selon r_spend + produit + objectif
     → pas de split fixe — ratio calculé dynamiquement

  ✅ AXE 5 — SCORING NORMALISÉ (safe_norm conservé + amélioré)
     → safe_norm sur toutes les formules
     → CPA ajouté comme KPI de scoring leads

  ✅ AXE 6 — CONTRAINTES MÉTIER FORTES
     → CPC trop élevé → pénalité −2 pts
     → clicks < seuil → pénalité −1.5 pts
     → ROAS < 1.0 → pénalité −1 pt + flag ELIMINER
     → conversions < 50 → pénalité −3 pts
     → CTR faible → pénalité −1 pt

  ✅ AXE 7 — EXPLORATION CONTRÔLÉE (conservée)
     → ±10% variation, seed fixe par stratégie

  ✅ AXE 8 — LOGIQUE BUSINESS RENFORCÉE
     → awareness : CTR↑, CPC↓, impressions↑
     → conversions : CR↑, ROAS↑
     → traffic : clicks↑, CPC↓
     → leads : CR↑, qualité audience

  ✅ AXE 9 — KPI RÉELS AJOUTÉS
     → impressions_est (déjà présent)
     → CPA = budget / conversions
     → efficiency = conversions / budget
     → reach_score = impressions / budget (awareness)

  ✅ AXE 10 — SORTIE CLAIRE
     → meilleure stratégie avec justification complète
     → corrélation + objectif + produit dans le trace XAI
     → rapport enrichi avec CPA et efficiency

Input  :
  - app/cas2-outputs/segmentation_outputs/cluster_profiles.csv
  - app/cas2-outputs/correlation_outputs/correlation_rules.json
  - user_input.json

Output :
  app/cas2-outputs/agent2_outputs/
    strategies.json
    xai_report.json
    agent2_report.txt
    strategies_comparison.png

Auteur : AdOptimizer AI — PFE 2024 (v4-saas-décideur)
===============================================================================
"""

import os
import json
import warnings
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CASE2_OUTPUTS_DIR = BASE_DIR / "cas2-outputs"

INPUT_PROFILES    = CASE2_OUTPUTS_DIR / "segmentation_outputs" / "cluster_profiles.csv"
INPUT_RULES       = CASE2_OUTPUTS_DIR / "correlation_outputs" / "correlation_rules.json"
OUTPUT_DIR        = CASE2_OUTPUTS_DIR / "agent2_outputs"

OUTPUT_STRATEGIES = OUTPUT_DIR / "strategies.json"
OUTPUT_XAI        = OUTPUT_DIR / "xai_report.json"
OUTPUT_REPORT     = OUTPUT_DIR / "agent2_report.txt"
OUTPUT_CHART      = OUTPUT_DIR / "strategies_comparison.png"

DEFAULT_USER_INPUT = {
    "objectif": "conversions",
    "budget": 5000,
    "plateforme": "meta",
    "produit": "ecommerce",
}

EXPLORATION_SEED  = 42
EXPLORATION_RATIO = 0.10   # ±10%

# ============================================================
# AXE 2 — VALEUR ET PROFIL PAR PRODUIT
# ============================================================
# Chaque produit a :
#   val       : valeur monétaire par conversion
#   cpc_mult  : CPC attendu vs référence (SaaS paie plus cher)
#   cr_mult   : taux de conversion relatif (ecommerce > SaaS)
#   ctr_mult  : CTR relatif (ecommerce visuel > finance)
#   budget_floor : budget minimum viable pour ce produit

PRODUCT_VALUE_MAP: Dict[str, float] = {
    "saas"      : 120.0,
    "formation" : 80.0,
    "ecommerce" : 25.0,
    "immobilier": 500.0,
    "finance"   : 200.0,
    "sante"     : 60.0,
    "autre"     : 20.0,
}

# AXE 2 — Profils produit complets (CPC, CR, CTR adaptés par secteur)
PRODUCT_PROFILE_MAP: Dict[str, Dict] = {
    "saas": {
        "cpc_mult"     : 1.40,   # SaaS → mots-clés chers
        "cr_mult"      : 0.80,   # cycle achat long → CR plus faible
        "ctr_mult"     : 0.90,   # audiences B2B → CTR plus faible
        "budget_floor" : 500.0,
        "description"  : "B2B SaaS — CPC élevé, cycle long, LTV haute",
    },
    "formation": {
        "cpc_mult"     : 1.10,
        "cr_mult"      : 1.00,
        "ctr_mult"     : 1.10,
        "budget_floor" : 300.0,
        "description"  : "Formation — audience motivée, CTR correct",
    },
    "ecommerce": {
        "cpc_mult"     : 0.90,   # ecom → CPCs compétitifs
        "cr_mult"      : 1.30,   # achat impulsif → CR plus élevé
        "ctr_mult"     : 1.20,   # visuels produits → CTR fort
        "budget_floor" : 200.0,
        "description"  : "E-commerce — achat impulsif, CR fort, visuels",
    },
    "immobilier": {
        "cpc_mult"     : 1.60,   # leads très chers
        "cr_mult"      : 0.50,   # cycle très long
        "ctr_mult"     : 0.80,
        "budget_floor" : 1000.0,
        "description"  : "Immobilier — CPC très élevé, lead de haute valeur",
    },
    "finance": {
        "cpc_mult"     : 1.50,
        "cr_mult"      : 0.70,
        "ctr_mult"     : 0.85,
        "budget_floor" : 800.0,
        "description"  : "Finance — secteur réglementé, coûts élevés",
    },
    "sante": {
        "cpc_mult"     : 1.20,
        "cr_mult"      : 0.90,
        "ctr_mult"     : 1.00,
        "budget_floor" : 400.0,
        "description"  : "Santé — secteur sensible, confiance clé",
    },
    "autre": {
        "cpc_mult"     : 1.00,
        "cr_mult"      : 1.00,
        "ctr_mult"     : 1.00,
        "budget_floor" : 100.0,
        "description"  : "Produit générique",
    },
}

def get_product_profile(produit: str) -> Dict:
    """Retourne le profil produit correspondant (AXE 2)."""
    prod = produit.lower().strip()
    for key, profile in PRODUCT_PROFILE_MAP.items():
        if key in prod:
            return profile
    return PRODUCT_PROFILE_MAP["autre"]

# ============================================================
# AXE 1 — PROFILS D'OBJECTIF POUR LA GÉNÉRATION
# ============================================================

OBJECTIVE_GENERATION_PROFILES: Dict[str, Dict] = {
    "conversions": {
        "description"    : "Maximiser CR et ROAS",
        "cpc_adj"        : 1.00,
        "ctr_adj"        : 1.00,
        "cr_boost"       : 1.20,   # CR boosté +20%
        "budget_usage"   : 1.00,
        "clicks_priority": False,
        "kpi_priority"   : ["conversion_rate", "roas_est", "conversions_est"],
        "business_logic" : "ROAS > 1.5 obligatoire | CR max | CPC maîtrisé",
    },
    "awareness": {
        "description"    : "Maximiser impressions et CTR",
        "cpc_adj"        : 0.80,   # CPC réduit → portée max
        "ctr_adj"        : 1.35,   # CTR amplifié
        "cr_boost"       : 0.75,   # CR réduit (volume > qualité)
        "budget_usage"   : 1.00,
        "clicks_priority": True,
        "kpi_priority"   : ["impressions_est", "CTR_cible", "reach_score"],
        "business_logic" : "Impressions max | CTR fort | CPA non critique",
    },
    "traffic": {
        "description"    : "Maximiser clicks à CPC bas",
        "cpc_adj"        : 0.75,   # CPC très réduit
        "ctr_adj"        : 1.15,
        "cr_boost"       : 0.85,
        "budget_usage"   : 1.00,
        "clicks_priority": True,
        "kpi_priority"   : ["clicks_est", "CTR_cible", "CPC_cible"],
        "business_logic" : "Clicks max | CPC minimal | Volume > Qualité",
    },
    "leads": {
        "description"    : "Maximiser leads qualifiés",
        "cpc_adj"        : 1.10,   # qualité > volume → CPC +10%
        "ctr_adj"        : 0.90,   # ciblage précis → CTR plus faible
        "cr_boost"       : 1.40,   # CR très boosté
        "budget_usage"   : 0.88,   # budget réduit → qualité
        "clicks_priority": False,
        "kpi_priority"   : ["conversion_rate", "conversions_est", "cpa_est"],
        "business_logic" : "CR max | CPA bas | Qualité audience prioritaire",
    },
}

# ============================================================
# AXE 1+5 — FORMULES DE SCORING NORMALISÉES PAR OBJECTIF
# ============================================================

def safe_norm(value, ref):
    """Normalise proprement par rapport à une référence (évite les biais d'échelle)."""
    return value / ref if ref > 0 else 0.0


SCORING_FORMULAS: Dict[str, Dict] = {
    "conversions": {
        "label"   : "ROAS/3×0.5 + conv/100×0.3 − CPC/2×0.2  [safe_norm]",
        "weights" : {"roas": 0.5, "conversions": 0.3, "cpc_penalty": 0.2},
        "function": lambda s: (
            safe_norm(s["roas_est"], 3)          * 0.5
            + safe_norm(s["conversions_est"], 100) * 0.3
            - safe_norm(s["CPC_cible"], 2)         * 0.2
        ),
        "kpi_focus": ["conversions_est", "roas_est", "CPC_cible"],
    },
    "awareness": {
        "label"   : "CTR/0.05×0.5 + imp/100k×0.35 + reach_score×0.15  [safe_norm]",
        "weights" : {"ctr": 0.5, "impressions": 0.35, "reach": 0.15},
        # AXE 9 — reach_score = impressions / budget (awareness spécifique)
        "function": lambda s: (
            safe_norm(s["CTR_cible"], 0.05)          * 0.50
            + safe_norm(s["impressions_est"], 100_000) * 0.35
            + safe_norm(s.get("reach_score", 0), 100)  * 0.15
        ),
        "kpi_focus": ["impressions_est", "CTR_cible", "reach_score"],
    },
    "traffic": {
        "label"   : "CTR/0.05×0.6 + clicks/1k×0.25 − CPC/2×0.15  [safe_norm]",
        "weights" : {"ctr": 0.6, "clicks": 0.25, "cpc_penalty": 0.15},
        "function": lambda s: (
            safe_norm(s["CTR_cible"], 0.05)      * 0.60
            + safe_norm(s["clicks_est"], 1_000)  * 0.25
            - safe_norm(s["CPC_cible"], 2)       * 0.15
        ),
        "kpi_focus": ["clicks_est", "CTR_cible", "CPC_cible"],
    },
    "leads": {
        "label"   : "conv/100×0.4 + CR/0.1×0.35 + (1−CPA/50)×0.25  [safe_norm]",
        "weights" : {"conversions": 0.4, "conv_rate": 0.35, "cpa": 0.25},
        # AXE 9 — CPA intégré dans le scoring leads
        "function": lambda s: (
            safe_norm(s["conversions_est"], 100)             * 0.40
            + safe_norm(s["conversion_rate"], 0.1)            * 0.35
            + (1.0 - safe_norm(s.get("cpa_est", 50), 50))    * 0.25
        ),
        "kpi_focus": ["conversions_est", "conversion_rate", "cpa_est"],
    },
}


def normalize_scores_minmax(strategies: List[dict],
                              target_min: float = 6.0,
                              target_max: float = 9.5) -> List[dict]:
    """
    AXE 5 — Normalisation min-max des scores bruts.
    Garantit la comparabilité entre stratégies de même plateforme.
    """
    raw = [s["score_brut"] for s in strategies]
    s_min, s_max = min(raw), max(raw)
    for s in strategies:
        if s_max > s_min:
            norm = target_min + (s["score_brut"] - s_min) / (s_max - s_min) * (target_max - target_min)
        else:
            norm = (target_min + target_max) / 2.0
        s["score_potentiel"] = round(norm, 2)
    return strategies

# ============================================================
# AXE 7 — MOTEUR D'EXPLORATION CONTRÔLÉE
# ============================================================

class ExplorationEngine:
    """±10% variation reproductible par seed fixe."""

    def __init__(self, ratio: float = EXPLORATION_RATIO, base_seed: int = EXPLORATION_SEED):
        self.ratio     = ratio
        self.base_seed = base_seed

    def _rng(self, strat_id: str) -> random.Random:
        seed = self.base_seed + sum(ord(c) for c in strat_id)
        return random.Random(seed)

    def apply(self, strat_id: str, cpc: float, cr: float,
              ref_ctr: float) -> Tuple[float, float, float, List[str]]:
        rng = self._rng(strat_id)
        d_cpc = rng.uniform(-self.ratio, self.ratio)
        d_cr  = rng.uniform(-self.ratio, self.ratio)
        d_ctr = rng.uniform(-self.ratio, self.ratio)

        cpc_e = max(0.05, cpc     * (1 + d_cpc))
        cr_e  = max(0.001, cr     * (1 + d_cr))
        ctr_e = max(0.001, ref_ctr * (1 + d_ctr))

        trace = [
            f"Exploration CPC : {cpc:.4f} → {cpc_e:.4f}  (Δ={d_cpc:+.2%})",
            f"Exploration CR  : {cr:.4f} → {cr_e:.4f}  (Δ={d_cr:+.2%})",
            f"Exploration CTR : {ref_ctr:.4f} → {ctr_e:.4f}  (Δ={d_ctr:+.2%})",
        ]
        return cpc_e, cr_e, ctr_e, trace


EXPLORER = ExplorationEngine()

# ============================================================
# MODULE XAI — EXPLICABILITÉ
# ============================================================

class ExplainabilityEngine:
    """
    Génère rapport XAI complet : feature_importances, decision_trace,
    counterfactuals, confidence_level.
    AXE 10 — trace enrichie avec objectif + produit + corrélations.
    """

    def explain_strategy(self, strategy: dict, key_factors: dict,
                          objectif: str, ref_cpc: float) -> dict:
        formula     = SCORING_FORMULAS[objectif]
        importances = key_factors["importance"]
        r_cpc    = importances.get("CPC",             -0.49)
        r_clicks = importances.get("clicks",          +0.89)
        r_cr     = importances.get("conversion_rate", +0.57)
        r_ctr    = importances.get("CTR",             +0.41)
        r_spend  = importances.get("spend",           +0.57)

        # Feature importances normalisées
        raw_imp = {
            "clicks"         : abs(r_clicks),
            "conversion_rate": abs(r_cr),
            "CPC"            : abs(r_cpc),
            "spend"          : abs(r_spend),
            "CTR"            : abs(r_ctr),
        }
        total = sum(raw_imp.values())
        feature_importances = {k: round(v / total, 4) for k, v in raw_imp.items()}

        obj_profile  = OBJECTIVE_GENERATION_PROFILES[objectif]
        prod_profile = get_product_profile(strategy["produit"])

        # AXE 10 — Decision trace enrichi
        decision_trace = [
            f"[INPUT]    objectif='{objectif}' | produit='{strategy['produit']}' | "
            f"budget={strategy['budget']:,.0f}€ | plateforme='{strategy['plateforme']}'",
            f"[BUSINESS] {obj_profile['business_logic']}",
            f"[PRODUIT]  {prod_profile['description']} | "
            f"CPC×{prod_profile['cpc_mult']:.2f} CR×{prod_profile['cr_mult']:.2f} CTR×{prod_profile['ctr_mult']:.2f}",
            f"[REF]      HIGH_PERFORMANCE → CTR={strategy['CTR_cible']:.4f} "
            f"CPC_ref={ref_cpc:.4f} CR_ref={strategy['conversion_rate']:.4f}",
            f"[CORR]     r_clicks={r_clicks:+.4f} | r_ctr={r_ctr:+.4f} | "
            f"r_spend={r_spend:+.4f} | r_cr={r_cr:+.4f} | r_cpc={r_cpc:+.4f}",
            f"[SCORE]    formule : {formula['label']}",
            f"[SCORE]    score_brut={strategy['score_brut']:.4f} "
            f"→ score_norm={strategy['score_potentiel']:.2f}/10",
            f"[KPI]      ROAS={strategy['roas_est']:.3f} | "
            f"conv={strategy['conversions_est']} | "
            f"CPA={strategy.get('cpa_est', 0):.2f}€ | "
            f"efficiency={strategy.get('efficiency', 0):.4f}",
        ]
        if strategy["contraintes"]:
            for c in strategy["contraintes"]:
                decision_trace.append(f"[CONTRAINTE] {c}")

        counterfactuals = self._compute_counterfactuals(strategy, objectif, ref_cpc)
        confidence      = self._compute_confidence(strategy, objectif, ref_cpc)

        return {
            "strategy_id"        : strategy["id"],
            "feature_importances": feature_importances,
            "decision_trace"     : decision_trace,
            "counterfactuals"    : counterfactuals,
            "confidence_level"   : confidence,
            "xai_summary"        : self._build_summary(strategy, feature_importances, confidence),
        }

    def _compute_counterfactuals(self, s: dict, objectif: str,
                                  ref_cpc: float) -> List[dict]:
        formula_fn = SCORING_FORMULAS[objectif]["function"]
        results    = []

        # Scénario 1 : CPC −10%
        s1 = dict(s)
        s1["CPC_cible"]       = s["CPC_cible"] * 0.90
        s1["clicks_est"]      = round(s["budget"] / s1["CPC_cible"])
        s1["conversions_est"] = round(s1["clicks_est"] * s["conversion_rate"])
        s1["roas_est"]        = round(
            (s1["conversions_est"] * s["val_per_conversion"]) / s["budget"], 3
        ) if s["budget"] > 0 else 0
        s1["cpa_est"]         = round(s["budget"] / s1["conversions_est"], 2) if s1["conversions_est"] > 0 else 9999
        score1 = round(formula_fn(s1), 4)
        results.append({
            "scenario"      : "CPC −10%",
            "delta_score"   : round(score1 - s["score_brut"], 4),
            "new_kpis"      : {"CPC_cible": round(s1["CPC_cible"], 4),
                               "clicks_est": s1["clicks_est"],
                               "conversions_est": s1["conversions_est"],
                               "roas_est": s1["roas_est"]},
            "interpretation": "CPC réduit → clicks ↑ → ROAS ↑" if score1 > s["score_brut"] else "Effet limité",
        })

        # Scénario 2 : CR +15%
        s2 = dict(s)
        s2["conversion_rate"] = s["conversion_rate"] * 1.15
        s2["conversions_est"] = round(s["clicks_est"] * s2["conversion_rate"])
        s2["roas_est"]        = round(
            (s2["conversions_est"] * s["val_per_conversion"]) / s["budget"], 3
        ) if s["budget"] > 0 else 0
        s2["cpa_est"]         = round(s["budget"] / s2["conversions_est"], 2) if s2["conversions_est"] > 0 else 9999
        score2 = round(formula_fn(s2), 4)
        results.append({
            "scenario"      : "Conv. rate +15%",
            "delta_score"   : round(score2 - s["score_brut"], 4),
            "new_kpis"      : {"conversion_rate": round(s2["conversion_rate"], 4),
                               "conversions_est": s2["conversions_est"],
                               "roas_est": s2["roas_est"]},
            "interpretation": "Meilleur ciblage → qualité audience ↑",
        })

        # Scénario 3 : Budget +20%
        s3 = dict(s)
        s3["budget"]          = s["budget"] * 1.20
        s3["clicks_est"]      = round(s3["budget"] / s["CPC_cible"])
        s3["conversions_est"] = round(s3["clicks_est"] * s["conversion_rate"])
        s3["impressions_est"] = round(s3["clicks_est"] / s["CTR_cible"]) if s["CTR_cible"] > 0 else s["impressions_est"]
        s3["roas_est"]        = round(
            (s3["conversions_est"] * s["val_per_conversion"]) / s3["budget"], 3
        ) if s3["budget"] > 0 else 0
        s3["cpa_est"]         = round(s3["budget"] / s3["conversions_est"], 2) if s3["conversions_est"] > 0 else 9999
        score3 = round(formula_fn(s3), 4)
        results.append({
            "scenario"      : "Budget +20%",
            "delta_score"   : round(score3 - s["score_brut"], 4),
            "new_kpis"      : {"budget": round(s3["budget"], 2),
                               "clicks_est": s3["clicks_est"],
                               "conversions_est": s3["conversions_est"],
                               "roas_est": s3["roas_est"]},
            "interpretation": "Budget scale → volume proportionnel",
        })

        return results

    def _compute_confidence(self, s: dict, objectif: str, ref_cpc: float) -> Dict:
        reasons = []
        score   = 100

        if s["CPC_cible"] > ref_cpc * 1.3:
            score -= 20; reasons.append("CPC élevé vs référence (−20)")
        if s["conversions_est"] < 50 and objectif != "awareness":
            score -= 25; reasons.append("Conversions faibles (−25)")
        if s["roas_est"] < 1.0 and objectif not in ("awareness", "traffic"):
            score -= 30; reasons.append("ROAS < 1.0 → perte sèche (−30)")
        if s["CTR_cible"] < 0.005:
            score -= 10; reasons.append("CTR très faible (−10)")
        if s.get("cpa_est", 0) > 200 and objectif == "leads":
            score -= 15; reasons.append("CPA élevé pour leads (−15)")

        level = "HIGH" if score >= 75 else "MEDIUM" if score >= 50 else "LOW"
        return {
            "score"  : max(0, score),
            "level"  : level,
            "reasons": reasons if reasons else ["Tous les KPI dans les normes"],
        }

    def _build_summary(self, s: dict, importances: dict, confidence: dict) -> str:
        top_feature = max(importances, key=importances.get)
        return (
            f"Stratégie [{s['id']}] pilotée par '{top_feature}' "
            f"(importance={importances[top_feature]:.2%}). "
            f"Confiance : {confidence['level']} ({confidence['score']}/100). "
            f"ROAS={s['roas_est']:.2f} | Conv={s['conversions_est']} | "
            f"CPA={s.get('cpa_est', 0):.2f}€ | Efficiency={s.get('efficiency', 0):.4f}"
        )

    def generate_full_report(self, strategies: List[dict], key_factors: dict,
                              objectif: str, ref_cpc: float) -> dict:
        explanations = [
            self.explain_strategy(s, key_factors, objectif, ref_cpc)
            for s in strategies
        ]
        return {
            "metadata": {
                "agent"          : "Agent 2 XAI Engine v4",
                "objectif"       : objectif,
                "n_strategies"   : len(strategies),
                "best_strategy"  : strategies[0]["id"],
            },
            "explanations"              : explanations,
            "global_feature_importance" : self._global_importance(key_factors),
        }

    def _global_importance(self, key_factors: dict) -> dict:
        imp   = key_factors["importance"]
        total = sum(abs(v) for v in imp.values())
        return {k: round(abs(v) / total, 4) for k, v in imp.items()}


XAI = ExplainabilityEngine()

# ============================================================
# DONNÉES SYNTHÉTIQUES DE FALLBACK
# ============================================================

SYNTHETIC_PROFILES = pd.DataFrame([
    {"label": "HIGH_PERFORMANCE",   "CTR": 0.035, "CPC": 1.00,
     "conversion_rate": 0.035, "ROAS": 3.00,
     "spend": 4200.0, "impressions": 120_000, "conversions": 145, "n_campaigns": 38},
    {"label": "MEDIUM_PERFORMANCE", "CTR": 0.018, "CPC": 1.80,
     "conversion_rate": 0.018, "ROAS": 1.80,
     "spend": 2100.0, "impressions": 60_000,  "conversions": 42,  "n_campaigns": 51},
    {"label": "LOW_PERFORMANCE",    "CTR": 0.008, "CPC": 2.90,
     "conversion_rate": 0.009, "ROAS": 0.90,
     "spend":  800.0, "impressions": 22_000,  "conversions": 7,   "n_campaigns": 23},
])

SYNTHETIC_RULES = {
    "metadata": {"n_relations": 5, "source": "synthetic_fallback"},
    "regles_agent2": {
        "variables_cles": [
            {"variable": "clicks",          "correlation_conversions": +0.8867, "impact": "fort positif"},
            {"variable": "spend",           "correlation_conversions": +0.5683, "impact": "modéré positif"},
            {"variable": "conversion_rate", "correlation_conversions": +0.5737, "impact": "modéré positif"},
            {"variable": "CPC",             "correlation_conversions": -0.4929, "impact": "modéré négatif"},
            {"variable": "CTR",             "correlation_conversions": +0.4100, "impact": "faible positif"},
        ],
        "regles_action": [
            {"action": "Augmenter les clicks → boost conversions",  "priorite": "haute"},
            {"action": "Réduire le CPC → meilleure rentabilité",    "priorite": "haute"},
            {"action": "Optimiser le conversion_rate",               "priorite": "normale"},
            {"action": "Augmenter le spend si budget disponible",    "priorite": "normale"},
            {"action": "Améliorer le CTR par ciblage précis",       "priorite": "normale"},
        ],
    },
    "importance_vs_conversions": {
        "clicks": 0.8867, "spend": 0.5683,
        "conversion_rate": 0.5737, "CPC": -0.4929, "CTR": 0.4100,
    },
}

# ============================================================
# UTILITAIRES
# ============================================================

def sep(c: str = "=", n: int = 72) -> str:
    return c * n

def log(msg: str) -> None:
    text = str(msg)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

def fmt_eur(v: float) -> str:
    return f"{v:,.2f}€"

def fmt_int(v: float) -> str:
    return f"{int(v):,}"

# ============================================================
# ÉTAPE 1 — CHARGEMENT DES INPUTS
# ============================================================

def load_inputs() -> Tuple[pd.DataFrame, dict]:
    log(sep()); log("ÉTAPE 1 — CHARGEMENT DES INPUTS"); log(sep())

    if os.path.exists(INPUT_PROFILES):
        profiles_df = pd.read_csv(INPUT_PROFILES)
        log(f"  Profils   : {INPUT_PROFILES}")
    else:
        profiles_df = SYNTHETIC_PROFILES.copy()
        log(f"  ⚠️  Profils introuvables → données synthétiques")

    if os.path.exists(INPUT_RULES):
        with open(INPUT_RULES, "r", encoding="utf-8") as f:
            rules_data = json.load(f)
        log(f"  Règles    : {INPUT_RULES}")
    else:
        rules_data = SYNTHETIC_RULES
        log(f"  ⚠️  Règles introuvables → règles synthétiques")

    log(f"  Clusters  : {profiles_df['label'].tolist()}")
    log(f"  Relations : {rules_data['metadata']['n_relations']}")
    return profiles_df, rules_data

# ============================================================
# ÉTAPE 2 — EXTRACTION PROFIL HIGH_PERFORMANCE
# ============================================================

def extract_high_profile(profiles_df: pd.DataFrame) -> dict:
    log(sep()); log("ÉTAPE 2 — EXTRACTION PROFIL HIGH_PERFORMANCE"); log(sep())

    high = profiles_df[profiles_df["label"].str.upper().str.contains("HIGH")]
    if high.empty:
        log("  ⚠️  Aucun cluster HIGH → ROAS max utilisé")
        high = profiles_df.loc[[profiles_df["ROAS"].idxmax()]]

    profile = high.iloc[0].to_dict()
    log(f"  Référence : {profile.get('label', 'HIGH')}")
    for k, fmt in [("CTR",":.4f"),("CPC",":.4f"),("conversion_rate",":.4f"),
                   ("ROAS",":.4f"),("spend",":.2f"),("conversions",":.0f")]:
        v = profile.get(k, 0)
        log(f"    {k:<20} : {v:{fmt[1:]}}")
    return profile

# ============================================================
# ÉTAPE 3 — EXTRACTION DES FACTEURS CLÉS
# ============================================================

def extract_key_factors(rules_data: dict) -> dict:
    log(sep()); log("ÉTAPE 3 — FACTEURS CLÉS (Corrélation)"); log(sep())

    variables_cles = rules_data["regles_agent2"]["variables_cles"]
    regles_action  = rules_data["regles_agent2"]["regles_action"]
    importance     = rules_data["importance_vs_conversions"]

    log("  Variables clés :")
    for v in variables_cles:
        bar = "█" * int(abs(v["correlation_conversions"]) * 20)
        log(f"    {v['variable']:20s}  r={v['correlation_conversions']:+.4f}  {bar}")

    log("\n  Règles d'action :")
    for r in regles_action:
        prio = "[HAUTE]" if r["priorite"] == "haute" else "[NORM.]"
        log(f"    {prio} {r['action']}")

    return {
        "variables_cles": variables_cles,
        "importance"    : importance,
        "regles_action" : regles_action,
        # AXE 3 — toutes les corrélations extraites
        "r_clicks"      : importance.get("clicks",          +0.8867),
        "r_spend"       : importance.get("spend",           +0.5683),
        "r_cr"          : importance.get("conversion_rate", +0.5737),
        "r_cpc"         : importance.get("CPC",             -0.4929),
        "r_ctr"         : importance.get("CTR",             +0.4100),
    }

# ============================================================
# ÉTAPE 4 — RÉSOLUTION VALEUR/CONVERSION
# ============================================================

def resolve_value_per_conversion(produit: str, user_input: Optional[dict] = None) -> float:
    """AXE 2 — Priorité à la valeur custom user, puis PRODUCT_VALUE_MAP."""
    if user_input and "value_per_conversion" in user_input:
        return float(user_input["value_per_conversion"])
    prod = produit.lower().strip()
    for key, val in PRODUCT_VALUE_MAP.items():
        if key in prod:
            return val
    return PRODUCT_VALUE_MAP["autre"]

# ============================================================
# ÉTAPE 5 — AJUSTEMENTS PLATEFORME
# ============================================================

def get_platform_adjustments(plateforme: str) -> dict:
    return {
        "meta"  : {"cpc_mult": 1.00, "ctr_mult": 1.00},
        "google": {"cpc_mult": 1.20, "ctr_mult": 1.50},
    }.get(plateforme, {"cpc_mult": 1.00, "ctr_mult": 1.00})

# ============================================================
# ÉTAPE 6 — RATIOS BUDGÉTAIRES DYNAMIQUES
# ============================================================

def compute_budget_ratios(key_factors: dict) -> dict:
    """
    AXE 3+4 — Ratios calculés depuis toutes les corrélations.
    r_spend pilote directement le budget_usage de la stratégie équilibre.
    """
    r_clicks  = abs(key_factors["r_clicks"])
    r_spend   = abs(key_factors["r_spend"])
    r_cr      = abs(key_factors["r_cr"])
    r_cpc_neg = abs(key_factors["r_cpc"])
    r_ctr     = abs(key_factors["r_ctr"])

    ratio_a = 1.00
    # AXE 3 — r_cpc pilote la réduction de budget coût
    ratio_b = round(max(0.60, 1.0 - r_cpc_neg * 0.5), 2)
    # AXE 3 — r_spend + r_cr pilotent le ratio équilibre
    ratio_c = round(max(0.70, min(0.95,
        (r_spend + r_cr) / (r_clicks + r_spend + r_cr)
    )), 2)

    log(f"  Ratios budgétaires (dynamiques — AXE 3+4) :")
    log(f"    A (Volume)    : {ratio_a*100:.0f}%  r_clicks={r_clicks:.3f}")
    log(f"    B (Coût)      : {ratio_b*100:.0f}%  r_cpc={r_cpc_neg:.3f}")
    log(f"    C (Équilibre) : {ratio_c*100:.0f}%  r_spend={r_spend:.3f}  r_ctr={r_ctr:.3f}")

    return {"A": ratio_a, "B": ratio_b, "C": ratio_c}

# ============================================================
# ÉTAPE 7 — CONTRAINTES MÉTIER FORTES
# ============================================================

def apply_smart_constraints(score: float, strategy: dict,
                             ref_cpc: float, objectif: str) -> Tuple[float, List[str]]:
    """
    AXE 6 — 6 contraintes métier avec pénalités + flag élimination ROAS.
    """
    alerts = []
    cpc    = strategy["CPC_cible"]
    conv   = strategy["conversions_est"]
    roas   = strategy["roas_est"]
    ctr    = strategy["CTR_cible"]
    clicks = strategy["clicks_est"]
    cpa    = strategy.get("cpa_est", 0)

    # Contrainte 1 — CPC trop élevé
    if cpc > ref_cpc * 1.5:
        score -= 2.0
        alerts.append(f"CPC trop élevé ({cpc:.3f}€ > 1.5×ref {ref_cpc:.3f}€) → −2 pts")

    # Contrainte 2 — Conversions faibles (sauf awareness)
    if objectif not in ("awareness",) and conv < 50:
        score -= 3.0
        alerts.append(f"Conversions faibles ({int(conv)} < 50 seuil) → −3 pts")

    # Contrainte 3 — ROAS < 1 → pénalité + flag élimination
    if objectif not in ("awareness", "traffic") and roas < 1.0:
        score -= 1.0
        alerts.append(f"ROAS < 1.0 ({roas:.3f}) → campagne non rentable → −1 pt [ÉLIMINER]")

    # Contrainte 4 — CTR très faible
    if ctr < 0.005 and objectif not in ("leads",):
        score -= 1.0
        alerts.append(f"CTR très faible ({ctr:.4f} < 0.005) → −1 pt")

    # Contrainte 5 — Volume trafic minimum
    if clicks < 100:
        score -= 1.5
        alerts.append(f"Trafic insuffisant ({int(clicks)} clicks < 100) → −1.5 pts")

    # Contrainte 6 — CPA trop élevé pour leads
    if objectif == "leads" and cpa > 200:
        score -= 2.0
        alerts.append(f"CPA trop élevé pour leads ({cpa:.0f}€ > 200€) → −2 pts")

    return round(score, 4), alerts

# ============================================================
# ÉTAPE 8 — GÉNÉRATION D'UNE STRATÉGIE (v4 core)
# ============================================================

def build_one_strategy_v4(strat_id: str, strat_type: str, plateforme: str,
                           budget_allocated: float,
                           cpc_base: float, cr_base: float, ref_ctr: float,
                           ref_cpc: float, val_per_conv: float,
                           objectif: str, produit: str,
                           key_factors: dict, budget_ratio_label: str,
                           obj_profile: dict,
                           prod_profile: dict,
                           apply_exploration: bool = True) -> dict:
    """
    Construction v4 — objectif + produit + toutes corrélations.

    AXE 1 : objectif pilote cpc_adj, ctr_adj, cr_boost
    AXE 2 : produit pilote cpc_mult, cr_mult, ctr_mult
    AXE 3 : r_cpc, r_cr, r_ctr, r_clicks, r_spend utilisés
    AXE 4 : budget_usage dynamique via r_spend
    AXE 6 : CPC borné [0.75×, 1.20×] (réduction max −25%)
    AXE 7 : exploration ±10%
    AXE 9 : CPA + efficiency + reach_score calculés
    """
    r_cpc   = abs(key_factors["r_cpc"])
    r_cr    = abs(key_factors["r_cr"])
    r_ctr   = abs(key_factors["r_ctr"])
    r_spend = abs(key_factors["r_spend"])

    # ── AXE 2 — Application profil produit sur la base ───────
    cpc_base_prod = cpc_base   * prod_profile["cpc_mult"]
    cr_base_prod  = cr_base    * prod_profile["cr_mult"]
    ctr_base_prod = ref_ctr    * prod_profile["ctr_mult"]

    # ── AXE 3+6 — CPC/CR pilotés par corrélations ────────────
    if strat_type == "volume":
        # Volume → CPC légèrement augmenté pour portée max
        cpc_corr = cpc_base_prod * (1.0 + r_ctr * 0.1)    # r_ctr booste légèrement
        cr_corr  = cr_base_prod
    elif strat_type == "cout":
        # Coût → CPC réduit piloté par r_cpc, borné [0.75×, 1.20×]
        cpc_corr = cpc_base_prod * (1.0 - r_cpc * 0.5)
        cpc_corr = max(cpc_corr, cpc_base_prod * 0.75)     # plancher −25%
        cpc_corr = min(cpc_corr, cpc_base_prod * 1.20)     # plafond  +20%
        cr_corr  = cr_base_prod * (1.0 + r_cr)             # CR boosté par r_cr
    else:  # equilibre
        # Équilibre → CPC neutre, CR légèrement amélioré via r_cr
        cpc_corr = cpc_base_prod
        cr_corr  = cr_base_prod * (1.0 + r_cr * 0.3)

    # ── AXE 1 — Objectif dans la génération ──────────────────
    cpc_final = cpc_corr     * obj_profile["cpc_adj"]
    ctr_final = ctr_base_prod * obj_profile["ctr_adj"]
    cr_final  = cr_corr      * obj_profile["cr_boost"]

    # ── AXE 4 — Budget usage dynamique via r_spend ───────────
    # r_spend fort → on peut dépenser plus, r_spend faible → prudence
    spend_factor  = obj_profile["budget_usage"] * (0.85 + r_spend * 0.30)
    spend_factor  = min(spend_factor, 1.00)         # jamais > 100%
    budget_u      = budget_allocated * spend_factor

    # ── AXE 7 — Exploration ±10% ─────────────────────────────
    exploration_trace = []
    if apply_exploration:
        cpc_final, cr_final, ctr_final, exploration_trace = EXPLORER.apply(
            strat_id, cpc_final, cr_final, ctr_final
        )

    # ── Métriques dérivées ────────────────────────────────────
    clicks      = max(1, round(budget_u / cpc_final))
    impressions = max(1, round(clicks / ctr_final))
    conversions = round(clicks * cr_final)
    roas_est    = round((conversions * val_per_conv) / budget_u, 3) if budget_u > 0 else 0.0

    # AXE 9 — KPI réels : CPA, efficiency, reach_score
    cpa_est        = round(budget_u / conversions, 2) if conversions > 0 else 9999.0
    efficiency     = round(conversions / budget_u, 4) if budget_u > 0 else 0.0
    reach_score    = round(impressions / budget_u, 4) if budget_u > 0 else 0.0   # awareness

    strategy = {
        "id"               : strat_id,
        "nom"              : f"Stratégie {strat_type.capitalize()} ({plateforme.capitalize()})",
        "type"             : strat_type,
        "objectif"         : objectif,
        "plateforme"       : plateforme,
        "produit"          : produit,
        "budget"           : round(budget_u, 2),
        "budget_ratio"     : budget_ratio_label,
        "CTR_cible"        : round(ctr_final, 5),
        "CPC_cible"        : round(cpc_final, 4),
        "impressions_est"  : impressions,
        "clicks_est"       : clicks,
        "conversions_est"  : conversions,
        "conversion_rate"  : round(cr_final, 5),
        "roas_est"         : roas_est,
        "val_per_conversion": val_per_conv,
        # AXE 9 — KPI réels
        "cpa_est"          : cpa_est,
        "efficiency"       : efficiency,
        "reach_score"      : reach_score,
        # Métadonnées
        "exploration_applied": apply_exploration,
        "exploration_trace"  : exploration_trace,
        "focus": {
            "volume"   : ["clicks", "impressions", "spend"],
            "cout"     : ["CPC", "conversion_rate", "ROAS"],
            "equilibre": ["clicks", "conversion_rate", "ROAS"],
        }.get(strat_type, ["clicks", "conversions"]),
    }

    # ── Score brut (normalisé après) ─────────────────────────
    formula_fn    = SCORING_FORMULAS.get(objectif, SCORING_FORMULAS["conversions"])["function"]
    score_brut    = round(formula_fn(strategy), 6)
    score_after_c, alerts = apply_smart_constraints(score_brut, strategy, ref_cpc, objectif)

    strategy["score_brut"]             = score_brut
    strategy["score_after_constraints"] = score_after_c
    strategy["score_potentiel"]         = score_after_c   # normalisé ensuite
    strategy["contraintes"]             = alerts

    # ── Justification XAI lite ────────────────────────────────
    r_clicks_v = key_factors["r_clicks"]
    r_cpc_v    = key_factors["r_cpc"]
    r_cr_v     = key_factors["r_cr"]
    r_ctr_v    = key_factors["r_ctr"]
    r_spend_v  = key_factors["r_spend"]

    strategy["justification"] = [
        f"[AXE1]  Objectif '{objectif}' → cpc_adj={obj_profile['cpc_adj']:.2f} "
        f"ctr_adj={obj_profile['ctr_adj']:.2f} cr_boost={obj_profile['cr_boost']:.2f}",
        f"[AXE1]  Business logic : {obj_profile['business_logic']}",
        f"[AXE2]  Produit '{produit}' → cpc×{prod_profile['cpc_mult']:.2f} "
        f"cr×{prod_profile['cr_mult']:.2f} ctr×{prod_profile['ctr_mult']:.2f}",
        f"[AXE2]  {prod_profile['description']}",
        f"[AXE3]  r_clicks={r_clicks_v:+.4f} | r_ctr={r_ctr_v:+.4f} | "
        f"r_spend={r_spend_v:+.4f} | r_cr={r_cr_v:+.4f} | r_cpc={r_cpc_v:+.4f}",
        f"[AXE4]  Budget dynamique : spend_factor={spend_factor:.3f} "
        f"(r_spend={r_spend:.3f}) → budget_u={budget_u:.2f}€",
        f"[AXE6]  CPC : base_prod={cpc_base_prod:.4f}€ → corr={cpc_corr:.4f}€ "
        f"→ final={cpc_final:.4f}€ (bornes [{cpc_base_prod*0.75:.4f}€, {cpc_base_prod*1.20:.4f}€])",
        f"[AXE9]  CPA={cpa_est:.2f}€ | efficiency={efficiency:.4f} | reach={reach_score:.4f}",
        f"[SCORE] brut={score_brut:.6f} → après contraintes={score_after_c:.6f}",
    ] + [f"[EXPLO] {t}" for t in exploration_trace]

    strategy["avantages"] = {
        "volume"   : ["Portée maximale", "Clicks élevés", f"CTR optimisé (r_ctr={abs(r_ctr_v):.2f})"],
        "cout"     : [f"CPC réduit (r_cpc={abs(r_cpc_v):.2f})", "CR boosté", "ROI optimal"],
        "equilibre": ["Risque faible", "KPI prévisibles", f"r_spend={r_spend_v:+.2f} exploité"],
    }.get(strat_type, [])

    strategy["inconvenients"] = {
        "volume"   : ["CPC légèrement plus élevé", "Rentabilité par click réduite"],
        "cout"     : ["Volume clicks réduit", "Moins d'impressions"],
        "equilibre": ["Pas de rupture de performance", "Innovation limitée"],
    }.get(strat_type, [])

    return strategy

# ============================================================
# ÉTAPE 9 — GÉNÉRATION COMPLÈTE
# ============================================================

def generate_strategies(user_input: dict, high_profile: dict,
                         key_factors: dict) -> List[dict]:
    log(sep()); log("ÉTAPE 9 — GÉNÉRATION DES STRATÉGIES CANDIDATES (v4)"); log(sep())

    budget     = float(user_input["budget"])
    plateforme = user_input["plateforme"]
    objectif   = user_input["objectif"]
    produit    = user_input["produit"]

    val_per_conv = resolve_value_per_conversion(produit, user_input)
    obj_profile  = OBJECTIVE_GENERATION_PROFILES[objectif]
    prod_profile = get_product_profile(produit)        # AXE 2
    formula_label = SCORING_FORMULAS[objectif]["label"]

    log(f"  Produit  '{produit}' → val/conv={val_per_conv}€ | {prod_profile['description']}")
    log(f"  Profil produit : CPC×{prod_profile['cpc_mult']:.2f} CR×{prod_profile['cr_mult']:.2f} CTR×{prod_profile['ctr_mult']:.2f}")
    log(f"  Objectif '{objectif}' → {obj_profile['description']}")
    log(f"  Business : {obj_profile['business_logic']}")
    log(f"  Scoring  : {formula_label}")

    splits = {
        "meta"  : [("meta",   1.00)],
        "google": [("google", 1.00)],
        "both"  : [("meta",   0.60), ("google", 0.40)],
    }.get(plateforme, [("meta", 1.00)])

    log(f"\n  Plateformes :")
    for plat, ratio in splits:
        log(f"    {plat:6s} → {ratio*100:.0f}%  =  {budget*ratio:,.0f}€")

    ratios = compute_budget_ratios(key_factors)

    ref_ctr = float(high_profile.get("CTR",             0.035))
    ref_cpc = float(high_profile.get("CPC",             1.000))
    ref_cr  = float(high_profile.get("conversion_rate", 0.035))

    all_strategies: List[dict] = []

    for plat, plat_ratio in splits:
        plat_budget = budget * plat_ratio
        adj         = get_platform_adjustments(plat)
        p_ref_cpc   = ref_cpc * adj["cpc_mult"]
        p_ref_ctr   = ref_ctr * adj["ctr_mult"]
        prefix      = "M" if plat == "meta" else "G"

        type_defs = [
            {"id": f"{prefix}A", "type": "volume",    "br": ratios["A"],
             "cpc_base": p_ref_cpc, "cr_base": ref_cr,
             "label": f"{ratios['A']*100:.0f}% budget {plat}"},
            {"id": f"{prefix}B", "type": "cout",      "br": ratios["B"],
             "cpc_base": p_ref_cpc, "cr_base": ref_cr,
             "label": f"{ratios['B']*100:.0f}% budget {plat} (dynamique)"},
            {"id": f"{prefix}C", "type": "equilibre", "br": ratios["C"],
             "cpc_base": p_ref_cpc, "cr_base": ref_cr,
             "label": f"{ratios['C']*100:.0f}% budget {plat} (dynamique)"},
        ]

        plat_strats = []
        for td in type_defs:
            s = build_one_strategy_v4(
                strat_id           = td["id"],
                strat_type         = td["type"],
                plateforme         = plat,
                budget_allocated   = plat_budget * td["br"],
                cpc_base           = td["cpc_base"],
                cr_base            = td["cr_base"],
                ref_ctr            = p_ref_ctr,
                ref_cpc            = p_ref_cpc,
                val_per_conv       = val_per_conv,
                objectif           = objectif,
                produit            = produit,
                key_factors        = key_factors,
                budget_ratio_label = td["label"],
                obj_profile        = obj_profile,
                prod_profile       = prod_profile,   # AXE 2
                apply_exploration  = True,
            )
            plat_strats.append(s)

        normalize_scores_minmax(plat_strats, target_min=6.0, target_max=9.5)
        all_strategies.extend(plat_strats)

    all_strategies.sort(key=lambda x: x["score_potentiel"], reverse=True)
    best = all_strategies[0]

    log(f"\n  Stratégies générées :")
    log(f"  {'ID':<4} {'Type':<11} {'Plat':<7} {'Budget':>8} {'Conv':>5} {'ROAS':>6} "
        f"{'CPA':>7} {'Eff':>8} {'Score':>7}")
    log(f"  {sep('-', 72)}")
    for s in all_strategies:
        flag = " ←BEST" if s["id"] == best["id"] else ""
        log(f"  [{s['id']:<2}] {s['type']:<11} {s['plateforme']:<7} "
            f"{s['budget']:>7,.0f}€ {s['conversions_est']:>5} "
            f"{s['roas_est']:>6.2f} {s.get('cpa_est',0):>6.1f}€ "
            f"{s.get('efficiency',0):>8.4f} {s['score_potentiel']:>6.2f}{flag}")

    return all_strategies

# ============================================================
# ÉTAPE 10 — RECOMMANDATION FINALE
# ============================================================

def generate_recommendation(best: dict, user_input: dict,
                              key_factors: dict) -> str:
    obj         = user_input["objectif"]
    obj_profile = OBJECTIVE_GENERATION_PROFILES[obj]

    obj_texts = {
        "conversions": f"maximiser les conversions (ROAS={best['roas_est']:.2f}, CPA={best.get('cpa_est',0):.2f}€)",
        "awareness"  : f"maximiser la visibilité ({best['impressions_est']:,} impressions, reach={best.get('reach_score',0):.2f})",
        "traffic"    : f"générer du trafic ({best['clicks_est']:,} clicks, CPC={best['CPC_cible']:.3f}€)",
        "leads"      : f"capturer des leads qualifiés (CR={best['conversion_rate']*100:.2f}%, CPA={best.get('cpa_est',0):.2f}€)",
    }

    lines = [
        f"STRATÉGIE RECOMMANDÉE : [{best['id']}] {best['nom']}",
        f"",
        f"  ➤ Objectif       : {obj_texts.get(obj, obj)}",
        f"  ➤ Business logic : {obj_profile['business_logic']}",
        f"  ➤ Plateforme     : {best['plateforme'].capitalize()}",
        f"  ➤ Budget alloué  : {fmt_eur(best['budget'])} ({best['budget_ratio']})",
        f"  ➤ CPC cible      : {fmt_eur(best['CPC_cible'])}",
        f"  ➤ CTR cible      : {best['CTR_cible']*100:.3f}%",
        f"  ➤ Impressions    : {fmt_int(best['impressions_est'])}",
        f"  ➤ Clicks est.    : {fmt_int(best['clicks_est'])}",
        f"  ➤ Conversions    : {fmt_int(best['conversions_est'])}",
        f"  ➤ Conv. rate     : {best['conversion_rate']*100:.3f}%",
        f"  ➤ ROAS estimé    : {best['roas_est']:.3f}",
        f"  ➤ CPA estimé     : {fmt_eur(best.get('cpa_est', 0))}  [AXE9]",
        f"  ➤ Efficiency     : {best.get('efficiency', 0):.4f} conv/€  [AXE9]",
        f"  ➤ Score final    : {best['score_potentiel']:.2f}/10",
        f"",
        f"  PLAN D'ACTION (5 étapes) :",
        f"    1. Lancer test budget  = {fmt_eur(best['budget'] * 0.20)} (20%) pendant 7 jours",
        f"    2. Surveiller CPC      : alerter si > {fmt_eur(best['CPC_cible'] * 1.3)}",
        f"    3. Surveiller CPA      : alerter si > {fmt_eur(best.get('cpa_est', 99) * 1.25)} après J+7",
        f"    4. Scale si ROAS       > {best['roas_est']:.2f} → budget ×2 progressif",
        f"    5. A/B créatifs sur    : {', '.join(best['focus'])}",
    ]

    if best["contraintes"]:
        lines += ["", "  ⚠️  ALERTES ACTIVES :"]
        for a in best["contraintes"]:
            lines.append(f"    • {a}")

    return "\n".join(lines)

# ============================================================
# ÉTAPE 11 — VISUALISATION 6 PANNEAUX
# ============================================================

def plot_strategies_comparison(strategies: List[dict], user_input: dict) -> None:
    n       = len(strategies)
    labels  = [s["id"] for s in strategies]
    budgets = [s["budget"] for s in strategies]
    convs   = [s["conversions_est"] for s in strategies]
    scores  = [s["score_potentiel"] for s in strategies]
    roas    = [s["roas_est"] for s in strategies]
    cpcs    = [s["CPC_cible"] for s in strategies]
    ctrs    = [s["CTR_cible"] * 100 for s in strategies]
    cpas    = [s.get("cpa_est", 0) for s in strategies]
    effs    = [s.get("efficiency", 0) for s in strategies]

    BASE_COLORS = ["#2980B9", "#27AE60", "#E67E22", "#8E44AD", "#E74C3C", "#16A085"]
    colors = [BASE_COLORS[i % len(BASE_COLORS)] for i in range(n)]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#F4F6F8")
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    prod_profile = get_product_profile(user_input["produit"])
    title_suffix = (
        f"Objectif={user_input['objectif'].upper()}  |  "
        f"Produit={user_input['produit']} ({prod_profile['description'][:25]})  |  "
        f"Budget={user_input['budget']:,}€  |  "
        f"Plateforme={user_input['plateforme'].upper()}"
    )
    fig.suptitle(
        f"Agent 2 v4 — Décideur SaaS — Comparaison Stratégies\n{title_suffix}",
        fontsize=11, fontweight="bold", y=0.99
    )

    def style(ax, title):
        ax.set_facecolor("#FFFFFF")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    def bar_labels(ax, bars, fmt="{:.0f}", suffix="", offset_ratio=0.02):
        ymin, ymax = ax.get_ylim()
        offset = (ymax - ymin) * offset_ratio
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h) + suffix, ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    x = np.arange(n)
    w = 0.5

    # [0] Budget
    ax = axes[0]
    bars = ax.bar(x, budgets, width=w, color=colors, edgecolor="white", linewidth=0.8)
    style(ax, "Budget alloué (€)  [AXE4 dynamique]")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(budgets) * 1.18)
    bar_labels(ax, bars, "{:,.0f}", "€")

    # [1] Conversions
    ax = axes[1]
    bars = ax.bar(x, convs, width=w, color=colors, edgecolor="white", linewidth=0.8)
    style(ax, "Conversions estimées  [AXE1+2]")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(convs) * 1.18)
    bar_labels(ax, bars, "{:.0f}")

    # [2] ROAS
    ax = axes[2]
    bars = ax.bar(x, roas, width=w, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=1.0, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.7, label="Seuil ROAS=1")
    style(ax, "ROAS estimé  [AXE6 — élimination <1]")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(roas) * 1.20 if max(roas) > 0 else 2)
    bar_labels(ax, bars, "{:.3f}")
    ax.legend(fontsize=8)

    # [3] Score normalisé
    ax = axes[3]
    ax.set_facecolor("#FFFFFF"); ax.spines[["top", "right"]].set_visible(False)
    y_pos = np.arange(n)
    hbars = ax.barh(y_pos, scores, color=colors, edgecolor="white", linewidth=0.8, height=0.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 10)
    ax.axvline(x=7.75, color="#888888", linestyle=":", linewidth=1.0, alpha=0.5, label="Médiane")
    ax.set_title("Score normalisé (/10)  [AXE5]", fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8)
    for bar, val in zip(hbars, scores):
        ax.text(val + 0.12, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8, fontweight="bold")

    # [4] CPA vs CPC (AXE 9)
    ax = axes[4]
    ax.set_facecolor("#FFFFFF"); ax.spines[["top", "right"]].set_visible(False)
    for i, s in enumerate(strategies):
        cpa_val = min(s.get("cpa_est", 9999), 500)    # cap à 500 pour lisibilité
        ax.scatter(cpcs[i], cpa_val, color=colors[i], s=140, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(labels[i], (cpcs[i], cpa_val),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=colors[i], fontweight="bold")
    ax.set_xlabel("CPC (€)  [AXE2+3 produit+corrélation]", fontsize=9)
    ax.set_ylabel("CPA (€)  [AXE9]", fontsize=9)
    ax.set_title("CPC vs CPA  [AXE2+9]", fontsize=10, fontweight="bold", pad=8)
    ax.grid(alpha=0.25, linestyle="--")

    # [5] Efficiency (conv/€) — AXE 9
    ax = axes[5]
    bars = ax.bar(x, effs, width=w, color=colors, edgecolor="white", linewidth=0.8)
    style(ax, "Efficiency (conv/€)  [AXE9]")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(effs) * 1.20 if max(effs) > 0 else 0.01)
    bar_labels(ax, bars, "{:.4f}")

    patches = [mpatches.Patch(color=colors[i], label=f"[{labels[i]}] {strategies[i]['nom']}")
               for i in range(n)]
    fig.legend(handles=patches, loc="lower center", ncol=min(n, 3),
               fontsize=9, bbox_to_anchor=(0.5, -0.03), framealpha=0.95)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log(f"  Graphique : {OUTPUT_CHART}")

# ============================================================
# ÉTAPE 12 — SAUVEGARDE JSON + XAI + RAPPORT TXT
# ============================================================

def save_results(strategies: List[dict], user_input: dict,
                 high_profile: dict, key_factors: dict,
                 recommendation: str, xai_report: dict) -> None:
    log(sep()); log("ÉTAPE 12 — SAUVEGARDE DES RÉSULTATS"); log(sep())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    formula      = SCORING_FORMULAS.get(user_input["objectif"], SCORING_FORMULAS["conversions"])
    prod_profile = get_product_profile(user_input["produit"])

    json_output = {
        "metadata": {
            "agent"            : "Agent 2 v4-SaaS-Décideur",
            "cas"              : "Cas 2 — Nouvelle Campagne",
            "version"          : "4.0-saas-décideur",
            "objectif"         : user_input["objectif"],
            "produit"          : user_input["produit"],
            "produit_profile"  : prod_profile["description"],
            "scoring_formula"  : formula["label"],
            "exploration_ratio": f"±{EXPLORATION_RATIO*100:.0f}%",
            "n_strategies"     : len(strategies),
            "user_input"       : user_input,
            "axes_applied"     : [
                "AXE1-objectif-partout", "AXE2-produit-profil",
                "AXE3-toutes-correlations", "AXE4-budget-intelligent",
                "AXE5-scoring-normalise-safe_norm", "AXE6-contraintes-fortes",
                "AXE7-exploration-controlee", "AXE8-logique-business",
                "AXE9-kpi-reels-cpa-efficiency", "AXE10-sortie-claire-xai",
            ],
        },
        "profil_high_reference": {k: high_profile.get(k)
                                   for k in ["CTR","CPC","conversion_rate","ROAS","spend"]},
        "facteurs_cles"        : key_factors["variables_cles"],
        "strategies"           : strategies,
        "meilleure_strategie"  : strategies[0]["id"] if strategies else None,
        "recommandation"       : recommendation,
    }
    with open(OUTPUT_STRATEGIES, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False, default=str)
    log(f"  JSON stratégies : {OUTPUT_STRATEGIES}")

    with open(OUTPUT_XAI, "w", encoding="utf-8") as f:
        json.dump(xai_report, f, indent=2, ensure_ascii=False, default=str)
    log(f"  JSON XAI        : {OUTPUT_XAI}")

    lines = [
        sep("="),
        "RAPPORT AGENT 2 v4-SAAS-DÉCIDEUR — DÉCISION STRATÉGIQUE",
        sep("="),
        f"Version          : 4.0-saas-décideur",
        f"Axes appliqués   : AXE1 (objectif partout) | AXE2 (profil produit)",
        f"                   AXE3 (toutes corrélations) | AXE4 (budget intelligent)",
        f"                   AXE5 (safe_norm) | AXE6 (contraintes fortes)",
        f"                   AXE7 (exploration ±{EXPLORATION_RATIO*100:.0f}%) | AXE8 (logique business)",
        f"                   AXE9 (CPA + efficiency + reach) | AXE10 (XAI enrichi)",
        f"Objectif         : {user_input['objectif']}",
        f"Budget           : {user_input['budget']:,} €",
        f"Plateforme       : {user_input['plateforme']}",
        f"Produit          : {user_input['produit']} — {prod_profile['description']}",
        f"Profil produit   : CPC×{prod_profile['cpc_mult']:.2f} CR×{prod_profile['cr_mult']:.2f} CTR×{prod_profile['ctr_mult']:.2f}",
        f"Formule scoring  : {formula['label']}",
        f"Business logic   : {OBJECTIVE_GENERATION_PROFILES[user_input['objectif']]['business_logic']}",
        "",
        sep("-"), "PROFIL HIGH_PERFORMANCE", sep("-"),
    ]
    for k in ["CTR", "CPC", "conversion_rate", "ROAS", "spend"]:
        lines.append(f"  {k:<20} : {high_profile.get(k, 0)}")

    lines += ["", sep("-"), "CORRÉLATIONS EXPLOITÉES  [AXE3]", sep("-")]
    for v in key_factors["variables_cles"]:
        lines.append(f"  → {v['variable']:20s}  r={v['correlation_conversions']:+.4f}  ({v['impact']})")

    lines += ["", sep("-"), "STRATÉGIES (triées par score)", sep("-")]
    for s in strategies:
        lines += [
            f"", f"[{s['id']}] {s['nom'].upper()}",
            f"  Type             : {s['type']}",
            f"  Plateforme       : {s['plateforme']}",
            f"  Budget           : {s['budget']:,.2f}€ ({s['budget_ratio']}) [AXE4 dynamique]",
            f"  CPC cible [AXE6] : {s['CPC_cible']:.4f}€",
            f"  CTR cible [AXE1] : {s['CTR_cible']:.5f}  ({s['CTR_cible']*100:.3f}%)",
            f"  Conv. rate[AXE2] : {s['conversion_rate']:.5f}  ({s['conversion_rate']*100:.3f}%)",
            f"  Impressions      : {s['impressions_est']:,}",
            f"  Clicks           : {s['clicks_est']:,}",
            f"  Conversions      : {s['conversions_est']:,}",
            f"  ROAS estimé      : {s['roas_est']:.3f}",
            f"  CPA estimé [AXE9]: {s.get('cpa_est', 0):.2f}€",
            f"  Efficiency [AXE9]: {s.get('efficiency', 0):.4f} conv/€",
            f"  Reach score      : {s.get('reach_score', 0):.4f}",
            f"  Val/conversion   : {s['val_per_conversion']}€",
            f"  Score brut       : {s['score_brut']:.6f}",
            f"  Score normalisé  : {s['score_potentiel']:.2f}/10  [AXE5]",
            f"  Focus KPI        : {', '.join(s['focus'])}",
            f"  Avantages        : {', '.join(s['avantages'])}",
            f"  Inconvénients    : {', '.join(s['inconvenients'])}",
            f"  Justification :",
        ]
        for j in s["justification"]:
            lines.append(f"    • {j}")
        if s["contraintes"]:
            lines.append(f"  Alertes [AXE6] :")
            for a in s["contraintes"]:
                lines.append(f"    ⚠️  {a}")

    lines += [
        "", sep("="), "RECOMMANDATION FINALE  [AXE10]", sep("="),
        recommendation,
        "", sep("-"), "XAI — Rapport externalisé  [AXE10]", sep("-"),
        f"  Fichier : {OUTPUT_XAI}",
        f"  Contenu : feature_importances | decision_trace (objectif+produit+corrélations) | counterfactuals | confidence",
        "", sep("-"), "PROCHAINE ÉTAPE : FEATURE ENGINEERING", sep("-"),
        "  Stratégies → features ML → Prédicteur (ROAS, CPA, CTR, CPC)",
        "", sep("="), "FIN DU RAPPORT — Agent 2 v4 SaaS Décideur", sep("="),
    ]

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  Rapport TXT     : {OUTPUT_REPORT}")

# ============================================================
# CHARGEMENT USER INPUT
# ============================================================

def load_user_input(path: str = "user_input.json") -> dict:
    DEFAULTS = {"objectif": "conversions", "budget": 5000,
                "plateforme": "meta", "produit": "ecommerce"}

    if not os.path.exists(path):
        log(f"  ⚠️  {path} introuvable → valeurs par défaut")
        data = {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    for k, d in DEFAULTS.items():
        if k not in data or not str(data[k]).strip():
            log(f"  ℹ️  '{k}' absent → {d}")
            data[k] = d

    data["budget"]     = max(float(data["budget"]), 100.0)
    data["plateforme"] = str(data["plateforme"]).lower().strip()
    data["objectif"]   = str(data["objectif"]).lower().strip()
    data["produit"]    = str(data["produit"]).strip()

    if data["plateforme"] not in ["meta", "google", "both"]:
        data["plateforme"] = "meta"
    if data["objectif"] not in ["conversions", "leads", "traffic", "awareness"]:
        data["objectif"] = "conversions"

    log(sep()); log("INPUT UTILISATEUR"); log(sep())
    log(f"  Fichier     : {path}")
    log(f"  Objectif    : {data['objectif']}")
    log(f"  Budget      : {data['budget']:,} €")
    log(f"  Plateforme  : {data['plateforme']}")
    log(f"  Produit     : {data['produit']}")

    return data


def normalize_user_input(user_input=None) -> dict:
    """
    Normalise l'input utilisateur pour FastAPI.
    """
    data = dict(user_input or {})

    for key, default in DEFAULT_USER_INPUT.items():
        if key not in data or data[key] is None or data[key] == "":
            log(f"  '{key}' absent -> valeur par defaut : {default}")
            data[key] = default

    data["budget"]     = max(float(data["budget"]), 100.0)
    data["plateforme"] = str(data["plateforme"]).lower().strip()
    data["objectif"]   = str(data["objectif"]).lower().strip()
    data["produit"]    = str(data["produit"]).strip()

    if data["plateforme"] not in ["meta", "google", "both"]:
        log(f"  Plateforme '{data['plateforme']}' inconnue -> 'meta' utilise")
        data["plateforme"] = "meta"
    if data["objectif"] not in ["conversions", "leads", "traffic", "awareness"]:
        log(f"  Objectif '{data['objectif']}' inconnu -> 'conversions' utilise")
        data["objectif"] = "conversions"

    if "value_per_conversion" in data and data["value_per_conversion"] not in [None, ""]:
        data["value_per_conversion"] = float(data["value_per_conversion"])

    return data


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_agent2_pipeline(user_input: dict) -> List[dict]:
    user_input = normalize_user_input(user_input)
    log(sep("=")); log("  AGENT 2 v4-SAAS-DÉCIDEUR [AdOptimizer AI]"); log(sep("="))
    log(f"  Objectif   : {user_input['objectif']}")
    log(f"  Budget     : {user_input['budget']:,} €")
    log(f"  Plateforme : {user_input['plateforme']}")
    log(f"  Produit    : {user_input['produit']}")
    log(f"  Axes       : AXE1(objectif) AXE2(produit) AXE3(corrélations) AXE4(budget)")
    log(f"               AXE5(safe_norm) AXE6(contraintes) AXE7(exploration)")
    log(f"               AXE8(business) AXE9(CPA+efficiency) AXE10(XAI)")

    profiles_df, rules_data = load_inputs()
    high_profile            = extract_high_profile(profiles_df)
    key_factors             = extract_key_factors(rules_data)
    strategies              = generate_strategies(user_input, high_profile, key_factors)

    log(sep()); log("ÉTAPE XAI — MODULE D'EXPLICABILITÉ  [AXE10]"); log(sep())
    ref_cpc    = float(high_profile.get("CPC", 1.0))
    xai_report = XAI.generate_full_report(strategies, key_factors,
                                           user_input["objectif"], ref_cpc)
    log(f"  XAI généré pour {len(strategies)} stratégies")
    log(f"  Feature importances : {xai_report['global_feature_importance']}")

    log(sep()); log("RECOMMANDATION FINALE  [AXE10]"); log(sep())
    recommendation = generate_recommendation(strategies[0], user_input, key_factors)
    log(recommendation)

    log(sep()); log("VISUALISATION"); log(sep())
    plot_strategies_comparison(strategies, user_input)

    save_results(strategies, user_input, high_profile, key_factors,
                 recommendation, xai_report)

    log(sep("=")); log("  RÉSUMÉ FINAL"); log(sep("="))
    best = strategies[0]
    log(f"  {len(strategies)} stratégies | objectif+produit+corrélations+contraintes = vrai décideur")
    log(f"  {'ID':<4} {'Nom':<36} {'Budget':>8} {'Conv':>5} {'ROAS':>6} "
        f"{'CPA':>7} {'Eff':>9} {'Score':>7}")
    log(f"  {sep('-', 84)}")
    for s in strategies:
        flag = "  ← RECOMMANDÉE" if s["id"] == best["id"] else ""
        log(f"  [{s['id']:<2}] {s['nom']:<36} {s['budget']:>7,.0f}€ "
            f"{s['conversions_est']:>5} {s['roas_est']:>6.2f} "
            f"{s.get('cpa_est',0):>6.1f}€ {s.get('efficiency',0):>9.4f} "
            f"{s['score_potentiel']:>6.2f}{flag}")

    log("")
    log(f"  Outputs ({OUTPUT_DIR}/) :")
    for fname in [OUTPUT_STRATEGIES, OUTPUT_XAI, OUTPUT_REPORT, OUTPUT_CHART]:
        size = os.path.getsize(fname) if os.path.exists(fname) else 0
        log(f"    → {fname}  ({size:,} o)")

    log(sep("=")); log("  AGENT 2 v4 TERMINÉ — Moteur décision SaaS opérationnel"); log(sep("="))
    return strategies

# ============================================================
# WRAPPER FASTAPI / N8N
# ============================================================

def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def run_case2_strategy(user_input=None) -> dict:
    """
    Point d'entree FastAPI pour le cas 2 : decision strategique.

    Body attendu :
      objectif, budget, plateforme, produit
      value_per_conversion optionnel
    """
    try:
        normalized_input = normalize_user_input(user_input)
        strategies = run_agent2_pipeline(normalized_input)
        best_strategy = strategies[0] if strategies else None

        saved_payload = {}
        if os.path.exists(OUTPUT_STRATEGIES):
            with open(OUTPUT_STRATEGIES, "r", encoding="utf-8") as f:
                saved_payload = json.load(f)

        xai_payload = {}
        if os.path.exists(OUTPUT_XAI):
            with open(OUTPUT_XAI, "r", encoding="utf-8") as f:
                xai_payload = json.load(f)

        return {
            "status": "success",
            "message": "Strategies cas 2 generees",
            "input": _json_safe(normalized_input),
            "output_dir": str(OUTPUT_DIR),
            "output_files": {
                "strategies": str(OUTPUT_STRATEGIES),
                "xai": str(OUTPUT_XAI),
                "report": str(OUTPUT_REPORT),
                "chart": str(OUTPUT_CHART),
            },
            "n_strategies": len(strategies),
            "best_strategy": _json_safe(best_strategy),
            "recommendation": saved_payload.get("recommandation"),
            "xai_summary": xai_payload.get("metadata", {}),
            "strategies": _json_safe(strategies),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input": _json_safe(user_input or {}),
        }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    USER_INPUT = load_user_input("user_input.json")
    result = run_case2_strategy(USER_INPUT)
    print(result)
