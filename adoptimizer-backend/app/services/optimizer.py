# =============================================================================
# TOOL 5 — RL OPTIMIZER PRO FINAL V2
# Recommandations qualitatives + quantitatives
# - Recommandations seulement pour campagnes problématiques
# - Gère mono-canal vs multi-canal
# - RL PPO + reward améliorée + Causal Guard + Business Guard
# - Ajout : budget_shift_pct, budget_shift_amount, budgets avant/après
# =============================================================================

import os
import sys
import json
import joblib
import logging
import warnings
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def install_if_missing(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install_if_missing("gymnasium")
install_if_missing("stable-baselines3", "stable_baselines3")

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

from pathlib import Path

BASE_DIR = Path("app")

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DIR = BASE_DIR / "models"

DATASET_PATH = BASE_DIR / "data" / "dataset_model_ready.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

HEALTH_PATH = OUTPUT_DIR / "campaign_health_score.json"
CAUSAL_PATH = OUTPUT_DIR / "causal_effects.json"
CORR_PATH = OUTPUT_DIR / "correlations.json"

OUT_PLAN = OUTPUT_DIR / "optimization_plan.json"
OUT_REPORT = OUTPUT_DIR / "optimizer_report.txt"
OUT_POLICY = OUTPUT_DIR / "policy"
OUT_CURVES = OUTPUT_DIR / "training_curves.png"
OUT_LOG = OUTPUT_DIR / "optimizer.log"

TOTAL_TIMESTEPS = 1000
MAX_STEPS = 20
RANDOM_STATE = 42

MIN_CAMPAIGN_SPEND = 5.0
MAX_CAMPAIGN_SPEND = 5000.0
CAUSAL_CONF_THRESHOLD = 0.80

# Bornes de réallocation
MIN_REALLOCATION_PCT = 5.0
MAX_REALLOCATION_PCT = 25.0

ACTION_NAMES = {
    0: "increase_budget_10pct",
    1: "decrease_budget_10pct",
    2: "maintain_budget",
    3: "reallocate_meta_to_google",
    4: "reallocate_google_to_meta",
    5: "pause_campaign",
}

ACTION_LABELS = {
    0: "+10% budget",
    1: "-10% budget",
    2: "Maintenir",
    3: "Réallouer Meta → Google",
    4: "Réallouer Google → Meta",
    5: "Pause campagne",
}

ROOT_CAUSE_ENCODING = {
    "no_clear_cause": 0.0,
    "direct_budget_impact": 1.0,
    "ad_saturation": 2.0,
    "halo_effect": 3.0,
    "cannibalization": 4.0,
    "delayed_branding": 5.0,
    "budget_inefficiency": 6.0,
}


# =============================================================================
# LOGGING
# =============================================================================

if OUT_LOG.exists():
    OUT_LOG.unlink()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUT_LOG, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger("rl_optimizer")


# =============================================================================
# LOADERS
# =============================================================================

def load_json(path: Path, default):
    if not path.exists():
        log.warning(f"Fichier manquant : {path}")
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "campaign_id"])

    if "platform" in df.columns:
        df["platform"] = df["platform"].astype(str).str.lower()

    if "roas" not in df.columns and {"conversion_value", "spend"}.issubset(df.columns):
        df["roas"] = df["conversion_value"] / df["spend"].clip(lower=1)

    if "ctr_calc" not in df.columns and {"clicks", "impressions"}.issubset(df.columns):
        df["ctr_calc"] = df["clicks"] / df["impressions"].clip(lower=1)

    if "cpc_calc" not in df.columns and {"spend", "clicks"}.issubset(df.columns):
        df["cpc_calc"] = df["spend"] / df["clicks"].clip(lower=1)

    if "cpa" not in df.columns and {"spend", "conversions"}.issubset(df.columns):
        df["cpa"] = df["spend"] / df["conversions"].clip(lower=1)

    if "conversion_rate" not in df.columns and {"conversions", "clicks"}.issubset(df.columns):
        df["conversion_rate"] = df["conversions"] / df["clicks"].clip(lower=1)

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    if "campaign_age_days" not in df.columns:
        df["campaign_age_days"] = (
            df["date"] - df.groupby("campaign_id")["date"].transform("min")
        ).dt.days

    log.info(f"Dataset chargé : {len(df):,} lignes | {df['campaign_id'].nunique()} campagnes")
    return df


def load_triggered_health():
    health_data = load_json(HEALTH_PATH, {"campaigns": []})
    triggered = {}

    for item in health_data.get("campaigns", []):
        if not isinstance(item, dict):
            continue

        cid = item.get("campaign_id")
        if not cid:
            continue

        if bool(item.get("trigger_causal_ai", False)):
            triggered[cid] = {
                "campaign_id": cid,
                "global_campaign_id": item.get("global_campaign_id"),
                "platform": str(item.get("platform", "")).lower(),
                "health_score": float(item.get("health_score", 50)),
                "health_status": item.get("status", "UNKNOWN"),
                "components": item.get("components", {}),
            }

    log.info(f"Campagnes problématiques détectées : {len(triggered)}")
    return triggered


def get_multi_channel_pairs(df):
    if "global_campaign_id" not in df.columns or "platform" not in df.columns:
        return set()

    pivot = (
        df.groupby(["global_campaign_id", "platform"])
        .size()
        .unstack(fill_value=0)
    )

    if "meta" not in pivot.columns or "google" not in pivot.columns:
        return set()

    return set(pivot[(pivot["meta"] > 0) & (pivot["google"] > 0)].index.tolist())


# =============================================================================
# CAMPAIGN STATS
# =============================================================================

def build_campaign_stats(df, triggered_health, recent_days=14):
    cutoff = df["date"].max() - pd.Timedelta(days=recent_days)
    recent_df = df[df["date"] >= cutoff].copy()

    records = []

    for cid, health in triggered_health.items():
        g = recent_df[recent_df["campaign_id"] == cid].copy()

        if g.empty:
            g = df[df["campaign_id"] == cid].copy()

        if g.empty:
            continue

        g = g.sort_values("date")

        platform = (
            str(g["platform"].dropna().iloc[-1]).lower()
            if "platform" in g.columns
            else health.get("platform")
        )

        gcid = (
            g["global_campaign_id"].dropna().iloc[-1]
            if "global_campaign_id" in g.columns and not g["global_campaign_id"].dropna().empty
            else health.get("global_campaign_id", cid)
        )

        record = {
            "campaign_id": cid,
            "global_campaign_id": gcid,
            "platform": platform,

            "health_score": health["health_score"],
            "health_status": health["health_status"],
            "prediction_score": float(health.get("components", {}).get("prediction_score", 50)),
            "anomaly_score": float(health.get("components", {}).get("anomaly_score", 50)),
            "trend_score": float(health.get("components", {}).get("trend_score", 50)),

            "roas": float(g["roas"].mean()) if "roas" in g.columns else 0.0,
            "conversions": float(g["conversions"].mean()) if "conversions" in g.columns else 0.0,
            "spend": float(g["spend"].mean()) if "spend" in g.columns else 0.0,
            "impressions": float(g["impressions"].mean()) if "impressions" in g.columns else 0.0,
            "clicks": float(g["clicks"].mean()) if "clicks" in g.columns else 0.0,
            "ctr_calc": float(g["ctr_calc"].mean()) if "ctr_calc" in g.columns else 0.0,
            "cpc_calc": float(g["cpc_calc"].mean()) if "cpc_calc" in g.columns else 0.0,
            "cpa": float(g["cpa"].mean()) if "cpa" in g.columns else 0.0,
            "conversion_rate": float(g["conversion_rate"].mean()) if "conversion_rate" in g.columns else 0.0,
            "is_weekend": int(g["is_weekend"].mode()[0]) if "is_weekend" in g.columns else 0,
            "month": int(g["month"].mode()[0]) if "month" in g.columns else 1,
            "campaign_age_days": int(g["campaign_age_days"].max()) if "campaign_age_days" in g.columns else 0,
            "campaign_status": g["campaign_status"].iloc[-1] if "campaign_status" in g.columns else "ACTIVE",
        }

        if "daily_budget" in g.columns:
            record["daily_budget"] = float(g["daily_budget"].mean())
        else:
            record["daily_budget"] = max(record["spend"], 1.0)

        record["budget_utilization"] = record["spend"] / max(record["daily_budget"], 1.0)

        records.append(record)

    log.info(f"Campagnes envoyées vers Optimizer : {len(records)}")
    return records


# =============================================================================
# CAUSAL GUARD
# =============================================================================

class CausalGuard:
    def __init__(self, causal_data, threshold=CAUSAL_CONF_THRESHOLD):
        self.threshold = threshold
        self.index = self._build_index(causal_data)

    def _safe_effect(self, effects, key):
        obj = effects.get(key, {})
        if isinstance(obj, dict):
            return float(obj.get("effect", 0.0) or 0.0)
        return 0.0

    def _build_index(self, causal_data):
        index = {}

        for r in causal_data.get("results", []):
            cid = r.get("campaign_id")
            if not cid:
                continue

            diag = r.get("diagnosis", {})
            effects = r.get("causal_effects", {})

            index[cid] = {
                "root_cause": diag.get("root_cause", "no_clear_cause"),
                "confidence": float(diag.get("confidence", 0.0) or 0.0),
                "direct_effect": self._safe_effect(effects, "direct_spend_to_conversions"),
                "saturation_effect": self._safe_effect(effects, "saturation_quadratic"),
                "lag_effect": self._safe_effect(effects, "lagged_spend_to_conversions"),
                "cross_effect": self._safe_effect(effects, "cross_channel_effect"),
                "evidence": diag.get("evidence", ""),
            }

        return index

    def get_info(self, campaign_id):
        return self.index.get(campaign_id, {
            "root_cause": "no_clear_cause",
            "confidence": 0.0,
            "direct_effect": 0.0,
            "saturation_effect": 0.0,
            "lag_effect": 0.0,
            "cross_effect": 0.0,
            "evidence": "",
        })

    def allowed_actions(self, campaign):
        cid = campaign["campaign_id"]
        platform = campaign.get("platform", "")
        is_multi = bool(campaign.get("is_multi_channel", False))

        info = self.get_info(cid)
        cause = info["root_cause"]
        conf = info["confidence"]

        allowed = {0, 1, 2, 5}

        if is_multi:
            if platform == "meta":
                allowed.add(3)
            elif platform == "google":
                allowed.add(4)

        if conf >= self.threshold:
            if cause == "direct_budget_impact":
                allowed = allowed & {0, 2}
            elif cause == "ad_saturation":
                allowed = allowed & {1, 2}
            elif cause == "budget_inefficiency":
                allowed = allowed & {1, 2, 5}
            elif cause == "delayed_branding":
                allowed = allowed & {0, 2}
            elif cause == "halo_effect":
                allowed = allowed & ({0, 2, 3, 4} if is_multi else {0, 2})
            elif cause == "cannibalization":
                allowed = allowed & ({1, 2, 3, 4} if is_multi else {1, 2})
            elif cause == "no_clear_cause":
                allowed = allowed & {1, 2, 5}

        if campaign.get("health_status") != "CRITICAL":
            allowed.discard(5)

        if not allowed:
            allowed = {2}

        return allowed

    def filter_action(self, campaign, action):
        allowed = self.allowed_actions(campaign)

        if action in allowed:
            return action, False, ""

        return 2, True, f"action_{ACTION_NAMES.get(action)}_blocked"


# =============================================================================
# PREDICTOR SIMULATOR
# =============================================================================

class PredictorSimulator:
    def __init__(self, model_path):
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle prédicteur introuvable : {model_path}")

        self.bundle = joblib.load(model_path)

        self.models_by_target = None
        self.model = None
        self.feature_columns = None
        self.target_columns = None
        self.imputer = None

        if isinstance(self.bundle, dict):
            self.models_by_target = self.bundle.get("models_by_target")

            self.model = (
                self.bundle.get("model")
                or self.bundle.get("best_model")
                or self.bundle.get("estimator")
            )

            self.feature_columns = (
                self.bundle.get("feature_columns")
                or self.bundle.get("feature_cols")
                or self.bundle.get("features")
            )

            self.target_columns = (
                self.bundle.get("target_columns")
                or self.bundle.get("target_cols")
                or self.bundle.get("targets")
            )

            self.imputer = (
                self.bundle.get("feature_imputer")
                or self.bundle.get("imputer")
            )
        else:
            self.model = self.bundle

        if self.feature_columns is None:
            self.feature_columns = load_pickle(MODEL_DIR / "feature_columns.pkl")

        if self.target_columns is None:
            self.target_columns = load_pickle(MODEL_DIR / "target_columns.pkl")

        if self.imputer is None:
            self.imputer = load_pickle(MODEL_DIR / "feature_imputer.pkl")

        if self.feature_columns is None:
            raise ValueError("feature_columns introuvables.")

        if self.models_by_target is None and self.model is None:
            raise ValueError("best_model.pkl ne contient pas de modèle exploitable.")

        if self.target_columns is None and self.models_by_target is not None:
            self.target_columns = list(self.models_by_target.keys())

        log.info("Prédicteur chargé.")
        log.info(f"Features : {len(self.feature_columns)}")

    def _prepare_features(self, context, new_spend):
        row = dict(context)
        row["spend"] = new_spend

        clicks = max(float(row.get("clicks", 1.0)), 1.0)
        impressions = max(float(row.get("impressions", 1.0)), 1.0)
        daily_budget = max(float(row.get("daily_budget", new_spend)), 1.0)

        row["ctr_calc"] = clicks / impressions
        row["cpc_calc"] = new_spend / clicks
        row["budget_utilization"] = new_spend / daily_budget
        row["spend_squared"] = new_spend ** 2

        X = pd.DataFrame([{col: row.get(col, 0.0) for col in self.feature_columns}])

        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if self.imputer is not None:
            X = pd.DataFrame(self.imputer.transform(X), columns=self.feature_columns)

        return X

    def _get_value(self, preds, names, fallback):
        for name in names:
            if name in preds:
                return float(preds[name])
        return float(fallback)

    def predict(self, context, new_spend, causal_info):
        X = self._prepare_features(context, new_spend)

        preds = {}

        if self.models_by_target is not None:
            for target, model in self.models_by_target.items():
                try:
                    preds[str(target)] = float(model.predict(X)[0])
                except Exception:
                    pass
        else:
            output = np.asarray(self.model.predict(X))
            if output.ndim == 2:
                output = output[0]

            for target, value in zip(self.target_columns, output):
                preds[str(target)] = float(value)

        roas = self._get_value(
            preds,
            ["target_roas_h14", "target_roas_h7", "target_roas_h3", "roas_h14", "roas_h7", "roas"],
            context.get("roas", 0.0),
        )

        conversions = self._get_value(
            preds,
            [
                "target_conversions_h14",
                "target_conversions_h7",
                "target_conversions_h3",
                "conversions_h14",
                "conversions_h7",
                "conversions",
            ],
            context.get("conversions", 0.0),
        )

        old_spend = max(float(context.get("spend", new_spend)), 1.0)
        delta_spend_pct = (new_spend - old_spend) / old_spend

        cause = causal_info.get("root_cause", "no_clear_cause")
        direct = float(causal_info.get("direct_effect", 0.0))

        conversions += direct * delta_spend_pct * 10.0

        if cause == "ad_saturation" and delta_spend_pct > 0:
            roas *= 0.90
            conversions *= 0.95

        if cause == "direct_budget_impact" and delta_spend_pct > 0:
            conversions *= 1.05

        if cause == "budget_inefficiency" and delta_spend_pct > 0:
            roas *= 0.90

        return max(0.01, float(roas)), max(0.0, float(conversions))


# =============================================================================
# RL ENV
# =============================================================================

class AdBudgetEnv(gym.Env):
    metadata = {"render_modes": []}
    STATE_DIM = 15

    def __init__(self, campaigns, simulator, causal_guard, max_steps=MAX_STEPS):
        super().__init__()

        self.campaigns = campaigns
        self.simulator = simulator
        self.causal_guard = causal_guard
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self.STATE_DIM,),
            dtype=np.float32
        )

        self.rng = np.random.default_rng(RANDOM_STATE)
        self.stats = self._compute_stats()

        self.campaign = None
        self.step_count = 0
        self.previous_action = None

    def _compute_stats(self):
        keys = ["roas", "conversions", "spend", "ctr_calc", "cpc_calc"]
        stats = {}

        for key in keys:
            vals = np.array([float(c.get(key, 0.0)) for c in self.campaigns])
            stats[key] = (float(vals.mean()), max(float(vals.std()), 1e-6))

        return stats

    def _norm(self, value, key):
        mean, std = self.stats[key]
        return float(np.clip((float(value) - mean) / std, -5, 5))

    def _build_state(self):
        c = self.campaign
        causal = self.causal_guard.get_info(c["campaign_id"])

        state = np.array([
            self._norm(c.get("roas", 0), "roas"),
            self._norm(c.get("conversions", 0), "conversions"),
            self._norm(c.get("spend", 0), "spend"),
            self._norm(c.get("ctr_calc", 0), "ctr_calc"),
            self._norm(c.get("cpc_calc", 0), "cpc_calc"),

            float(c.get("health_score", 50)) / 100.0,
            float(c.get("prediction_score", 50)) / 100.0,
            float(c.get("anomaly_score", 50)) / 100.0,
            float(c.get("trend_score", 50)) / 100.0,

            ROOT_CAUSE_ENCODING.get(causal["root_cause"], 0.0) / 6.0,
            float(causal["confidence"]),
            float(np.clip(causal["direct_effect"], -5, 5)),
            float(np.clip(causal["saturation_effect"], -5, 5)),
            float(np.clip(causal["cross_effect"], -5, 5)),
            1.0 if c.get("is_multi_channel") else 0.0,
        ], dtype=np.float32)

        return np.clip(state, -5, 5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.campaign = dict(self.rng.choice(self.campaigns))
        self.step_count = 0
        self.previous_action = None

        return self._build_state(), {}

    def _action_to_spend(self, action):
        current_spend = float(self.campaign["spend"])

        if action == 0:
            new_spend = current_spend * 1.10
        elif action == 1:
            new_spend = current_spend * 0.90
        elif action == 2:
            new_spend = current_spend
        elif action == 3:
            new_spend = current_spend * 0.90
        elif action == 4:
            new_spend = current_spend * 0.90
        elif action == 5:
            new_spend = 0.0
        else:
            new_spend = current_spend

        violation = 0.0

        if action != 5:
            if new_spend < MIN_CAMPAIGN_SPEND:
                new_spend = MIN_CAMPAIGN_SPEND
                violation += 0.5

            if new_spend > MAX_CAMPAIGN_SPEND:
                new_spend = MAX_CAMPAIGN_SPEND
                violation += 0.5

        return new_spend, violation

    def step(self, action):
        action = int(action)

        action_final, filtered, reason = self.causal_guard.filter_action(self.campaign, action)

        old_spend = float(self.campaign["spend"])
        old_roas = float(self.campaign["roas"])
        old_conv = float(self.campaign["conversions"])

        new_spend, violation = self._action_to_spend(action_final)

        causal_info = self.causal_guard.get_info(self.campaign["campaign_id"])

        if action_final == 5:
            new_roas, new_conv = 0.01, 0.0
        else:
            new_roas, new_conv = self.simulator.predict(
                context=self.campaign,
                new_spend=new_spend,
                causal_info=causal_info
            )

        delta_roas = new_roas - old_roas
        delta_conv = new_conv - old_conv

        roas_gain = delta_roas / max(abs(old_roas), 0.01)
        conv_gain = delta_conv / max(abs(old_conv), 0.1)
        spend_change = abs(new_spend - old_spend) / max(old_spend, 1.0)
        risk = 1 - (float(self.campaign.get("health_score", 50)) / 100.0)

        reward = (
            2.0 * roas_gain
            + 1.5 * conv_gain
            - 0.5 * spend_change
            - 0.8 * risk
            - 2.0 * violation
        )

        if self.previous_action is not None and action_final == self.previous_action:
            reward -= 0.2

        if action_final == 0 and (roas_gain < 0.05 or conv_gain <= 0):
            reward -= 2.0

        if action_final == 2 and causal_info.get("root_cause") == "no_clear_cause":
            reward += 0.5

        if filtered:
            reward -= 0.7

        if action_final == 5 and self.campaign.get("health_status") != "CRITICAL":
            reward -= 5.0

        self.campaign["spend"] = new_spend
        self.campaign["roas"] = new_roas
        self.campaign["conversions"] = new_conv
        self.previous_action = action_final

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        info = {
            "action_original": action,
            "action_final": action_final,
            "filtered": filtered,
            "filter_reason": reason,
            "delta_roas": delta_roas,
            "delta_conv": delta_conv,
        }

        return self._build_state(), float(reward), terminated, truncated, info


# =============================================================================
# TRAINING CALLBACK
# =============================================================================

class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0
        self.action_counts = np.zeros(6)

    def _on_step(self):
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        actions = self.locals.get("actions", [])

        self.current_reward += float(rewards[0])

        for a in np.atleast_1d(actions):
            self.action_counts[int(a)] += 1

        if dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        return True


# =============================================================================
# PLOT
# =============================================================================

def plot_training(callback):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if callback.episode_rewards:
        rewards = pd.Series(callback.episode_rewards)
        axes[0].plot(rewards, alpha=0.35)
        axes[0].plot(rewards.rolling(max(5, len(rewards)//20), min_periods=1).mean())
        axes[0].set_title("Reward PPO")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(alpha=0.3)

    total = callback.action_counts.sum() or 1
    pct = callback.action_counts / total * 100

    axes[1].bar([ACTION_LABELS[i] for i in range(6)], pct)
    axes[1].set_title("Distribution des actions")
    axes[1].set_ylabel("%")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_CURVES, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# FINAL BUSINESS GUARD
# =============================================================================

def apply_final_business_guard(action_id, campaign, causal_info, old_roas, old_conv, new_roas, new_conv):
    constraints = []
    filtered = False

    roas_pct = ((new_roas - old_roas) / max(abs(old_roas), 0.01)) * 100
    conv_delta = new_conv - old_conv
    conv_pct = (conv_delta / max(abs(old_conv), 0.1)) * 100

    root_cause = causal_info.get("root_cause", "no_clear_cause")
    health_status = campaign.get("health_status", "UNKNOWN")

    bad_roas = roas_pct < -5
    bad_conv_abs = conv_delta < -0.05
    bad_conv_strong = conv_delta < -0.25 or conv_pct < -10

    if action_id == 0 and bad_roas and bad_conv_abs:
        action_id = 2
        filtered = True
        constraints.append("increase_blocked_negative_roas_and_conversions")

    elif action_id == 0 and bad_conv_strong:
        action_id = 2
        filtered = True
        constraints.append("increase_blocked_strong_conversion_drop")

    elif action_id == 0 and root_cause == "no_clear_cause" and health_status == "CRITICAL":
        action_id = 1
        filtered = True
        constraints.append("increase_blocked_critical_no_clear_cause")

    elif action_id == 0 and root_cause == "no_clear_cause" and bad_roas:
        action_id = 2
        filtered = True
        constraints.append("increase_blocked_no_clear_cause_negative_roas")

    elif action_id == 0 and root_cause == "no_clear_cause":
        if roas_pct < 15 or conv_delta < 0.20:
            action_id = 2
            filtered = True
            constraints.append("increase_blocked_no_clear_cause_insufficient_gain")

    if action_id == 5 and health_status != "CRITICAL":
        action_id = 2
        filtered = True
        constraints.append("pause_blocked_not_critical")

    return action_id, filtered, constraints


# =============================================================================
# QUANTITATIVE BUDGET DECISION
# =============================================================================

def compute_budget_adjustment(action_id, campaign, causal_info, roas_pct, conv_delta):
    current_spend = float(campaign.get("spend", 0.0))
    health_score = float(campaign.get("health_score", 50.0))
    confidence = float(causal_info.get("confidence", 0.0))
    root_cause = causal_info.get("root_cause", "no_clear_cause")

    severity = np.clip((100.0 - health_score) / 100.0, 0.0, 1.0)
    gain_factor_pct = np.clip(abs(roas_pct), 5.0, 30.0)

    adjustment = {
        "adjustment_type": "none",
        "shift_pct": 0.0,
        "shift_amount": 0.0,
        "current_budget": round(current_spend, 2),
        "recommended_budget": round(current_spend, 2),
        "source_channel": None,
        "target_channel": None,
        "quantitative_explanation": "Aucun ajustement budgétaire recommandé."
    }

    if action_id == 0:
        pct = 10.0
        amount = current_spend * pct / 100.0
        adjustment.update({
            "adjustment_type": "increase",
            "shift_pct": pct,
            "shift_amount": round(amount, 2),
            "recommended_budget": round(current_spend + amount, 2),
            "quantitative_explanation": f"Augmenter le budget de {pct:.1f}% soit +{amount:.2f}."
        })

    elif action_id == 1:
        pct = 10.0
        amount = current_spend * pct / 100.0
        adjustment.update({
            "adjustment_type": "decrease",
            "shift_pct": pct,
            "shift_amount": round(amount, 2),
            "recommended_budget": round(max(current_spend - amount, 0.0), 2),
            "quantitative_explanation": f"Réduire le budget de {pct:.1f}% soit -{amount:.2f}."
        })

    elif action_id == 2:
        adjustment.update({
            "adjustment_type": "maintain",
            "shift_pct": 0.0,
            "shift_amount": 0.0,
            "recommended_budget": round(current_spend, 2),
            "quantitative_explanation": "Maintenir le budget actuel sans changement quantitatif."
        })

    elif action_id in [3, 4]:
        pct = gain_factor_pct * confidence * severity

        if conv_delta < -0.5:
            pct *= 0.75

        if root_cause == "halo_effect":
            pct *= 1.00
        elif root_cause == "cannibalization":
            pct *= 1.10
        elif root_cause == "budget_inefficiency":
            pct *= 0.90
        elif root_cause == "direct_budget_impact":
            pct *= 0.80

        pct = float(np.clip(pct, MIN_REALLOCATION_PCT, MAX_REALLOCATION_PCT))
        amount = current_spend * pct / 100.0

        if action_id == 3:
            source, target = "meta", "google"
        else:
            source, target = "google", "meta"

        adjustment.update({
            "adjustment_type": "reallocation",
            "shift_pct": round(pct, 1),
            "shift_amount": round(amount, 2),
            "current_budget": round(current_spend, 2),
            "recommended_budget": round(max(current_spend - amount, 0.0), 2),
            "source_channel": source,
            "target_channel": target,
            "quantitative_explanation": (
                f"Déplacer {pct:.1f}% du budget {source} vers {target}, "
                f"soit environ {amount:.2f}."
            )
        })

    elif action_id == 5:
        adjustment.update({
            "adjustment_type": "pause",
            "shift_pct": 100.0,
            "shift_amount": round(current_spend, 2),
            "recommended_budget": 0.0,
            "quantitative_explanation": (
                f"Mettre la campagne en pause : budget ramené de {current_spend:.2f} à 0."
            )
        })

    return adjustment


# =============================================================================
# INFERENCE
# =============================================================================

def priority_from_campaign(campaign):
    if campaign.get("health_status") == "CRITICAL":
        return "high"
    if campaign.get("health_score", 100) < 50:
        return "medium"
    return "low"


def build_explanation(action_name, campaign, causal_info, roas_pct, conv_delta, budget_adjustment=None):
    cause = causal_info["root_cause"]
    q = ""

    if budget_adjustment:
        q = " " + budget_adjustment.get("quantitative_explanation", "")

    if action_name == "increase_budget_10pct":
        return (
            f"Augmenter le budget car le simulateur prévoit un gain acceptable. "
            f"Cause={cause}, ROAS attendu {roas_pct:+.1f}%, conversions {conv_delta:+.2f}.{q}"
        )

    if action_name == "decrease_budget_10pct":
        return (
            f"Réduire le budget car la performance est risquée ou inefficace. "
            f"Cause={cause}, ROAS attendu {roas_pct:+.1f}%, conversions {conv_delta:+.2f}.{q}"
        )

    if action_name == "maintain_budget":
        return (
            f"Maintenir le budget : signal causal insuffisant ou impact prédit défavorable. "
            f"Cause={cause}, ROAS attendu {roas_pct:+.1f}%, conversions {conv_delta:+.2f}.{q}"
        )

    if action_name == "reallocate_meta_to_google":
        return (
            f"Réallouer Meta → Google car la campagne est multi-canal et le signal inter-canal le permet. "
            f"Cause={cause}.{q}"
        )

    if action_name == "reallocate_google_to_meta":
        return (
            f"Réallouer Google → Meta car la campagne est multi-canal et le signal inter-canal le permet. "
            f"Cause={cause}.{q}"
        )

    if action_name == "pause_campaign":
        return f"Pause recommandée car la campagne est CRITICAL. Validation humaine obligatoire.{q}"

    return "Recommandation générée par l’agent RL."


def predict_recommendation(model, campaign, env):
    env.campaign = dict(campaign)
    env.step_count = 0
    env.previous_action = None

    obs = env._build_state()

    votes = []

    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        votes.append(int(action))

    raw_action = Counter(votes).most_common(1)[0][0]

    action_final, filtered_causal, reason = env.causal_guard.filter_action(campaign, raw_action)

    causal_info = env.causal_guard.get_info(campaign["campaign_id"])

    old_roas = float(campaign["roas"])
    old_conv = float(campaign["conversions"])

    constraints = []
    if reason:
        constraints.append(reason)

    new_spend, violation = env._action_to_spend(action_final)

    if action_final == 5:
        new_roas, new_conv = 0.01, 0.0
    else:
        new_roas, new_conv = env.simulator.predict(
            context=campaign,
            new_spend=new_spend,
            causal_info=causal_info
        )

    guarded_action, filtered_business, business_constraints = apply_final_business_guard(
        action_final,
        campaign,
        causal_info,
        old_roas,
        old_conv,
        new_roas,
        new_conv
    )

    if filtered_business:
        action_final = guarded_action
        constraints.extend(business_constraints)

    new_spend, violation = env._action_to_spend(action_final)

    if action_final == 5:
        new_roas, new_conv = 0.01, 0.0
    else:
        new_roas, new_conv = env.simulator.predict(
            context=campaign,
            new_spend=new_spend,
            causal_info=causal_info
        )

    roas_pct = ((new_roas - old_roas) / max(abs(old_roas), 0.01)) * 100
    conv_delta = new_conv - old_conv

    action_name = ACTION_NAMES[action_final]

    budget_adjustment = compute_budget_adjustment(
        action_id=action_final,
        campaign=campaign,
        causal_info=causal_info,
        roas_pct=roas_pct,
        conv_delta=conv_delta
    )

    return {
        "campaign_id": campaign["campaign_id"],
        "global_campaign_id": campaign.get("global_campaign_id"),
        "platform": campaign.get("platform"),
        "is_multi_channel": bool(campaign.get("is_multi_channel", False)),

        "health_score": round(float(campaign.get("health_score", 50)), 2),
        "health_status": campaign.get("health_status"),

        "root_cause": causal_info["root_cause"],
        "causal_confidence": round(float(causal_info["confidence"]), 3),
        "causal_evidence": causal_info.get("evidence", ""),

        "recommended_action": action_name,
        "action_label": ACTION_LABELS[action_final],
        "action_id": action_final,

        "current_state": {
            "roas": round(old_roas, 4),
            "conversions": round(old_conv, 2),
            "spend": round(float(campaign["spend"]), 2),
        },

        "expected_impact": {
            "expected_roas": round(new_roas, 4),
            "expected_conversions": round(new_conv, 2),
            "expected_spend": round(new_spend, 2),
            "delta_roas_pct": round(roas_pct, 2),
            "delta_conversions": round(conv_delta, 2),
        },

        "budget_adjustment": budget_adjustment,

        "priority": priority_from_campaign(campaign),
        "constraints_applied": constraints,
        "explanation": build_explanation(
            action_name,
            campaign,
            causal_info,
            roas_pct,
            conv_delta,
            budget_adjustment
        ),
        "fallback_used": bool(filtered_causal or filtered_business),
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# REPORT
# =============================================================================

def build_report(plan, metadata):
    lines = [
        "=" * 70,
        "TOOL 5 — RL OPTIMIZER PRO REPORT",
        "Version : qualitative + quantitative",
        f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        f"Campagnes problématiques optimisées : {len(plan)}",
        f"Mode : {metadata['mode']}",
        f"Timesteps : {metadata['timesteps']}",
        "",
    ]

    for priority in ["high", "medium", "low"]:
        group = [p for p in plan if p["priority"] == priority]

        if not group:
            continue

        lines.append("-" * 70)
        lines.append(f"PRIORITÉ {priority.upper()}")
        lines.append("-" * 70)

        for p in group:
            lines.append(f"[{p['campaign_id']}] {p['platform']} | {p['action_label']}")
            lines.append(f"  Multi-canal : {p['is_multi_channel']}")
            lines.append(f"  Health      : {p['health_score']} ({p['health_status']})")
            lines.append(f"  Cause       : {p['root_cause']} ({p['causal_confidence']:.0%})")
            lines.append(f"  ROAS actuel : {p['current_state']['roas']}")
            lines.append(f"  ROAS prévu  : {p['expected_impact']['expected_roas']} ({p['expected_impact']['delta_roas_pct']:+.1f}%)")
            lines.append(f"  Conv prévu  : {p['expected_impact']['expected_conversions']} ({p['expected_impact']['delta_conversions']:+.2f})")
            lines.append(f"  Spend prévu : {p['expected_impact']['expected_spend']}")

            ba = p.get("budget_adjustment", {})
            lines.append("  --- Décision quantitative ---")
            lines.append(f"  Type ajustement     : {ba.get('adjustment_type')}")
            lines.append(f"  Pourcentage         : {ba.get('shift_pct')}%")
            lines.append(f"  Montant             : {ba.get('shift_amount')}")
            lines.append(f"  Budget actuel       : {ba.get('current_budget')}")
            lines.append(f"  Budget recommandé   : {ba.get('recommended_budget')}")

            if ba.get("source_channel") and ba.get("target_channel"):
                lines.append(f"  Source              : {ba.get('source_channel')}")
                lines.append(f"  Destination         : {ba.get('target_channel')}")

            lines.append(f"  Décision budget     : {ba.get('quantitative_explanation')}")
            lines.append(f"  Explication         : {p['explanation']}")

            if p["constraints_applied"]:
                lines.append(f"  Contraintes         : {', '.join(p['constraints_applied'])}")

            lines.append("")

    return "\n".join(lines)


# =============================================================================
# RUN
# =============================================================================

def run():
    log.info("=" * 70)
    log.info("TOOL 5 — RL OPTIMIZER PRO FINAL V2")
    log.info("=" * 70)

    df = load_dataset()
    health = load_triggered_health()
    causal_data = load_json(CAUSAL_PATH, {"results": []})
    correlations = load_json(CORR_PATH, {})

    if not health:
        raise RuntimeError("Aucune campagne problématique trouvée dans campaign_health_score.json")

    multi_pairs = get_multi_channel_pairs(df)

    campaigns = build_campaign_stats(df, health, recent_days=14)

    for c in campaigns:
        c["is_multi_channel"] = c.get("global_campaign_id") in multi_pairs

    campaigns = [
        c for c in campaigns
        if str(c.get("campaign_status", "ACTIVE")).upper() != "PAUSED"
    ]

    if not campaigns:
        raise RuntimeError("Aucune campagne active à optimiser.")

    simulator = PredictorSimulator(BEST_MODEL_PATH)
    causal_guard = CausalGuard(causal_data)

    log.info(f"Causal campaigns indexées : {len(causal_guard.index)}")
    log.info(f"Campagnes multi-canal : {sum(1 for c in campaigns if c['is_multi_channel'])}")
    log.info(f"Campagnes mono-canal  : {sum(1 for c in campaigns if not c['is_multi_channel'])}")

    env = AdBudgetEnv(
        campaigns=campaigns,
        simulator=simulator,
        causal_guard=causal_guard,
        max_steps=MAX_STEPS,
    )

    try:
        check_env(env, warn=True)
        log.info("check_env OK")
    except Exception as e:
        log.warning(f"check_env warning : {e}")

    monitored_env = Monitor(env)
    callback = TrainingCallback()

    model = PPO(
        "MlpPolicy",
        monitored_env,
        learning_rate=2e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=8,
        gamma=0.97,
        gae_lambda=0.95,
        ent_coef=0.05,
        seed=RANDOM_STATE,
        verbose=1,
    )

    log.info(f"Training PPO : {TOTAL_TIMESTEPS} timesteps")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    model.save(str(OUT_POLICY))
    plot_training(callback)

    plan = []

    for campaign in campaigns:
        try:
            rec = predict_recommendation(model, campaign, env)
            plan.append(rec)
        except Exception as e:
            log.error(f"Erreur recommandation {campaign['campaign_id']} : {e}")

    priority_order = {"high": 0, "medium": 1, "low": 2}

    plan.sort(
        key=lambda x: (
            priority_order.get(x["priority"], 3),
            x["health_score"]
        )
    )

    metadata = {
        "tool": "Tool 5 — RL Optimizer PRO V2",
        "mode": "Offline RL PPO + PredictorSimulator + CausalGuard + HealthScore filter + FinalBusinessGuard + QuantitativeBudgetDecision",
        "algorithm": "PPO Stable-Baselines3",
        "timesteps": TOTAL_TIMESTEPS,
        "filter_source": str(HEALTH_PATH),
        "filter_rule": "trigger_causal_ai == True",
        "causal_source": str(CAUSAL_PATH),
        "predictor_model": str(BEST_MODEL_PATH),
        "n_problem_campaigns": len(campaigns),
        "n_recommendations": len(plan),
        "n_multi_channel": sum(1 for c in campaigns if c["is_multi_channel"]),
        "n_mono_channel": sum(1 for c in campaigns if not c["is_multi_channel"]),
        "n_fallback_used": sum(1 for p in plan if p["fallback_used"]),
        "generated_at": datetime.now().isoformat(),
    }

    with open(OUT_PLAN, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "optimization_plan": plan,
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    report = build_report(plan, metadata)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    action_counts = Counter([p["recommended_action"] for p in plan])

    log.info("-" * 70)
    log.info(f"Recommandations générées : {len(plan)}")
    log.info(f"Actions                  : {dict(action_counts)}")
    log.info(f"Fallbacks                : {metadata['n_fallback_used']}")
    log.info(f"Multi-canal              : {metadata['n_multi_channel']}")
    log.info(f"Mono-canal               : {metadata['n_mono_channel']}")
    log.info(f"JSON                     : {OUT_PLAN}")
    log.info(f"Rapport                  : {OUT_REPORT}")
    log.info(f"Policy                   : {OUT_POLICY}.zip")
    log.info(f"Courbes                  : {OUT_CURVES}")
    log.info("=" * 70)

    return {
        "metadata": metadata,
        "optimization_plan": plan,
    }


if __name__ == "__main__":
    print(run())


def run_optimizer():
    try:
        result = run()
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }