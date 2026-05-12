"""
inference.py — Interface de prediction AdOptimizer AI
Genere automatiquement par tool2_training_VV3.py
Modele : LightGBM  |  15 targets : 5 KPIs x 3 horizons
"""
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/best_model.pkl"


def _load_bundle():
    return joblib.load(MODEL_PATH)


def predict_future_kpis(features_dict: dict) -> dict:
    """
    Predit les 5 KPIs sur J+3, J+7, J+14 pour une campagne donnee.

    Input  : features_dict — dictionnaire avec les features du jour courant
             (memes colonnes que feature_cols du bundle)

    Output : {
        "roas"        : {"J+3": 2.1,  "J+7": 1.9,  "J+14": 1.7},
        "conversions" : {"J+3": 45,   "J+7": 98,   "J+14": 187},
        "cpa"         : {"J+3": 12.3, "J+7": 13.1, "J+14": 14.2},
        "ctr"         : {"J+3": 0.03, "J+7": 0.028,"J+14": 0.025},
        "cpc"         : {"J+3": 0.41, "J+7": 0.43, "J+14": 0.46},
    }
    """
    bundle      = _load_bundle()
    models      = bundle["models_by_target"]
    imputer     = bundle["imputer"]
    feats       = bundle["feature_cols"]
    target_cols = bundle["target_cols"]

    X     = pd.DataFrame([features_dict]).reindex(columns=feats)
    X_imp = pd.DataFrame(imputer.transform(X), columns=feats)

    raw = {}
    for target, est in models.items():
        raw[target] = float(est.predict(X_imp)[0])

    result = {}
    for metric in ['roas', 'conversions', 'cpa', 'ctr', 'cpc']:
        result[metric] = {
            "J+3" : raw.get(f"target_{metric}_h3",  None),
            "J+7" : raw.get(f"target_{metric}_h7",  None),
            "J+14": raw.get(f"target_{metric}_h14", None),
        }
    return result


def format_prediction_for_chatbot(prediction: dict,
                                   campaign_name: str = "") -> str:
    """
    Formate les predictions pour affichage dans le chatbot.
    Retourne une chaine de texte tabulee prete a l'emploi.
    """
    kpi_labels = {'roas': 'Return On Ad Spend', 'conversions': 'Conversions', 'cpa': 'Cout par Acquisition', 'ctr': 'Click Through Rate', 'cpc': 'Cout par Clic'}
    formatters = {
        "roas"        : lambda x: f"{x:.2f}x",
        "conversions" : lambda x: f"{int(round(x))}",
        "cpa"         : lambda x: f"{x:.2f} EUR",
        "ctr"         : lambda x: f"{x*100:.2f}%",
        "cpc"         : lambda x: f"{x:.3f} EUR",
    }

    lines = []
    if campaign_name:
        lines.append(f"Predictions pour : {campaign_name}")
        lines.append("")
    lines.append("-" * 56)
    lines.append(f"{"KPI":<22} {"J+3":<11} {"J+7":<11} {"J+14":<11}")
    lines.append("-" * 56)

    for metric, horizons in prediction.items():
        fmt   = formatters.get(metric, lambda x: f"{x:.4f}")
        label = kpi_labels.get(metric, metric)
        v3    = fmt(horizons["J+3"])  if horizons["J+3"]  is not None else "N/A"
        v7    = fmt(horizons["J+7"])  if horizons["J+7"]  is not None else "N/A"
        v14   = fmt(horizons["J+14"]) if horizons["J+14"] is not None else "N/A"
        lines.append(f"{label:<22} {v3:<11} {v7:<11} {v14:<11}")

    lines.append("-" * 56)
    return "\n".join(lines)


def get_feature_importance(top_n: int = 15) -> pd.DataFrame:
    """
    Charge et retourne le top N des features les plus importantes.
    Necessite que feature_importance.csv ait ete genere par le training.
    """
    fi_path = "models/feature_importance.csv"
    if not pd.io.common.file_exists(fi_path):
        return pd.DataFrame(columns=["feature", "importance"])
    fi = pd.read_csv(fi_path)
    return fi.head(top_n)


if __name__ == "__main__":
    import json

    sample = {
        "spend": 150.0, "impressions": 8500, "clicks": 210,
        "conversions": 7, "daily_budget": 200.0,
        "lifetime_budget": 18000.0, "campaign_age_days": 15,
        "spend_lag1": 142.0, "clicks_lag1": 198,
        "impressions_lag1": 8200, "conversions_lag1": 6,
        "ctr_calc_lag1": 0.0241, "cpc_calc_lag1": 0.717,
        "roas_lag1": 1.82, "conversion_value_lag1": 258.4,
        "spend_roll7m": 138.5, "ctr_calc_roll7m": 0.0235,
        "conversions_roll7m": 5.8, "roas_roll7m": 1.75,
        "dow_sin": 0.782, "dow_cos": 0.623,
        "doy_sin": 0.415, "doy_cos": 0.910,
    }

    pred = predict_future_kpis(sample)
    print(format_prediction_for_chatbot(pred, "Campagne Meta — Test"))
    print("\nResultat brut JSON :")
    print(json.dumps(pred, indent=2))

    print("\nTop 10 features :")
    print(get_feature_importance(10).to_string(index=False))
