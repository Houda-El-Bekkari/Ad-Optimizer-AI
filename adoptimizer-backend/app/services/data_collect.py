import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

np.random.seed(123)
random.seed(123)

START_DATE = datetime(2024, 1, 1)
DAYS = 90
OUTPUT_FILE = "app/data/raw_ads_dataset_strict.csv"
NA_PLATFORM = "NA_platform"

def maybe_nan(value, rate=0.02):
    return np.nan if random.random() < rate else value

def natural_demand_factor(date):
    doy = date.timetuple().tm_yday
    seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * doy / 365)
    weekly = 1.05 if date.weekday() >= 5 else 1.0
    return seasonal * weekly

def conversion_value(conversions, avg_value):
    if conversions <= 0:
        return 0.0
    return round(conversions * random.uniform(avg_value * 0.85, avg_value * 1.10), 2)

def problem_factor(day):
    if day < 60:
        return {
            "ctr_mult": 1.0,
            "cpc_mult": 1.0,
            "conv_mult": 1.0,
            "value_mult": 1.0,
            "spend_mult": 1.0
        }
    elif day < 75:
        return {
            "ctr_mult": 0.80,
            "cpc_mult": 1.25,
            "conv_mult": 0.70,
            "value_mult": 0.80,
            "spend_mult": 1.10
        }
    else:
        return {
            "ctr_mult": 0.55,
            "cpc_mult": 1.65,
            "conv_mult": 0.35,
            "value_mult": 0.55,
            "spend_mult": 1.25
        }

def generate_bad_campaign_two_channels():
    rows = []

    gcamp = {
        "global_campaign_id": "GCAMP_BAD_001",
        "campaign_objective": "conversion",
        "campaign_status": "ACTIVE",
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date": (START_DATE + timedelta(days=120)).strftime("%Y-%m-%d"),
        "budget_type": "daily",
        "location": "Morocco | Casablanca"
    }

    configs = {
        "meta": {
            "campaign_id": "META_BAD_001",
            "adset_id": "ADSET_BAD_001",
            "ad_id": "AD_META_BAD_001",
            "daily_budget": 220,
            "base_ctr": 0.030,
            "base_cpc": 0.70,
            "base_conv_rate": 0.040,
            "ad_format": "video",
            "avg_value": 85
        },
        "google": {
            "campaign_id": "GOOG_BAD_001",
            "adset_id": "ADGROUP_BAD_001",
            "ad_id": "GAD_BAD_001",
            "daily_budget": 240,
            "base_ctr": 0.060,
            "base_cpc": 1.05,
            "base_conv_rate": 0.050,
            "ad_format": "responsive_search",
            "avg_value": 90
        }
    }

    for platform, cfg in configs.items():
        for day in range(DAYS):
            date = START_DATE + timedelta(days=day)

            demand = natural_demand_factor(date)
            noise = random.uniform(0.85, 1.15)
            pf = problem_factor(day)

            shock = 1.0
            if day in [78, 79, 80, 86, 87]:
                shock = 0.45

            spend = round(
                cfg["daily_budget"]
                * random.uniform(0.85, 1.02)
                * demand
                * noise
                * pf["spend_mult"],
                2
            )

            ctr = max(
                0.001,
                cfg["base_ctr"]
                * random.uniform(0.90, 1.08)
                * demand
                * pf["ctr_mult"]
                * shock
            )

            cpc = max(
                0.10,
                cfg["base_cpc"]
                * random.uniform(0.95, 1.10)
                * pf["cpc_mult"]
            )

            clicks = max(1, int(spend / cpc))
            impressions = max(clicks + 1, int(clicks / ctr)) if ctr > 0 else 0

            conv_rate = (
                cfg["base_conv_rate"]
                * random.uniform(0.90, 1.08)
                * demand
                * pf["conv_mult"]
                * shock
            )

            conversions = max(0, int(clicks * conv_rate))

            conv_value = conversion_value(
                conversions,
                avg_value=cfg["avg_value"] * pf["value_mult"]
            )

            cpm = round(spend / impressions * 1000, 2) if impressions > 0 else 0.0

            common = {
                "global_campaign_id": gcamp["global_campaign_id"],
                "platform": platform,
                "campaign_id": cfg["campaign_id"],
                "adset_id": cfg["adset_id"],
                "ad_id": cfg["ad_id"],
                "date": date.strftime("%Y-%m-%d"),
                "campaign_objective": gcamp["campaign_objective"],
                "campaign_status": gcamp["campaign_status"],
                "start_date": gcamp["start_date"],
                "end_date": gcamp["end_date"],
                "budget_type": gcamp["budget_type"],
                "daily_budget": cfg["daily_budget"],
                "lifetime_budget": cfg["daily_budget"] * DAYS,
                "spend": maybe_nan(spend),
                "impressions": maybe_nan(impressions),
                "clicks": maybe_nan(clicks),
                "conversions": maybe_nan(conversions),
                "conversion_value": maybe_nan(conv_value),
                "CTR": maybe_nan(round(ctr, 5)),
                "CPC": maybe_nan(round(cpc, 3)),
                "CPM": maybe_nan(cpm),
                "device": random.choice(["mobile", "desktop", "tablet"]),
                "location": gcamp["location"],
                "ad_format": cfg["ad_format"],
                "primary_text": "Campaign test with realistic degradation and anomalies."
            }

            if platform == "meta":
                frequency = round(random.uniform(2.4, 4.8) if day >= 60 else random.uniform(1.5, 2.4), 2)
                reach = int(impressions / max(frequency, 1.0))

                likes = int(impressions * random.uniform(0.004, 0.012) * pf["ctr_mult"])
                comments = int(likes * random.uniform(0.04, 0.10))
                shares = int(likes * random.uniform(0.02, 0.06))

                row = {
                    **common,
                    "reach": maybe_nan(reach),
                    "frequency": maybe_nan(frequency),
                    "link_clicks": maybe_nan(int(clicks * 0.85)),
                    "likes": maybe_nan(likes),
                    "comments": maybe_nan(comments),
                    "shares": maybe_nan(shares),
                    "post_engagement": maybe_nan(likes + comments + shares),
                    "video_views": maybe_nan(int(impressions * random.uniform(0.25, 0.45))),
                    "add_to_cart": maybe_nan(int(conversions * random.uniform(1.2, 2.0))),
                    "purchases": maybe_nan(int(conversions * random.uniform(0.5, 0.85))),
                    "age": "25-34",
                    "gender": "all",
                    "network": NA_PLATFORM,
                    "keyword": NA_PLATFORM,
                    "match_type": NA_PLATFORM,
                    "search_term": NA_PLATFORM,
                    "quality_score": NA_PLATFORM,
                }
            else:
                quality_score = random.choice([7, 8, 9]) if day < 60 else random.choice([4, 5, 6])
                if day >= 75:
                    quality_score = random.choice([2, 3, 4])

                row = {
                    **common,
                    "reach": NA_PLATFORM,
                    "frequency": NA_PLATFORM,
                    "link_clicks": NA_PLATFORM,
                    "likes": NA_PLATFORM,
                    "comments": NA_PLATFORM,
                    "shares": NA_PLATFORM,
                    "post_engagement": NA_PLATFORM,
                    "video_views": NA_PLATFORM,
                    "add_to_cart": NA_PLATFORM,
                    "purchases": NA_PLATFORM,
                    "age": NA_PLATFORM,
                    "gender": NA_PLATFORM,
                    "network": "search",
                    "keyword": "best_marketing_solution",
                    "match_type": "phrase",
                    "search_term": "best marketing solution",
                    "quality_score": quality_score,
                }

            rows.append(row)

    df = pd.DataFrame(rows)

    dup = df.sample(frac=0.015, random_state=123)
    df = pd.concat([df, dup], ignore_index=True)

    df = df.sort_values(["global_campaign_id", "platform", "date"]).reset_index(drop=True)

    # ✅ création dossier backend
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Dataset mauvais généré")
    print(f"Fichier : {OUTPUT_FILE}")
    print(f"Lignes : {len(df)}")

    return df


# ✅ FONCTION APPELÉE PAR FASTAPI / N8N
def run_data_collect():
    try:
        df = generate_bad_campaign_two_channels()

        return {
            "status": "success",
            "message": "Dataset généré avec campagne problématique Meta + Google",
            "output_file": OUTPUT_FILE,
            "rows": len(df),
            "global_campaigns": int(df["global_campaign_id"].nunique()),
            "technical_campaigns": int(df["campaign_id"].nunique())
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }