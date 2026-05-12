# -*- coding: utf-8 -*-
"""
===============================================================================
AdOptimizer AI — Agent Marketing Global (Agent 1)
===============================================================================
Modes disponibles :
  learning  → RAG pédagogique (ChromaDB + LLaMA 3)
  analysis  → XAI (explications campagne active depuis xai_explanations.json)

UTILISATION :
  agent_marketing_global("C'est quoi le ROAS ?",              mode="learning")
  agent_marketing_global("Pourquoi mon CPA est élevé ?",      mode="analysis")
  agent_marketing_global("Explique les anomalies",            mode="auto")

FASTAPI :
  payload → { "question": "...", "mode": "learning" | "analysis" | "auto" }
===============================================================================
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
import numpy as np

from dotenv import load_dotenv
load_dotenv()

conversation_history = []
# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION CHEMINS
# ============================================================
BASE_DIR    = Path(__file__).resolve().parent.parent   # → app/
DATA_DIR    = BASE_DIR / "data"
VECTOR_DIR  = BASE_DIR / "vector_db"
OUTPUTS_DIR = BASE_DIR / "outputs"

XAI_PATH = OUTPUTS_DIR / "xai_explanations.json"

DOC_DIRS = {
    "general"   : DATA_DIR / "general_marketing",
    "strategic" : DATA_DIR / "strategic_marketing",
    "ads"       : DATA_DIR / "digital_ads",
}

VECTOR_DIRS = {
    "general"   : VECTOR_DIR / "general",
    "strategic" : VECTOR_DIR / "strategic",
    "ads"       : VECTOR_DIR / "ads",
}

for d in list(DOC_DIRS.values()) + list(VECTOR_DIRS.values()) + [OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# CLÉS API
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️  GROQ_API_KEY non définie.")

# ============================================================
# LLM
# ============================================================
client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ============================================================
# EMBEDDINGS (singleton)
# ============================================================
class EmbeddingModel:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return cls._instance

embeddings = None

# ============================================================
# CHARGEMENT DOCUMENTS & VECTOR DB
# ============================================================

def load_docs(path: Path) -> list:
    docs = []
    if not path.exists():
        return docs
    for file in path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())
            logger.info(f"  ✅ {file.name} chargé")
        except Exception as e:
            logger.error(f"  ❌ {file.name} : {e}")
    return docs


def create_or_load_db(chunks: list, persist_dir: Path, name: str):
    embedding_model = EmbeddingModel.get_instance()
    if persist_dir.exists() and any(persist_dir.iterdir()):
        logger.info(f"  🔄 {name} : rechargée depuis disque")
        return Chroma(persist_directory=str(persist_dir), embedding_function=embedding_model)
    logger.info(f"  ⏳ {name} : création en cours...")
    return Chroma.from_documents(chunks, embedding_model, persist_directory=str(persist_dir))


def init_vector_dbs() -> dict:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    dbs = {}
    for name, doc_path in DOC_DIRS.items():
        docs   = load_docs(doc_path)
        chunks = splitter.split_documents(docs)
        dbs[name] = create_or_load_db(chunks, VECTOR_DIRS[name], name)
        logger.info(f"  DB '{name}' : {len(chunks)} chunks")
    return dbs


DOMAIN_DB_MAP = None


def get_domain_db_map() -> dict:
    global DOMAIN_DB_MAP
    if DOMAIN_DB_MAP is None:
        DOMAIN_DB_MAP = init_vector_dbs()
    return DOMAIN_DB_MAP

# ============================================================
# ROUTER INTELLIGENT
# ============================================================
CONFIDENCE_THRESHOLD = 0.40
MULTI_DOMAIN_GAP     = 0.02


def is_marketing_question(question: str) -> bool:
    prompt = f"""Is this question related to marketing, advertising, branding, sales,
customer behaviour, pricing, product, promotion or business strategy?
Answer ONLY YES or NO.

Question: {question}"""
    try:
        ans = ask_llm(prompt).strip().upper()
        return ans.startswith("YES")
    except:
        return True


def route_question(question: str, k_probe: int = 5, verbose: bool = False):
    scores = {}
    domain_db_map = get_domain_db_map()
    for domain, db in domain_db_map.items():
        results = db.similarity_search_with_score(question, k=k_probe)
        if not results:
            scores[domain] = 0.0
            continue
        domain_scores = [1 / (1 + dist) for _, dist in results]
        weights       = np.linspace(1.0, 0.5, len(domain_scores))
        scores[domain] = float(np.average(domain_scores, weights=weights))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = ranked[0]

    if verbose:
        print("🧭 Scores:", {k: round(v, 3) for k, v in ranked})

    if best_score < CONFIDENCE_THRESHOLD:
        return [], scores

    selected = [best_domain]
    for d, s in ranked[1:2]:
        if best_score - s < MULTI_DOMAIN_GAP:
            selected.append(d)

    return selected, scores


# ============================================================
# MODE LEARNING — RAG pédagogique
# ============================================================

def ask_agent(question: str, k: int = 4, verbose: bool = False) -> dict:
    k = min(k, 8)

    if not is_marketing_question(question):
        return {
            "answer" : "Cette question est hors domaine marketing.",
            "domain" : [],
            "sources": [],
            "mode"   : "out_of_domain"
        }

    domains, scores = route_question(question, verbose=verbose)

    if verbose:
        print("📌 Domaines sélectionnés:", domains)
        print("📊 Scores:", {k: round(v, 3) for k, v in scores.items()})

    if not domains:
        response = ask_llm(question)
        return {
            "answer" : response,
            "domain" : ["fallback_llm"],
            "sources": [],
            "mode"   : "llm_fallback"
        }

    docs = []
    seen = set()
    domain_db_map = get_domain_db_map()
    for d in domains:
        retriever = domain_db_map[d].as_retriever(search_kwargs={"k": k})
        for doc in retriever.invoke(question):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)

    if not docs:
        response = ask_llm(question)
        return {
            "answer" : response,
            "domain" : domains,
            "sources": [],
            "mode"   : "llm_fallback"
        }

    context = "\n\n---\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "unknown") for d in docs]))

    full_prompt = f"""You are a senior marketing expert.

Use ONLY the context below to answer.
If information is missing, say: "I don't know".

Response structure:
1. Definition
2. Explanation
3. Importance
4. Example
5. Practical advice

Context:
{context}

Question:
{question}"""

    response = ask_llm(full_prompt)

    return {
        "answer" : response,
        "domain" : domains,
        "sources": sources,
        "mode"   : "rag"
    }


# ============================================================
# MODE ANALYSIS — XAI
# ============================================================

def load_xai() -> list:
    """Charge les explications XAI depuis le fichier JSON."""
    if not XAI_PATH.exists():
        logger.warning(f"⚠️  XAI file introuvable : {XAI_PATH}")
        return []
    with open(XAI_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("xai_explanations", [])


def explain_with_xai(question: str) -> str:
    """
    Mode Analyse système :
    Utilise UNIQUEMENT la campagne active (xai_data[:1])
    pour éviter le bruit des campagnes historiques.
    """
    xai_data = load_xai()

    if not xai_data:
        return (
            "⚠️ Aucune donnée XAI disponible. "
            "Lancez d'abord le pipeline Tool XAI pour générer les explications."
        )

    # ✅ Filtrage : on garde seulement la campagne active (index 0)
    # Les campagnes suivantes sont historiques → bruit inutile pour le LLM
    active_campaigns = xai_data

    # Construire contexte XAI structuré
    context = ""
    for c in active_campaigns:
        campaign_id = c.get("campaign_id", "N/A")
        platform    = c.get("platform", "N/A")
        summary     = c.get("xai_summary", "")
        health      = c.get("health_explanation", {})
        causal      = c.get("causal_explanation", {})
        optimizer   = c.get("optimizer_explanation", {})

        context += f"""
━━━ Campagne : {campaign_id} | Plateforme : {platform} ━━━
Résumé XAI      : {summary}
Health Score    : {health.get("health_score", "N/A")} — {health.get("status", "")}
Raisons santé   : {", ".join(health.get("main_reasons", []))}
Analyse causale : {causal.get("summary", "")}
Plan optimizer  : {optimizer.get("summary", "")}
"""

    prompt = f"""You are an expert marketing data analyst specializing in digital advertising performance.

You have access to AI-generated campaign analysis data below.
Use it to answer the user's question with precision and business reasoning.

Campaign Analysis Data:
{context}

User Question:
{question}

Instructions:
- Be concise (max 120–150 words)
- Avoid repetition
- Focus on actionable insights
- Use a professional business tone

Structure your answer as:

1. Diagnostic (2–3 sentences max)
2. Root causes (max 2–3 key drivers)
3. Recommended action (clear and actionable)
4. Risks / trade-offs (if any)
5. If multiple campaigns: briefly compare performance and impact

Your answer must be data-driven, precise, and decision-oriented.
"""

    try:
        return ask_llm(prompt)
    except Exception as e:
        logger.error(f"LLM error in XAI mode: {e}")
        return f"Erreur lors de l'analyse XAI : {str(e)}"


# ============================================================
# INTENT DETECTION (fallback auto si mode non fourni)
# ============================================================

def detect_intent(user_input: str) -> str:
    """
    Détecte automatiquement le mode si non spécifié.
    Retourne : learning | analysis
    """
    prompt = f"""Classify this request into ONE category:

LEARNING  → user wants to learn marketing concepts, understand metrics, get explanations
ANALYSIS  → user asks about campaign performance, anomalies, health score, why metrics changed

Answer ONLY: LEARNING or ANALYSIS

Request: {user_input}"""

    try:
        response = ask_llm(prompt).strip().upper()
        if "ANALYSIS" in response:
            return "analysis"
        else:
            return "learning"
    except:
        return "learning"


# ============================================================
# AGENT PRINCIPAL — 2 MODES
# ============================================================

def agent_marketing_global(user_input: str, mode: str = "auto") -> dict:
    """
    Point d'entrée principal de l'agent.

    Paramètres :
      user_input : question de l'utilisateur
      mode       : "learning" | "analysis" | "auto"

    Retourne :
      { "answer": str, "mode": str, "status": "success" | "error" }
    """
    print(f"\n🤖 AdOptimizer AI")
    print(f"📥 Question : {user_input}")

    # Auto-détection si mode non spécifié
    if mode == "auto":
        mode = detect_intent(user_input)
        print(f"🧠 Mode auto-détecté : {mode}")
    else:
        print(f"🎯 Mode choisi : {mode}")

    try:
        # ──────────────────────────────────────────
        # 📚 MODE LEARNING — RAG pédagogique
        # ──────────────────────────────────────────
        if mode == "learning":
            result = ask_agent(user_input)
            return {
                "answer" : result["answer"],
                "mode"   : "learning",
                "domain" : result.get("domain", []),
                "sources": result.get("sources", []),
                "status" : "success"
            }

        # ──────────────────────────────────────────
        # 📊 MODE ANALYSIS — XAI
        # ──────────────────────────────────────────
        elif mode == "analysis":
            answer = explain_with_xai(user_input)
            return {
                "answer" : answer,
                "mode"   : "analysis",
                "status" : "success"
            }

        else:
            return {
                "answer" : f"Mode '{mode}' invalide. Utilisez : learning, analysis, auto.",
                "mode"   : "error",
                "status" : "error"
            }

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {
            "answer" : f"Erreur interne : {str(e)}",
            "mode"   : mode,
            "status" : "error"
        }


# ============================================================
# WRAPPER FASTAPI / N8N
# ============================================================

def run_agent(question: str, mode: str = "auto") -> dict:
    global conversation_history

    # 🔹 Construire contexte
    history_text = ""
    for msg in conversation_history[-4:]:
        history_text += msg + "\n"

    full_question = f"""
Conversation précédente:
{history_text}

Nouvelle question:
{question}
"""

    # 🔥 LA CORRECTION ICI
    if mode == "analysis":
        result = agent_marketing_global(question, mode)
    else:
        result = agent_marketing_global(full_question, mode)

    # 🔹 sauvegarde
    conversation_history.append(f"User: {question}")
    conversation_history.append(f"Assistant: {result.get('answer', '')[:300]}")

    return result


# ============================================================
# CAS 1 - DASHBOARD SUMMARY LLM
# ============================================================
# Cette section reformule les resultats techniques pour l'interface
# Campaign Optimization. Elle ne modifie pas le chatbot run_agent().

def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _dashboard_action_label(payload: dict) -> str:
    return (
        payload.get("action_label")
        or payload.get("recommended_action")
        or "Action recommandee"
    )


def _is_maintain_action(payload: dict) -> bool:
    action = str(payload.get("recommended_action") or "").lower()
    action_label = _dashboard_action_label(payload).lower()
    return "maintain" in action or action_label.startswith("maint")


def _build_dashboard_summary_fallback(payload: dict) -> str:
    action_label = _dashboard_action_label(payload)
    root_cause = payload.get("root_cause_label") or payload.get("root_cause")
    expected = payload.get("expected_impact") or {}
    health_score = payload.get("health_score")
    health_status = payload.get("health_status") or "a surveiller"

    delta_roas = _safe_float(expected.get("delta_roas_pct"))
    delta_conversions = _safe_float(expected.get("delta_conversions"))
    expected_roas = _safe_float(expected.get("expected_roas"))

    if _is_maintain_action(payload):
        if expected_roas and expected_roas < 1:
            return (
                "L'IA recommande de maintenir le budget sous surveillance, "
                "pas de scaler la campagne. La rentabilite reste faible "
                f"(ROAS prevu {expected_roas:.2f}x), meme si certains signaux peuvent s'ameliorer. "
                "Il faut surveiller les prochaines 48 heures et revoir le ciblage, "
                "les creatives ou l'offre avant toute hausse de budget."
            )

        if delta_roas > 0 and delta_conversions < 0:
            return (
                "L'IA recommande de maintenir le budget sous surveillance. "
                f"Le scenario peut ameliorer le ROAS d'environ {delta_roas:.1f}%, "
                "mais il risque de reduire le volume de conversions. "
                "Il ne faut pas augmenter le budget tant que le volume reste fragile."
            )

        return (
            "L'IA recommande de maintenir la configuration actuelle. "
            "Les signaux ne justifient pas encore une modification automatique du budget, "
            "donc la meilleure action est de surveiller les performances avant d'intervenir. "
            "Il vaut mieux attendre un signal plus stable avant de modifier le budget."
        )

    return (
        f"Action recommandee : {action_label}. "
        f"La campagne est classee {health_status} avec un health score de {health_score}. "
        f"La cause principale identifiee est {root_cause}. "
        "L'action doit etre appliquee progressivement avec suivi du ROAS, du CPA et des conversions."
    )


def _build_dashboard_summary_prompt(payload: dict) -> str:
    compact_payload = {
        "campaign_id": payload.get("campaign_id"),
        "platform": payload.get("platform"),
        "health_score": payload.get("health_score"),
        "health_status": payload.get("health_status"),
        "anomaly_level": payload.get("anomaly_level"),
        "current_kpis": payload.get("current_kpis"),
        "predicted_kpis": payload.get("predicted_kpis"),
        "root_cause": payload.get("root_cause"),
        "root_cause_label": payload.get("root_cause_label"),
        "recommended_action": payload.get("recommended_action"),
        "action_label": payload.get("action_label"),
        "expected_impact": payload.get("expected_impact"),
        "budget_adjustment": payload.get("budget_adjustment"),
        "top_anomalies": payload.get("top_anomalies"),
    }

    return f"""Tu es l'assistant marketing d'AdOptimizer AI.

Objectif: reformuler une recommandation technique pour un utilisateur non technique.
Tu ne dois PAS changer la decision recommandee.
Tu ne dois PAS inventer de nouveaux chiffres.
Tu ne dois PAS mentionner backend, XAI, causal model, RL, JSON, pipeline ou fichiers.

Donnees techniques:
{json.dumps(compact_payload, ensure_ascii=False, indent=2)}

Reponds en francais clair, en 2 ou 3 phrases maximum.
La reponse sera affichee dans une carte "AI Recommended Action".
Si l'action est "maintain_budget", ne donne pas l'impression que tout va bien.
Explique que la bonne decision est de maintenir le budget sous surveillance, sans augmenter.
Si expected_impact.delta_conversions est negatif, tu dois explicitement mentionner le compromis:
le ROAS peut s'ameliorer, mais le volume de conversions risque de baisser.
Ne presente jamais une recommandation comme entierement positive si delta_conversions est negatif.
Si expected_impact.expected_roas est inferieur a 1, dis clairement que la rentabilite reste faible
et qu'il ne faut pas scaler la campagne pour le moment.
Ne commence pas une phrase par "Conseil business", car l'interface affiche deja ce bloc separement.
Retourne uniquement le texte final, sans markdown et sans titre.
"""


def build_dashboard_summary(payload: dict, use_llm: bool = True) -> tuple[str, str]:
    fallback = _build_dashboard_summary_fallback(payload)

    if not use_llm:
        return fallback, "fallback_disabled"

    if not GROQ_API_KEY:
        return fallback, "fallback_no_api_key"

    try:
        prompt = _build_dashboard_summary_prompt(payload)
        summary = ask_llm(prompt).strip().strip('"').strip("'").strip()
        if not summary:
            return fallback, "fallback_empty_llm_response"
        return summary, "llm"
    except Exception as e:
        logger.error(f"Dashboard summary LLM error: {e}")
        return fallback, "fallback_llm_error"


def run_dashboard_summary(payload: dict, use_llm: bool = True) -> dict:
    summary, generation_mode = build_dashboard_summary(payload, use_llm=use_llm)
    result = dict(payload)
    result["dashboard_summary"] = summary
    result["dashboard_summary_generation_mode"] = generation_mode
    result["dashboard_summary_generated_at"] = datetime.now().isoformat()
    return result


# ============================================================
# CAS 2 - FINAL RESPONSE LLM
# ============================================================
# Cette section est separee du chatbot cas 1.
# Elle ne modifie pas run_agent(), agent_marketing_global(), learning ou analysis.

CASE2_XAI_PATH = BASE_DIR / "cas2-outputs" / "xai_outputs" / "case2_xai_explanation.json"
CASE2_FINAL_DIR = BASE_DIR / "cas2-outputs" / "final_outputs"
CASE2_FINAL_RESPONSE_PATH = CASE2_FINAL_DIR / "case2_final_response.json"


def _load_case2_xai() -> dict:
    if not CASE2_XAI_PATH.exists():
        raise FileNotFoundError(
            f"Fichier XAI cas 2 introuvable : {CASE2_XAI_PATH}. "
            "Lancez d'abord case2_xai."
        )

    with open(CASE2_XAI_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_case2_money(value) -> str:
    try:
        return f"{float(value):,.2f} EUR"
    except (TypeError, ValueError):
        return "N/A"


def _format_case2_percent(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _case2_has_multi_platform_plan(xai_data: dict) -> bool:
    plan = xai_data.get("multi_platform_plan")
    return bool(plan and plan.get("channels"))


def _format_case2_channel(channel: dict) -> str:
    kpis = channel.get("kpis", {})
    platform = str(channel.get("platform", "canal")).capitalize()
    strategy = channel.get("name") or channel.get("strategy_id") or "strategie recommandee"
    budget = _format_case2_money(channel.get("budget"))
    roas = kpis.get("roas_j14")
    cpa = _format_case2_money(kpis.get("cpa_j14"))
    conversions = kpis.get("conversions_j14")

    return (
        f"{platform}: {strategy}, budget {budget}, "
        f"ROAS {roas}x, {conversions} conversions, CPA {cpa}"
    )


def _build_case2_fallback_message(xai_data: dict) -> str:
    if _case2_has_multi_platform_plan(xai_data):
        plan = xai_data.get("multi_platform_plan", {})
        channels = plan.get("channels", [])
        channel_text = "; ".join(_format_case2_channel(channel) for channel in channels)
        reserve = _format_case2_money(plan.get("reserve_budget"))
        total_budget = _format_case2_money(plan.get("total_budget"))

        return (
            f"Nous recommandons un plan Meta Ads + Google Ads avec un budget total de {total_budget}. "
            f"{channel_text}. Une reserve de {reserve} est conservee pour ajuster la reallocation apres les premiers resultats. "
            "Ce plan permet de garder la meilleure strategie par canal tout en limitant le risque avant le scaling."
        )

    recommendation = xai_data.get("recommendation", {})
    kpis = recommendation.get("kpis", {})
    name = recommendation.get("name", "la strategie recommandee")
    platform = recommendation.get("platform", "la plateforme choisie")
    budget = _format_case2_money(recommendation.get("budget"))
    objective = recommendation.get("objective", "l'objectif choisi")

    return (
        f"Nous recommandons {name} sur {platform} avec un budget de {budget} "
        f"pour atteindre l'objectif {objective}. Cette recommandation est soutenue par "
        f"un ROAS predit de {kpis.get('roas_j14')}x a J+14, environ "
        f"{kpis.get('conversions_j14')} conversions et un CPA estime a "
        f"{_format_case2_money(kpis.get('cpa_j14'))}. Il est conseille de lancer "
        "un test initial, de surveiller le CPA et le ROAS, puis d'augmenter le budget "
        "progressivement si les resultats restent stables."
    )


def _build_case2_llm_prompt(xai_data: dict) -> str:
    recommendation = xai_data.get("recommendation", {})
    kpis = recommendation.get("kpis", {})
    why = xai_data.get("why", [])
    multi_platform_plan = xai_data.get("multi_platform_plan")
    multi_platform_why = xai_data.get("multi_platform_why", [])
    risks = xai_data.get("risks", [])
    action_plan = xai_data.get("action_plan", [])
    multi_platform_action_plan = xai_data.get("multi_platform_action_plan", [])
    plan_instruction = (
        "Le champ multi_platform_plan est present: la reponse DOIT presenter un plan Meta Ads + Google Ads "
        "avec les budgets par canal et la reserve. Ne pas reformuler comme si une seule plateforme etait choisie."
        if _case2_has_multi_platform_plan(xai_data)
        else "Le champ multi_platform_plan est absent: la reponse peut presenter la strategie unique recommandee."
    )

    return f"""Tu es un assistant marketing pour AdOptimizer AI.

Ta mission: reformuler la recommandation finale pour un utilisateur non technique.
Tu ne dois PAS choisir une autre strategie.
Tu ne dois PAS parler du backend, du pipeline, des fichiers, du ranking ou des calculs internes.
Utilise uniquement les donnees ci-dessous.
{plan_instruction}

Strategie recommandee:
- Nom: {recommendation.get("name")}
- Type: {recommendation.get("type")}
- Plateforme: {recommendation.get("platform")}
- Objectif: {recommendation.get("objective")}
- Produit: {recommendation.get("product")}
- Budget: {_format_case2_money(recommendation.get("budget"))}

KPIs predits a J+14:
- ROAS: {kpis.get("roas_j14")}x
- Conversions: {kpis.get("conversions_j14")}
- CPA: {_format_case2_money(kpis.get("cpa_j14"))}
- CTR: {_format_case2_percent(kpis.get("ctr_j14"))}
- CPC: {_format_case2_money(kpis.get("cpc_j14"))}

Plan multi-plateforme:
{json.dumps(multi_platform_plan, ensure_ascii=False, indent=2)}

Pourquoi le plan multi-plateforme:
{json.dumps(multi_platform_why, ensure_ascii=False, indent=2)}

Pourquoi:
{json.dumps(why, ensure_ascii=False, indent=2)}

Risques:
{json.dumps(risks, ensure_ascii=False, indent=2)}

Plan d'action:
{json.dumps(action_plan, ensure_ascii=False, indent=2)}

Plan d'action multi-plateforme:
{json.dumps(multi_platform_action_plan, ensure_ascii=False, indent=2)}

Reponds en francais, en 1 paragraphe clair de 80 a 120 mots.
La reponse doit etre directement affichable dans Angular.
Ne retourne que le texte final, sans JSON, sans markdown, sans titre.
"""


def _generate_case2_final_message(xai_data: dict, use_llm: bool = True) -> tuple[str, str]:
    if not use_llm:
        return _build_case2_fallback_message(xai_data), "fallback_disabled"

    if not GROQ_API_KEY:
        return _build_case2_fallback_message(xai_data), "fallback_no_api_key"

    try:
        prompt = _build_case2_llm_prompt(xai_data)
        message = ask_llm(prompt).strip()
        message = message.strip('"').strip("'").strip()
        if not message:
            return _build_case2_fallback_message(xai_data), "fallback_empty_llm_response"
        return message, "llm"
    except Exception as e:
        logger.error(f"Case2 final LLM error: {e}")
        return _build_case2_fallback_message(xai_data), "fallback_llm_error"


def build_case2_final_response(use_llm: bool = True) -> dict:
    xai_data = _load_case2_xai()
    final_message, generation_mode = _generate_case2_final_message(xai_data, use_llm=use_llm)

    return {
        "title": xai_data.get("title", "Recommandation de nouvelle campagne"),
        "generated_at": datetime.now().isoformat(),
        "generation_mode": generation_mode,
        "final_message": final_message,
        "recommendation": xai_data.get("recommendation"),
        "best_by_platform": xai_data.get("best_by_platform", {}),
        "multi_platform_plan": xai_data.get("multi_platform_plan"),
        "why": xai_data.get("why", []),
        "multi_platform_why": xai_data.get("multi_platform_why", []),
        "risks": xai_data.get("risks", []),
        "action_plan": xai_data.get("action_plan", []),
        "multi_platform_action_plan": xai_data.get("multi_platform_action_plan", []),
        "confidence": xai_data.get("confidence", {}),
    }


def run_case2_final_response(use_llm: bool = True) -> dict:
    try:
        CASE2_FINAL_DIR.mkdir(parents=True, exist_ok=True)
        final_response = build_case2_final_response(use_llm=use_llm)

        with open(CASE2_FINAL_RESPONSE_PATH, "w", encoding="utf-8") as f:
            json.dump(final_response, f, indent=2, ensure_ascii=False)

        return {
            "status": "success",
            "message": "Reponse finale cas 2 generee avec succes",
            "output_file": str(CASE2_FINAL_RESPONSE_PATH),
            "data": final_response,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "input_file": str(CASE2_XAI_PATH),
        }


# ============================================================
# POINT D'ENTRÉE SCRIPT DIRECT
# ============================================================

if __name__ == "__main__":

    tests = [
        ("C'est quoi le ROAS ?",                       "learning"),
        ("Pourquoi mon CPA est élevé sur Google ?",    "analysis"),
        ("Explique les anomalies de ma campagne",       "auto"),
        ("Que signifie un CTR faible ?",                "auto"),
    ]

    for question, mode in tests:
        print("\n" + "=" * 70)
        result = agent_marketing_global(question, mode=mode)
        print(f"Mode    : {result['mode']}")
        print(f"Réponse : {result['answer'][:300]}...")
        print("=" * 70)
 
