# -*- coding: utf-8 -*-
"""
===============================================================================
AdOptimizer AI - Agent Marketing Global (Agent 1) - PREMIUM UPGRADE
===============================================================================
Modes disponibles :
  learning  -> RAG pédagogique mentor-like (ChromaDB + LLaMA 3)
  analysis  -> Consultant décisionnel XAI (campagne active + memoire)

UTILISATION :
  agent_marketing_global("C'est quoi le ROAS ?",              mode="learning")
  agent_marketing_global("Pourquoi mon CPA est élevé ?",      mode="analysis")
  agent_marketing_global("Explique les anomalies",            mode="auto")

FASTAPI :
  payload -> { "question": "...", "mode": "learning" | "analysis" | "auto" }

ARCHITECTURE PRESERVEE :
  - RAG (ChromaDB multi-domaines)
  - Memoire conversationnelle semantique
  - Retrieval semantique multi-requetes
  - Profiling learner adaptatif
  - Routing intelligent
  - FastAPI wrapper (run_agent)
  - Outputs JSON (xai_explanations.json, case2_final_response.json)
  - Separation learning / analysis / cas2

UPGRADES PREMIUM v2 (sans casser l'existant) :
  - Learning : mentor pedagogique dynamique en 2 etapes LLM controlees
    Etape 1 (small LLM, ~150 tok): resolution anaphore + typage question
    Etape 2 (small LLM): reponse mentor + memoire + suggestions
    -> resout les bugs : labels visibles, perte de reference ("l'ameliorer"),
       repetitions sur concepts deja maitrises, style scolaire generique
  - Decision : consultant senior conversationnel avec memoire,
    meme etape d'anaphore pour les follow-ups ("et les risques ?", "et pour Meta ?")
    interpretation business des KPIs, trade-offs, risk reasoning
===============================================================================
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

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
BASE_DIR    = Path(__file__).resolve().parent.parent   # -> app/
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
# CLES API
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY non definie.")

# ============================================================
# LLM
# ============================================================
client = Groq(api_key=GROQ_API_KEY)

SMALL_LLM_MODEL = "llama-3.1-8b-instant"
BIG_LLM_MODEL = "llama-3.3-70b-versatile"


def _ask_groq_model(model: str, prompt: str, temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


def ask_small_llm(prompt: str, temperature: float = 0.1) -> str:
    return _ask_groq_model(SMALL_LLM_MODEL, prompt, temperature=temperature)


def ask_big_llm(prompt: str, temperature: float = 0.3) -> str:
    return _ask_groq_model(BIG_LLM_MODEL, prompt, temperature=temperature)


def ask_llm(prompt: str, temperature: float = 0.3) -> str:
    """Backward-compatible default for existing non-Learning code paths."""
    return ask_big_llm(prompt, temperature=temperature)


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "rate_limit" in text or "rate limit" in text


def _friendly_rate_limit_message() -> str:
    return (
        "Le moteur IA a atteint la limite de tokens du fournisseur pour le moment. "
        "Reessaie dans quelques minutes. Le Learning Mode utilise maintenant moins "
        "d'appels LLM pour reduire ce risque."
    )

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
            logger.info(f"  {file.name} charge")
        except Exception as e:
            logger.error(f"  {file.name} : {e}")
    return docs


def create_or_load_db(chunks: list, persist_dir: Path, name: str):
    embedding_model = EmbeddingModel.get_instance()
    if persist_dir.exists() and any(persist_dir.iterdir()):
        logger.info(f"  {name} : rechargee depuis disque")
        return Chroma(persist_directory=str(persist_dir), embedding_function=embedding_model)
    logger.info(f"  {name} : creation en cours...")
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
    best_domain, best_score = ranked[0] if ranked else ("semantic_llm", 0.0)

    if verbose:
        print("Scores:", {k: round(v, 3) for k, v in ranked})

    if best_score <= 0:
        return [], scores

    selected = [domain for domain, score in ranked if score > 0][:3]

    return selected, scores


# ============================================================
# MEMOIRE CONVERSATIONNELLE (LEARNING + DECISION)
# ============================================================

class ConversationMemory:
    """Semantic memory shared by Learning Mode and Decision Mode."""

    MAX_HISTORY = 12

    def __init__(self):
        self.history: list[dict[str, str]] = []
        self.summary: str = ""
        self.current_topic: str = ""
        self.last_standalone_query: str = ""
        self.learning_trajectory: list[str] = []
        self.mastered_concepts: list[str] = []
        self.confusion_areas: list[str] = []
        self.user_goals: list[str] = []
        self.profile: dict[str, Any] = {}
        # Decision-specific lightweight memory
        self.decision_focus: str = ""          # e.g. "CPA elevation Google Ads"
        self.discussed_kpis: list[str] = []    # KPIs already explained
        self.discussed_actions: list[str] = [] # recommendations already justified

    def add_exchange(self, user_msg: str, assistant_msg: str):
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        self.history = self.history[-(self.MAX_HISTORY * 2):]

    def recent_history_text(self, chars_per_msg: int = 300) -> str:
        lines = []
        for msg in self.history[-8:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:chars_per_msg]}")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "current_topic": self.current_topic,
            "last_standalone_query": self.last_standalone_query,
            "learning_trajectory": self.learning_trajectory[-10:],
            "mastered_concepts": self.mastered_concepts[-12:],
            "confusion_areas": self.confusion_areas[-8:],
            "user_goals": self.user_goals[-8:],
            "profile": self.profile,
            "decision_focus": self.decision_focus,
            "discussed_kpis": self.discussed_kpis[-10:],
            "discussed_actions": self.discussed_actions[-6:],
            "recent_history": self.recent_history_text(),
        }

    def compact_snapshot(self) -> str:
        """Snapshot textuel compact (~600 chars) pour injection dans prompt unifie."""
        parts = []
        if self.current_topic:
            parts.append(f"Sujet actuel: {self.current_topic[:80]}")
        if self.mastered_concepts:
            parts.append(f"Concepts maitrises: {', '.join(self.mastered_concepts[-6:])}")
        if self.confusion_areas:
            parts.append(f"Zones de confusion: {', '.join(self.confusion_areas[-3:])}")
        if self.user_goals:
            parts.append(f"Objectifs: {', '.join(str(g) for g in self.user_goals[-3:])}")
        profile_maturity = self.profile.get("marketing_maturity", "")
        if profile_maturity:
            parts.append(f"Niveau: {str(profile_maturity)[:60]}")
        if self.history:
            parts.append(f"Historique recent:\n{self.recent_history_text(200)}")
        return "\n".join(parts)[:700]

    def decision_snapshot(self) -> str:
        """Snapshot compact dedie au mode Decision (~500 chars)."""
        parts = []
        if self.decision_focus:
            parts.append(f"Focus de la conversation: {self.decision_focus[:100]}")
        if self.discussed_kpis:
            parts.append(f"KPIs deja abordes: {', '.join(self.discussed_kpis[-5:])}")
        if self.discussed_actions:
            parts.append(f"Recommandations deja discutees: {', '.join(self.discussed_actions[-3:])}")
        if self.history:
            parts.append(f"Echanges recents:\n{self.recent_history_text(180)}")
        return "\n".join(parts)[:600]

    def merge_update(self, update: dict[str, Any]):
        if not isinstance(update, dict):
            return
        self.summary = str(update.get("summary") or self.summary)[:1400]
        self.current_topic = str(update.get("current_topic") or self.current_topic)[:160]
        for attr, key in (
            ("learning_trajectory", "learning_trajectory"),
            ("mastered_concepts", "mastered_concepts"),
            ("confusion_areas", "confusion_areas"),
            ("user_goals", "user_goals"),
            ("discussed_kpis", "discussed_kpis"),
            ("discussed_actions", "discussed_actions"),
        ):
            values = update.get(key, [])
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                existing = getattr(self, attr)
                for value in values:
                    text = str(value).strip()
                    if text and text not in existing:
                        existing.append(text)
                setattr(self, attr, existing[-20:])
        if update.get("decision_focus"):
            self.decision_focus = str(update["decision_focus"])[:200]
        profile = update.get("profile")
        if isinstance(profile, dict):
            self.profile.update(profile)

    def reset(self):
        self.__init__()


conversation_memory = ConversationMemory()


def _strip_json_markdown(raw: str) -> str:
    return raw.strip().replace("```json", "").replace("```", "").strip()


def _safe_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(_strip_json_markdown(raw))
    except Exception:
        return default


def _ask_json(prompt: str, default: Any, temperature: float = 0.1) -> Any:
    try:
        data = _safe_json(ask_small_llm(prompt, temperature=temperature), default)
        return data if data is not None else default
    except Exception:
        return default


def _compact_json(data: Any, max_chars: int = 4500) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)[:max_chars]


def _safe_score(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _embedding_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    try:
        model = EmbeddingModel.get_instance()
        left_vec = np.array(model.embed_query(left), dtype=float)
        right_vec = np.array(model.embed_query(right), dtype=float)
        denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
        return float(np.dot(left_vec, right_vec) / denom) if denom else 0.0
    except Exception:
        return 0.0


def _lexical_overlap_score(query: str, text: str) -> float:
    query_terms = {t.lower().strip(".,;:!?()[]{}") for t in query.split() if len(t) > 2}
    text_terms = {t.lower().strip(".,;:!?()[]{}") for t in text.split() if len(t) > 2}
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


def _detect_language(text: str) -> str:
    """Detection legere FR/EN basee sur des marqueurs frequents (0 LLM)."""
    if not text:
        return "fr"
    t = text.lower()
    fr_markers = [" le ", " la ", " les ", " un ", " une ", " des ", " du ", " est ",
                  "pourquoi", "comment", "qu'est", "c'est quoi", "mon ", "ma ", "mes ",
                  " et ", " ou ", "campagne", "publicit"]
    en_markers = [" the ", " is ", " are ", " what ", " why ", " how ", " my ", " our ",
                  " and ", "campaign", " ads ", "should"]
    fr_score = sum(1 for m in fr_markers if m in f" {t} ")
    en_score = sum(1 for m in en_markers if m in f" {t} ")
    return "en" if en_score > fr_score else "fr"


# ============================================================
# RESOLUTION D'ANAPHORE (etape 1 du flux Learning / Decision)
# ============================================================
# Probleme resolu :
#   Quand l'utilisateur ecrit "Et comment l'ameliorer ?" juste apres avoir
#   parle du CPA, le modele de generation principale repondait parfois sur
#   le CTR ou un autre sujet recemment vu. Cette etape reformule la question
#   en autonome (resolution des pronoms / sujets implicites) AVANT le RAG
#   et avant la generation de la reponse mentor.
#
# Heuristique rapide :
#   Si la question contient deja un sujet explicite (CPA, ROAS, CTR, etc.)
#   ou est suffisamment longue (> 6 mots distinctifs), on saute le LLM
#   et on renvoie la question telle quelle. Economie de tokens.
# ============================================================

# Mots indiquant un suivi conversationnel court (probable anaphore)
_FOLLOWUP_MARKERS = [
    "et comment", "et pourquoi", "et quoi", "et la difference",
    "et le ", "et la ", "et les ", "et pour ", "et si ",
    "comment l'ameliorer", "comment l'optimiser", "comment le calculer",
    "et celui", "et celle", "donne un exemple", "donne-moi un exemple",
    "explique encore", "developpe", "approfondis", "et apres",
    "and how", "and why", "and what about", "what about",
    "tell me more", "explain further", "and the ",
]

# KPIs / concepts marketing explicites : si presents, pas besoin de resoudre
_EXPLICIT_TOPICS = [
    "roas", "roi", "cpa", "cpc", "ctr", "cpm", "cpl", "ltv", "aov",
    "conversion rate", "taux de conversion", "taux de clic",
    "google ads", "meta ads", "facebook ads", "tiktok ads",
    "linkedin ads", "audience", "creatives", "creative", "ciblage",
    "budget", "bidding", "enchere", "retargeting", "lookalike",
    "funnel", "tunnel", "landing page", "attribution",
]


def _needs_anaphora_resolution(question: str) -> bool:
    """
    Decide si la question doit passer par l'etape de resolution.
    Retourne True si la question est probablement un follow-up court
    sans sujet explicite.
    """
    q = question.strip().lower()
    word_count = len(q.split())

    # Question tres courte (<= 6 mots) -> probable suivi
    if word_count <= 6:
        return True

    # Contient un marqueur de suivi
    if any(m in q for m in _FOLLOWUP_MARKERS):
        # Mais si le sujet est deja explicite dans la phrase, pas besoin
        if any(t in q for t in _EXPLICIT_TOPICS):
            return False
        return True

    return False


def resolve_question_context(question: str, memory: ConversationMemory) -> dict[str, Any]:
    """
    Etape 1 du flux : resoud les anaphores et type la question.
    Appel small LLM uniquement si necessaire (heuristique en amont).

    Retourne :
      {
        "resolved_question": str,    # question reformulee en autonome
        "is_follow_up": bool,        # True si suivi conversationnel
        "anaphora_target": str,      # sujet detecte ou ""
        "question_type": str,        # definition / mechanism / strategy /
                                     # troubleshooting / comparison / coaching / followup
        "skipped_llm": bool,         # True si l'etape LLM a ete sautee
      }
    """
    default = {
        "resolved_question": question,
        "is_follow_up": False,
        "anaphora_target": "",
        "question_type": "general",
        "skipped_llm": True,
    }

    # Si pas d'historique, rien a resoudre
    if not memory.history:
        return default

    # Heuristique : eviter l'appel LLM si question deja autonome
    if not _needs_anaphora_resolution(question):
        # On profite quand meme de cette branche pour identifier un sujet
        # explicite dans la question, utile pour eviter les repetitions.
        q_lower = question.lower()
        target = ""
        for topic in _EXPLICIT_TOPICS:
            if topic in q_lower:
                target = topic.upper() if len(topic) <= 5 else topic
                break
        return {
            "resolved_question": question,
            "is_follow_up": False,
            "anaphora_target": target,
            "question_type": "general",
            "skipped_llm": True,
        }

    # Construire un contexte minimal pour la resolution
    last_topic = memory.current_topic or ""
    recent = memory.recent_history_text(180)
    mastered = ", ".join(memory.mastered_concepts[-5:]) if memory.mastered_concepts else ""

    prompt = f"""Tu es un module de comprehension contextuelle pour un assistant marketing.

Ta seule tache : reformuler la question de l'utilisateur en une question AUTONOME et COMPLETE,
en resolvant les pronoms et les references implicites a partir de l'historique.

Sujet de la conversation precedente : {last_topic if last_topic else "(non defini)"}
Concepts deja abordes : {mastered if mastered else "(aucun)"}

Historique recent :
{recent if recent else "(premiere question)"}

Question actuelle de l'utilisateur :
{question}

Reponds UNIQUEMENT avec un objet JSON sur une seule ligne, sans markdown, sans backticks :
{{"resolved_question":"la question reformulee en autonome, comprehensible sans contexte","is_follow_up":true_ou_false,"anaphora_target":"le sujet resolu (ex: CPA, ROAS, ciblage Meta, etc.) ou chaine vide","question_type":"definition|mechanism|strategy|troubleshooting|comparison|coaching|followup|general"}}

Regles strictes :
- Si la question est deja autonome, repete-la presque telle quelle dans resolved_question.
- Si l'utilisateur dit "et comment l'ameliorer" apres avoir parle du CPA, resolved_question doit etre "Comment ameliorer le CPA d'une campagne ?".
- N'invente pas de sujet : si rien dans l'historique ne permet de resoudre, mets anaphora_target vide et resolved_question = question originale.
- Question pure de definition (ex "c'est quoi le ROAS") -> question_type = "definition", is_follow_up = false.
- Question de suivi sur un concept deja explique -> question_type = "followup".
"""

    try:
        raw = ask_small_llm(prompt, temperature=0.0).strip()
        # Nettoyage backticks
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Extraction premier objet JSON
        candidate = ""
        for line in raw.splitlines():
            s = line.strip()
            if s.startswith("{"):
                candidate = s
                break
        if not candidate or not candidate.endswith("}"):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                candidate = match.group(0)
        if not candidate:
            return default

        data = json.loads(candidate)
        if not isinstance(data, dict):
            return default

        resolved = str(data.get("resolved_question") or question).strip()
        # Garde-fou : si la resolution donne une question vide ou trop courte,
        # fallback sur la question originale.
        if len(resolved) < 4:
            resolved = question

        return {
            "resolved_question": resolved,
            "is_follow_up": bool(data.get("is_follow_up", False)),
            "anaphora_target": str(data.get("anaphora_target") or "").strip()[:80],
            "question_type": str(data.get("question_type") or "general").strip().lower()[:30],
            "skipped_llm": False,
        }
    except Exception as e:
        logger.warning(f"resolve_question_context failed, fallback: {e}")
        return default


def _semantic_retrieval_queries(question: str, semantic_intent: dict[str, Any]) -> list[str]:
    queries = [question]
    for key in ("retrieval_needs", "concepts", "conceptual_gaps"):
        values = semantic_intent.get(key, [])
        if isinstance(values, str):
            values = [values]
        for value in values:
            text = str(value).strip()
            if text:
                queries.append(f"{question} {text}")
    business_context = semantic_intent.get("business_context", {})
    if isinstance(business_context, dict):
        context_terms = " ".join(str(v) for v in business_context.values() if v)
        if context_terms:
            queries.append(f"{question} {context_terms}")
    unique = []
    for query in queries:
        if query not in unique:
            unique.append(query)
    return unique[:6]


def contextual_semantic_retrieval(question: str, semantic_intent: dict[str, Any], memory: ConversationMemory, k: int = 4) -> dict[str, Any]:
    domain_db_map = get_domain_db_map()
    queries = _semantic_retrieval_queries(question, semantic_intent)
    candidates = []
    seen = set()
    cluster_scores = {name: 0.0 for name in domain_db_map}
    for cluster_name, db in domain_db_map.items():
        for query_index, query in enumerate(queries):
            try:
                results = db.similarity_search_with_score(query, k=max(3, min(k, 6)))
            except Exception:
                results = []
            for rank, (doc, distance) in enumerate(results):
                fingerprint = doc.page_content[:500]
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                semantic_score = 1 / (1 + float(distance))
                lexical_score = _lexical_overlap_score(query, doc.page_content[:1200])
                continuity_score = max(
                    _embedding_similarity(doc.page_content[:700], memory.current_topic),
                    _embedding_similarity(doc.page_content[:700], memory.summary),
                    _embedding_similarity(doc.page_content[:700], memory.last_standalone_query),
                )
                final_score = semantic_score + 0.08 * lexical_score + 0.06 * continuity_score + 0.04 / (rank + 1) + 0.02 / (query_index + 1)
                candidates.append({"doc": doc, "cluster": cluster_name, "score": final_score})
                cluster_scores[cluster_name] = max(cluster_scores[cluster_name], final_score)
    selected = sorted(candidates, key=lambda item: item["score"], reverse=True)[:8]
    docs = [item["doc"] for item in selected]
    context = "\n\n---\n\n".join(f"[Knowledge excerpt {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs))
    sources, clusters = [], []
    for item in selected:
        if item["cluster"] not in clusters:
            clusters.append(item["cluster"])
    for doc in docs:
        source = str(doc.metadata.get("source", "unknown"))
        if source not in sources:
            sources.append(source)
    return {
        "docs": docs,
        "context": context,
        "sources": sources,
        "domain": clusters or ["semantic_llm"],
        "scores": {name: round(score, 4) for name, score in cluster_scores.items()},
    }


# ============================================================
# MODE LEARNING - MENTOR MARKETING PEDAGOGIQUE PREMIUM v2
# ============================================================
# Architecture preservee : RAG semantique en amont, memoire conversationnelle,
# profiling adaptatif, sortie unifiee avec bloc de memoire pour merge_update().
#
# Fixes v2 (resout les bugs observes en test reel) :
#   1) Labels "BLOC 1 / BLOC 2" qui apparaissaient dans la sortie
#      -> remplacement par marqueurs techniques ===ANSWER=== / ===MEMORY===
#         + post-cleaning defensif dans le parser
#   2) Perte de reference ("Et comment l'ameliorer ?" -> mauvais sujet)
#      -> question_context resolu en AMONT par resolve_question_context()
#         et injecte ici sous forme de "question reformulee"
#   3) Repetitions sur concepts deja maitrises (CPA explique 3 fois)
#      -> regle dure : si mastered_concepts contient le concept de la question,
#         varier l'angle (cas d'usage, edge case, lien metier) au lieu de redefinir
#   4) Style scolaire ("C'est une mesure importante car...")
#      -> anti-patterns explicites + 2 exemples de TON mentor (pas du contenu)
#
# Toujours 1 appel small LLM ici (l'etape 1 d'anaphore est un appel separe).
# ============================================================


# Exemples de ton mentor (style uniquement, pas contenu) - injectes dans le prompt
# pour debloquer le modele 8B d'un registre purement scolaire.
_MENTOR_STYLE_EXAMPLES = """Exemple de ton mentor (style uniquement, ne pas copier le contenu) :

Ex A - Question definition :
"Le ROAS, c'est juste le rapport entre ce que tu encaisses et ce que tu depenses en pub. Si tu mets 100 euros et que ca te ramene 300 euros de CA, ton ROAS est de 3. La vraie question derriere ce chiffre c'est : a partir de quel ROAS ta campagne devient rentable une fois la marge enlevee ? Parce que c'est la qu'on bascule du vanity au business."

Ex B - Question d'optimisation :
"Pour faire baisser un CPA, il y a trois leviers et un seul d'entre eux marche vraiment selon le contexte. Le premier c'est le creative, qui agit sur le CTR. Le deuxieme c'est la landing, qui agit sur le taux de conversion. Le troisieme c'est l'audience. Avant de tout changer, regarde ou ton funnel decroche : si ton CTR est correct mais que les conversions ne suivent pas, c'est rarement le creative qui est en cause."
"""


def build_unified_learning_prompt(
    question: str,
    retrieved: dict[str, Any],
    memory: ConversationMemory,
    question_context: dict[str, Any] | None = None,
) -> str:
    """
    Prompt unifie premium v2 : mentor marketing senior avec anti-bugs.
    1 appel small LLM, raisonnement pedagogique interne avant la reponse.

    question_context vient de resolve_question_context() :
      - resolved_question : question reformulee en autonome
      - is_follow_up      : True si suivi conversationnel
      - anaphora_target   : sujet detecte
      - question_type     : definition / mechanism / strategy / troubleshooting /
                            comparison / coaching / followup / general
    """
    memory_block = memory.compact_snapshot()
    raw_rag = retrieved.get("context") or ""
    rag_context = (raw_rag[:1800] + "\n[...]") if len(raw_rag) > 1800 else (raw_rag or "(aucun extrait RAG disponible pour cette question)")
    user_lang = _detect_language(question)
    lang_directive = (
        "Reponds en francais naturel et professionnel."
        if user_lang == "fr"
        else "Answer in clear, professional English matching the user's language."
    )

    ctx = question_context or {}
    resolved_q = ctx.get("resolved_question") or question
    is_follow_up = bool(ctx.get("is_follow_up"))
    anaphora_target = str(ctx.get("anaphora_target") or "").strip()
    question_type = str(ctx.get("question_type") or "general").strip()

    # Bloc contextuel : informe le modele de ce que l'utilisateur veut vraiment
    context_block_parts = [f"Question telle qu'ecrite par l'utilisateur : {question}"]
    if resolved_q.strip().lower() != question.strip().lower():
        context_block_parts.append(f"Question reformulee en autonome (a traiter) : {resolved_q}")
    if anaphora_target:
        context_block_parts.append(f"Sujet resolu : {anaphora_target}")
    if is_follow_up:
        context_block_parts.append("Type d'interaction : SUIVI conversationnel (l'utilisateur prolonge le sujet precedent)")
    context_block_parts.append(f"Categorie de question : {question_type}")
    context_block = "\n".join(context_block_parts)

    # Bloc anti-repetition : si le sujet de la question est dans mastered_concepts,
    # on l'indique explicitement.
    mastered = memory.mastered_concepts[-8:] if memory.mastered_concepts else []
    target_lower = (anaphora_target or resolved_q).lower()
    already_known = [c for c in mastered if c and c.lower() in target_lower]
    anti_repeat_block = ""
    if already_known:
        anti_repeat_block = (
            f"\nIMPORTANT - Concepts deja maitrises par l'utilisateur : {', '.join(already_known)}.\n"
            "Ne redefinis PAS ces concepts. Traite-les comme des acquis. Apporte plutot un angle nouveau :\n"
            "  - un cas d'usage sectoriel (ecommerce, SaaS, lead gen)\n"
            "  - un piege courant ou un edge case\n"
            "  - le lien avec un autre KPI ou levier marketing\n"
            "  - une regle pratique de media buyer\n"
        )

    return f"""Tu es le Learning Mode d'AdOptimizer AI.
Tu es un MENTOR marketing digital senior. Pas un chatbot generique, pas un manuel.
Tu enseignes comme un coach qui a deja gere des centaines de campagnes Meta Ads, Google Ads,
TikTok Ads et LinkedIn Ads, et qui sait expliquer simplement sans jamais etre condescendant.

================= POSTURE ET PRINCIPES =================
- Tu n'es PAS un dictionnaire. Tu enseignes en racontant comment ca marche reellement.
- Tu adaptes la profondeur a la maturite detectee de l'utilisateur (debutant / intermediaire / avance).
- Tu evites TOUTE repetition : si un concept est deja maitrise, tu t'appuies dessus, tu ne le redefinis pas.
- Tu utilises des exemples concrets et plausibles quand ca aide vraiment (pas systematiquement).
- Tu fais des liens explicites entre concepts (ex: CPA depend du CTR, du CR et du CPC).
- Tu raisonnes business : trade-offs, signaux faibles, ce qu'un bon media buyer regarderait.
- Tu acceptes l'incertitude ("ca depend de ton secteur, de ton funnel, etc.") quand c'est honnete.
- Tu fluidifies la conversation : si l'utilisateur revient sur un point, tu y reponds directement.

================= STYLE MENTOR (NE PAS COPIER LE CONTENU, S'INSPIRER DU TON) =================
{_MENTOR_STYLE_EXAMPLES}

================= ANTI-PATTERNS A EVITER ABSOLUMENT =================
Ces formulations sonnent scolaires et robotiques. INTERDITES :
- "C'est une mesure importante car elle permet de..."
- "Cela vous permet de savoir si..."
- "Il est important de noter que..."
- "En resume, X evalue... tandis que Y evalue..."
- "Pour calculer X, vous divisez A par B" (sauf si la question demande EXPRESSEMENT la formule)
- Toute structure du type "definition + formule + exemple + conclusion" : c'est un manuel, pas un mentor.

================= RAISONNEMENT INTERNE (NE PAS AFFICHER) =================
1. Quelle est REELLEMENT la question (utiliser la "Question reformulee" si fournie) ?
2. Type de question et profondeur attendue ?
3. Que sait deja l'utilisateur ? Que ne pas redire ?
4. Quel angle apporte le plus de valeur ICI ?
5. Quelle ouverture mentor naturelle prolonge l'apprentissage sur le SUJET EXACT ?

================= HORS-SUJET =================
Si la question n'a aucun lien avec le marketing, la publicite digitale, la croissance ou la strategie
business, reponds brievement : tu es specialise en marketing digital. Pas de longue explication.

================= UTILISATION DU RAG =================
- Les extraits RAG sont une SOURCE DE VERIFICATION, pas un texte a recopier.
- Si pertinents : appuie-toi dessus, reformule avec tes mots.
- Si partiels : complete avec les meilleures pratiques du metier.
- Ne cite JAMAIS "selon les extraits" ou "d'apres le RAG" : reste fluide.

================= REGLES DE STYLE =================
- Ton : chaleureux, expert, direct. Tu peux dire "ok regarde", "le truc a comprendre",
  "en pratique", mais sans abus.
- Phrases courtes, paragraphes courts (2-4 phrases max par paragraphe).
- Aucun bullet point ni liste numerotee, SAUF si la question demande explicitement plusieurs leviers
  d'action distincts.
- Termine par UNE ouverture mentor naturelle et specifique au sujet exact discute.
- {lang_directive}

================= CONTEXTE DE LA QUESTION =================
{context_block}
{anti_repeat_block}
================= MEMOIRE CONVERSATIONNELLE =================
{memory_block if memory_block else "(premiere question, aucun contexte anterieur)"}

================= EXTRAITS RAG =================
{rag_context}

================= FORMAT DE SORTIE STRICT =================
Tu produis DEUX sections separees par un marqueur technique.

CRITIQUE : ta reponse commence DIRECTEMENT par la premiere phrase mentor.
N'ecris JAMAIS de label, titre, ou prefixe au debut. Pas de "BLOC 1", pas de "Reponse :", pas de
"Voici ma reponse", pas de "En tant que mentor...". Tu commences directement par le contenu.

Apres ta reponse pedagogique, sur une nouvelle ligne, ecris exactement :
===MEMORY===
Puis un objet JSON valide sur UNE SEULE LIGNE (pas de markdown, pas de backticks) :
{{"summary":"resume conversationnel en 1 phrase","current_topic":"sujet principal detecte","learning_trajectory":["intitule court de la question"],"mastered_concepts":["concept manifestement maitrise"],"confusion_areas":["zone de flou si pertinent"],"user_goals":[],"profile":{{"marketing_maturity":"descripteur (debutant curieux / praticien intermediaire / media buyer avance)","maturity_score":0.0}},"suggestions":["question de suivi specifique au sujet exact","autre question de suivi specifique"]}}

Regles JSON :
- Sur UNE SEULE LIGNE.
- Les suggestions sont des prolongations REELLES, specifiques au sujet du tour (jamais generiques).
- maturity_score entre 0.0 et 1.0.
"""


def _clean_answer_leakage(answer: str) -> str:
    """
    Nettoyage defensif : supprime les labels de prompt que le modele 8B
    pourrait recopier en debut de reponse (observe sur llama-3.1-8b-instant).
    Robuste a plusieurs variantes.
    """
    if not answer:
        return answer

    cleaned = answer.strip()

    # Liste de prefixes parasites (insensibles a la casse, regex)
    leakage_patterns = [
        r"^\s*bloc\s*1\s*[-:]*\s*(?:ta reponse pedagogique(?: mentor-like)?\.?)?\s*",
        r"^\s*bloc\s*2\s*[-:]*\s*.*$",  # ligne entiere si presente
        r"^\s*===\s*answer\s*===\s*",
        r"^\s*===\s*memory\s*===\s*",
        r"^\s*reponse\s*[-:]\s*",
        r"^\s*voici (?:ma|la) reponse\s*[-:]?\s*",
        r"^\s*en tant que mentor[^\n.]*[.,]?\s*",
        r"^\s*ta reponse pedagogique(?: mentor-like)?\.?\s*",
    ]

    for pattern in leakage_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Supprimer "BLOC 2" et tout ce qui suit en fin de chaine (si le marqueur memoire a echoue)
    cleaned = re.sub(r"\n+\s*bloc\s*2\s*[-:]*.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    return cleaned.strip()


def _parse_unified_response(raw: str) -> tuple[str, dict[str, Any], list[str]]:
    """
    Parse la reponse unifiee du LLM en (answer, memory_update_dict, suggestions_list).
    Robuste a :
      - marqueur ===MEMORY=== (v2) ou %%MEMORY_UPDATE%% (v1, backward compat)
      - JSON multi-lignes ou avec backticks
      - labels parasites en debut de reponse (BLOC 1, etc.)
    """
    # Detecter le marqueur de memoire (v2 ou v1)
    marker = None
    for candidate_marker in ("===MEMORY===", "%%MEMORY_UPDATE%%"):
        if candidate_marker in raw:
            marker = candidate_marker
            break

    if marker is None:
        return _clean_answer_leakage(raw), {}, []

    parts = raw.split(marker, 1)
    answer = _clean_answer_leakage(parts[0])
    json_block = parts[1].strip() if len(parts) > 1 else ""

    # Nettoyage backticks
    json_block = json_block.replace("```json", "").replace("```", "").strip()

    # Strategie 1 : premiere ligne commencant par {
    candidate = ""
    for line in json_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            candidate = stripped
            break

    # Strategie 2 : si multi-ligne, regex sur premier objet {...}
    if not candidate or not candidate.endswith("}"):
        match = re.search(r"\{.*\}", json_block, re.DOTALL)
        if match:
            candidate = match.group(0)

    memory_update: dict[str, Any] = {}
    suggestions: list[str] = []
    if candidate:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                memory_update = {k: v for k, v in data.items() if k != "suggestions"}
                raw_suggestions = data.get("suggestions", [])
                if isinstance(raw_suggestions, list):
                    suggestions = [str(s).strip()[:160] for s in raw_suggestions if str(s).strip()][:3]
        except Exception:
            pass

    return answer, memory_update, suggestions


def ask_agent(question: str, k: int = 4, verbose: bool = False) -> dict:
    """
    Point d'entree Learning Mode - mentor pedagogique premium v2.

    Flux (architecture preservee + fix anaphore) :
      1. resolve_question_context()        -> 1 small LLM si necessaire (anaphore)
      2. contextual_semantic_retrieval()   -> 0 LLM (embeddings + scoring)
         (utilise la resolved_question pour un meilleur retrieval)
      3. build_unified_learning_prompt()
         + ask_small_llm()                 -> 1 small LLM (mentor + memoire)
      4. _parse_unified_response()         -> 0 LLM (parsing local)
      5. memory.merge_update()             -> 0 LLM (mise a jour locale)
    """
    k = min(k, 8)

    # Etape 1 : resolution d'anaphore (utilise small LLM uniquement si necessaire)
    question_context = resolve_question_context(question, conversation_memory)
    resolved_q = question_context.get("resolved_question") or question

    if verbose:
        print("Question context:", {
            "resolved": resolved_q,
            "is_follow_up": question_context.get("is_follow_up"),
            "target": question_context.get("anaphora_target"),
            "type": question_context.get("question_type"),
            "skipped_llm": question_context.get("skipped_llm"),
        })

    semantic_intent_for_retrieval: dict[str, Any] = {}

    # Etape 2 : RAG sur la question resolue pour un meilleur scoring
    retrieved = contextual_semantic_retrieval(
        resolved_q, semantic_intent_for_retrieval, conversation_memory, k=k
    )

    if verbose:
        print("Semantic knowledge clusters:", retrieved["domain"])
        print("Semantic retrieval scores:", retrieved["scores"])

    # Etape 3 : prompt mentor v2 avec contexte de question + anti-repetition
    prompt = build_unified_learning_prompt(
        question, retrieved, conversation_memory, question_context=question_context
    )
    # Temperature mentor : chaleureux sans deriver.
    raw_answer = ask_small_llm(prompt, temperature=0.55).strip()

    answer, memory_update, suggestions = _parse_unified_response(raw_answer)

    if memory_update:
        memory_update.setdefault("current_topic", (question_context.get("anaphora_target") or resolved_q)[:80])
        conversation_memory.merge_update(memory_update)
    else:
        conversation_memory.current_topic = (question_context.get("anaphora_target") or resolved_q)[:80]
    conversation_memory.last_standalone_query = resolved_q

    learner_profile = memory_update.get("profile") or conversation_memory.profile
    current_topic = memory_update.get("current_topic") or conversation_memory.current_topic

    return {
        "answer": answer,
        "domain": retrieved["domain"],
        "sources": retrieved["sources"],
        "mode": "rag_semantic" if retrieved["docs"] else "llm_semantic",
        "intent": memory_update.get("current_topic", "semantic learning request"),
        "intent_confidence": _safe_score(
            memory_update.get("profile", {}).get("maturity_score") if isinstance(memory_update.get("profile"), dict) else None,
            0.75
        ),
        "user_level": (
            memory_update.get("profile", {}).get("marketing_maturity", "adaptive learner")
            if isinstance(memory_update.get("profile"), dict)
            else conversation_memory.profile.get("marketing_maturity", "adaptive learner")
        ),
        "topic": current_topic,
        "suggestions": suggestions,
        "semantic_query": resolved_q,
        "retrieval_scores": retrieved["scores"],
        "profile": learner_profile,
        # Debug context exposes resolution details (utile pour FastAPI + Angular)
        "question_context": {
            "resolved_question": resolved_q,
            "is_follow_up": question_context.get("is_follow_up", False),
            "anaphora_target": question_context.get("anaphora_target", ""),
            "question_type": question_context.get("question_type", "general"),
        },
    }


# Backward compat : ancien nom interne conserve.
def _deprecated_static_rag_agent(question: str, k: int = 4, verbose: bool = False) -> dict:
    return ask_agent(question, k=k, verbose=verbose)


# ============================================================
# MODE ANALYSIS - CONSULTANT MARKETING SENIOR XAI
# ============================================================
# Architecture preservee :
#   - xai_explanations.json (campagne active uniquement)
#   - health score / causal / optimizer / anomalies
#   - 1 appel big LLM (70B) pour le raisonnement decisionnel
#
# Upgrade premium :
#   - Persona consultant senior (pas template numerote)
#   - Detection dynamique du TYPE de question decisionnelle :
#       diagnostic (pourquoi), justification (pourquoi cette reco),
#       trade-off (risques), planification (next steps), comparaison,
#       interpretation KPI, suivi conversationnel
#   - Memoire conversationnelle injectee -> follow-ups naturels
#       ("et les risques ?", "et pour Meta ?", "explique encore le CPA")
#   - Detection de langue (FR par defaut, mirror EN si l'utilisateur ecrit EN)
#   - Sortie unifiee : reponse + bloc %%DECISION_UPDATE%% pour memoire
# ============================================================


def load_xai() -> list:
    """Charge les explications XAI depuis le fichier JSON."""
    if not XAI_PATH.exists():
        logger.warning(f"XAI file introuvable : {XAI_PATH}")
        return []
    with open(XAI_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("xai_explanations", [])


def _format_xai_for_consultant(xai_data: list) -> tuple[str, dict[str, Any]]:
    """
    Reformate la campagne active en un brief structure pour le consultant.
    Retourne (brief_text, raw_active_payload) pour reutilisation downstream.
    """
    if not xai_data:
        return "", {}

    active = xai_data[0] if isinstance(xai_data, list) else xai_data
    if not isinstance(active, dict):
        return "", {}

    campaign_id = active.get("campaign_id", "N/A")
    platform = active.get("platform", "N/A")
    summary = active.get("xai_summary", "")
    health = active.get("health_explanation", {}) or {}
    causal = active.get("causal_explanation", {}) or {}
    optimizer = active.get("optimizer_explanation", {}) or {}
    anomalies = active.get("top_anomalies") or active.get("anomalies") or []
    current_kpis = active.get("current_kpis", {}) or {}
    predicted_kpis = active.get("predicted_kpis", {}) or {}
    expected_impact = active.get("expected_impact", {}) or optimizer.get("expected_impact", {}) or {}

    def _fmt_kv(d: dict, limit: int = 8) -> str:
        if not isinstance(d, dict) or not d:
            return "(non disponible)"
        items = list(d.items())[:limit]
        return ", ".join(f"{k}={v}" for k, v in items)

    anomaly_text = ""
    if isinstance(anomalies, list) and anomalies:
        rows = []
        for a in anomalies[:5]:
            if isinstance(a, dict):
                rows.append(
                    f"- {a.get('kpi', a.get('metric', 'kpi'))} : "
                    f"{a.get('severity', a.get('level', ''))} | "
                    f"{a.get('description', a.get('explanation', ''))}"
                )
            else:
                rows.append(f"- {a}")
        anomaly_text = "\n".join(rows)
    else:
        anomaly_text = "(aucune anomalie significative remontee)"

    brief = f"""CAMPAGNE ACTIVE
  id        : {campaign_id}
  plateforme: {platform}
  resume XAI: {summary or "(non fourni)"}

SANTE CAMPAGNE
  health_score   : {health.get("health_score", "N/A")}
  statut         : {health.get("status", "N/A")}
  raisons        : {", ".join(health.get("main_reasons", [])) or "(non fourni)"}

KPIs ACTUELS
  {_fmt_kv(current_kpis)}

KPIs PREDITS
  {_fmt_kv(predicted_kpis)}

ANALYSE CAUSALE
  resume         : {causal.get("summary", "(non fourni)")}
  drivers cles   : {", ".join(causal.get("key_drivers", causal.get("drivers", []))) or "(non fourni)"}

PLAN OPTIMIZER (RL)
  action recommandee : {optimizer.get("recommended_action", optimizer.get("action", "(non fourni)"))}
  resume             : {optimizer.get("summary", "(non fourni)")}
  impact attendu     : {_fmt_kv(expected_impact)}

ANOMALIES PRIORITAIRES
{anomaly_text}
"""

    return brief, active


def build_decision_prompt(
    question: str,
    brief: str,
    raw_active: dict[str, Any],
    memory: ConversationMemory,
    question_context: dict[str, Any] | None = None,
) -> str:
    """
    Prompt consultant senior dynamique v2. Pas de template visible.
    Le modele raisonne en interne puis repond comme un vrai consultant.
    Accepte question_context pour resolution d'anaphore ("et les risques ?").
    """
    memory_block = memory.decision_snapshot()
    user_lang = _detect_language(question)
    lang_directive = (
        "Reponds en francais naturel et professionnel."
        if user_lang == "fr"
        else "Answer in clear, professional English matching the user's language."
    )

    ctx = question_context or {}
    resolved_q = ctx.get("resolved_question") or question
    is_follow_up = bool(ctx.get("is_follow_up"))
    anaphora_target = str(ctx.get("anaphora_target") or "").strip()

    context_block_parts = [f"Question telle qu'ecrite : {question}"]
    if resolved_q.strip().lower() != question.strip().lower():
        context_block_parts.append(f"Question reformulee en autonome (a traiter) : {resolved_q}")
    if anaphora_target:
        context_block_parts.append(f"Sujet resolu : {anaphora_target}")
    if is_follow_up:
        context_block_parts.append("Type d'interaction : SUIVI de la conversation precedente")
    context_block = "\n".join(context_block_parts)

    return f"""Tu es le Decision Mode d'AdOptimizer AI.
Tu joues le role d'un CONSULTANT MARKETING SENIOR specialise en performance ads.
Tu as deja conseille des marques DTC, SaaS B2B et e-commerce sur Meta Ads et Google Ads.
Tu lis les outputs de notre stack interne (health score, causal AI, RL optimizer, anomaly detection)
et tu les traduis en insights business clairs pour un decideur.

================= POSTURE =================
- Tu ne recites pas les chiffres bruts : tu les INTERPRETES.
- Tu expliques le POURQUOI business derriere les recommandations IA.
- Tu mentionnes les trade-offs et risques quand c'est honnete (ex: scaler ROAS peut tuer le volume).
- Tu n'inventes JAMAIS de chiffres absents du brief. Si une donnee manque, tu le dis franchement.
- Tu evites le jargon technique interne (XAI, RL, causal model, pipeline, JSON, backend, ranking).
- Tu parles le langage du media buyer / growth manager / CMO.
- Tu adaptes le niveau de detail au TYPE de question posee.
- Tu maintiens la continuite conversationnelle.

================= ANTI-PATTERNS A EVITER =================
Pas de structure rigide visible (interdit : "1. Diagnostic 2. Causes 3. Action").
Pas de label en debut de reponse (interdit : "BLOC 1", "Reponse :", "En tant que consultant...").
Pas de jargon interne. Pas de chiffres inventes.

================= RAISONNEMENT INTERNE (NE PAS AFFICHER) =================
1. Quelle est REELLEMENT la question (utiliser "Question reformulee" si fournie) ?
2. Type de question ? (diagnostic / justification de reco / trade-off / next steps /
   interpretation KPI / comparaison / suivi)
3. Quels elements du brief sont REELLEMENT utiles ici ?
4. Que sait deja l'utilisateur ? Que ne pas repeter ?
5. La reco IA est-elle confirmee par les chiffres ? Justifie ou nuance.
6. Vrais risques si l'utilisateur applique l'action ?
7. Prochaine etape concrete ?

================= STYLE =================
- Reponse compacte mais dense : 100 a 180 mots selon la profondeur.
- Prose fluide. Liste courte autorisee UNIQUEMENT si la question demande explicitement plusieurs
  leviers distincts ("que faire ?", "quelles options ?").
- Termine par UNE ouverture pertinente et specifique.
- Si recommandation = maintain_budget : explique que c'est de la PRUDENCE, pas un signal positif.
- Si expected_impact.delta_conversions negatif : signale explicitement le trade-off ROAS vs volume.
- Si expected_roas < 1 : dis franchement que la rentabilite reste fragile.
- Si donnee manque dans le brief : dis-le, n'invente pas.
- {lang_directive}

================= CONTEXTE DE LA QUESTION =================
{context_block}

================= MEMOIRE CONVERSATIONNELLE =================
{memory_block if memory_block else "(premiere question sur cette campagne)"}

================= BRIEF CAMPAGNE (donnees XAI) =================
{brief if brief else "(aucune donnee de campagne active disponible)"}

================= FORMAT DE SORTIE STRICT =================
CRITIQUE : ta reponse commence DIRECTEMENT par la premiere phrase de consultant.
N'ecris JAMAIS de label, titre, ou prefixe au debut.

Apres ta reponse, sur une nouvelle ligne, ecris exactement :
===DECISION===
Puis un objet JSON valide sur UNE SEULE LIGNE (pas de markdown, pas de backticks) :
{{"decision_focus":"focus business mis a jour en 1 ligne","discussed_kpis":["kpi mentionne"],"discussed_actions":["recommandation justifiee ou nuancee"],"summary":"resume de l'echange en 1 phrase","current_topic":"sujet decisionnel actuel","suggestions":["question de suivi concrete","autre question de suivi concrete"]}}

Regles JSON :
- UNE SEULE LIGNE.
- Suggestions = prolongations naturelles, jamais generiques.
- Liste vide [] autorisee.
"""


def _parse_decision_response(raw: str) -> tuple[str, dict[str, Any], list[str]]:
    """
    Parse la reponse du Decision Mode.
    Supporte le marqueur v2 ===DECISION=== et le v1 %%DECISION_UPDATE%% (backward compat).
    Nettoie aussi les labels parasites en debut de reponse.
    """
    marker = None
    for candidate_marker in ("===DECISION===", "%%DECISION_UPDATE%%"):
        if candidate_marker in raw:
            marker = candidate_marker
            break

    if marker is None:
        return _clean_answer_leakage(raw), {}, []

    parts = raw.split(marker, 1)
    answer = _clean_answer_leakage(parts[0])
    json_block = parts[1].strip() if len(parts) > 1 else ""
    json_block = json_block.replace("```json", "").replace("```", "").strip()

    candidate = ""
    for line in json_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            candidate = stripped
            break
    if not candidate or not candidate.endswith("}"):
        match = re.search(r"\{.*\}", json_block, re.DOTALL)
        if match:
            candidate = match.group(0)

    memory_update: dict[str, Any] = {}
    suggestions: list[str] = []
    if candidate:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                memory_update = {k: v for k, v in data.items() if k != "suggestions"}
                raw_suggestions = data.get("suggestions", [])
                if isinstance(raw_suggestions, list):
                    suggestions = [str(s).strip()[:160] for s in raw_suggestions if str(s).strip()][:3]
        except Exception:
            pass

    return answer, memory_update, suggestions


def explain_with_xai(question: str) -> str:
    """
    Mode Analyse premium - consultant senior conversationnel.
    Architecture preservee : utilise xai_data[0] (campagne active) + big LLM 70B.
    Ajout : memoire conversationnelle pour les follow-ups + persona consultant.

    Note: La signature retourne str pour preserver la compatibilite. Le wrapper
    dans agent_marketing_global() recupere les suggestions et la memoire via
    explain_with_xai_full().
    """
    result = explain_with_xai_full(question)
    return result["answer"]


def explain_with_xai_full(question: str) -> dict:
    """
    Version complete du mode Analysis : retourne answer + memoire + suggestions.
    Utilise par agent_marketing_global() pour enrichir la sortie.
    Inclut la resolution d'anaphore pour les follow-ups ("et les risques ?").
    """
    xai_data = load_xai()

    if not xai_data:
        return {
            "answer": (
                "Aucune donnee XAI disponible pour le moment. "
                "Lance d'abord le pipeline d'analyse pour que je puisse t'expliquer "
                "ce qui se passe sur ta campagne active."
            ),
            "suggestions": [],
            "discussed_kpis": [],
            "discussed_actions": [],
            "question_context": {
                "resolved_question": question,
                "is_follow_up": False,
                "anaphora_target": "",
                "question_type": "general",
            },
        }

    brief, raw_active = _format_xai_for_consultant(xai_data)

    # Etape 1 : resolution d'anaphore (utile pour "et les risques ?", "et pour Meta ?")
    question_context = resolve_question_context(question, conversation_memory)
    resolved_q = question_context.get("resolved_question") or question

    try:
        prompt = build_decision_prompt(
            question, brief, raw_active, conversation_memory,
            question_context=question_context,
        )
        # Big LLM (70B) reste sur le Decision Mode : raisonnement business + nuance.
        raw = ask_big_llm(prompt, temperature=0.35).strip()
    except Exception as e:
        logger.error(f"LLM error in Decision Mode: {e}")
        if _is_rate_limit_error(e):
            return {
                "answer": _friendly_rate_limit_message(),
                "suggestions": [],
                "discussed_kpis": [],
                "discussed_actions": [],
                "question_context": question_context,
            }
        return {
            "answer": f"Erreur lors de l'analyse decisionnelle : {str(e)}",
            "suggestions": [],
            "discussed_kpis": [],
            "discussed_actions": [],
            "question_context": question_context,
        }

    answer, memory_update, suggestions = _parse_decision_response(raw)

    if memory_update:
        memory_update.setdefault("current_topic", (question_context.get("anaphora_target") or resolved_q)[:120])
        conversation_memory.merge_update(memory_update)
    else:
        conversation_memory.current_topic = (question_context.get("anaphora_target") or resolved_q)[:120]
    conversation_memory.last_standalone_query = resolved_q

    return {
        "answer": answer,
        "suggestions": suggestions,
        "discussed_kpis": memory_update.get("discussed_kpis", []),
        "discussed_actions": memory_update.get("discussed_actions", []),
        "decision_focus": memory_update.get("decision_focus", conversation_memory.decision_focus),
        "topic": memory_update.get("current_topic", conversation_memory.current_topic),
        "question_context": {
            "resolved_question": resolved_q,
            "is_follow_up": question_context.get("is_follow_up", False),
            "anaphora_target": question_context.get("anaphora_target", ""),
            "question_type": question_context.get("question_type", "general"),
        },
    }


# ============================================================
# INTENT DETECTION (fallback auto si mode non fourni)
# ============================================================

def detect_intent(user_input: str) -> str:
    """
    Detecte automatiquement le mode si non specifie.
    Retourne : learning | analysis
    Heuristique legere + fallback LLM, avec biais memoire.
    """
    text = user_input.lower()

    # Heuristique rapide : mots clefs decisionnels forts
    decision_markers = [
        "campagne", "campaign", "ma campagne", "mon roas", "mon cpa", "mon ctr",
        "pourquoi mon", "why is my", "anomalie", "anomaly", "health score",
        "performance", "scale", "scaler", "augmenter le budget", "baisser le budget",
        "recommandation", "recommendation", "optimizer", "active campaign",
    ]
    learning_markers = [
        "c'est quoi", "what is", "definition", "explique-moi", "explain me",
        "comment ca marche", "how does", "difference entre", "difference between",
        "apprendre", "learn", "comprendre",
    ]

    decision_score = sum(1 for m in decision_markers if m in text)
    learning_score = sum(1 for m in learning_markers if m in text)

    if decision_score > learning_score and decision_score >= 1:
        return "analysis"
    if learning_score > decision_score and learning_score >= 1:
        return "learning"

    # Biais memoire : si la conversation est deja decisionnelle, rester decisionnel
    if conversation_memory.decision_focus and not learning_markers:
        return "analysis"

    # Fallback LLM (rare)
    prompt = f"""Classify this user message into ONE category:

LEARNING  -> the user wants to learn marketing concepts, understand metrics, get definitions or explanations of how things work
ANALYSIS  -> the user asks about their actual campaign performance, anomalies, why a KPI changed, what action to take, justification of a recommendation

Answer ONLY: LEARNING or ANALYSIS

Message: {user_input}"""

    try:
        response = ask_llm(prompt).strip().upper()
        if "ANALYSIS" in response:
            return "analysis"
        return "learning"
    except Exception:
        return "learning"


# ============================================================
# AGENT PRINCIPAL - 2 MODES
# ============================================================

def agent_marketing_global(user_input: str, mode: str = "auto") -> dict:
    """
    Point d'entree principal de l'agent.

    Parametres :
      user_input : question de l'utilisateur
      mode       : "learning" | "analysis" | "auto"

    Retourne :
      { "answer": str, "mode": str, "status": "success" | "error" | "rate_limited", ... }
    """
    print(f"\nAdOptimizer AI")
    print(f"Question : {user_input}")

    if mode == "auto":
        mode = detect_intent(user_input)
        print(f"Mode auto-detecte : {mode}")
    else:
        print(f"Mode choisi : {mode}")

    try:
        # ------------------------------------------
        # MODE LEARNING - mentor pedagogique premium
        # ------------------------------------------
        if mode == "learning":
            result = ask_agent(user_input)
            conversation_memory.add_exchange(user_input, result.get("answer", "")[:500])
            return {
                "answer"            : result["answer"],
                "mode"              : "learning",
                "domain"            : result.get("domain", []),
                "sources"           : result.get("sources", []),
                "intent"            : result.get("intent", ""),
                "intent_confidence" : result.get("intent_confidence", 0.0),
                "user_level"        : result.get("user_level", "adaptive learner"),
                "topic"             : result.get("topic", ""),
                "suggestions"       : result.get("suggestions", []),
                "semantic_query"    : result.get("semantic_query", user_input),
                "retrieval_scores"  : result.get("retrieval_scores", {}),
                "profile"           : result.get("profile", {}),
                "question_context"  : result.get("question_context", {}),
                "status"            : "success",
            }

        # ------------------------------------------
        # MODE ANALYSIS - consultant senior premium
        # ------------------------------------------
        elif mode == "analysis":
            decision_result = explain_with_xai_full(user_input)
            conversation_memory.add_exchange(user_input, decision_result.get("answer", "")[:500])
            return {
                "answer"           : decision_result["answer"],
                "mode"             : "analysis",
                "suggestions"      : decision_result.get("suggestions", []),
                "discussed_kpis"   : decision_result.get("discussed_kpis", []),
                "discussed_actions": decision_result.get("discussed_actions", []),
                "decision_focus"   : decision_result.get("decision_focus", ""),
                "topic"            : decision_result.get("topic", ""),
                "question_context" : decision_result.get("question_context", {}),
                "status"           : "success",
            }

        else:
            return {
                "answer" : f"Mode '{mode}' invalide. Utilisez : learning, analysis, auto.",
                "mode"   : "error",
                "status" : "error"
            }

    except Exception as e:
        logger.error(f"Agent error: {e}")
        if _is_rate_limit_error(e):
            return {
                "answer" : _friendly_rate_limit_message(),
                "mode"   : mode,
                "status" : "rate_limited"
            }
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

    history_text = ""
    for msg in conversation_history[-4:]:
        history_text += msg + "\n"

    # Note: on conserve la signature et le comportement (passage de `question`
    # nu, car la memoire conversationnelle interne fait deja le travail de
    # continuite. history_text reste disponible pour debug / log externe.)
    _ = history_text  # conserve pour compat backward

    result = agent_marketing_global(question, mode)

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
# POINT D'ENTREE SCRIPT DIRECT
# ============================================================

if __name__ == "__main__":

    tests = [
        ("C'est quoi le ROAS ?",                       "learning"),
        ("Pourquoi mon CPA est eleve sur Google ?",    "analysis"),
        ("Explique les anomalies de ma campagne",       "auto"),
        ("Que signifie un CTR faible ?",                "auto"),
    ]

    for question, mode in tests:
        print("\n" + "=" * 70)
        result = agent_marketing_global(question, mode=mode)
        print(f"Mode    : {result['mode']}")
        print(f"Reponse : {result['answer'][:300]}...")
        print("=" * 70)