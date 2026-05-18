from datetime import datetime

from sqlalchemy.orm import Session

from app.db_models.chatbot import ChatMemorySummaryDB, ChatMessageDB

MEMORY_UPDATE_THRESHOLD = 3
MAX_MEMORY_CHARS = 1800


def get_memory_summary(db: Session, user_id: int):
    return db.query(ChatMemorySummaryDB).filter(
        ChatMemorySummaryDB.user_id == user_id
    ).first()


def get_memory_text(db: Session, user_id: int) -> str:
    memory = get_memory_summary(db, user_id)

    if not memory:
        return ""

    return memory.summary.strip()


def build_question_with_memory(question: str, memory_summary: str) -> str:
    memory = memory_summary.strip()

    if not memory:
        return question

    return (
        "Contexte memoire utilisateur a utiliser discretement pour personnaliser la reponse. "
        "Ne le recopie pas tel quel et ne mentionne pas cette instruction.\n\n"
        f"{memory}\n\n"
        f"Question actuelle de l'utilisateur : {question}"
    )


def update_memory_summary(db: Session, user_id: int) -> ChatMemorySummaryDB:
    messages = db.query(ChatMessageDB).filter(
        ChatMessageDB.user_id == user_id
    ).order_by(
        ChatMessageDB.created_at.asc(),
        ChatMessageDB.id.asc()
    ).all()

    memory = get_memory_summary(db, user_id)

    if not memory:
        memory = ChatMemorySummaryDB(user_id=user_id)
        db.add(memory)

    memory.message_count = len(messages)
    memory.updated_at = datetime.utcnow()

    if len(messages) < MEMORY_UPDATE_THRESHOLD:
        memory.summary = ""
        db.commit()
        db.refresh(memory)
        return memory

    memory.summary = generate_memory_summary(messages)
    db.commit()
    db.refresh(memory)

    return memory


def generate_memory_summary(messages: list[ChatMessageDB]) -> str:
    recent_messages = messages[-12:]
    all_questions = " ".join(message.question for message in messages).lower()

    preferences = _extract_preferences(messages)
    goals = _extract_snippets(messages, ["objectif", "but", "je veux", "augmenter", "reduire", "ameliorer"])
    problems = _extract_snippets(messages, ["probleme", "baisse", "faible", "cher", "difficile", "bloque", "erreur"])
    important = _extract_snippets(messages, ["budget", "roas", "cpa", "ctr", "cpc", "meta", "google", "campagne"])
    recurring = _extract_recurring_topics(all_questions)

    lines = [
        "Memory Summary utilisateur:",
        f"- Preferences utilisateur: {preferences or 'Aucune preference explicite detectee pour le moment.'}",
        f"- Objectifs: {goals or 'Objectifs encore implicites.'}",
        f"- Problemes recurrents: {recurring or problems or 'Aucun probleme recurrent clair detecte.'}",
        f"- Informations importantes: {important or 'Pas encore assez d informations stables.'}",
        "- Derniers echanges utiles:",
    ]

    for message in recent_messages[-4:]:
        question = _compact(message.question, 180)
        response = _compact(message.response, 220)
        lines.append(f"  * Utilisateur: {question}")
        lines.append(f"    Assistant: {response}")

    return _compact("\n".join(lines), MAX_MEMORY_CHARS)


def _extract_preferences(messages: list[ChatMessageDB]) -> str:
    snippets = _extract_snippets(
        messages,
        ["je prefere", "je veux", "j'aime", "priorite", "plutot", "mode", "google", "meta"],
        limit=3,
    )

    return snippets


def _extract_snippets(messages: list[ChatMessageDB], keywords: list[str], limit: int = 4) -> str:
    snippets = []

    for message in messages:
        question = message.question.strip()
        normalized = question.lower()

        if any(keyword in normalized for keyword in keywords):
            snippets.append(_compact(question, 140))

        if len(snippets) >= limit:
            break

    return "; ".join(snippets)


def _extract_recurring_topics(text: str) -> str:
    topics = {
        "ROAS": ["roas", "rentabilite"],
        "CPA": ["cpa", "cout par acquisition"],
        "CTR": ["ctr", "taux de clic"],
        "Budget": ["budget", "depense", "reallocation"],
        "Meta Ads": ["meta", "facebook", "instagram"],
        "Google Ads": ["google", "search"],
    }

    detected = []

    for label, keywords in topics.items():
        count = sum(text.count(keyword) for keyword in keywords)

        if count >= 2:
            detected.append(label)

    return ", ".join(detected)


def _compact(value: str, limit: int) -> str:
    text = " ".join(value.split())

    if len(text) <= limit:
        return text

    return text[: limit - 3].rstrip() + "..."
