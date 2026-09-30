import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def _extractive_answer(query, contexts):
    if not contexts:
        return "I could not find supporting context."
    text = contexts[0]["text"]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    q_words = set(re.findall(r"\b\w+\b", query.lower()))
    ranked = sorted(
        sentences,
        key=lambda s: len(q_words & set(re.findall(r"\b\w+\b", s.lower()))),
        reverse=True,
    )
    return " ".join(ranked[:3]).strip()

def _generate_gemini(query, contexts):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    context_text = "\n\n".join(
        f"[{i+1}] {x['text']}" for i, x in enumerate(contexts)
    )

    prompt = f"""You are the answer-generation component of RAG Doctor.

Answer the question using ONLY the supplied context.
- Be concise and factual.
- Do not invent information.
- If the context is insufficient, explicitly say that.
- Cite supporting passages using [1], [2], etc.

Question:
{query}

Retrieved context:
{context_text}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 400
        }
    }

    try:
        response = requests.post(
            GEMINI_API_URL.format(model=model),
            params={"key": api_key},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        return answer.strip()
    except Exception:
        return None

def generate_answer(query, contexts):
    # Gemini is the preferred optional generator.
    gemini_answer = _generate_gemini(query, contexts)
    if gemini_answer:
        return gemini_answer, "gemini"

    # Keep the project runnable without an API key.
    return _extractive_answer(query, contexts), "extractive-fallback"
