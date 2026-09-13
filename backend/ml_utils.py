import os

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from campus_rag import retrieve_campus_context
from rag.vector_store import search_collection, upsert_documents

# Trained ML assets: category classifier, priority regressor, dedup vectorizer.
# Rule-based heuristics below are only used as a fallback signal for routing and admin summaries.
load_dotenv()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, "category_model.joblib")
if not os.path.exists(CATEGORY_MODEL_PATH):
    raise FileNotFoundError("No trained category model found in backend/ml/models")

_category_model = joblib.load(CATEGORY_MODEL_PATH)
_priority_model = joblib.load(os.path.join(MODEL_DIR, "priority_model.joblib"))
_dedup_vectorizer = joblib.load(os.path.join(MODEL_DIR, "dedup_vectorizer.joblib"))

URGENT_WORDS = [
    "fire", "spark", "sparking", "shock", "flood", "burst", "unsafe",
    "emergency", "exposed", "no security", "blocked exit", "gas leak",
    "ragging", "beating", "beat", "fight", "fighting", "assault", "assaulting",
    "bully", "bullying", "harass", "harassment", "attack", "blood", "threat",
    "weapon", "knife", "injury", "hazing", "abuse", "violence"
]

SAFETY_OVERRIDE_TOKENS = [
    "ragging", "rag", "beating", "beat", "fight", "fighting", "assault",
    "assaulting", "bully", "bullying", "harass", "harassment", "attack",
    "blood", "threat", "weapon", "knife", "injury", "hazing", "abuse",
    "abusing", "violence", "security", "gate", "safety", "fire", "cctv",
    "exit", "unsafe", "emergency", "staircase", "guard"
]

DUPLICATE_THRESHOLD = 0.4
TEAM_DEPARTMENTS = {
    "wifi": "IT Team",
    "electrical": "Electrical Team",
    "plumbing": "Maintenance Team",
    "hostel": "Hostel Team",
    "safety": "Safety Team",
    "other": "Facilities Team",
}
ALLOWED_CATEGORIES = set(TEAM_DEPARTMENTS)


def infer_department(category: str | None, description: str = "") -> str:
    text = (description or "").lower()
    if any(token in text for token in ["ragging", "rag", "beating", "beat", "fight", "fighting", "assault", "assaulting", "bully", "bullying", "harass", "harassment", "attack", "blood", "threat", "weapon", "knife", "injury", "hazing", "abuse", "violence", "security", "safety", "fire"]):
        return "Safety Team"
    if any(token in text for token in ["wifi", "internet", "network", "router", "lan", "server", "connection"]):
        return "IT Team"
    if any(token in text for token in ["water", "leak", "drain", "toilet", "tap", "pipe", "flood", "bathroom", "washroom", "sewer"]):
        return "Maintenance Team"
    if any(token in text for token in ["power", "electricity", "fan", "light", "socket", "switch", "short", "spark", "ac", "circuit"]):
        return "Electrical Team"
    if any(token in text for token in ["hostel", "room", "mess", "cleaning", "laundry", "noise", "pest", "furniture"]):
        return "Hostel Team"
    if any(token in text for token in ["security", "gate", "safety", "fire", "cctv", "exit", "unsafe", "emergency", "staircase"]):
        return "Safety Team"
    if category:
        return TEAM_DEPARTMENTS.get(category.lower(), "Facilities Team")
    return "Facilities Team"


def _generate_with_gemini(prompt: str) -> str | None:
    if not GEMINI_API_KEY:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text:
            return text.strip()
        return None
    except Exception:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return None
        except Exception:
            return None


def _fallback_category_from_text(text: str) -> str:
    text_l = (text or "").lower()
    if any(token in text_l for token in ["ragging", "rag", "beating", "beat", "fight", "fighting", "assault", "assaulting", "bully", "bullying", "harass", "harassment", "attack", "blood", "threat", "weapon", "knife", "injury", "hazing", "abuse", "violence"]):
        return "safety"
    if any(token in text_l for token in ["wifi", "internet", "network", "router", "lan", "server", "connection", "signal"]):
        return "wifi"
    if any(token in text_l for token in ["water", "leak", "drain", "toilet", "tap", "pipe", "flood", "bathroom", "washroom", "sewer"]):
        return "plumbing"
    if any(token in text_l for token in ["power", "electricity", "fan", "light", "socket", "switch", "short", "spark", "ac", "circuit"]):
        return "electrical"
    if any(token in text_l for token in ["hostel", "room", "mess", "cleaning", "laundry", "noise", "pest", "furniture", "bed", "gate"]):
        return "hostel"
    if any(token in text_l for token in ["security", "gate", "safety", "fire", "cctv", "exit", "unsafe", "emergency", "staircase", "guard"]):
        return "safety"
    return "other"


def classify_category(text: str) -> str:
    if not text or not str(text).strip():
        return "other"

    text_l = str(text).lower()
    if any(token in text_l for token in ["ragging", "rag", "beating", "beat", "fight", "fighting", "assault", "assaulting", "bully", "bullying", "harass", "harassment", "attack", "blood", "threat", "weapon", "knife", "injury", "hazing", "abuse", "violence"]):
        return "safety"

    prediction = str(_category_model.predict([text])[0]).strip().lower()
    if prediction in ALLOWED_CATEGORIES and prediction != "other":
        return prediction
    return _fallback_category_from_text(text)


def compute_urgency(text: str) -> float:
    text_l = (text or "").lower()
    CRITICAL_SAFETY = ["ragging", "beating", "assault", "attack", "weapon", "knife", "blood", "fire", "gas leak", "violence"]
    if any(w in text_l for w in CRITICAL_SAFETY):
        return 1.0
    score = 0.15
    for w in URGENT_WORDS:
        if w in text_l:
            score += 0.3
    return float(min(score, 1.0))


def compute_priority(urgency: float, frequency: int, days_open: int) -> float:
    x = np.array([[urgency, frequency, days_open]])
    score = float(_priority_model.predict(x)[0])
    if urgency >= 0.9:
        score = max(score, 78.5 + (urgency - 0.9) * 20.0)
    return float(np.clip(score, 0, 100))


def _location_match(loc_a: str, loc_b: str) -> float:
    a, b = (loc_a or "").strip().lower(), (loc_b or "").strip().lower()
    if a == b:
        return 1.0
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def find_duplicate(new_text: str, new_category: str, new_location: str, open_complaints: list):
    """Duplicate detection is a hybrid semantic vector search + location-match heuristic."""
    if not open_complaints:
        return None

    new_text_clean = (new_text or "").strip().lower()
    new_loc_clean = (new_location or "").strip().lower()
    for cid, text, cat, loc in open_complaints:
        if text and loc and text.strip().lower() == new_text_clean and loc.strip().lower() == new_loc_clean:
            return cid

    candidates = [c for c in open_complaints if c[2] == new_category or c[2] == "other" or new_category == "other"]
    if not candidates:
        return None


    campus_context = retrieve_campus_context(new_location, new_text)
    campus_boost = 0.0
    if campus_context:
        campus_boost = min(campus_context[0]["score"] / 10.0, 0.35)

    texts = [c[1] for c in candidates]
    vecs = _dedup_vectorizer.transform(texts + [new_text])
    text_sims = cosine_similarity(vecs[-1], vecs[:-1])[0]

    best_id, best_score = None, 0.0
    for (cid, _, _, loc), text_sim in zip(candidates, text_sims):
        loc_sim = _location_match(new_location, loc)
        context_match = 1.0 if campus_context and any(item["name"].lower() in (loc or "").lower() for item in campus_context) else 0.0
        score = (0.50 * text_sim) + (0.35 * loc_sim) + (0.15 * context_match) + campus_boost
        if score > best_score:
            best_score, best_id = score, cid
    if best_score >= DUPLICATE_THRESHOLD:
        return best_id
    return None



def draft_resolution_message(student_name: str, category: str, description: str) -> str:
    """Gemini is optional; if no key is configured we fall back to a template."""
    prompt = (
        "Write a polite campus complaint resolution message in a friendly but professional tone. "
        f"Student name: {student_name}. Category: {category}. Complaint: {description}. "
        "Keep it brief, empathetic, and mention that the issue has been reviewed."
    )
    ai_text = _generate_with_gemini(prompt)
    if ai_text:
        return ai_text

    return (
        f"Hi {student_name},\n\n"
        f"Your complaint regarding \"{description.strip()}\" (category: {category}) "
        f"has been reviewed and resolved by the {category} team. "
        "Thank you for reporting this — it helps us keep the campus running smoothly.\n\n"
        "If the issue recurs, please submit a new report on Compass.\n\n"
        "— Compass Admin"
    )


def answer_admin_query(query: str, complaints: list[dict]) -> str:
    """Answer a simple admin query using aggregate filters and a semantic complaint retrieval pass."""
    q = (query or "").strip().lower()
    if not q:
        return "Please ask a question about the complaints data."
    if not complaints:
        return "There are no complaints matching that query yet."

    filtered = complaints
    try:
        from rag.vector_store import search_collection

        hits = search_collection(q, collection_name="complaints", k=min(8, len(complaints)))
        if hits:
            ids = {str(hit.get("id")) for hit in hits if hit.get("id") is not None}
            filtered = [c for c in complaints if str(c.get("id") or c.get("_id")) in ids] or filtered
    except Exception:
        pass

    if "resolved" in q:
        filtered = [c for c in filtered if (c.get("status") or "").lower() == "resolved"]
    elif "in progress" in q or "in_progress" in q or "progress" in q:
        filtered = [c for c in filtered if (c.get("status") or "").lower() == "in_progress"]
    elif "submitted" in q:
        filtered = [c for c in filtered if (c.get("status") or "").lower() == "submitted"]

    location_tokens = [token for token in q.split() if token and len(token) > 2]
    for token in location_tokens:
        matches = [c for c in filtered if token in (c.get("location") or "").lower()]
        if matches:
            filtered = matches
            break

    category_tokens = [cat for cat in ALLOWED_CATEGORIES if cat in q]
    for cat in category_tokens:
        filtered = [c for c in filtered if (c.get("category") or "").lower() == cat]

    if not filtered:
        return "There are no complaints matching that query right now."

    by_location = {}
    by_category = {}
    for item in filtered:
        loc = item.get("location") or "unknown"
        cat = item.get("category") or "other"
        by_location[loc] = by_location.get(loc, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1

    top_location = max(by_location, key=by_location.get)
    top_category = max(by_category, key=by_category.get)
    average_priority = sum(float(c.get("priority_score") or 0) for c in filtered) / len(filtered)

    prompt = (
        "Answer this admin question succinctly in natural language using the aggregate facts. "
        f"Question: {query}. "
        f"Matching complaints: {len(filtered)}. "
        f"Top location: {top_location}. "
        f"Top category: {top_category}. "
        f"Average priority: {average_priority:.1f}."
    )
    ai_text = _generate_with_gemini(prompt)
    if ai_text:
        return ai_text

    return (
        f"There are {len(filtered)} complaints matching that question. "
        f"The most reported location is {top_location} and the most common category is {top_category}. "
        f"The average priority score is {average_priority:.1f}."
    )
