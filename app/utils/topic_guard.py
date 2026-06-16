"""
app/utils/topic_guard.py
──────────────────────────────────────────────────────────────
Keyword-based topic guard that blocks clearly off-topic queries
before they reach the LLM.

Design rationale:
  - Lightweight first-pass filter using keyword matching.
  - Prevents wasting Groq API tokens on irrelevant questions.
  - The LLM prompt itself also enforces topic restrictions as a
    second layer.
  - Can be extended to use a small classification model for
    greater accuracy (Phase 2 enhancement).
"""

# ── Water quality topic keywords ─────────────────────────────
WATER_QUALITY_KEYWORDS: frozenset[str] = frozenset({
    # Core domain
    "water", "aqua", "river", "lake", "reservoir", "pond", "stream",
    "groundwater", "wastewater", "drinking", "potable",
    # Parameters
    "ph", "dissolved oxygen", "do", "bod", "biological oxygen demand",
    "tds", "total dissolved solids", "turbidity", "nitrate", "nitrite",
    "coliform", "bacteria", "contaminant", "pollutant", "pollution",
    # Metrics
    "wqi", "water quality index", "quality index", "score",
    # Chemistry
    "alkalinity", "hardness", "chloride", "fluoride", "arsenic",
    "lead", "mercury", "heavy metal", "chemical", "parameter",
    # Standards
    "who", "guideline", "standard", "safe", "limit", "threshold",
    "acceptable", "permissible", "regulation", "EPA",
    # Ecology
    "aquatic", "ecosystem", "fish", "algae", "eutrophication",
    "hypoxic", "anoxic", "dissolved",
    # Actions
    "treatment", "filtration", "purification", "disinfection",
    "chlorination", "monitoring", "testing", "measure", "sample",
    # WQI categories
    "excellent", "good", "medium", "bad", "poor",
    # Geographic context
    "body", "bodies", "source", "location", "area", "zone",
})

# ── Keywords that strongly indicate off-topic content ─────────
OFF_TOPIC_STRONG_SIGNALS: frozenset[str] = frozenset({
    # Politics
    "election", "president", "minister", "government", "parliament",
    "vote", "political", "party", "democrat", "republican",
    # Sports
    "cricket", "football", "soccer", "basketball", "tennis", "ipl",
    "match", "tournament", "player", "score", "goal", "team",
    # Entertainment
    "movie", "film", "actor", "actress", "celebrity", "music",
    "song", "concert", "netflix", "bollywood", "hollywood",
    # Finance
    "stock", "market", "bitcoin", "crypto", "forex", "invest",
    # General knowledge (that would never overlap with water quality)
    "recipe", "cooking", "fashion", "travel", "hotel",
})

# ── Fallback response ─────────────────────────────────────────
OFF_TOPIC_RESPONSE: str = (
    "I am a Water Quality Assistant and can only answer questions "
    "related to water quality data, WQI, water bodies, pollution "
    "indicators, and WHO water quality standards. "
    "Please ask me something related to water quality!"
)


from app.services.data_loader import load_data
from app.services.simple_retriever import STOP_WORDS, normalize_text
import re

_known_wb_names = None
_known_locations = None
_known_wb_tokens = None
_known_loc_tokens = None

def _get_known_entities():
    global _known_wb_names, _known_locations
    if _known_wb_names is None:
        try:
            df = load_data()
            _known_wb_names = {str(n).lower().strip() for n in df["Water Body Name"].dropna() if len(str(n).strip()) > 2}
            _known_locations = {str(l).lower().strip() for l in df["Location"].dropna() if len(str(l).strip()) > 2}
        except Exception:
            _known_wb_names = set()
            _known_locations = set()
    return _known_wb_names, _known_locations


def _get_known_tokens():
    global _known_wb_tokens, _known_loc_tokens
    if _known_wb_tokens is None:
        wb_names, locations = _get_known_entities()
        _known_wb_tokens = set()
        for name in wb_names:
            norm = normalize_text(name)
            for token in norm.split():
                if token not in STOP_WORDS and len(token) > 2:
                    _known_wb_tokens.add(token)
        _known_loc_tokens = set()
        for loc in locations:
            norm = normalize_text(loc)
            for token in norm.split():
                if token not in STOP_WORDS and len(token) > 2:
                    _known_loc_tokens.add(token)
    return _known_wb_tokens, _known_loc_tokens


def is_water_quality_related(query: str) -> bool:
    """
    Determine whether a query is related to water quality topics.

    Allows water body names and locations to pass even if general keywords
    such as WQI, pH, DO, TDS are absent.
    """
    query_lower = query.lower().strip()
    tokens = set(query_lower.split())

    # Pass 1: Check for strong off-topic signals (multi-word and single-word)
    for signal in OFF_TOPIC_STRONG_SIGNALS:
        if signal in query_lower:
            return False

    # Pass 2: Check for at least one water-quality keyword
    for keyword in WATER_QUALITY_KEYWORDS:
        if keyword in query_lower:
            return True

    # Pass 3: Check if query contains any known water body name or location (word-bounded / token intersection)
    query_norm = normalize_text(query_lower)
    query_words = set(query_norm.split())
    wb_tokens, loc_tokens = _get_known_tokens()

    if query_words.intersection(wb_tokens) or query_words.intersection(loc_tokens):
        return True

    # Edge case: very short queries (1–2 words) might be ambiguous.
    # Default to allowing them through to the LLM (which will also filter).
    if len(tokens) <= 2:
        return True

    # Default: block if no water-quality keywords found in a longer query
    return False


def get_off_topic_response() -> str:
    """Return the standard off-topic response message."""
    return OFF_TOPIC_RESPONSE
