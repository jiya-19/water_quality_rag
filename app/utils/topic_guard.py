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


def is_water_quality_related(query: str) -> bool:
    """
    Determine whether a query is related to water quality topics.

    Uses a two-pass check:
      1. Reject if strong off-topic signals are found.
      2. Accept if water quality keywords are present.

    Args:
        query: The raw user query string.

    Returns:
        True if query is likely water-quality related,
        False if it should be blocked.
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

    # Edge case: very short queries (1–2 words) might be ambiguous.
    # Default to allowing them through to the LLM (which will also filter).
    if len(tokens) <= 2:
        return True

    # Default: block if no water-quality keywords found in a longer query
    return False


def get_off_topic_response() -> str:
    """Return the standard off-topic response message."""
    return OFF_TOPIC_RESPONSE
