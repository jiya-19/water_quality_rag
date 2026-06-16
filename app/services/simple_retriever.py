import re
import rapidfuzz
import pandas as pd
from typing import Any, Dict, List
from app.services.data_loader import load_data
from app.data.who_guidelines import WHO_KNOWLEDGE_BASE
from app.core.logger import logger

# ── Stop Words ────────────────────────────────────────────────────────
STOP_WORDS = {
    "what", "is", "the", "water", "quality", "of", "river", "lake", "pond", 
    "reservoir", "stream", "creek", "at", "in", "on", "for", "limit", "standard", 
    "acceptable", "permissible", "threshold", "safe", "who", "guideline", 
    "guidelines", "standards", "limits", "does", "do", "how", "much", "many", 
    "about", "tell", "me", "show", "get", "retrieve", "wqi", "category", "score", 
    "value", "level", "levels", "parameter", "parameters", "which", "bodies", "body",
    "wqi_category", "and", "or", "a", "an", "to"
}

def normalize_text(text: str) -> str:
    """
    Lowercases text and replaces non-alphanumeric characters with spaces,
    collapsing multiple spaces.
    """
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    return " ".join(cleaned.split())

def filter_query(query: str) -> str:
    """
    Remove stop words from the query to yield a string of important tokens.
    """
    norm = normalize_text(query)
    tokens = norm.split()
    filtered_tokens = [t for t in tokens if t not in STOP_WORDS]
    if not filtered_tokens:
        return norm
    return " ".join(filtered_tokens)

class SimpleRetriever:
    """
    In-memory database retriever using pandas and RapidFuzz.
    Supports exact, partial, and fuzzy string matching across
    water bodies, locations, and WHO guidelines.
    """

    def __init__(self, csv_path: str | None = None) -> None:
        logger.info("Initializing SimpleRetriever...")
        self.dataset = load_data(csv_path)
        self.who_guidelines = WHO_KNOWLEDGE_BASE
        
        # Cache WHO Guidelines precomputations
        self.cached_who_guidelines = []
        for item in self.who_guidelines:
            topic = str(item.get("topic", "")).strip()
            content = str(item.get("content", "")).strip()
            self.cached_who_guidelines.append({
                "topic": topic,
                "content": content,
                "topic_lower": topic.lower(),
                "topic_norm": normalize_text(topic),
                "content_lower": content.lower()
            })

        # Cache dataset records and precomputations
        self.records = []
        for row in self.dataset.to_dict("records"):
            water_body = str(row.get("Water Body Name", "")).strip()
            location = str(row.get("Location", "")).strip()
            
            # Format natural language content string once and cache it
            content = (
                f"Water body: {water_body} located in {location} "
                f"(Latitude: {row.get('Latitude', 'N/A')}, Longitude: {row.get('Longitude', 'N/A')}). "
                f"Year = {row.get('Year', 'Unknown')}. "
                f"Water quality parameters: pH = {row.get('pH', 'N/A')}, "
                f"Dissolved Oxygen (DO) = {row.get('Dissolved Oxygen (DO)', 'N/A')} mg/L, "
                f"Biological Oxygen Demand (BOD) = {row.get('Biological Oxygen Demand (BOD)', 'N/A')} mg/L, "
                f"Total Dissolved Solids (TDS) = {row.get('Total Dissolved Solids (TDS)', 'N/A')} mg/L, "
                f"Turbidity = {row.get('Turbidity', 'N/A')} NTU, "
                f"Nitrate = {row.get('Nitrate', 'N/A')} mg/L, "
                f"Coliform = {row.get('Coliform', 'N/A')} CFU/100 mL. "
                f"Water Quality Index (WQI) = {row.get('Water Quality Index (WQI)', 'N/A')}. "
                f"Water Quality Category: {row.get('Water Quality Category', 'Unknown')}."
            )
            
            row_dict = {
                "source": "dataset",
                "water_body": water_body,
                "location": location,
                "latitude": float(row.get("Latitude", 0.0) or 0.0),
                "longitude": float(row.get("Longitude", 0.0) or 0.0),
                "year": int(row.get("Year", 0) or 0),
                "ph": float(row.get("pH", 0.0) or 0.0),
                "do": float(row.get("Dissolved Oxygen (DO)", 0.0) or 0.0),
                "bod": float(row.get("Biological Oxygen Demand (BOD)", 0.0) or 0.0),
                "tds": float(row.get("Total Dissolved Solids (TDS)", 0.0) or 0.0),
                "turbidity": float(row.get("Turbidity", 0.0) or 0.0),
                "nitrate": float(row.get("Nitrate", 0.0) or 0.0),
                "coliform": float(row.get("Coliform", 0.0) or 0.0),
                "wqi": float(row.get("Water Quality Index (WQI)", 0.0) or 0.0),
                "wqi_category": str(row.get("Water Quality Category", "")),
                "content": content,
                "_wb_lower": water_body.lower(),
                "_wb_norm": normalize_text(water_body),
                "_loc_lower": location.lower(),
                "_loc_norm": normalize_text(location)
            }
            self.records.append(row_dict)

        logger.info(f"SimpleRetriever initialized with {len(self.records)} dataset records and {len(self.cached_who_guidelines)} WHO guideline topics.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search both the dataset and the WHO guidelines using a scoring strategy
        comprising exact, partial, location, and RapidFuzz fuzzy matches.
        """
        q_lower = query.lower().strip()
        q_norm = normalize_text(query)
        q_filtered = filter_query(query)
        
        candidates = []
        has_dataset_match = False

        # ── 1. Search dataset rows ─────────────────────────────────────────────
        for row in self.records:
            wb_lower = row["_wb_lower"]
            wb_norm = row["_wb_norm"]
            loc_lower = row["_loc_lower"]
            loc_norm = row["_loc_norm"]

            score = 0.0
            matched_entity = False

            # 1. Exact water body name match
            wb_fuzzy = 0.0
            if wb_norm == q_filtered:
                score += 150.0
                wb_fuzzy = 100.0
                matched_entity = True
            # 2. Partial water body name match (word-bounded)
            elif len(wb_norm) > 2 and len(q_filtered) > 2 and (re.search(rf"\b{re.escape(wb_norm)}\b", q_filtered) or re.search(rf"\b{re.escape(q_filtered)}\b", wb_norm)):
                score += 100.0
                wb_token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, wb_lower)
                wb_part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, wb_lower)
                wb_fuzzy = max(wb_token_set, wb_part_ratio)
                matched_entity = True
            else:
                wb_token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, wb_lower)
                wb_part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, wb_lower)
                # Penalize partial ratio for short strings to avoid false positive substring matches
                if len(wb_lower) < 4 or len(q_filtered) < 4:
                    wb_part_ratio *= 0.4
                wb_fuzzy = max(wb_token_set, wb_part_ratio)

            # 3. Exact location match
            loc_fuzzy = 0.0
            if loc_norm == q_filtered:
                score += 120.0
                loc_fuzzy = 100.0
                matched_entity = True
            # 4. Partial location match (word-bounded)
            elif len(loc_norm) > 2 and len(q_filtered) > 2 and (re.search(rf"\b{re.escape(loc_norm)}\b", q_filtered) or re.search(rf"\b{re.escape(q_filtered)}\b", loc_norm)):
                score += 80.0
                loc_token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, loc_lower)
                loc_part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, loc_lower)
                loc_fuzzy = max(loc_token_set, loc_part_ratio)
                matched_entity = True
            else:
                loc_token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, loc_lower)
                loc_part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, loc_lower)
                # Penalize partial ratio for short strings to avoid false positive substring matches
                if len(loc_lower) < 4 or len(q_filtered) < 4:
                    loc_part_ratio *= 0.4
                loc_fuzzy = max(loc_token_set, loc_part_ratio)

            if matched_entity:
                has_dataset_match = True

            fuzzy_score = max(wb_fuzzy, loc_fuzzy)
            score += fuzzy_score

            candidates.append({
                "score": score,
                "data": {
                    "source": "dataset",
                    "water_body": row["water_body"],
                    "location": row["location"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "year": row["year"],
                    "ph": row["ph"],
                    "do": row["do"],
                    "bod": row["bod"],
                    "tds": row["tds"],
                    "turbidity": row["turbidity"],
                    "nitrate": row["nitrate"],
                    "coliform": row["coliform"],
                    "wqi": row["wqi"],
                    "wqi_category": row["wqi_category"],
                    "content": row["content"]
                }
            })

        # ── 2. Search WHO guidelines (Only if trigger words are present) ──────
        who_triggers = {"who", "guideline", "standard", "limit", "permissible", "acceptable", "threshold", "safe", "guidelines", "standards", "limits"}
        search_who = any(trigger in q_lower for trigger in who_triggers)

        if search_who:
            for item in self.cached_who_guidelines:
                topic = item["topic"]
                content = item["content"]
                topic_lower = item["topic_lower"]
                topic_norm = item["topic_norm"]
                content_lower = item["content_lower"]

                score = 0.0

                # Exact matching on topic
                fuzzy_score = 0.0
                if topic_norm == q_norm:
                    score += 150.0
                    fuzzy_score = 100.0
                # Partial matching on topic (word-bounded)
                elif len(topic_norm) > 1 and re.search(rf"\b{re.escape(topic_norm)}\b", q_norm):
                    score += 100.0
                    token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, topic_lower)
                    part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, topic_lower)
                    content_token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, content_lower)
                    content_part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, content_lower)
                    fuzzy_score = max(token_set, part_ratio, content_token_set * 0.8, content_part_ratio * 0.8)
                else:
                    token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, topic_lower)
                    part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, topic_lower)
                    content_token_set = rapidfuzz.fuzz.token_set_ratio(q_filtered, content_lower)
                    content_part_ratio = rapidfuzz.fuzz.partial_ratio(q_filtered, content_lower)
                    fuzzy_score = max(token_set, part_ratio, content_token_set * 0.8, content_part_ratio * 0.8)

                score += fuzzy_score

                # WHO guidelines specific keyword boost
                if any(term in q_lower for term in ["who", "guideline", "standard", "limit", "threshold", "acceptable"]):
                    # If topic matches parameters in query
                    if any(param in topic_lower for param in ["ph", "tds", "turbidity", "nitrate", "do", "bod", "coliform", "wqi"]):
                        score += 50.0
                    else:
                        score += 20.0

                # Cap score if a dataset match exists, preventing WHO from outranking dataset
                if has_dataset_match:
                    score = min(score, 50.0)

                candidates.append({
                    "score": score,
                    "data": {
                        "source": "who_guidelines",
                        "water_body": "",
                        "location": "",
                        "topic": topic,
                        "wqi": None,
                        "wqi_category": "",
                        "content": content
                    }
                })

        # Sort candidates by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Inject score inside candidates data dict for retrieval-score-based topic detection
        for c in candidates:
            c["data"]["score"] = c["score"]

        # Return top_k candidates' data
        return [c["data"] for c in candidates[:top_k]]
