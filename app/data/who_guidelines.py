"""
app/data/who_guidelines.py
──────────────────────────────────────────────────────────────
Static WHO (World Health Organisation) drinking-water quality
guidelines embedded as Python strings.

These are converted to LangChain Documents and merged into the
FAISS vector store alongside the CSV dataset so that questions
about acceptable parameter ranges are answered accurately.

Source: WHO Guidelines for Drinking-water Quality, 4th edition
(https://www.who.int/publications/i/item/9789241549950)
"""

WHO_KNOWLEDGE_BASE: list[dict] = [
    {
        "topic": "pH",
        "content": (
            "WHO pH Guidelines for Drinking Water: "
            "The WHO recommends a pH range of 6.5 to 8.5 for safe drinking water. "
            "Water with pH below 6.5 is considered acidic and may cause corrosion "
            "of pipes, leading to metal leaching. Water with pH above 8.5 is alkaline "
            "and may taste bitter. Extreme pH levels (below 4 or above 11) can be "
            "directly harmful to human health. The ideal drinking water pH is 7.0 "
            "(neutral). pH outside the 6.5–8.5 range is classified as a water "
            "quality concern."
        ),
    },
    {
        "topic": "TDS (Total Dissolved Solids)",
        "content": (
            "WHO TDS Guidelines for Drinking Water: "
            "WHO does not set a formal health-based guideline for TDS but provides "
            "palatability thresholds. TDS below 300 mg/L is considered excellent. "
            "TDS between 300–600 mg/L is considered good. TDS between 600–900 mg/L "
            "is considered fair. TDS between 900–1200 mg/L is considered poor. "
            "TDS above 1200 mg/L is considered unacceptable. The US EPA secondary "
            "standard is 500 mg/L. High TDS may indicate presence of harmful "
            "dissolved salts, heavy metals, or other contaminants."
        ),
    },
    {
        "topic": "Turbidity",
        "content": (
            "WHO Turbidity Guidelines for Drinking Water: "
            "WHO recommends turbidity below 1 NTU (Nephelometric Turbidity Unit) "
            "for treated drinking water. Turbidity up to 5 NTU is the maximum "
            "permissible level in many national standards. High turbidity (above 5 NTU) "
            "indicates suspended particles, possible microbial contamination, and "
            "interference with disinfection processes. Turbidity above 10 NTU is "
            "a serious health risk. Clear water should be 0–1 NTU. Turbidity is a "
            "key indicator of water treatment effectiveness."
        ),
    },
    {
        "topic": "Nitrate",
        "content": (
            "WHO Nitrate Guidelines for Drinking Water: "
            "WHO sets the guideline value for nitrate at 50 mg/L (as NO3). "
            "The US EPA and European Union standard is 10 mg/L as nitrogen (NO3-N), "
            "equivalent to approximately 44 mg/L as nitrate. High nitrate levels "
            "(above 50 mg/L) can cause methaemoglobinaemia (blue baby syndrome) in "
            "infants under 6 months. Nitrate above 50 mg/L is particularly dangerous "
            "for pregnant women, infants, and immunocompromised individuals. "
            "Nitrate contamination commonly results from agricultural runoff, "
            "fertilizer use, and sewage discharge. Levels below 10 mg/L are considered "
            "safe for all population groups."
        ),
    },
    {
        "topic": "Dissolved Oxygen (DO)",
        "content": (
            "Dissolved Oxygen (DO) Water Quality Guidelines: "
            "Dissolved Oxygen is a critical indicator of aquatic ecosystem health, "
            "not a direct WHO drinking-water health guideline. DO levels above 6 mg/L "
            "indicate good water quality suitable for most aquatic life. "
            "DO between 4–6 mg/L indicates moderate stress on aquatic organisms. "
            "DO between 2–4 mg/L indicates hypoxic conditions and poor water quality. "
            "DO below 2 mg/L is anoxic and unsuitable for most aquatic life. "
            "High DO (above 8 mg/L) is associated with clean, healthy water bodies. "
            "Low DO is often caused by organic pollution, BOD loading, and algal blooms."
        ),
    },
    {
        "topic": "BOD (Biological Oxygen Demand)",
        "content": (
            "BOD Water Quality Guidelines and Interpretation: "
            "BOD (Biological Oxygen Demand) measures the amount of oxygen consumed "
            "by biological organisms decomposing organic matter in water. "
            "BOD below 1 mg/L indicates very clean water (unpolluted rivers/lakes). "
            "BOD between 1–2 mg/L indicates clean water. "
            "BOD between 2–3 mg/L indicates doubtful purity, some pollution. "
            "BOD between 3–5 mg/L indicates moderately polluted water. "
            "BOD above 5 mg/L indicates heavily polluted water. "
            "BOD above 10 mg/L indicates water receiving heavy organic waste. "
            "High BOD reduces dissolved oxygen and endangers aquatic life. "
            "WHO guideline for treated wastewater discharge: BOD below 10 mg/L."
        ),
    },
    {
        "topic": "Coliform Bacteria",
        "content": (
            "WHO Coliform Guidelines for Drinking Water: "
            "WHO guideline: Zero total coliform bacteria per 100 mL in drinking water. "
            "E. coli (or thermotolerant coliforms): must not be detectable in any "
            "100 mL sample of drinking water. Total coliforms: must not be detectable "
            "in 95% of samples taken throughout the year. Coliform presence indicates "
            "fecal contamination and potential presence of pathogens such as cholera, "
            "typhoid, and dysentery. Coliform count above 0 CFU/100 mL requires "
            "immediate investigation and disinfection. This is the most critical "
            "microbiological indicator for drinking water safety."
        ),
    },
    {
        "topic": "Water Quality Index (WQI)",
        "content": (
            "Water Quality Index (WQI) Interpretation and Categories: "
            "WQI is a composite score aggregating multiple water quality parameters "
            "into a single number from 0 to 100. "
            "WQI 90–100: Excellent water quality — suitable for drinking with minimal "
            "treatment. All parameters within WHO guidelines. "
            "WQI 70–89: Good water quality — suitable for drinking, minor parameter "
            "deviations, standard treatment required. "
            "WQI 50–69: Medium water quality — suitable for irrigation and industrial "
            "use, requires treatment before drinking. "
            "WQI 25–49: Bad water quality — high pollution levels, significant "
            "treatment needed, unsuitable for direct human consumption. "
            "WQI 0–24: Very bad / unfit — severely polluted, not suitable for any "
            "domestic use without extensive treatment."
        ),
    },
    {
        "topic": "General Water Pollution Indicators",
        "content": (
            "Key Water Pollution Indicators and Their Significance: "
            "pH deviation from 6.5–8.5 indicates chemical pollution or acid rain. "
            "High TDS (above 500 mg/L) suggests dissolved salts, heavy metals, or "
            "industrial effluents. High turbidity (above 5 NTU) indicates suspended "
            "solids, sediment, or microbial contamination. Elevated nitrate (above "
            "50 mg/L) indicates agricultural runoff or sewage contamination. "
            "Low DO (below 4 mg/L) indicates organic pollution and oxygen depletion. "
            "High BOD (above 5 mg/L) indicates heavy organic waste loading. "
            "Coliform presence indicates microbiological contamination from fecal "
            "sources. These parameters together form the basis of WQI calculation "
            "and water body health assessment."
        ),
    },
]
