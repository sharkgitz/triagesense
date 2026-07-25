"""Central configuration: thresholds, model name, paths, KB, colors.

Kept as the single swap point for anything environment- or provider-specific
(Section 0.5 / Section 3 of POC_BUILD_SPEC.md).
"""
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DB_PATH = BASE / "triage_log.db"

GROQ_MODEL = "llama-3.1-8b-instant"

CONFIDENCE_THRESHOLD = 0.60

# Business-impact constants (Section 7.3)
AVG_HANDLE_MINUTES_HUMAN = 6
LOADED_COST_PER_MIN = 0.35

URGENCY_COLORS = {
    "Low": "#0EA5A4",
    "Medium": "#F59E0B",
    "High": "#F97316",
    "Critical": "#DC2626",
}

# Brand palette (used for UI theming / charts)
BRAND = {
    "navy": "#12284C",
    "coral": "#FF5C39",
    "teal": "#0EA5A4",
    "slate": "#64748B",
    "bg": "#F5F7FB",
    "border": "#E2E8F0",
}

# Small hard-coded knowledge base grounding Branch 2 (Claim Status/Coverage Enquiry).
KNOWLEDGE_BASE = {
    "claim_processing_time": "Standard claims are processed within 10-15 business days of submission.",
    "coverage_check": "Coverage can be verified via the member portal or by providing your member ID and the date of service.",
    "appeal_window": "Members have 180 days from the date of an adverse determination to file an appeal.",
    "prior_auth_turnaround": "Routine prior-authorization requests are turned around within 3-5 business days; urgent requests within 24-72 hours.",
    "benefits_summary": "Benefit details, including deductibles and co-pays, are outlined in the member's Summary of Benefits document.",
    "explanation_of_benefits": "An Explanation of Benefits (EOB) is sent after a claim is processed, showing what was billed, covered, and owed.",
    "network_providers": "In-network providers can be located using the online provider directory to minimize out-of-pocket costs.",
    "premium_payment": "Premium payments are due on the 1st of each month, with a 30-day grace period before coverage lapses.",
}
