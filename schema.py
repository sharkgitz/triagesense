"""Pydantic data contracts (Section 5) + tolerant enum coercion (Section 0.5).

Everything else in the app depends on these types, so they must never raise
on malformed input - coercion always degrades to a safe default instead.
"""
from enum import Enum

from pydantic import BaseModel, Field


class RequestType(str, Enum):
    BILLING_DISPUTE = "Billing/Claim Dispute"
    CLAIM_ENQUIRY = "Claim Status/Coverage Enquiry"
    SERVICE_REQUEST = "Prior-Auth/Appointment Service Request"
    ESCALATION = "Complaint/Urgent Escalation"
    UNKNOWN = "Unknown/Needs Human Review"


class Urgency(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class Classification(BaseModel):
    request_type: RequestType
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    signals: list[str]
    extracted_fields: dict


class RemediationResult(BaseModel):
    case_id: str
    request_type: RequestType
    urgency: Urgency
    confidence: float
    steps_executed: list[str]
    outputs: dict
    human_in_loop: bool
    timestamp: str
    agent_reasoning: str = ""
    planned_tools: list[str] = []


def _norm(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("-", "").replace("/", "")


def to_request_type(raw: str) -> RequestType:
    for rt in RequestType:
        if _norm(rt.value) == _norm(raw or ""):
            return rt
    return RequestType.UNKNOWN


def to_urgency(raw: str) -> Urgency:
    for u in Urgency:
        if _norm(u.value) == _norm(raw or ""):
            return u
    return Urgency.LOW


def clamp_confidence(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))
