"""Branch dispatch + remediation (Section 6.2).

`remediate` classifies nothing itself - it takes an existing Classification
and runs the matching branch's simulated multi-step workflow.
"""
import secrets
from datetime import date, datetime, timezone
from typing import Callable, Optional

from classifier import generate_response, retrieve_kb
from config import CONFIDENCE_THRESHOLD
from schema import Classification, RemediationResult, RequestType, Urgency


def _default_id_factory() -> str:
    return f"TS-{date.today().strftime('%Y%m%d')}-{secrets.token_hex(2)}"


def _default_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _handle_billing_dispute(text, cls, llm_call):
    draft = generate_response(text, cls, llm_call)
    steps = [
        "Acknowledged receipt to member",
        "Escalated to senior claims handler (queue routing)",
        "Created case with priority flag in audit log",
        "Set 2-hour follow-up SLA timer",
    ]
    outputs = {
        "draft_response": draft,
        "routing": "Senior Claims Handler Queue",
        "case_log_entry": "Priority flag set",
        "sla": "2 hours",
    }
    return steps, outputs


def _handle_claim_enquiry(text, cls, llm_call):
    draft = generate_response(text, cls, llm_call)
    kb_used = retrieve_kb(text, k=2)
    steps = [
        "Identified sub-topic (status vs coverage vs benefits)",
        "Retrieved relevant knowledge-base facts",
        "Generated grounded answer and sent auto-response",
        "Logged as auto-resolved",
    ]
    outputs = {
        "draft_response": draft,
        "knowledge_base_used": kb_used,
        "status": "auto-resolved",
    }
    return steps, outputs


def _handle_service_request(text, cls, llm_call):
    draft = generate_response(text, cls, llm_call)
    steps = [
        "Extracted required details (member ID, procedure/date)",
        "Routed to relevant department (Utilization Mgmt / Scheduling)",
        "Generated confirmation to member",
        "Set SLA timer",
    ]
    outputs = {
        "extracted_fields": cls.extracted_fields,
        "routing": "Utilization Management / Scheduling",
        "draft_response": draft,
        "sla": "48 hours",
    }
    return steps, outputs


def _handle_escalation(text, cls, llm_call):
    draft = generate_response(text, cls, llm_call)
    steps = [
        "Flagged for human review immediately",
        "Drafted urgent empathetic acknowledgement",
        "Notified supervisor",
        "Paused auto-resolution (human-in-the-loop)",
    ]
    outputs = {
        "supervisor_alert": "Supervisor notified",
        "draft_response": draft,
        "human_in_loop": True,
    }
    return steps, outputs


def _handle_unknown(text, cls, llm_call):
    draft = generate_response(text, cls, llm_call)
    steps = [
        "Flagged as unknown/ambiguous request",
        "Routed to human review queue",
    ]
    outputs = {
        "draft_response": draft,
        "status": "pending human review",
    }
    return steps, outputs


DISPATCH = {
    RequestType.BILLING_DISPUTE: _handle_billing_dispute,
    RequestType.CLAIM_ENQUIRY: _handle_claim_enquiry,
    RequestType.SERVICE_REQUEST: _handle_service_request,
    RequestType.ESCALATION: _handle_escalation,
    RequestType.UNKNOWN: _handle_unknown,
}


def _should_human_review(cls: Classification) -> bool:
    # Policy enforced in code, identical for the deterministic and agentic paths:
    # financial disputes, complaints/escalations, and anything unknown or
    # critical goes to a human. Routine enquiries and service requests, when the
    # classifier is confident, are auto-handled.
    if cls.request_type in (
        RequestType.BILLING_DISPUTE,
        RequestType.ESCALATION,
        RequestType.UNKNOWN,
    ):
        return True
    if cls.urgency == Urgency.CRITICAL:
        return True
    return cls.confidence < CONFIDENCE_THRESHOLD


def remediate(
    text: str,
    cls: Classification,
    llm_call: Optional[Callable[[str, str], str]] = None,
    id_factory: Optional[Callable[[], str]] = None,
    now: Optional[Callable[[], str]] = None,
) -> RemediationResult:
    id_factory = id_factory or _default_id_factory
    now = now or _default_now

    handler = DISPATCH.get(cls.request_type, _handle_unknown)
    steps_executed, outputs = handler(text, cls, llm_call)

    return RemediationResult(
        case_id=id_factory(),
        request_type=cls.request_type,
        urgency=cls.urgency,
        confidence=cls.confidence,
        steps_executed=steps_executed,
        outputs=outputs,
        human_in_loop=_should_human_review(cls),
        timestamp=now(),
    )
