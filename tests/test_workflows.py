import pytest

from schema import Classification, RequestType, Urgency
import workflows


def _cls(request_type, urgency=Urgency.MEDIUM, confidence=0.9):
    return Classification(
        request_type=request_type,
        urgency=urgency,
        confidence=confidence,
        rationale="test rationale",
        signals=["test signal"],
        extracted_fields={"member_id": "12345"},
    )


@pytest.fixture(autouse=True)
def mock_generate_response(monkeypatch):
    monkeypatch.setattr(workflows, "generate_response", lambda text, cls, llm_call=None: "mocked draft response")


@pytest.mark.parametrize(
    "request_type",
    [
        RequestType.BILLING_DISPUTE,
        RequestType.CLAIM_ENQUIRY,
        RequestType.SERVICE_REQUEST,
        RequestType.ESCALATION,
    ],
)
def test_each_branch_produces_steps_and_outputs(request_type):
    cls = _cls(request_type)
    result = workflows.remediate("some request text", cls)
    assert len(result.steps_executed) >= 2
    assert result.outputs  # non-empty dict
    assert result.request_type == request_type


def test_escalation_is_always_human_in_loop():
    cls = _cls(RequestType.ESCALATION, urgency=Urgency.CRITICAL, confidence=0.95)
    result = workflows.remediate("I need urgent help", cls)
    assert result.human_in_loop is True


def test_low_confidence_forces_human_in_loop_even_non_escalation():
    cls = _cls(RequestType.BILLING_DISPUTE, urgency=Urgency.HIGH, confidence=0.4)
    result = workflows.remediate("I was charged twice", cls)
    assert result.human_in_loop is True


def test_high_confidence_non_escalation_is_not_human_in_loop():
    cls = _cls(RequestType.CLAIM_ENQUIRY, urgency=Urgency.LOW, confidence=0.9)
    result = workflows.remediate("what's my claim status", cls)
    assert result.human_in_loop is False


def test_unknown_request_type_forces_human_in_loop():
    cls = _cls(RequestType.UNKNOWN, urgency=Urgency.LOW, confidence=0.9)
    result = workflows.remediate("garbage", cls)
    assert result.human_in_loop is True


def test_injected_id_factory_produces_fixed_case_id():
    cls = _cls(RequestType.BILLING_DISPUTE)
    result = workflows.remediate(
        "some text", cls, id_factory=lambda: "TS-FIXED-0001"
    )
    assert result.case_id == "TS-FIXED-0001"
