import json

from conftest import fake_llm, sequenced_llm
from schema import Classification, RequestType, Urgency
from classifier import classify, rule_based_fallback, retrieve_kb, generate_response


VALID_BILLING_JSON = json.dumps({
    "request_type": "Billing/Claim Dispute",
    "urgency": "High",
    "confidence": 0.87,
    "rationale": "Member reports being charged twice for the same claim.",
    "signals": ["charged twice", "refund"],
    "extracted_fields": {"amount": "150"},
})


def test_retrieve_kb_returns_relevant_fact():
    facts = retrieve_kb("How long does it take for my claim to be processed?", k=2)
    assert any("processed within" in f for f in facts)
    assert 1 <= len(facts) <= 2


def test_retrieve_kb_never_empty_on_gibberish():
    facts = retrieve_kb("zzzzz qqqqq", k=2)
    assert len(facts) >= 1  # falls back to a general fact, never empty


def test_enquiry_response_falls_back_to_grounded_kb_line():
    cls = Classification(
        request_type=RequestType.CLAIM_ENQUIRY, urgency=Urgency.LOW, confidence=0.9,
        rationale="", signals=[], extracted_fields={},
    )
    # LLM returns empty -> must fall back to a KB-grounded line, not a generic template
    out = generate_response("When will my claim be processed?", cls, llm_call=fake_llm(""))
    assert "knowledge base" in out.lower()


def test_valid_json_fake_produces_correct_classification():
    result = classify("I was charged twice, please refund me.", llm_call=fake_llm(VALID_BILLING_JSON))
    assert result.request_type == RequestType.BILLING_DISPUTE
    assert result.urgency == Urgency.HIGH
    assert result.confidence == 0.87


def test_malformed_then_valid_json_recovers_via_retry():
    llm = sequenced_llm(["not json at all", VALID_BILLING_JSON])
    result = classify("I was charged twice, please refund me.", llm_call=llm)
    assert result.request_type == RequestType.BILLING_DISPUTE
    assert result.confidence == 0.87


def test_always_garbage_falls_back_to_rule_based():
    llm = fake_llm("still not json")
    result = classify("I was charged twice for my claim", llm_call=llm)
    assert result.confidence == 0.40
    assert "fallback" in result.rationale.lower()


def test_empty_input_short_circuits_without_calling_llm():
    calls = {"count": 0}

    def counting_llm(system_prompt, user_text):
        calls["count"] += 1
        return VALID_BILLING_JSON

    result = classify("", llm_call=counting_llm)
    assert result.request_type == RequestType.UNKNOWN
    assert calls["count"] == 0


def test_empty_input_routes_to_human_review_semantics():
    result = classify("   ", llm_call=fake_llm(VALID_BILLING_JSON))
    assert result.request_type == RequestType.UNKNOWN
    assert result.confidence < 0.60


def test_rule_based_fallback_billing_dispute():
    result = rule_based_fallback("I was charged twice for my claim")
    assert result.request_type == RequestType.BILLING_DISPUTE
    assert result.urgency == Urgency.HIGH


def test_rule_based_fallback_claim_enquiry():
    result = rule_based_fallback("What is the coverage status of my claim?")
    assert result.request_type == RequestType.CLAIM_ENQUIRY
    assert result.urgency == Urgency.LOW


def test_rule_based_fallback_service_request():
    result = rule_based_fallback("I need a prior authorization for my appointment")
    assert result.request_type == RequestType.SERVICE_REQUEST
    assert result.urgency == Urgency.MEDIUM


def test_rule_based_fallback_escalation():
    result = rule_based_fallback("This is an emergency, I need urgent help now")
    assert result.request_type == RequestType.ESCALATION
    assert result.urgency == Urgency.CRITICAL


def test_rule_based_fallback_unknown_on_no_keywords():
    result = rule_based_fallback("xyz completely unrelated text")
    assert result.request_type == RequestType.UNKNOWN
