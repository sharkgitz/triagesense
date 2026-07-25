from schema import (
    RequestType,
    Urgency,
    Classification,
    to_request_type,
    to_urgency,
    clamp_confidence,
)


def test_to_request_type_matches_normalized():
    assert to_request_type("Billing / Claim Dispute") == RequestType.BILLING_DISPUTE
    assert to_request_type("Billing/Claim Dispute") == RequestType.BILLING_DISPUTE


def test_to_request_type_unknown_on_garbage():
    assert to_request_type("garbage") == RequestType.UNKNOWN
    assert to_request_type("") == RequestType.UNKNOWN
    assert to_request_type(None) == RequestType.UNKNOWN


def test_to_urgency_case_insensitive():
    assert to_urgency("critical") == Urgency.CRITICAL
    assert to_urgency("CRITICAL") == Urgency.CRITICAL


def test_to_urgency_defaults_to_low_on_empty_or_garbage():
    assert to_urgency("") == Urgency.LOW
    assert to_urgency(None) == Urgency.LOW
    assert to_urgency("nonsense") == Urgency.LOW


def test_confidence_clamps_high_and_low():
    assert clamp_confidence(1.5) == 1.0
    assert clamp_confidence(-0.2) == 0.0
    assert clamp_confidence(0.75) == 0.75


def test_confidence_clamps_bad_input_to_zero():
    assert clamp_confidence(None) == 0.0
    assert clamp_confidence("not a number") == 0.0


def test_classification_constructs_within_valid_range():
    c = Classification(
        request_type=RequestType.BILLING_DISPUTE,
        urgency=Urgency.HIGH,
        confidence=clamp_confidence(1.5),
        rationale="test",
        signals=[],
        extracted_fields={},
    )
    assert c.confidence == 1.0
