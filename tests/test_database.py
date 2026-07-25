from database import init_db, log_case, get_all_cases, redact_pii
from schema import RemediationResult, RequestType, Urgency


def _make_result(case_id="TS-20260724-abcd", outputs=None):
    return RemediationResult(
        case_id=case_id,
        request_type=RequestType.BILLING_DISPUTE,
        urgency=Urgency.HIGH,
        confidence=0.8,
        steps_executed=["Acknowledge receipt", "Escalate to senior handler"],
        outputs=outputs if outputs is not None else {"draft_response": "hello", "routing": "senior"},
        human_in_loop=False,
        timestamp="2026-07-24T00:00:00",
    )


def test_log_case_round_trip(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    result = _make_result()

    log_case(result, raw_text="I was charged twice", db_path=db_path)
    df = get_all_cases(db_path=db_path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["case_id"] == "TS-20260724-abcd"
    assert row["request_type"] == "Billing/Claim Dispute"
    assert row["urgency"] == "High"
    assert row["confidence"] == 0.8
    assert row["steps_executed"] == ["Acknowledge receipt", "Escalate to senior handler"]
    assert row["outputs"] == {"draft_response": "hello", "routing": "senior"}
    assert bool(row["human_in_loop"]) is False
    assert row["timestamp"] == "2026-07-24T00:00:00"


def test_get_all_cases_empty_db_returns_empty_dataframe(tmp_path):
    db_path = tmp_path / "empty.db"
    init_db(db_path)
    df = get_all_cases(db_path=db_path)
    assert len(df) == 0


def test_redact_pii_masks_email_phone_and_digits():
    text = "reach me a@b.com or 9876543210, member 123456789"
    redacted = redact_pii(text)
    assert "a@b.com" not in redacted
    assert "9876543210" not in redacted
    assert "123456789" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


def test_redact_pii_handles_empty_and_none():
    assert redact_pii("") == ""
    assert redact_pii(None) == ""


def test_logged_case_stores_redacted_text_only(tmp_path):
    db_path = tmp_path / "test2.db"
    init_db(db_path)
    result = _make_result(case_id="TS-20260724-efgh")
    raw_text = "reach me a@b.com or 9876543210, member 123456789"

    log_case(result, raw_text=raw_text, db_path=db_path)
    df = get_all_cases(db_path=db_path)
    stored_text = df.iloc[0]["raw_text_redacted"]

    assert "a@b.com" not in stored_text
    assert "9876543210" not in stored_text
    assert "123456789" not in stored_text
