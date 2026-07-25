import pandas as pd

from config import AVG_HANDLE_MINUTES_HUMAN, LOADED_COST_PER_MIN
from business import compute_business_impact, compute_service_tiers, TIER_MINUTES_SAVED


def _make_df(automated_count, human_count, confidences=None):
    total = automated_count + human_count
    confidences = confidences or [0.9] * total
    return pd.DataFrame(
        {
            "human_in_loop": [False] * automated_count + [True] * human_count,
            "confidence": confidences,
        }
    )


def test_hours_and_cost_saved_match_hand_computed_values():
    df = _make_df(automated_count=7, human_count=3)
    result = compute_business_impact(df, daily_volume=1000)

    expected_minutes = 7 * AVG_HANDLE_MINUTES_HUMAN
    expected_hours = expected_minutes / 60
    expected_cost = expected_minutes * LOADED_COST_PER_MIN

    assert result["automated_cases"] == 7
    assert result["human_cases"] == 3
    assert result["automation_rate"] == 0.7
    assert result["hours_saved"] == expected_hours
    assert result["cost_saved"] == expected_cost


def test_projection_scales_with_daily_volume():
    df = _make_df(automated_count=5, human_count=5)
    result = compute_business_impact(df, daily_volume=1000)

    expected_projected_automated = 1000 * 0.5
    expected_projected_minutes = expected_projected_automated * AVG_HANDLE_MINUTES_HUMAN
    expected_projected_hours = expected_projected_minutes / 60

    assert result["projected_daily_hours_saved"] == expected_projected_hours


def test_empty_dataframe_returns_zeroes_no_crash():
    df = pd.DataFrame({"human_in_loop": [], "confidence": []})
    result = compute_business_impact(df, daily_volume=1000)

    assert result["total_cases"] == 0
    assert result["automation_rate"] == 0.0
    assert result["hours_saved"] == 0.0
    assert result["cost_saved"] == 0.0


def _tier_df():
    # 2 auto-resolved enquiries, 1 AI-handled billing dispute, 1 escalation
    return pd.DataFrame(
        {
            "request_type": [
                "Claim Status/Coverage Enquiry", "Claim Status/Coverage Enquiry",
                "Billing/Claim Dispute", "Complaint/Urgent Escalation",
            ],
            "urgency": ["Low", "Low", "High", "Critical"],
            "human_in_loop": [False, False, True, True],
            "confidence": [0.9, 0.9, 0.9, 0.9],
        }
    )


def test_service_tiers_split_correctly():
    tiers = compute_service_tiers(_tier_df())
    assert tiers["auto_resolved"] == 2
    assert tiers["ai_handled"] == 1        # billing dispute: human approves, not escalated
    assert tiers["escalated"] == 1         # complaint/critical -> escalated
    assert tiers["ai_touched_pct"] == 1.0


def test_service_tiers_weighted_saving_matches_formula():
    tiers = compute_service_tiers(_tier_df())
    expected_minutes = (
        2 * TIER_MINUTES_SAVED["auto_resolved"]
        + 1 * TIER_MINUTES_SAVED["ai_handled"]
        + 1 * TIER_MINUTES_SAVED["escalated"]
    )
    assert tiers["weighted_hours_saved"] == expected_minutes / 60


def test_service_tiers_empty_no_crash():
    df = pd.DataFrame({"request_type": [], "urgency": [], "human_in_loop": [], "confidence": []})
    tiers = compute_service_tiers(df)
    assert tiers["total"] == 0
    assert tiers["auto_resolved"] == 0
    assert tiers["weighted_hours_saved"] == 0.0
