"""Business-impact calculator (Section 7.3).

Turns the audit log into automation-rate / hours-saved / cost-saved figures,
plus a projection at a configurable daily volume.
"""
import pandas as pd

from config import AVG_HANDLE_MINUTES_HUMAN, LOADED_COST_PER_MIN

# Minutes of human effort saved per case, by service tier (documented assumptions).
# Auto-resolved: the AI handles it end-to-end -> full handle time saved.
# AI-handled: the AI classifies, drafts, routes and prepares the case; a human only
#   reviews/approves -> roughly half the handle time saved.
# Escalated: a human owns the case, but the AI still triaged and routed it -> a small
#   saving from not having to read and sort it manually.
TIER_MINUTES_SAVED = {
    "auto_resolved": AVG_HANDLE_MINUTES_HUMAN,
    "ai_handled": AVG_HANDLE_MINUTES_HUMAN * 0.5,
    "escalated": 1.0,
}


def compute_service_tiers(df: pd.DataFrame, daily_volume: int = 1000) -> dict:
    """Break the audit log into three honest service tiers instead of a binary
    automated/not-automated split, and compute a weighted time/cost saving.

    - auto_resolved: fully automated (no human in the loop)
    - ai_handled:    AI did the work, a human approves (human-in-loop, not escalated)
    - escalated:     high-risk (complaint/urgent or critical) -> owned by a human
    """
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "auto_resolved": 0, "ai_handled": 0, "escalated": 0,
            "auto_resolved_pct": 0.0, "ai_handled_pct": 0.0, "escalated_pct": 0.0,
            "ai_touched_pct": 0.0,
            "weighted_hours_saved": 0.0, "weighted_cost_saved": 0.0,
            "projected_daily_hours_saved": 0.0, "projected_daily_cost_saved": 0.0,
        }

    human = df["human_in_loop"].astype(bool)
    is_escalation = df["request_type"].astype(str).str.contains("Escalation", case=False, na=False)
    is_critical = df["urgency"].astype(str).str.lower().eq("critical")
    escalated_mask = human & (is_escalation | is_critical)
    auto_mask = ~human
    ai_handled_mask = human & ~escalated_mask

    auto_resolved = int(auto_mask.sum())
    ai_handled = int(ai_handled_mask.sum())
    escalated = int(escalated_mask.sum())

    minutes = (
        auto_resolved * TIER_MINUTES_SAVED["auto_resolved"]
        + ai_handled * TIER_MINUTES_SAVED["ai_handled"]
        + escalated * TIER_MINUTES_SAVED["escalated"]
    )
    hours_saved = minutes / 60
    cost_saved = minutes * LOADED_COST_PER_MIN

    per_case_minutes = minutes / total
    projected_minutes = daily_volume * per_case_minutes
    projected_hours = projected_minutes / 60
    projected_cost = projected_minutes * LOADED_COST_PER_MIN

    return {
        "total": total,
        "auto_resolved": auto_resolved,
        "ai_handled": ai_handled,
        "escalated": escalated,
        "auto_resolved_pct": auto_resolved / total,
        "ai_handled_pct": ai_handled / total,
        "escalated_pct": escalated / total,
        "ai_touched_pct": 1.0,  # every case is triaged by the system
        "weighted_hours_saved": hours_saved,
        "weighted_cost_saved": cost_saved,
        "projected_daily_hours_saved": projected_hours,
        "projected_daily_cost_saved": projected_cost,
    }


def compute_business_impact(df: pd.DataFrame, daily_volume: int = 1000) -> dict:
    total = len(df)
    if total == 0:
        return {
            "total_cases": 0,
            "automated_cases": 0,
            "human_cases": 0,
            "automation_rate": 0.0,
            "avg_confidence": 0.0,
            "minutes_saved": 0.0,
            "hours_saved": 0.0,
            "cost_saved": 0.0,
            "projected_daily_automated": 0.0,
            "projected_daily_hours_saved": 0.0,
            "projected_daily_cost_saved": 0.0,
        }

    automated_count = int((df["human_in_loop"] == False).sum())
    human_count = int((df["human_in_loop"] == True).sum())
    automation_rate = automated_count / total
    avg_confidence = float(df["confidence"].mean())

    minutes_saved = automated_count * AVG_HANDLE_MINUTES_HUMAN
    hours_saved = minutes_saved / 60
    cost_saved = minutes_saved * LOADED_COST_PER_MIN

    projected_automated = daily_volume * automation_rate
    projected_minutes_saved = projected_automated * AVG_HANDLE_MINUTES_HUMAN
    projected_hours_saved = projected_minutes_saved / 60
    projected_cost_saved = projected_minutes_saved * LOADED_COST_PER_MIN

    return {
        "total_cases": total,
        "automated_cases": automated_count,
        "human_cases": human_count,
        "automation_rate": automation_rate,
        "avg_confidence": avg_confidence,
        "minutes_saved": minutes_saved,
        "hours_saved": hours_saved,
        "cost_saved": cost_saved,
        "projected_daily_automated": projected_automated,
        "projected_daily_hours_saved": projected_hours_saved,
        "projected_daily_cost_saved": projected_cost_saved,
    }
