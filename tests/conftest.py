import pytest


def fake_llm(response: str):
    """Returns an llm_call that always yields `response`."""
    def _call(system_prompt: str, user_text: str) -> str:
        return response
    return _call


def sequenced_llm(responses: list[str]):
    """Returns each response in turn (for testing the retry path)."""
    state = {"i": 0}
    def _call(system_prompt: str, user_text: str) -> str:
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r
    return _call
