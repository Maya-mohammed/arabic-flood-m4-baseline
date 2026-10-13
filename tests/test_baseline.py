"""Offline structural tests; these DO NOT verify real Gemini API connectivity or accuracy."""
import json
from types import SimpleNamespace
import pytest
from baseline import classify_tweet, is_consistent, has_allowed_labels


def fake_response(r, num):
    return SimpleNamespace(text=json.dumps(r), usage_metadata=SimpleNamespace(
        prompt_token_count=30, candidates_token_count=10,
        thoughts_token_count=0, total_token_count=40))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        return fake_response(self.replies.pop(0), len(self.calls))


def test_three_calls_and_metrics():
    expected = {"relevance": "Related", "information_type": "Caution and advice"}
    client = FakeClient([expected, expected, expected])
    result = classify_tweet("سيول قوية وتحذير من مجاري الأودية", client=client)
    assert len(client.calls) == result["successful_model_calls"] == result["api_attempts"] == 3
    assert result["relevance"] == "Related"
    assert result["information_type"] == "Caution and advice"
    assert result["total_tokens"] == 120
    assert result["repairs"] == 0
    assert result["initial_consistency_violation"] is False


def test_validator_is_one_way():
    assert is_consistent({"relevance": "Related", "information_type": "Not applicable"})
    assert not is_consistent({"relevance": "Not related", "information_type": "Caution and advice"})
    assert is_consistent({"relevance": "Not related", "information_type": "Not applicable"})


def test_final_inconsistency_triggers_bounded_repair():
    valid = {"relevance": "Related", "information_type": "Not applicable"}
    inconsistent = {"relevance": "Not related", "information_type": "Caution and advice"}
    fixed = {"relevance": "Not related", "information_type": "Not applicable"}
    client = FakeClient([valid, valid, inconsistent, fixed])
    result = classify_tweet("أمطار مستمرة", client=client)
    assert result["repairs"] == 1
    assert result["successful_model_calls"] == 4
    assert is_consistent(result)


def test_empty_input():
    with pytest.raises(ValueError, match="nonempty"):
        classify_tweet(" ", client=FakeClient([]))
