"""Offline tests. They do NOT call Gemini, so they check logic, not accuracy."""
import json
from types import SimpleNamespace

import pytest

import baseline
import evaluate
from baseline import classify_tweet, is_consistent
from labels import INFO_TYPE_LABELS, RELEVANCE_LABELS, label_guide


def fake_response(r):
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
        return fake_response(self.replies.pop(0))


def test_three_calls_and_metrics():
    expected = {"relevance": "Related", "information_type": "Caution and advice"}
    client = FakeClient([expected, expected, expected])
    result = classify_tweet("سيول قوية وتحذير من مجاري الأودية", client=client)
    assert len(client.calls) == result["successful_model_calls"] == result["api_attempts"] == 3
    assert (result["relevance"], result["information_type"]) == ("Related", "Caution and advice")
    assert result["total_tokens"] == 120
    assert result["repairs"] == 0
    assert result["initial_consistency_violation"] is False


def test_stage_predictions_are_returned():
    first = {"relevance": "Related", "information_type": "Caution and advice"}
    later = {"relevance": "Related", "information_type": "Affected individuals"}
    result = classify_tweet("tweet", client=FakeClient([first, later, later]))
    assert result["initial_prediction"] == first
    assert result["self_review_prediction"] == later
    assert [s["stage"] for s in result["stages"]] == ["initial", "self_review", "final"]


def test_prompts_contain_every_label_definition_and_no_temperature():
    pair = {"relevance": "Related", "information_type": "Not applicable"}
    client = FakeClient([pair, pair, pair])
    classify_tweet("tweet", client=client)
    for _, prompt, config in client.calls:
        assert label_guide() in prompt
        assert "temperature" not in config
    for label in RELEVANCE_LABELS + INFO_TYPE_LABELS:
        assert label in label_guide()


def test_validator_is_one_way():
    assert is_consistent({"relevance": "Related", "information_type": "Not applicable"})
    assert not is_consistent({"relevance": "Not related", "information_type": "Caution and advice"})
    assert is_consistent({"relevance": "Not related", "information_type": "Not applicable"})


def test_initial_violation_is_recorded():
    bad = {"relevance": "Not related", "information_type": "Caution and advice"}
    ok = {"relevance": "Not related", "information_type": "Not applicable"}
    result = classify_tweet("tweet", client=FakeClient([bad, ok, ok]))
    assert result["initial_consistency_violation"] is True
    assert result["repairs"] == 0


def test_final_inconsistency_triggers_bounded_repair():
    valid = {"relevance": "Related", "information_type": "Not applicable"}
    inconsistent = {"relevance": "Not related", "information_type": "Caution and advice"}
    fixed = {"relevance": "Not related", "information_type": "Not applicable"}
    result = classify_tweet("أمطار مستمرة", client=FakeClient([valid, valid, inconsistent, fixed]))
    assert result["repairs"] == 1
    assert result["successful_model_calls"] == 4
    assert is_consistent(result)


def test_repair_is_bounded():
    valid = {"relevance": "Related", "information_type": "Not applicable"}
    bad = {"relevance": "Not related", "information_type": "Caution and advice"}
    with pytest.raises(ValueError, match="bounded repair"):
        classify_tweet("tweet", client=FakeClient([valid, valid, bad, bad]))


def test_empty_input():
    with pytest.raises(ValueError, match="nonempty"):
        classify_tweet(" ", client=FakeClient([]))


def test_failed_case_counts_as_wrong(monkeypatch):
    def boom(text, seed=None):
        raise RuntimeError("503 UNAVAILABLE")
    monkeypatch.setattr(evaluate, "classify_tweet", boom)
    row = evaluate.run_case(1, {"case_id": "X", "tweet": "t", "gold_relevance": "Related",
                                "gold_information_type": "Caution and advice"}, None)
    assert row["exact_match"] is False and row["valid_final_pair"] is False
    assert "RuntimeError" in row["error"]
    assert "0/1" in evaluate.summary([row])
