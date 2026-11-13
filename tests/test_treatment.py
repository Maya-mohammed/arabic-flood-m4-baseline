"""Offline tests for the three-agent treatment and the shared runner. No API calls."""
import csv
import json
import sys
from types import SimpleNamespace

import pytest

import evaluate
import treatment
from labels import INFO_TYPE_DEFINITIONS, RELEVANCE_DEFINITIONS

REL = {"relevance": "Related"}
INFO = {"information_type": "Caution and advice"}
PAIR = {"relevance": "Related", "information_type": "Caution and advice"}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(
            prompt_token_count=30, candidates_token_count=10,
            thoughts_token_count=None, total_token_count=40))


def test_three_calls_and_specialist_pair():
    client = FakeClient([REL, INFO, PAIR])
    r = treatment.classify_tweet("tweet", client=client, seed=42)
    assert len(client.calls) == r["successful_model_calls"] == r["api_attempts"] == 3
    assert (r["relevance"], r["information_type"]) == ("Related", "Caution and advice")
    assert r["initial_prediction"] == PAIR
    assert r["self_review_prediction"] is None
    assert r["initial_consistency_violation"] is False
    assert r["judge_changed_pair"] is False
    assert r["total_tokens"] == 120 and r["reasoning_tokens"] is None


def test_each_specialist_sees_only_its_own_definitions():
    client = FakeClient([REL, INFO, PAIR])
    treatment.classify_tweet("tweet", client=client)
    rel_prompt, info_prompt, judge_prompt = (c[1] for c in client.calls)
    for text in RELEVANCE_DEFINITIONS.values():
        assert text in rel_prompt and text not in info_prompt and text in judge_prompt
    for text in INFO_TYPE_DEFINITIONS.values():
        assert text in info_prompt and text not in rel_prompt and text in judge_prompt
    assert "MUST be Not applicable" in judge_prompt
    assert "MUST be Not applicable" not in rel_prompt + info_prompt
    assert json.dumps(REL) in judge_prompt and json.dumps(INFO) in judge_prompt


def test_specialists_never_see_each_other():
    client = FakeClient([REL, INFO, PAIR])
    treatment.classify_tweet("tweet", client=client)
    info_prompt = client.calls[1][1]
    assert "Relevance Agent:" not in info_prompt and json.dumps(REL) not in info_prompt


def test_same_settings_and_seed_on_every_call():
    client = FakeClient([REL, INFO, PAIR])
    treatment.classify_tweet("tweet", client=client, seed=123)
    for model, _, config in client.calls:
        assert model == treatment.MODEL_NAME
        assert config == {"response_mime_type": "application/json", "seed": 123}


def test_inconsistent_specialist_pair_is_recorded_and_judge_change_detected():
    client = FakeClient([{"relevance": "Not related"}, INFO,
                         {"relevance": "Not related", "information_type": "Not applicable"}])
    r = treatment.classify_tweet("tweet", client=client)
    assert r["initial_consistency_violation"] is True
    assert r["judge_changed_pair"] is True


def test_specialist_bad_output_gets_one_repair():
    client = FakeClient(['{"relevance": "Maybe"}', REL, INFO, PAIR])
    r = treatment.classify_tweet("tweet", client=client)
    assert r["repairs"] == 1 and r["successful_model_calls"] == 4


def test_inconsistent_judge_pair_is_repaired_once_then_fails():
    bad = {"relevance": "Not related", "information_type": "Caution and advice"}
    client = FakeClient([REL, INFO, bad, bad])
    with pytest.raises(ValueError, match="judge"):
        treatment.classify_tweet("tweet", client=client)
    assert len(client.calls) == 4


def test_runner_uses_treatment_and_fills_treatment_columns(monkeypatch):
    def fake(text, seed=None):
        return {**PAIR, "initial_prediction": PAIR, "self_review_prediction": None,
                "initial_consistency_violation": False, "judge_changed_pair": False,
                "latency_seconds": 1.0, "successful_model_calls": 3, "api_attempts": 3,
                "repairs": 0, "input_tokens": 90, "output_tokens": 30,
                "reasoning_tokens": None, "total_tokens": 120}
    monkeypatch.setattr(treatment, "classify_tweet", fake)
    case = {"ID": "1", "tweet_text": "t", "Relevance": "Related", "Information Type": "Caution and advice"}
    row = evaluate.run_case(1, case, 42, "treatment")
    assert row["system"] == "treatment" and row["seed"] == 42
    assert row["exact_match"] is True and row["judge_changed_pair"] is False
    assert row["review_relevance"] == "" and row["error"] == ""


class QuotaError(Exception):
    code = 429


def test_quota_error_stops_instead_of_counting_as_failure(monkeypatch):
    def quota(text, seed=None):
        raise QuotaError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(evaluate, "classify_tweet", quota)
    with pytest.raises(evaluate.QuotaStop):
        evaluate.run_case(1, {"case_id": "X", "tweet": "t"}, None)


def test_resume_continues_without_repeating_cases(tmp_path, monkeypatch):
    cases = tmp_path / "cases.csv"
    with open(cases, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "tweet_text", "Relevance", "Information Type"])
        for i in range(3):
            w.writerow([f"id{i}", f"tweet {i}", "Related", "Caution and advice"])
    out = tmp_path / "seed_42.csv"
    seen = []

    def fake(text, seed=None):
        seen.append(text)
        if text == "tweet 1" and seen.count("tweet 1") == 1:
            raise QuotaError("429")
        return {**PAIR, "initial_prediction": PAIR, "self_review_prediction": PAIR,
                "initial_consistency_violation": False, "latency_seconds": 1.0,
                "successful_model_calls": 3, "api_attempts": 3, "repairs": 0,
                "input_tokens": 90, "output_tokens": 30, "reasoning_tokens": None,
                "total_tokens": 120}
    monkeypatch.setattr(evaluate, "classify_tweet", fake)
    argv = ["evaluate.py", "--cases", str(cases), "--out", str(out), "--seed", "42"]

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stop:
        evaluate.main()
    assert stop.value.code == 2
    assert len(evaluate.read_cases(str(out))) == 1        # quota case not recorded

    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    evaluate.main()
    rows = evaluate.read_cases(str(out))
    assert [r["case_id"] for r in rows] == ["id0", "id1", "id2"]
    assert seen == ["tweet 0", "tweet 1", "tweet 1", "tweet 2"]   # id0 not run again
    assert all(r["error"] == "" for r in rows)
