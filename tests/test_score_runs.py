"""Offline tests for score_runs.py, using small hand-made runs."""
import math

import pandas as pd
import pytest

import score_runs as sr

GOLD = [
    ("A", "Related", "Affected individuals"),
    ("B", "Related", "Caution and advice"),
    ("C", "Not related", "Not applicable"),
    ("D", "Related", "Not applicable"),
]


def golden():
    return pd.DataFrame(GOLD, columns=["ID", "Relevance", "Information Type"]).set_index("ID")


def run_rows(preds, errors=None):
    """preds: list of (relevance, information_type) in GOLD order."""
    errors = errors or [""] * len(GOLD)
    rows = []
    for (cid, g_rel, g_info), (p_rel, p_info), err in zip(GOLD, preds, errors):
        failed = bool(err)
        rows.append({
            "case_id": cid, "gold_relevance": g_rel, "gold_information_type": g_info,
            "initial_relevance": "" if failed else p_rel,
            "initial_information_type": "" if failed else p_info,
            "pred_relevance": "" if failed else p_rel,
            "pred_information_type": "" if failed else p_info,
            "exact_match": str(not failed and (p_rel, p_info) == (g_rel, g_info)),
            "initial_violation": "False", "error": err,
            "model_calls": "" if failed else "3", "api_attempts": "" if failed else "3",
            "repairs": "" if failed else "0", "input_tokens": "" if failed else "1000",
            "output_tokens": "" if failed else "50", "reasoning_tokens": "",
            "total_tokens": "" if failed else "1050", "latency_seconds": "" if failed else "2.0",
        })
    return pd.DataFrame(rows).set_index("case_id")


def test_all_correct():
    m = sr.score_run(run_rows([(r, i) for _, r, i in GOLD]), None)["metrics"]
    assert m["joint_exact_match"] == 1 and m["valid_output_rate"] == 1
    assert m["relevance_macro_f1"] == 1


def test_failed_request_counts_as_wrong_and_invalid():
    preds = [(r, i) for _, r, i in GOLD]
    m = sr.score_run(run_rows(preds, errors=["", "", "", "ServerError: 503"]), None)["metrics"]
    assert m["joint_exact_match"] == 0.75
    assert m["valid_output_rate"] == 0.75
    assert m["failed_requests"] == 1
    assert m["mean_model_calls"] == 3          # usage means use returned cases only
    assert m["cases_with_first_output"] == 3


def test_inconsistent_final_pair_is_invalid():
    preds = [(r, i) for _, r, i in GOLD]
    preds[2] = ("Not related", "Caution and advice")   # breaks Not related => Not applicable
    m = sr.score_run(run_rows(preds), None)["metrics"]
    assert m["valid_output_rate"] == 0.75 and m["joint_exact_match"] == 0.75


def test_macro_f1_known_value():
    # Relevance: gold R,R,N,R; predicted R,R,R,R.
    # Related F1 = 2*3/(2*3+1+0) = 6/7; Not related F1 = 0; macro = 3/7.
    preds = [("Related", "Affected individuals"), ("Related", "Caution and advice"),
             ("Related", "Not applicable"), ("Related", "Not applicable")]
    m = sr.score_run(run_rows(preds), None)["metrics"]
    assert math.isclose(m["relevance_macro_f1"], 3 / 7)


def test_cost_uses_given_prices():
    m = sr.score_run(run_rows([(r, i) for _, r, i in GOLD]), (0.10, 0.40))["metrics"]
    assert math.isclose(m["mean_cost_usd"], (1000 * 0.10 + 50 * 0.40) / 1e6)


def test_reasoning_tokens_not_returned_is_nan_not_zero():
    m = sr.score_run(run_rows([(r, i) for _, r, i in GOLD]), None)["metrics"]
    assert math.isnan(m["mean_reasoning_tokens"])


def test_run_with_missing_case_is_rejected(tmp_path):
    path = tmp_path / "seed_1.csv"
    run_rows([(r, i) for _, r, i in GOLD]).iloc[:3].reset_index().to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not match"):
        sr.load_run(path, golden())


def test_run_with_different_gold_label_is_rejected(tmp_path):
    path = tmp_path / "seed_1.csv"
    rows = run_rows([(r, i) for _, r, i in GOLD]).reset_index()
    rows.loc[0, "gold_information_type"] = "Caution and advice"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="gold labels differ"):
        sr.load_run(path, golden())


def test_seed_is_read_from_file_name(tmp_path):
    assert sr.seed_from_path(tmp_path / "seed_2026.csv") == 2026
    with pytest.raises(ValueError):
        sr.seed_from_path(tmp_path / "results.csv")


def test_treatment_judge_measures():
    # Specialist pairs (initial) vs final pairs, against GOLD:
    # A: pre wrong, final right  -> helpful override
    # B: pre right, final wrong  -> harmful override
    # C: pre = final = right     -> no override
    # D: pre wrong, final wrong (changed) -> override, neither helpful nor harmful
    run = run_rows([(r, i) for _, r, i in GOLD])
    run["system"] = "treatment"
    run.loc["A", ["initial_relevance", "initial_information_type"]] = ["Related", "Caution and advice"]
    run.loc["B", ["pred_relevance", "pred_information_type"]] = ["Related", "Other useful information"]
    run.loc["D", ["initial_relevance", "initial_information_type"]] = ["Not related", "Caution and advice"]
    run.loc["D", ["pred_relevance", "pred_information_type"]] = ["Not related", "Not applicable"]
    run.loc["D", "initial_violation"] = "True"
    m = sr.score_run(run, None)["metrics"]
    assert m["judge_override_rate"] == 0.75
    assert m["helpful_override_rate"] == 0.25 and m["harmful_override_rate"] == 0.25
    assert m["first_output_violation_rate"] == 0.25


def test_baseline_runs_get_no_judge_measures():
    m = sr.score_run(run_rows([(r, i) for _, r, i in GOLD]), None)["metrics"]
    assert "judge_override_rate" not in m
