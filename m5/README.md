# M5: Evaluation Harness

This folder holds the measurement instrument for the project: a frozen golden set with gold labels, automated scoring, and the single-agent baseline scored across three seeds. The multi-agent treatment (M6) will be scored on the same cases, with the same seeds and the same scoring code.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maya-mohammed/arabic-flood-m4-baseline/blob/main/M5_evaluation_colab.ipynb)

## Files

| File | Purpose |
|---|---|
| `m5/golden_set_ids.csv` | The 140 golden-set cases: tweet ID, gold labels and flood event (no tweet text) |
| `m5/runs/baseline/seed_<n>.csv` | Baseline output for every case, one file per seed (42, 123, 2026), as written by `evaluate.py` |
| `m5/runs/baseline_pilot/seed_42_pilot.csv` | Five-case pilot run made before the full runs, used in the seed check |
| `m5/results/baseline/` | Scoring output: `summary.csv`, `per_class_f1.csv` and confusion matrices per seed and for all seeds |
| `make_golden_set.py` | Draws the golden set from the reserved split and checks it against the frozen file |
| `score_runs.py` | Scores saved runs and reports each metric per seed, with mean and standard deviation |
| `M5_evaluation_colab.ipynb` | Runs the tests and the scoring in Colab; the API runs are optional and off by default |
| `tests/test_score_runs.py` | Offline tests for the scoring rules |

## How to run

Scoring needs no API key and no API calls:

```bash
python -m pytest -q tests
python score_runs.py --condition baseline \
    --runs m5/runs/baseline/seed_42.csv m5/runs/baseline/seed_123.csv m5/runs/baseline/seed_2026.csv \
    --out m5/results/baseline --price-input 0.30 --price-output 2.50
```

To rebuild the golden set with tweet text (needed only to repeat the API runs), download the corpus as described in the main README, then:

```bash
python prepare_data.py --csv private_data/ArabicFloods.csv
python make_golden_set.py
python evaluate.py --cases private_data/golden_set.csv --out m5/runs/rerun/seed_42.csv --seed 42 --pause 10
```

`make_golden_set.py` stops without writing anything if its sample differs from `m5/golden_set_ids.csv`. The file with tweet text goes to `private_data/`, which Git ignores.

## Golden set

**Source.** The cases come from the reserved split made by `prepare_data.py` in M4 (3,223 tweets). This split was not used while building the baseline. None of the 140 IDs is in the development split.

**Ground truth.** The gold labels are the original human labels from the Arabic Twitter Corpus for Flood Detection (Alharbi & Lee, 2019). We did not relabel any tweet.

**Sampling.** 20 tweets per Information Type, drawn at random within each class (`random_state=2026`), giving 7 x 20 = 140 cases.

| Joint label | Cases |
|---|---:|
| Related / Affected individuals | 20 |
| Related / Infrastructure and utilities damage | 20 |
| Related / Donations and volunteering | 20 |
| Related / Caution and advice | 20 |
| Related / Sympathy and emotional support | 20 |
| Related / Other useful information | 20 |
| Not related / Not applicable | 16 |
| Related / Not applicable | 4 |

All four floods are present: Al Lith 45, Kuwait 35, Jordan 30, Qurayyat 30.

**Why this design and size.**

1. *Every class can be measured.* In a sample of 140 drawn in proportion to the reserved split, Donations and volunteering would get about 5 tweets and Other useful information about 8. That is too few to read a per-class F1. Taking 20 per Information Type gives every class the same support.
2. *Repeated runs.* Each case is run three times per condition (three seeds), with three planned model calls per run. 140 cases therefore means 1,260 planned calls per condition and 2,520 for the whole comparison. On the free tier, one seed of 140 cases used about one day of request quota.
3. *Related work.* The studies we reviewed in M3 test on much larger sets, for example 12,446 tweets in Kawarith [2] and a 15,160-tweet HumAID test split [4]. We cannot run sets of that size three times per condition on the free tier, so we use a smaller set and list its size as a limitation. The same studies shaped what the set covers: [1] found that accuracy drops on an unseen flood, and [4] found floods and fine-grained information types the hardest, so the set includes all four floods and all seven Information Types.

**What this design does not give.** The set is balanced by Information Type, not by the natural distribution of tweets, so its accuracy is not an estimate for the whole corpus. Because sampling was done within Information Type only, the Relevance split was left to chance: 16 Not related tweets (11%, against 25% in the reserved split) and only 4 Related / Not applicable tweets. Floods are also uneven within classes; for example, no Jordan tweet is labelled Caution and advice.

## Fixed settings

- Model: `gemini-3.5-flash-lite`, default settings, JSON output. `temperature`, `top_p` and `top_k` are not sent; Google's documentation for this model states that they are ignored. The thinking level stays at the model default, which Google lists as `minimal`.
- `baseline.py`, `labels.py` and `evaluate.py` are unchanged since the M4 submission (last code change: 13 October 2026).
- Seeds: 42, 123 and 2026. M6 uses the same three seeds.
- Prompts: our evaluation plan allowed prompt changes on the development split. We made none to the baseline after M4. To keep the comparison controlled, the M6 treatment follows the same rule: its prompts are written once and checked on a few development tweets for format and pipeline errors only. Neither system's prompts are changed after seeing golden-set results.

## Seed check

`evaluate.py` sends the seed with every request, but Google's page for this model does not mention a seed parameter, so we tested whether it is applied on our own runs.

- 23 of the 140 cases received a different final label pair under the three seeds.
- These 23 cases were run twice more with seed 42. Both reruns gave the same labels as each other (23 of 23) and as the saved seed 42 run (23 of 23). The rerun predictions were printed as counts in the notebook and not saved as files.
- The five-case pilot, run in a separate session before the full seed 42 run, matches that run at every stage and in output token counts (5 of 5).

In our runs, the same seed repeated the same output and different seeds changed some outputs, so the three runs are seeded repeats rather than unrelated runs. We did not test runs without a seed, and the result holds for this model during our runs only; Google can update a model behind the same name.

## Scoring rules

`score_runs.py` first checks that each run covers exactly the 140 golden-set cases, once each, with the same gold labels as `m5/golden_set_ids.csv`, and stops otherwise. It then recomputes every metric from the labels.

- **Joint exact match** (primary, end to end): both labels equal the gold labels.
- **Valid output**: the request returned, both labels are in the closed sets, and the pair follows the rule Not related => Not applicable. An invalid case counts as wrong in every accuracy and F1 measure; it is not dropped.
- **Macro-F1, Relevance and Information Type** (component level): F1 per class, averaged over the closed label set (2 and 7 classes). An invalid prediction is scored as a separate `NO_VALID_OUTPUT` label, which lowers recall of the gold class.
- **Per-class F1 and confusion matrices** for both labels.
- **First-output consistency violation rate** (trajectory level): share of cases whose first complete output, the initial call after at most one format repair, has Not related with any Information Type other than Not applicable.
- **Efficiency**: model calls, API attempts, repairs, input, output and reasoning tokens, and end-to-end latency, averaged over cases that returned a result.
- **Estimated cost per case**: input tokens x $0.30 and output tokens x $2.50 per million, the paid Standard prices Google lists for `gemini-3.5-flash-lite`. The runs used the free tier, so the actual cost was zero.

Every metric is reported per seed and as mean and sample standard deviation across the three seeds.

## Baseline results

140 cases x 3 seeds. Mean ± standard deviation across seeds.

| Metric | Mean ± SD | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---:|---:|---:|
| Joint exact match | 56.43% ± 1.89 | 54.29% | 57.86% | 57.14% |
| Valid output rate | 100% ± 0.00 | 100% | 100% | 100% |
| Relevance macro-F1 | 80.13% ± 1.44 | 78.68% | 80.16% | 81.55% |
| Information Type macro-F1 | 57.89% ± 2.08 | 55.50% | 59.33% | 58.83% |
| First-output consistency violations | 0% ± 0.00 | 0 of 140 | 0 of 140 | 0 of 140 |
| Model calls per case | 3.00 ± 0.00 | 3.00 | 3.00 | 3.00 |
| API attempts per case | 3.03 ± 0.01 | 3.01 | 3.04 | 3.04 |
| Repairs per case | 0.00 ± 0.00 | 0 | 0 | 0 |
| Input tokens per case | 1,091.6 ± 0.1 | 1,091.5 | 1,091.7 | 1,091.6 |
| Output tokens per case | 51.4 ± 0.4 | 51.0 | 51.8 | 51.4 |
| Reasoning tokens per case | not returned by the API | | | |
| Latency per case (s) | 17.11 ± 19.98 | 39.85 | 2.33 | 9.16 |
| Estimated cost per case (USD) | 0.000456 ± 0.000001 | 0.000455 | 0.000457 | 0.000456 |

No request failed in any run.

**Per-class F1** (mean ± SD across seeds):

| Label | F1 |
|---|---|
| Relevance: Related | 0.963 ± 0.003 |
| Relevance: Not related | 0.639 ± 0.029 |
| Affected individuals | 0.713 ± 0.009 |
| Infrastructure and utilities damage | 0.479 ± 0.086 |
| Donations and volunteering | 0.297 ± 0.040 |
| Caution and advice | 0.700 ± 0.047 |
| Sympathy and emotional support | 0.814 ± 0.012 |
| Other useful information | 0.416 ± 0.022 |
| Not applicable | 0.634 ± 0.043 |

**Main error patterns** (counts added over the three seeds, from `m5/results/baseline/`):

- *Not related tweets marked Related.* 23 of 48 Not related cases were labelled Related; only 5 of 372 Related cases were labelled Not related.
- *Too many tweets labelled Other useful information.* Out of 60 cases per class, 30 Donations and volunteering, 27 Infrastructure and utilities damage and 22 Caution and advice tweets were labelled Other useful information.

Both patterns are the ones seen on the small M4 test set, now on a larger and frozen set.

## Limitations

- The golden set is small and class-balanced, so its scores do not describe the natural mix of tweets. Not related and Related / Not applicable cases are few.
- Latency depends on the state of the Gemini service on the day of the run more than on the system: mean latency per case was 39.85 s for seed 42 against 2.33 s and 9.16 s for the other seeds, with the same code and cases. Latency includes waits after rate-limit or server errors inside a case, but not the pause between cases.
- The API did not return reasoning-token counts, so they are reported as not returned rather than as zero.
- Cost is an estimate at paid prices; the runs used the free tier.
- The team-written edge cases from M4 are not part of the golden set and are reported separately.

## References

Numbered as in our M3 related-work review.

[1] A. Alharbi and M. Lee, "Crisis detection from Arabic tweets," in Proc. 3rd Workshop Arabic Corpus Linguistics, Cardiff, U.K., Jul. 2019, pp. 72–79.

[2] A. Alharbi and M. Lee, "Kawarith: An Arabic Twitter corpus for crisis events," in Proc. 6th Arabic Natural Language Processing Workshop, Kyiv, Ukraine (Virtual), Apr. 2021, pp. 42–52.

[4] M. Imran, A. W. Ziaullah, K. Chen, and F. Ofli, "Evaluating robustness of LLMs on crisis-related microblogs across events, information types, and linguistic features," in Proc. ACM Web Conf. (WWW '25), 2025, doi: 10.1145/3696410.3714511.

Model information and prices: Google, "What's new in Gemini 3.6 Flash and 3.5 Flash-Lite," https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6
