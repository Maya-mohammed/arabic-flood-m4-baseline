# M6: Experimental Condition and Results

This folder holds the three-agent treatment runs and the comparison with the single-agent baseline from M5. Both conditions were run on the same 140 golden-set cases with the same three seeds (42, 123, 2026) and scored with the same code.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maya-mohammed/arabic-flood-m4-baseline/blob/main/M6_treatment_colab.ipynb)

## Files

| File | Purpose |
|---|---|
| `treatment.py` | The treatment: Relevance Agent, Information-Type Agent and Judge |
| `evaluate.py` | Runs either system on a labelled CSV (`--system baseline` or `--system treatment`); can resume a run stopped by the API quota |
| `score_runs.py` | Scores saved runs per seed; adds the Judge measures for treatment runs |
| `compare_runs.py` | Compares the two conditions seed by seed and writes the tables below |
| `M6_treatment_colab.ipynb` | Runs the checks, the treatment runs and the scoring in Colab |
| `tests/test_treatment.py` | Offline tests for the treatment and the runner |
| `m6/runs/treatment/seed_<n>.csv` | Treatment output for every case, one file per seed |
| `m6/checks/` | The two checks made before the main runs (see "Controlled comparison") |
| `m6/results/treatment/` | Treatment scores: summary, per-class F1, confusion matrices |
| `m6/results/comparison/` | Baseline versus treatment: `comparison.csv`, `comparison.md`, `per_class_f1.csv`, `paired_outcomes.csv`, `error_breakdown.csv` |

## How to run

Scoring and comparison need no API key:

```bash
python -m pytest -q tests
python score_runs.py --condition treatment \
    --runs m6/runs/treatment/seed_42.csv m6/runs/treatment/seed_123.csv m6/runs/treatment/seed_2026.csv \
    --out m6/results/treatment --price-input 0.30 --price-output 2.50
python compare_runs.py \
    --baseline m5/runs/baseline/seed_42.csv m5/runs/baseline/seed_123.csv m5/runs/baseline/seed_2026.csv \
    --treatment m6/runs/treatment/seed_42.csv m6/runs/treatment/seed_123.csv m6/runs/treatment/seed_2026.csv \
    --out m6/results/comparison --price-input 0.30 --price-output 2.50
```

To repeat the treatment runs, use the notebook, or rebuild the golden set as described in `m5/README.md` and run:

```bash
python evaluate.py --system treatment --cases private_data/golden_set.csv \
    --out m6/runs/rerun/seed_42.csv --seed 42 --pause 10 --resume
```

## Treatment design

**Agents.**

| Agent | Sees | Returns |
|---|---|---|
| Relevance Agent | The tweet and the two Relevance labels with their definitions | `{"relevance": ...}` |
| Information-Type Agent | The tweet and the seven Information Type labels with their definitions | `{"information_type": ...}` |
| Judge | The tweet, all labels and definitions, the one-way rule, and both specialist outputs | `{"relevance": ..., "information_type": ...}` |

All definitions are read from `labels.py`, the same file the baseline uses. The specialists do not see the one-way rule, because each predicts only one field; the Judge and the validator apply it.

**Topology.** Star: the two specialists work on the same tweet independently and never see each other's output. Both report to the Judge.

**Protocol.** One round. The specialists are called one after the other, then the Judge. There is no negotiation and no revision of specialist outputs. Three planned model calls per tweet, the same as the baseline.

**Termination rule.** A case ends when the Judge returns a valid label pair. Each stage may get one repair request if its output is malformed, uses a label outside the closed set, or, for the Judge, breaks the rule Not related => Not applicable. If any stage is still invalid after its repair, the case ends as failed and counts as wrong, as in the baseline.

**Why this design factor.** The baseline asks one generalist agent to predict both labels together and then review itself. The treatment splits the task by role. The studies we reviewed in M3 point to this question without testing it: [2] suggests filtering irrelevant tweets before information-type classification, and [3] asks whether combining two tasks in one prompt affects results. [5] also cites earlier evidence that gains from multi-agent systems over a single agent are often small. The specialists see only their own labels so that each role is focused on one decision; the Judge is the only agent that sees both labels and the rule, so it is the place where the two decisions are reconciled.

## Controlled comparison

- **Same cases and seeds.** Both conditions use the 140 cases in `m5/golden_set_ids.csv` and seeds 42, 123 and 2026. `compare_runs.py` stops if the two conditions do not cover the same seeds, and `score_runs.py` stops if a run does not cover exactly the golden-set cases with the same gold labels.
- **Same model and settings.** `gemini-3.5-flash-lite`, default settings, JSON output, the seed sent with every call. `treatment.py` imports the model name, retry and repair limits, client and final-pair validator from `baseline.py`.
- **Same scoring.** Both conditions are scored with `score_runs.py`; a failed or invalid case counts as wrong in both.
- **Unchanged baseline.** `baseline.py` and `labels.py` were not changed in M6. `evaluate.py` gained the `--system` and `--resume` options and the quota stop; an offline test with the same simulated model results, including failed requests, gave the same baseline output before and after the change.
- **Account and date check.** The treatment runs used a different Google account and later days than the baseline runs. Before the main runs, the baseline was run again on the first 5 golden-set cases with seed 42 on the new account. All 5 matched the saved M5 run at every stage and in token counts (`m6/checks/check_baseline_seed_42.csv`).
- **No prompt tuning.** As for the baseline, the treatment prompts were written once. They were checked on the 8 development cases used in M4 for format and pipeline errors only: 8 of 8 valid pairs, no repairs, no errors (`m6/checks/check_treatment_dev.csv`). `treatment.py`, `baseline.py`, `labels.py` and `evaluate.py` were then frozen at commit `2951fb3`, before the golden-set runs, and were not changed after seeing the results. Only the comparison script and the notebook were added later.
- **Quota stops.** The runs for seeds 123 and 2026 were stopped by the daily free-tier quota and resumed later with `--resume`. Each file holds every case exactly once, in golden-set order, with no failed requests.

## Results

Same 140 cases and seeds 42, 123 and 2026 for both conditions. Mean ± sample standard deviation across seeds. Difference = treatment minus baseline.

| Metric | Baseline | Treatment | Difference |
|---|---|---|---|
| Joint exact match | 56.43% ± 1.89 | 54.05% ± 0.41 | -2.38 points |
| Valid output rate | 100% ± 0.00 | 100% ± 0.00 | 0 |
| Failed requests | 0 | 0 | 0 |
| Relevance macro-F1 | 80.13% ± 1.44 | 80.05% ± 3.19 | -0.09 points |
| Information Type macro-F1 | 57.89% ± 2.08 | 53.86% ± 0.83 | -4.03 points |
| First-output consistency violations | 0% ± 0.00 | 3.81% ± 1.09 | +3.81 points |
| Judge override rate | n/a | 5.71% ± 0.71 | n/a |
| Helpful override rate | n/a | 1.90% ± 0.82 | n/a |
| Harmful override rate | n/a | 0.71% ± 0.00 | n/a |
| Model calls per case | 3.00 | 3.00 | 0 |
| API attempts per case | 3.03 ± 0.01 | 3.08 ± 0.07 | +0.05 |
| Repairs per case | 0 | 0 | 0 |
| Input tokens per case | 1,091.6 ± 0.1 | 812.5 ± 0.0 | -279.1 |
| Output tokens per case | 51.4 ± 0.4 | 34.1 ± 0.0 | -17.3 |
| Total tokens per case | 1,143.0 ± 0.4 | 846.6 ± 0.1 | -296.4 |
| Reasoning tokens per case | not returned | not returned | n/a |
| Latency per case (s) | 17.11 ± 19.98 | 4.03 ± 1.84 | see limitations |
| Estimated cost per case (USD) | 0.000456 ± 0.000001 | 0.000329 ± 0.000000 | -0.000127 |

Joint exact match per seed: baseline 54.29%, 57.86%, 57.14%; treatment 54.29%, 54.29%, 53.57% (seeds 42, 123, 2026).

For the treatment, the first complete output is the pair formed by the two specialists before the Judge, so the first-output violation rate is also the specialist-pair violation rate. The Judge measures are shares of all cases. Since no case failed at the Judge step, helpful minus harmful (1.90 - 0.71 = 1.19 points) is the gain in joint exact match that the Judge added over the specialist pair.

Cost uses the paid Standard prices Google lists for `gemini-3.5-flash-lite` (USD 0.30 per 1M input tokens, USD 2.50 per 1M output tokens). Both conditions ran on the free tier, so the actual cost was zero.

**Paired outcomes** (same case, same seed):

| Seed | Both correct | Baseline only | Treatment only | Both wrong |
|---|---:|---:|---:|---:|
| 42 | 66 | 10 | 10 | 54 |
| 123 | 70 | 11 | 6 | 53 |
| 2026 | 67 | 13 | 8 | 52 |

**Per-class F1** (mean across seeds):

| Label | Baseline | Treatment | Difference |
|---|---:|---:|---:|
| Relevance: Related | 0.963 | 0.951 | -0.012 |
| Relevance: Not related | 0.639 | 0.650 | +0.010 |
| Affected individuals | 0.713 | 0.667 | -0.046 |
| Infrastructure and utilities damage | 0.479 | 0.526 | +0.047 |
| Donations and volunteering | 0.297 | 0.285 | -0.012 |
| Caution and advice | 0.700 | 0.684 | -0.016 |
| Sympathy and emotional support | 0.814 | 0.769 | -0.044 |
| Other useful information | 0.416 | 0.272 | -0.144 |
| Not applicable | 0.634 | 0.566 | -0.068 |

**Answer to the research question.** The three-agent treatment did not reach higher joint exact-match accuracy than the single-agent baseline. On this golden set it was 2.38 points lower on average. When the two systems disagreed on a case, the baseline was the one that was right more often in two of the three seeds, and they were even in the third. With three seeds and 140 cases, we read this as no evidence for H1, not as proof that the treatment is worse. The treatment used about 26% fewer tokens and cost about 28% less per tweet.

## Failure analysis

Counts below are added over the three seeds (420 case runs per condition) and come from `m6/results/comparison/` and the run files.

**1. Relevance: the two systems fail in opposite directions.** Relevance macro-F1 is almost the same (80.1%), but the errors are different.

| Error | Baseline | Treatment |
|---|---:|---:|
| Not related tweet labelled Related | 23 | 15 |
| Related tweet labelled Not related | 5 | 21 |

The baseline's main relevance error, seen since M4, is marking tweets about something else as flood tweets. The treatment makes this error less often, but its Relevance Agent marks real flood tweets as Not related more often. All 21 of these treatment errors started with the Relevance Agent. Ten of them are tweets whose gold label is Other useful information, and five are Sympathy and emotional support. The Other useful information cases mention the flood indirectly: questions about a school suspension and an exam postponement in Al Lith, a media figure addressing members of parliament, and a news item about the Al Lith dam project. A likely reason is that the Relevance Agent decides from the Relevance definitions alone, without the Information Type labels that describe the kinds of content a flood tweet can carry. In both systems, most Not related tweets labelled Related were labelled Sympathy and emotional support or Not applicable. These tweets comment on a flood or a flooded place without reporting on it: a short poem about winter with a prayer for the dead, a question about flags at half-mast, a joke, criticism of officials. Some are close calls even for a human reader; one consists only of the words "Al Lith floods" and a link, and its gold label is Not related.

**2. The Judge mostly sides with the Relevance Agent.** The specialists disagreed (Relevance Agent: Not related, Information-Type Agent: a real information type) in 16 case runs. In 13 of these the gold label was Related, so the Information-Type Agent was right about relevance. The Judge kept Not related in 14 of the 16 and switched to Related in 2. Overall the Judge changed the specialist pair in 24 case runs (5.71%): 8 changes fixed a wrong pair, 3 broke a correct pair, and the rest changed one wrong pair into another. The Judge therefore added 1.19 points, but it did not use the disagreement between specialists as a signal that the Relevance Agent might be wrong.

**3. Information Type: different errors, not fewer.** The baseline over-uses Other useful information: 100 wrong labels of that kind, against 48 for the treatment. The treatment instead over-uses Not applicable (59 against 24) and Affected individuals (42 against 32). Its most common confusions are Donations and volunteering labelled Affected individuals (23) and Other useful information labelled Not applicable (23). The largest per-class drop is Other useful information (0.416 to 0.272), which links back to the relevance errors above.

**4. Where both systems fail.** Both systems were wrong on the same case in 52 to 54 cases per seed. By gold label, the hardest classes for both are Donations and volunteering (49 shared failures), Infrastructure and utilities damage (32) and Caution and advice (21). Reading the 17 Donations and volunteering tweets that both systems got wrong at least once shows why that class is hard: almost none are about public donations or volunteers. They report help from official bodies: the Makkah emir ordering a shelter committee, housing and food for people affected in Al Lith, medical clinics for affected villages, a traffic patrol helping a stranded family. Our one-line definition in `labels.py` ("Offers or requests of donations, assistance or volunteering services.") does not say whether reports of help already given by the authorities belong to this class, and the gold labels put them there. Both systems tend to label these tweets Other useful information or Affected individuals instead. Splitting the task by role did not help on them, since both systems read the same definition.

**5. Stability.** The treatment changed its final pair across seeds on 13 of the 140 cases, against 23 for the baseline, which is why its standard deviation is smaller.

## Limitations

- The golden set is small and balanced by Information Type (see `m5/README.md`). Not related cases are few (16), so relevance results rest on few examples.
- Three seeds give a rough view of variance. A difference of 2.38 points with these spreads is not enough to claim that one system is better.
- Latency is not a fair comparison here. The conditions ran on different days and latency varied a lot between runs: baseline mean latency was 39.85 s for seed 42, against 2.33 s and 9.16 s for seeds 123 and 2026. For seeds 123 and 2026, median latency was 1.73 s and 1.70 s for the baseline and 2.19 s and 2.00 s for the treatment. Latency also includes waits after rate-limit errors inside a case.
- The specialists are called one after the other. Calling them at the same time would lower the treatment's latency, but was not tried, to stay within the free-tier per-minute limit.
- Neither system's prompts were tuned. Different prompts, for example a Relevance Agent prompt that lists the kinds of flood-related content, could change the result.
- The definition of Not applicable in `labels.py` mentions tweets that are not related, so the Information-Type Agent has some information about relevance. Both systems see the same definition.
- The one-line label definitions are short. For Donations and volunteering, the definition does not cover reports of help from the authorities, which the gold labels include in this class (see the failure analysis).
- A few gold labels look arguable, for example a tweet that is only the words "Al Lith floods" and a link, labelled Not related.
- Both conditions use one model. The result may not hold for other models.
- The API did not return reasoning-token counts.

## References

Numbered as in our M3 related-work review.

[2] A. Alharbi and M. Lee, "Kawarith: An Arabic Twitter corpus for crisis events," in Proc. 6th Arabic Natural Language Processing Workshop, Kyiv, Ukraine (Virtual), Apr. 2021, pp. 42–52.

[3] E. L. McDaniel, S. Scheele, and J. Liu, "Zero-shot classification of crisis tweets using instruction-finetuned large language models," in Proc. IEEE Int. Humanitarian Technologies Conf. (IHTC), 2024, doi: 10.1109/IHTC61819.2024.10855037.

[5] M. Cemri et al., "Why do multi-agent LLM systems fail?," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track, 2025.
