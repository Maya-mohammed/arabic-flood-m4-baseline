# M4 Test Evidence - Baseline and Edge Cases

**Date:** 13 October 2026  |  **Model used in the Colab tests:** `gemini-3.5-flash-lite`  |  **System:** one generalist agent with three planned sequential calls (initial classification, self-review, final classification).

This note records the tests actually run for M4. The normal-case results were recorded in the team's Colab screenshots; the edge-case numbers below were checked against the exported CSV files. These are small development and robustness checks, **not the final M5 evaluation**.

**Version note:** these runs used an earlier prompt version **without label definitions** and the earlier runner scripts (`evaluate_dev.py` and `evaluate_edge_m4.py`). The repository now adds the definitions in `labels.py` and merges the two runners into `evaluate.py`, which counts failed requests as wrong for every test set. Results from the current version should be recorded as a separate run.

## 1. Source-labelled development cases

| Case | Original reference labels | Baseline prediction | Both correct? | Time (s) |
|---|---|---|:---:|---:|
| 1 | Related / Caution and advice | Related / Caution and advice | Yes | 77.52 |
| 2 | Related / Affected individuals | Related / Affected individuals | Yes | 71.89 |
| 3 | Not related / Not applicable | Related / Other useful information | No | 76.06 |
| 4 | Related / Infrastructure and utilities damage | Related / Infrastructure and utilities damage | Yes | 42.29 |
| 5 | Related / Donations and volunteering | Related / Affected individuals | No | 77.10 |
| 6 | Related / Not applicable | Related / Other useful information | No | 36.41 |
| 7 | Related / Other useful information | Related / Other useful information | Yes | 30.40 |
| 8 | Related / Sympathy and emotional support | Related / Sympathy and emotional support | Yes | 45.67 |

**Results:** 8/8 valid output pairs (100%); 5/8 joint exact matches (62.5%); mean latency 57.17 s per tweet.

The eight cases were selected for coverage of the seven Information Type classes, a Not related case, and the four flood events. This is a small, selected development sample, not a representative sample of the full dataset. The held-out evaluation partition was not used for these checks.

## 2. Separate, synthetic edge cases

The team prepared 12 candidate cases using the class task categories (English, Arabic dialects, mixed language, Arabizi, rain without a flood, short posts, and sarcasm), plus patterns seen in development errors. One ambiguous candidate, EDGE10, was excluded **before testing**; it has no scored result. One team member reviewed and approved the proposed labels for the remaining 11 cases before Gemini was called. These are team-reviewed proposed labels, **not labels supplied by the original dataset authors or independently double-annotated**.

| Case | Intended difficulty | Gold pair | Model result | Outcome | Time (s) |
|---|---|---|---|---|---:|
| EDGE01 | English / flood damage | Related / Infrastructure and utilities damage | Related / Infrastructure and utilities damage | Correct | 291.04 |
| EDGE02 | English / rain without a flood | Not related / Not applicable | Not related / Not applicable | Correct | 226.28 |
| EDGE03 | Arabic dialect / people affected | Related / Affected individuals | Related / Affected individuals | Correct | 233.69 |
| EDGE04 | Mixed Arabic-English / warning | Related / Caution and advice | Related / Caution and advice | Correct | 168.64 |
| EDGE05 | Arabizi / damaged bridge | Related / Infrastructure and utilities damage | Related / Infrastructure and utilities damage | Correct | 442.07 |
| EDGE06 | Arabic rain prayer | Not related / Not applicable | Not related / Not applicable | Correct | 323.03 |
| EDGE07 | Sarcasm / damage | Related / Infrastructure and utilities damage | Related / Infrastructure and utilities damage | Correct | 118.63 |
| EDGE08 | Very short / trapped people | Related / Affected individuals | Related / Affected individuals | Correct | 191.19 |
| EDGE09 | Unrelated road closure | Not related / Not applicable | Related / Caution and advice | Wrong prediction | 194.08 |
| EDGE11 | Metaphorical drowning | Not related / Not applicable | No final result (503) | API failure | Not recorded |
| EDGE12 | Volunteering / confusing affected people | Related / Donations and volunteering | No final result (503) | API failure | Not recorded |

**Original 11-case run:** 9/11 valid final label pairs (**81.8%**); 8/11 joint exact matches (**72.7%**, treating cases with no final result as incorrect); 1 wrong label pair (EDGE09); 2 API failures (EDGE11 and EDGE12). Among the nine requests with a final prediction, 8/9 were exact matches (**88.9%**). This last figure is conditional on a result being returned, **not** the end-to-end score.

- **EDGE09:** The post said the road was closed for maintenance, with no flood. Reference: `Not related / Not applicable`; Gemini predicted `Related / Caution and advice`. That was a classification error.
- **EDGE11 and EDGE12:** The original runs returned `ServerError: 503 UNAVAILABLE` rather than classification labels. The API message stated that the model was experiencing high demand. An additional **separate recovery attempt** on EDGE11 returned the same 503 error. EDGE12 was not retried. The original outcomes remain unchanged.

## 3. Model usage and latency

- For the **nine successfully classified edge cases only**: 27 successful model calls, 34 recorded API attempts (including retries), 0 recorded format repairs, and 5,279 total reported tokens.
- Mean end-to-end classification time over those **nine cases**: **243.18 seconds**. The failed cases have blank latency and attempt-count fields in the exported CSV, so these totals are **not** totals for all 11 requests.
- Reasoning-token counts were not provided in those records. USD cost has not been calculated because applicable billing prices were not verified.
- The intended baseline is three sequential LLM calls per case. Actual API attempts can be higher because transient service errors may trigger retries.

## 4. Gradio UI test

The team entered a new Arabic road-closure warning through Gradio, outside the CSV workflow. The displayed result was `Related / Caution and advice`, with 3 successful Gemini calls, 0 output repairs, 42.61 seconds, and 616 total tokens. This confirms one observed end-to-end interaction, not a claim about general accuracy or permanent hosting.

## 5. Interpretation and next milestones

- These were small preliminary M4 tests; neither percentage is a general population accuracy estimate, and the two sets should not be combined into one accuracy figure.
- The team set **99% valid output** as a goal in the class activity. The edge-case run did **not** reach that goal (9/11 valid). No claim is made that the 99% target has been achieved.
- Source corpus: 4,037 original rows and 4,028 cleaned rows (805 development and 3,223 held aside), as recorded in the project preparation run. These raw corpus rows are intentionally not published in the GitHub repository.
- The normal cases were run with `evaluate_dev.py`, which left failed requests out of the exact-match denominator (no request failed in that run, so the 5/8 result is not affected). The edge run used `evaluate_edge_m4.py`, which counts failed labelled requests as wrong. Both are now replaced by `evaluate.py`, which counts failures as wrong for every test set. The final M5 evaluator must use the same denominator in both conditions.
- M5 will freeze the test set and settings, prepare a golden set with at least 20 cases, and evaluate the baseline across three seeds. M6 will implement the two specialists plus Judge and compare it with the baseline. Those later-stage results are **not** part of M4.

**Evidence retained by the team:** `M4_Edge_Test_Evidence.zip` contains the 11-case input, per-batch output CSVs, the consolidated original-run results, and the EDGE11 recovery result. This private evidence backup does not need to be put on a public repository.

**Academic and data handling note:** If course policy requires disclosure of AI assistance in preparing code or test cases, disclose it accurately. Do not upload private API credentials or redistribute the source tweet corpus without the relevant permission.
