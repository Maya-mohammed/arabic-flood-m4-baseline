# M4 Test Evidence — preliminary development results

**Date:** 13 October 2026. **Condition:** one generalist agent, **three sequential Gemini calls per tweet**; model ID `gemini-3.5-flash-lite`. Results were observed in the team's Google Colab session, and reported from its console-output screenshots. The raw labelled corpus and case-result CSVs stay on the team's private machine/Colab; they are not included here owing to redistribution restrictions.

## Source-labelled normal test cases (development only)

| Case | Ground-truth label pair | Model-predicted label pair | Joint exact match | Latency (s) |
|---|---|---|---|---:|
| 1 | Related / Caution and advice | Related / Caution and advice | Yes | 77.5232 |
| 2 | Related / Affected individuals | Related / Affected individuals | Yes | 71.8950 |
| 3 | Not related / Not applicable | Related / Other useful information | No | 76.0574 |
| 4 | Related / Infrastructure and utilities damage | Related / Infrastructure and utilities damage | Yes | 42.2912 |
| 5 | Related / Donations and volunteering | Related / Affected individuals | No | 77.0959 |
| 6 | Related / Not applicable | Related / Other useful information | No | 36.4117 |
| 7 | Related / Other useful information | Related / Other useful information | Yes | 30.4010 |
| 8 | Related / Sympathy and emotional support | Related / Sympathy and emotional support | Yes | 45.6727 |

**Calculated from these eight recorded runs:** Valid final label pairs: **8/8 = 100%**. Joint exact-match: **5/8 = 62.5%**. Average latency: **57.17 seconds per tweet**. No statistical generalization should be made from eight hand-selected cases. The selection covers all eight valid joint label pairs and all four named flood events, rather than forming a representative random sample.

## Independent Gradio UI demonstration

A new, user-entered Arabic tweet about closing a road due to flooding was submitted through Gradio. Recorded result: `Related` / `Caution and advice`; **3 successful model calls**, **0 repairs**, **42.61 s**, **616 total tokens**. This demonstrates one end-to-end run of the user interface, but does not establish overall accuracy.

## Limits and next steps

- These are **development-set** cases; the separate evaluation partition was not scored here.
- The submitted class activity's edge cases are a separate synthetic robustness set, with human reference labels still awaiting team confirmation. No edge-case accuracy is claimed here.
- The 99% valid-output target is not established on the population by the observed 8/8 sample.
- No trustworthy USD cost figure can be asserted without the account-specific billing/pricing evidence.
- M5 requires a frozen golden set of at least 20 cases, automated scoring and repeated three-seed baseline measurement. None is claimed as complete here.
- Existing `evaluate_dev.py` omits API-failed cases from its *exact-match denominator* if the model produced no prediction. This did not affect the eight fully completed cases, but must be corrected and prespecified before formal M5 evaluation.
