# Arabic Flood-Crisis Tweet Classification — M4 Baseline

**Milestone:** CPCS683 M4 / single-agent baseline, live demonstration and reproducible repository.

## Scope and architecture

An Arabic flood-crisis tweet is classified into two closed-label fields: **Relevance** (`Related` / `Not related`) and **Information Type** (seven classes). The baseline is **one generalist agent** executing **three sequential calls to the same Gemini model**: (1) initial joint classification, (2) self-review, (3) final joint classification. It does not use multiple agents, model training, tools, retrieval, weather data or flood forecasting. See `baseline.py`.

Model ID used in the team's original submitted M4 notebook: `gemini-3.5-flash-lite`. This repository preserves that choice. Availability is specific to the Gemini project/account, so an actual authorized account must test it.

**Label rule:** `Not related` implies `Not applicable`; the reverse implication is **false**. Both `Related + Not applicable` and `Not related + Not applicable` are legal pairs.

## Installation / first-time setup

Requires Python 3.10+ and permission to access Google's Gemini API.

```bash
python -m pip install -r requirements.txt
```

Set `GEMINI_API_KEY` **privately in your shell/environment**. Never paste it into source files, a screenshot, the README, or a public repository. The `.env.example` is only a template and is not loaded automatically.

```bash
# macOS/Linux, run locally with your own key:
export GEMINI_API_KEY='YOUR_KEY_VALUE'

# Windows PowerShell:
$env:GEMINI_API_KEY='YOUR_KEY_VALUE'
```

**Google Colab:** Save the key in Colab **Secrets** under `GEMINI_API_KEY`, then run this Python cell before using the scripts:

```python
import os
from google.colab import userdata
secret = userdata.get("GEMINI_API_KEY")
if not secret:
    raise RuntimeError("GEMINI_API_KEY has not been added to Colab Secrets")
os.environ["GEMINI_API_KEY"] = secret
print("Gemini key available to the code (value not displayed)")
```

## Run a quick end-to-end classification

```bash
python -c "from baseline import classify_tweet; print(classify_tweet('تحذير من السيول يرجى الابتعاد عن الأودية'))"
```

Expected *structure*, not a guaranteed label: `relevance`, `information_type`, timing, 3 planned calls, actual successful model calls, repairs, and token usage. The actual labels must come from Gemini.

## Run user interface

```bash
python app.py
```

Open the local URL printed by Gradio. Enter a tweet not in the dataset, click **Classify tweet**, and verify BOTH labels appear. The diagnostic box displays latency, actual calls, repair count, and token count when supplied by Gemini.

In Colab, you may set `os.environ["GRADIO_SHARE"] = "1"` before launching `!python app.py` to obtain a temporary public Gradio URL. Anyone with that URL may trigger API usage and associated charges while your server runs; stop the demo afterward.

## Dataset and private local tests

This work refers to Alharbi & Lee (2019), *Crisis Detection from Arabic Tweets*, using the authors' `ArabicFloods.csv` (4,037 rows across Jordan, Kuwait, Qurayyat, Al Lith floods). The source corpus does not have a confirmed redistribution license in the provided proposal, so **do not upload it to the public repository**. Obtain it from the [original authors' repository](https://github.com/alaa-a-a/Arabic-Twitter-corpus-for-flood-detection), as documented in M2, and copy it to `private_data/ArabicFloods.csv` locally. Review the source's terms before use.

```bash
python prepare_data.py --csv private_data/ArabicFloods.csv
```

This script removes conflicting tweet IDs, collapses remaining duplicate IDs, and repairs the original team split's **one cross-split duplicate tweet text** without changing sizes or stratification: **4,028 clean rows**, **805 development**, **3,223 evaluation**. The source CSV stays unchanged. The resulting files are stored in `private_data/` and are git-ignored. `dev_5.csv` is an initial smoke-test selection; `normal_cases.csv` covers eight valid label pairs and all four source flood events.

```bash
python evaluate_dev.py --cases private_data/dev_5.csv --out private_data/dev_5_results.csv
python evaluate_dev.py --cases private_data/normal_cases.csv --out private_data/normal_results.csv
```

This script runs **real Gemini calls**, reports an outcome for each case, checks human gold labels if present, and stores label validity, first-pair consistency, token metadata and latency. The `private_data/evaluation.csv` holdout is **not** used for the M4 prompt design; it is preserved for later formal evaluation (M5). Never tune on the holdout.

## Edge cases (class activity)

`edge_cases_template.csv` provides **original, synthetic** examples for English, dialect, Arabizi, mixed language, rain without flooding, sarcasm, and very short messages. The gold columns are intentionally blank: **team members must manually agree on reference labels** before scoring. This robustness set should be reported **separately** from the Arabic corpus-based main evaluation, as committed in the class activity. Once agreed, run it as a separate input to `evaluate_dev.py`. Do not claim these are source-corpus samples.

## Automated code tests (do not require Gemini key)

```bash
python -m pip install pytest
python -m pytest -q tests
```

Passing local unit tests demonstrates program logic only; the user must still perform and record a real Gemini API call and an end-to-end Gradio demo.

## Research integrity

Review, understand, and test all code before submitting. Disclose AI-assisted code preparation and any other reused resources to the extent required by the course's academic-integrity policy. Do not claim an unperformed test or a fabricated classification score.

## Experiment logging and methodological boundary

- M4 delivers a working single-agent baseline. Planned calls are 3, but temporary API retries or *bounded* invalid-output repairs can add actual calls; the code logs the difference transparently.
- Output JSON is checked against closed labels and the one-way consistency rule. The first complete pair's violation is tracked before final correction.
- `usage_metadata` is used for input, output, reasoning and total token counts **when the API supplies them**; absent counts remain blank, not invented.
- **USD cost is not computed** without a verified price for the actual model and billing tier. Record the correct provider tariff before reporting any numeric cost.
- Three-seed comparison, ≥20-case golden set and multi-agent treatment are later milestones (M5/M6), not claims of completion in M4.
- The API key and protected corpus are excluded from Git. Read the `Evaluation_Task_Review.md` for limitations in the class activity's success criteria and wording.

## M4 live demo (five minutes)

1. Explain the task, controlled comparison, and single-agent 3-call flow.
2. Enter a new Arabic tweet in Gradio; submit and show the two-label classification and diagnostics.
3. Present 5 labelled development-case predictions with gold comparison and execution times (if completed).
4. Show the validation rule and an example of `Related + Not applicable` being valid.
5. Open the repository and show `README.md`, setup steps, and how to run `app.py`. State that the multi-agent treatment is scheduled for M6.

## Recorded M4 demonstration evidence (13 October 2026)

The team exercised this baseline in their own credentialed Google Colab session: a new user-entered tweet was classified through Gradio using **3 successful Gemini calls, 0 format repairs, 42.61 seconds, 616 total tokens**. For eight stratified, manually selected corpus-labelled **development** cases, **8/8 produced valid pairs and 5/8 exactly matched both gold labels (62.5%)**. These are preliminary *development smoke-test results*, **not a representative held-out evaluation score**. Details, conditions and limitations: [`M4_TEST_EVIDENCE.md`](M4_TEST_EVIDENCE.md). Any third party must perform their own live API test using their own authorized credential; unit tests do not test network connectivity.

> **Known evaluation limitation:** `evaluate_dev.py` presently reports exact-match over cases with a recorded prediction; failed API requests are logged but omitted from that accuracy denominator. All eight reported development runs completed successfully, so this limitation did not affect the reported 5/8. For M5, define the denominator to include all planned labelled cases, including failed runs as incorrect, and use the same protocol across conditions.
