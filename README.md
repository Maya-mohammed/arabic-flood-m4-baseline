# Arabic Flood-Crisis Tweet Classification — M4 Baseline

This repository contains our **single-agent baseline** for classifying flood-related tweets. For Milestone 4, the goal is to show that the baseline runs, accepts a new tweet through a user interface, and returns two classification labels. The multi-agent version is planned for a later milestone.

## What the system does

For each tweet, it predicts:

- **Relevance:** `Related` or `Not related`.
- **Information Type:** `Affected individuals`, `Infrastructure and utilities damage`, `Donations and volunteering`, `Caution and advice`, `Sympathy and emotional support`, `Other useful information`, or `Not applicable`.

The baseline uses **one generalist agent and three sequential calls** to the same Gemini model:

1. Initial classification of both labels.
2. Self-review of that classification.
3. Final classification of both labels.

We used `gemini-3.5-flash-lite` in our recorded M4 tests. Access and response time depend on the Gemini account and service availability. No model training, additional agents, or external data retrieval is involved in this baseline.

**Consistency rule:** If the Relevance label is `Not related`, Information Type must be `Not applicable`. The reverse is **not** required: `Related / Not applicable` is a valid pair in this project.

## Files in this repository

| File | Purpose |
|---|---|
| `baseline.py` | Single-agent classification with three sequential Gemini calls, output validation, and usage logging |
| `app.py` | Gradio interface for entering a new tweet |
| `prepare_data.py` | Cleans and splits the original dataset locally |
| `evaluate_dev.py` | Runs initial development tests |
| `evaluate_edge_m4.py` | Runs labelled edge cases and records failures as incorrect for end-to-end exact-match |
| `edge_cases_labelled_m4.csv` | Eleven team-reviewed synthetic edge cases and their proposed reference labels |
| `edge_cases_template.csv` | Earlier blank-label template; **not** the scored set |
| `M4_TEST_EVIDENCE.md` | Recorded M4 test results and limitations |
| `tests/` | Unit tests that do not call Gemini |
| `requirements.txt` | Required Python packages |
| `.env.example` and `.gitignore` | Example environment setting and excluded private files |

## Set up and run

Use Python **3.10 or newer**. Download this repository using **Code → Download ZIP**, extract it, and open a terminal inside the extracted project folder.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a Gemini API key in Google AI Studio if you do not already have one. Make it available as an environment variable named `GEMINI_API_KEY`; **do not add the key to any file committed to GitHub**.

On macOS/Linux:

```bash
export GEMINI_API_KEY='YOUR_PRIVATE_KEY'
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY='YOUR_PRIVATE_KEY'
```

If using **Google Colab**, upload the repository ZIP and extract it, then change into the extracted project directory and install `requirements.txt`. Add your key to **Colab Secrets** with the exact name `GEMINI_API_KEY` and enable notebook access. Run this cell before the Python scripts:

```python
import os
from google.colab import userdata
key = userdata.get("GEMINI_API_KEY")
if not key:
    raise RuntimeError("GEMINI_API_KEY is missing from Colab Secrets")
os.environ["GEMINI_API_KEY"] = key
print("Gemini key loaded without displaying it")
```

### Run the Gradio interface

From the project directory, run:

```bash
python app.py
```

Open the URL shown by Gradio, enter a new tweet, and click **Classify tweet**. The interface shows Relevance, Information Type, and basic run details. The text is sent to Google's Gemini API. **This is a classification demonstration, not a flood-warning service.**

In Colab, to create a temporary public Gradio link, set `os.environ["GRADIO_SHARE"] = "1"` in a Python cell before running `!python app.py`. Keep the session running while testing, and stop it afterward. Anyone with that link can submit requests that use your API account.

### Run unit tests

These tests do **not** need a Gemini API key:

```bash
python -m pip install pytest
python -m pytest -q tests
```

Passing unit tests checks code logic; it does not replace a real Gemini or Gradio test.

## Dataset and normal development tests

The source dataset is `ArabicFloods.csv`, referenced in our M2 proposal (Alharbi & Lee, 2019). It originally contains **4,037 rows** from four flood events (Jordan, Kuwait, Qurayyat, and Al Lith). We do **not** redistribute the corpus through this repository because redistribution permission has not been confirmed. Obtain it from the [source project](https://github.com/alaa-a-a/Arabic-Twitter-corpus-for-flood-detection) after reviewing its terms.

Create a local `private_data` folder and place the original CSV there as `private_data/ArabicFloods.csv`. You can create the folder with `mkdir -p private_data` on macOS/Linux/Colab, or `New-Item -ItemType Directory -Force private_data` in Windows PowerShell.

Run:

```bash
python prepare_data.py --csv private_data/ArabicFloods.csv
```

Our recorded preparation run produced **4,028 cleaned rows**, divided into **805 development** and **3,223 reserved evaluation** rows. The script removes conflicting/duplicate IDs and checks that the two partitions do not share IDs or identical tweet text. The original CSV is left unchanged. The reserved evaluation partition was **not used** for the reported M4 development tests.

To run the small development test from the prepared files:

```bash
python evaluate_dev.py --cases private_data/dev_5.csv --out private_data/dev_5_results.csv
```

To test the eight selected normal cases:

```bash
python evaluate_dev.py --cases private_data/normal_cases.csv --out private_data/normal_results.csv
```

These commands make real Gemini API calls and may take time or count against API limits. `evaluate_dev.py` currently reports exact-match among cases with recorded predictions; consult `M4_TEST_EVIDENCE.md` for this known limitation. All eight of our recorded normal cases returned valid predictions.

## Edge cases (class activity)

We created **12 candidate** edge cases to cover English, Arabic dialects, mixed Arabic-English, Arabizi, rain without flooding, sarcasm, very short tweets, and difficult meanings seen during development. We excluded **EDGE10** before testing because its proposed reference label was ambiguous. The remaining **11 synthetic cases** and their proposed labels were reviewed by one team member before running Gemini. These labels are **not** labels from the original corpus or independently double-annotated.

Run the labelled cases with:

```bash
python evaluate_edge_m4.py --cases edge_cases_labelled_m4.csv --out private_data/edge_11_results.csv
```

The edge-case evaluation is separate from the source-data evaluation. Its script writes results after each case and counts labelled requests with no final prediction as incorrect in the end-to-end exact-match rate. Actual model availability, delays, and retries can change the outcome on a new run.

## Recorded M4 results (13 October 2026)

| Test | Valid final label pairs | Both labels matched the reference |
|---|---:|---:|
| 8 selected normal development cases | 8/8 (100%) | 5/8 (62.5%) |
| 11 separately labelled synthetic edge cases, original run | 9/11 (81.8%) | 8/11 (72.7%) |

For the edge run, **EDGE09** returned an incorrect label pair; **EDGE11** and **EDGE12** returned `503 UNAVAILABLE` with no final prediction. A separate retry of EDGE11 also returned 503. We kept the original outcomes unchanged. The nine completed edge requests averaged **243.18 seconds**; latency for the two failed cases was not recorded. These are **small selected tests**, not a representative accuracy estimate for all tweets.

We also tested Gradio with a new user-entered Arabic tweet; the interface returned `Related / Caution and advice` with three successful Gemini calls, 42.61 seconds, and 616 reported tokens.

For per-case results, recorded API attempts, tokens and limitations, see **[`M4_TEST_EVIDENCE.md`](M4_TEST_EVIDENCE.md)**. We have not calculated USD cost without verified pricing for our actual Gemini account/model.

## Scope and limitations

- M4 provides a working **single-agent baseline** only. We have not implemented or compared the multi-agent treatment yet.
- Three Gemini calls are planned for each classification; retries or format repairs may lead to more actual API attempts. The code records available usage information.
- The **99% valid-output rate** in our class activity is a team goal, not a result reached by the edge-case run or a grading threshold stated in the M4 rubric.
- The larger golden set, three-seed evaluation, and single-agent versus multi-agent comparison belong to later milestones (M5/M6).
- Never upload the source corpus, local `private_data` files, real API keys, or private credentials to the repository. Follow course rules for acknowledging outside or AI-assisted work.
