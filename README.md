# Arabic Flood-Crisis Tweet Classification — M4 Baseline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maya-mohammed/arabic-flood-m4-baseline/blob/main/M4_baseline_colab.ipynb)

This repository contains our **single-agent baseline** for classifying Arabic flood-crisis tweets (CPCS683, Milestone 4). A user can type any tweet in a simple interface and get two labels back. The multi-agent treatment will be added in a later milestone.

## What the system does

For each tweet it predicts:

- **Relevance:** `Related` or `Not related`
- **Information Type:** `Affected individuals`, `Infrastructure and utilities damage`, `Donations and volunteering`, `Caution and advice`, `Sympathy and emotional support`, `Other useful information`, or `Not applicable`

One generalist agent makes **three sequential calls** to the same Gemini model:

1. Initial classification of both labels
2. Self-review of that classification
3. Final classification of both labels

Every prompt includes the closed label sets and a one-line definition of each label (`labels.py`). The multi-agent system will import the same file, so both systems see the same labels and definitions.

**Consistency rule (one way only):** if Relevance is `Not related`, Information Type must be `Not applicable`. The reverse is not required: `Related / Not applicable` is valid (452 tweets in the corpus have this pair).

**Model and settings:** `gemini-3.5-flash-lite` with its default settings and JSON output. We do not send `temperature`, because Gemini 3.5 models ignore it. No training, extra agents, or outside data retrieval are used.

**Repairs and retries:** if a stage returns malformed JSON or a label outside the closed sets, or if the final pair breaks the consistency rule, the agent gets **one** repair request for that stage. Rate-limit errors (429) are retried after waiting 15, 30, 45 and 60 seconds; temporary server errors (5xx) are retried after a short wait. Each request is tried at most 5 times. All calls, repairs and attempts are counted and reported.

## Files

| File | Purpose |
|---|---|
| `baseline.py` | The single-agent baseline: three calls, validation, bounded repair, usage logging |
| `labels.py` | Closed label sets and label definitions (shared with the future multi-agent system) |
| `app.py` | Gradio user interface: type a tweet or pick a test case |
| `M4_baseline_colab.ipynb` | Colab notebook that runs everything from start to finish |
| `prepare_data.py` | Cleans the original dataset and creates the development and reserved splits |
| `evaluate.py` | Runs the baseline on a labelled CSV and saves one result row per tweet |
| `edge_cases_labelled_m4.csv` | 11 team-written edge cases with reference labels |
| `M4_TEST_EVIDENCE.md` | Recorded M4 test results and limitations |
| `tests/` | Offline unit tests (no API calls) |
| `requirements.txt` | Python packages |
| `.env.example`, `.gitignore` | Key template and files kept out of Git |

## Run in Google Colab (easiest)

1. Create a Gemini API key in Google AI Studio.
2. In Colab, open the key icon on the left (**Secrets**), add a secret named `GEMINI_API_KEY`, and turn on notebook access.
3. Click the **Open in Colab** badge above and run the cells in order.

The notebook downloads the code and the dataset, runs the unit tests, classifies one tweet, runs the test cases, and opens the interface.

## Run on your own computer

Use Python 3.10 or newer.

```bash
git clone https://github.com/Maya-mohammed/arabic-flood-m4-baseline.git
cd arabic-flood-m4-baseline
python -m pip install -r requirements.txt
```

Set your key as an environment variable. Never put the key in a file that goes to GitHub.

```bash
export GEMINI_API_KEY='YOUR_PRIVATE_KEY'        # macOS / Linux
$env:GEMINI_API_KEY='YOUR_PRIVATE_KEY'          # Windows PowerShell
```

Start the interface and open the link it prints:

```bash
python app.py
```

Set `GRADIO_SHARE=1` before starting it if you need a temporary public link. Anyone with that link uses your API key, so stop the app after the demo.

Run the offline unit tests (no key needed):

```bash
python -m pip install pytest
python -m pytest -q tests
```

## Dataset

We use the Arabic Twitter Corpus for Flood Detection (Alharbi & Lee, 2019): 4,037 human-labelled tweets from four 2018 floods (Jordan, Kuwait, Qurayyat, Al Lith). The repository does not state a licence, so **we do not redistribute the corpus**. Download it from the [authors' repository](https://github.com/alaa-a-a/Arabic-Twitter-corpus-for-flood-detection):

```bash
git clone --depth 1 https://github.com/alaa-a-a/Arabic-Twitter-corpus-for-flood-detection.git source_data
mkdir -p private_data && cp source_data/ArabicFloods.csv private_data/
python prepare_data.py --csv private_data/ArabicFloods.csv
```

`prepare_data.py` gives **4,028 clean rows**: 7 tweet IDs appear twice in the corpus, 2 of them with conflicting labels (both copies removed) and 5 with the same labels (one copy kept). It then makes a stratified split: **805 development** rows and **3,223 reserved** rows, with no shared IDs or identical tweet text. The reserved rows were not used in M4; the M5 evaluation set will be sampled from them and frozen before formal evaluation. All files go to `private_data/`, which Git ignores.

## Test cases

**Normal cases:** `private_data/normal_cases.csv` has 8 development tweets, one for each valid label pair, covering all four floods.

```bash
python evaluate.py --cases private_data/normal_cases.csv --out private_data/normal_results.csv --pause 15
```

**Edge cases:** `edge_cases_labelled_m4.csv` has 11 team-written tweets: English, Arabic dialects, mixed Arabic-English, Arabizi, rain without a flood, sarcasm, very short tweets, and confusing cases seen during development. One team member reviewed the reference labels before Gemini was run. They are not labels from the original corpus and were not double-annotated. A twelfth candidate (EDGE10) was dropped before testing because its label was unclear.

```bash
python evaluate.py --cases edge_cases_labelled_m4.csv --out private_data/edge_results.csv --pause 15
```

`--pause 15` waits 15 seconds between tweets so a free-tier key does not hit its per-minute request limit (each tweet needs three requests). `evaluate.py` keeps every case in the results. A request that fails (for example an API error) counts as an invalid pair and, if the case has a reference label, as a wrong answer. Results are saved after each case. The edge cases also appear as clickable examples in the interface.

## Recorded M4 results

These results were recorded on 13 October 2026 with an **earlier prompt version that did not include label definitions**. The current prompts add the definitions in `labels.py`, so the numbers below describe that earlier version.

| Test | Valid final pairs | Both labels correct |
|---|---:|---:|
| 8 normal development cases | 8/8 (100%) | 5/8 (62.5%) |
| 11 edge cases | 9/11 (81.8%) | 8/11 (72.7%) |

In the edge run, EDGE09 got a wrong label pair, and EDGE11 and EDGE12 failed with `503 UNAVAILABLE` (model under high demand). Mean time per tweet was 57.17 s for the normal cases and 243.18 s for the nine completed edge cases. These are small selected tests, not an accuracy estimate for the whole dataset. Details are in [`M4_TEST_EVIDENCE.md`](M4_TEST_EVIDENCE.md).

## Scope and limitations

- M4 covers the single-agent baseline only. The multi-agent treatment and the comparison come in M5 and M6.
- The plan is three model calls per tweet. Repairs and retries can add requests; they are counted, not hidden.
- Response time depends on the Gemini service. During our tests, single tweets took from about 30 seconds to several minutes, and some requests failed with 503 errors.
- `classify_tweet` and `evaluate.py` accept a `seed`, but Google's documentation for this model does not confirm that the seed is applied. This will be checked before the three-seed runs in M5.
- The 99% valid-output target in our evaluation plan is a team goal. The M4 edge run did not reach it (9/11).
- USD cost is not calculated yet; token counts are recorded so it can be calculated from the official price list.
- This is a classification demo, not a flood-warning service.

## Data and AI-use notes

Do not upload the corpus, the `private_data/` folder, or any API key to this repository. Follow the course rules for acknowledging outside or AI-assisted work.
