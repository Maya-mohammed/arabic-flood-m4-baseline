"""Gradio interface for the single-agent baseline. Run: python app.py"""
import csv
import os
from pathlib import Path

import gradio as gr

from baseline import classify_tweet

EDGE_FILE = Path(__file__).with_name("edge_cases_labelled_m4.csv")


def load_test_cases():
    """Team-written edge cases (not from the source corpus), with reference labels."""
    if not EDGE_FILE.exists():
        return []
    with open(EDGE_FILE, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


TEST_CASES = load_test_cases()
REFERENCE = {c["tweet"]: f"{c['gold_relevance']} / {c['gold_information_type']}" for c in TEST_CASES}


def predict(tweet):
    try:
        result = classify_tweet(tweet)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        # Never show credentials or the full request.
        raise gr.Error(f"Classification error ({type(exc).__name__}). "
                       "Check the Gemini key and try again in a minute.") from exc

    first, review = result["initial_prediction"], result["self_review_prediction"]
    steps = (
        "| Step | Relevance | Information Type |\n|---|---|---|\n"
        f"| 1. Initial classification | {first['relevance']} | {first['information_type']} |\n"
        f"| 2. Self-review | {review['relevance']} | {review['information_type']} |\n"
        f"| 3. Final classification | **{result['relevance']}** | **{result['information_type']}** |"
    )
    tokens = result["total_tokens"]
    details = (
        f"Model: {result['model']} | Successful calls: {result['successful_model_calls']} "
        f"(planned 3) | API attempts: {result['api_attempts']} | Repairs: {result['repairs']} "
        f"| Time: {result['latency_seconds']:.2f}s "
        f"| Tokens: {tokens if tokens is not None else 'not returned'}"
    )
    reference = REFERENCE.get(tweet.strip(), "No reference label (new tweet)")
    return result["relevance"], result["information_type"], steps, reference, details


with gr.Blocks(title="Arabic Flood Tweet Classifier - M4 Baseline") as demo:
    gr.Markdown("# Arabic Flood Tweet Classifier\n"
                "### Single-agent baseline: one agent, three sequential Gemini calls")
    gr.Markdown("Type any tweet, including one that is not in the dataset, or pick a test case below. "
                "The text is sent to Google's Gemini API. This is a classification demo, "
                "not a flood-warning service.")
    tweet = gr.Textbox(label="Tweet text", placeholder="اكتب التغريدة هنا...", lines=4)
    classify = gr.Button("Classify tweet", variant="primary")
    with gr.Row():
        relevance = gr.Textbox(label="Relevance", interactive=False)
        information_type = gr.Textbox(label="Information Type", interactive=False)
    steps = gr.Markdown()
    reference = gr.Textbox(label="Reference label (test cases only)", interactive=False)
    diagnostics = gr.Textbox(label="Run details", interactive=False)
    if TEST_CASES:
        gr.Examples(
            examples=[[c["tweet"]] for c in TEST_CASES],
            inputs=[tweet],
            label="Test cases (team-written edge cases: English, dialects, mixed text, "
                  "Arabizi, rain without a flood, sarcasm, very short)",
        )
    classify.click(fn=predict, inputs=tweet,
                   outputs=[relevance, information_type, steps, reference, diagnostics])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")
