"""Interactive Gradio interface. Run: python app.py"""
import os
import gradio as gr
from baseline import classify_tweet


def predict(tweet):
    try:
        result = classify_tweet(tweet)
        tokens = result["total_tokens"]
        details = (
            f"Model: {result['model']} | LLM calls: {result['successful_model_calls']} "
            f"| Repairs: {result['repairs']} | Time: {result['latency_seconds']:.2f}s "
            f"| Tokens: {tokens if tokens is not None else 'not returned'}"
        )
        return result["relevance"], result["information_type"], details
    except Exception as exc:
        # Never print credentials or the full request to the user.
        raise gr.Error(f"Classification error ({type(exc).__name__}). Check Gemini credentials and server logs.") from exc


with gr.Blocks(title="Arabic Flood Tweet Classifier — M4 Baseline") as demo:
    gr.Markdown("# Arabic Flood Tweet Classifier\n### Single-Agent Baseline (three sequential Gemini calls)")
    gr.Markdown("Enter a NEW tweet, including one outside the dataset. The text is sent to Google's Gemini API for processing. This is not a flood warning service.")
    tweet = gr.Textbox(label="Tweet text", placeholder="اكتب التغريدة العربية هنا...", lines=4)
    classify = gr.Button("Classify tweet", variant="primary")
    with gr.Row():
        relevance = gr.Textbox(label="Relevance", interactive=False)
        information_type = gr.Textbox(label="Information Type", interactive=False)
    diagnostics = gr.Textbox(label="Run diagnostics", interactive=False)
    classify.click(fn=predict, inputs=tweet, outputs=[relevance, information_type, diagnostics])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")
