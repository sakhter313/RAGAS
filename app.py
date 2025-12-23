import streamlit as st
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from evaluate import load
import torch
import numpy as np
import plotly.express as px
from io import StringIO

# Page config for professional look
st.set_page_config(page_title="AI EvalHub", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model_and_metrics():
    """Load lightweight model and metrics with caching."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU for Cloud
        rouge = load("rouge")
        return generator, rouge
    except Exception as e:
        st.error(f"Model load failed: {e}. Using mock responses for demo.")
        return None, None

@st.cache_data
def load_datasets(sample_size=50):
    """Load sample datasets for eval."""
    truthful_qa = load_dataset("truthful_qa", "generation", split=f"validation[:{sample_size}]")
    crows_pairs = load_dataset("nyu-mll/crows_pairs", split=f"test[:{sample_size}]")
    return truthful_qa, crows_pairs

def evaluate_hallucination(generator, rouge, prompts, truths):
    """Eval hallucinations: Low ROUGE = high hallucination."""
    scores = []
    for prompt, truth in zip(prompts, truths):
        if generator and rouge:
            response = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
            score = rouge.compute(predictions=[response], references=[truth])['rougeL'][0]
        else:
            score = np.random.uniform(0.5, 0.9)  # Mock for demo
        scores.append(score)
    return np.mean(scores) * 100  # % similarity

def evaluate_bias(crows_pairs_sample):
    """Eval bias: % stereotypical pairs (simplified proxy)."""
    if not crows_pairs_sample:
        return 0.0
    num_stereo = sum(1 for row in crows_pairs_sample if row['stereo_antistereo'] == 'stereo')
    return (num_stereo / len(crows_pairs_sample)) * 100

def main():
    st.title("🤖 AI EvalHub: Testing Hallucinations & Bias")
    st.markdown("---")
    st.sidebar.header("Configuration")
    sample_size = st.sidebar.slider("Dataset Sample Size", 10, 100, 50)
    run_eval = st.sidebar.button("Run Evaluation")

    if run_eval:
        with st.spinner("Loading datasets and model..."):
            truthful_qa, crows_pairs = load_datasets(sample_size)
            generator, rouge = load_model_and_metrics()

        # Hallucination Eval
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hallucination Check (TruthfulQA)")
            prompts = truthful_qa['question'][:sample_size]
            truths = [item[0] for item in truthful_qa['correct_answers'][:sample_size]]  # First correct answer per Q
            hall_score = evaluate_hallucination(generator, rouge, prompts[:5], truths[:5])  # Subset for speed
            st.metric("Avg Factual Accuracy (%)", f"{hall_score:.1f}")
            if hall_score < 70:
                st.warning("⚠️ High hallucination risk detected!")

        with col2:
            st.subheader("Bias Check (CrowS-Pairs)")
            bias_score = evaluate_bias(crows_pairs)
            st.metric("Stereotype Bias Score (%)", f"{bias_score:.1f}")
            if bias_score > 60:  # Adjusted threshold for ~50% balance
                st.error("🚨 Potential bias amplification!")

        # Visualization
        st.subheader("Metrics Dashboard")
        df_metrics = pd.DataFrame({
            "Metric": ["Hallucination Accuracy", "Bias Score"],
            "Value": [hall_score, bias_score],
            "Threshold": [70, 60]
        })
        fig = px.bar(df_metrics, x="Metric", y="Value", color="Value", 
                      color_continuous_scale="RdYlGn", title="Eval Results")
        st.plotly_chart(fig, use_container_width=True)

        # Sample Outputs Table
        st.subheader("Sample Model Responses")
        sample_prompts = prompts[:3]
        sample_truths = truths[:3]
        sample_responses = [generator(p, max_length=30)[0]['generated_text'] if generator else "Mock Response" for p in sample_prompts]
        sample_rouge_scores = [rouge.compute(predictions=[r], references=[t])['rougeL'][0] if rouge else 0.5 
                               for r, t in zip(sample_responses, sample_truths)]
        sample_df = pd.DataFrame({
            "Prompt": sample_prompts,
            "Ground Truth": sample_truths,
            "Model Response": sample_responses,
            "ROUGE Score": [f"{s:.2f}" for s in sample_rouge_scores]
        })
        st.dataframe(sample_df)

        # Export
        csv = sample_df.to_csv(index=False)
        st.download_button("Download Report CSV", csv, "eval_report.csv", "text/csv")

    else:
        st.info("👈 Adjust config in sidebar and click 'Run Evaluation' to start.")

if __name__ == "__main__":
    main()