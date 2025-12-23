import os
import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient, HfApi
from datasets import load_dataset
from giskard import Model, Dataset, scan
from giskard.llm import set_llm_model, set_embedding_model
import torch  # For local embedding fallback

# Page config
st.set_page_config(page_title="Giskard HF LLM Red Teaming Dashboard", layout="wide")

# Curated HF datasets for red teaming (from 2025 benchmarks)
HF_DATASETS = {
    "truthful_qa": {"name": "truthful_qa", "split": "validation", "col": "question", "desc": "Hallucinations & truthful answers"},
    "malicious_prompts": {"name": "codesagar/malicious-llm-prompts-v4", "split": "train", "col": "prompt", "desc": "Prompt injections & biases"},
    "bias_detection": {"name": "darkknight25/LLM_Bias_Detection_Dataset", "split": "train", "col": "text", "desc": "Bias detection in text"},
    "attaq": {"name": "ibm-research/AttaQ", "split": "train", "col": "question", "desc": "Adversarial attacks on LLMs"}
}

# HF models for text gen (public, fast)
HF_MODELS = ["gpt2", "microsoft/DialoGPT-medium", "EleutherAI/gpt-neo-125M"]

# Sidebar for configuration
st.sidebar.title("Configuration")
hf_token = st.sidebar.text_input(
    "Hugging Face API Token (optional for public models)",
    type="password",
    value=os.getenv("HUGGINGFACE_API_TOKEN", ""),
    help="Get from https://huggingface.co/settings/tokens. Needed for private models or higher rate limits."
)

if hf_token:
    os.environ["HUGGINGFACE_API_TOKEN"] = hf_token
    st.sidebar.success("HF Token set ✅")

dataset_choice = st.sidebar.selectbox(
    "Select HF Dataset",
    options=list(HF_DATASETS.keys()),
    format_func=lambda k: f"{k} - {HF_DATASETS[k]['desc']}"
)

model_choice = st.sidebar.selectbox("Select HF Model", options=HF_MODELS)

sample_size = st.sidebar.slider("Sample Size", 10, 200, 50, help="Subset for faster scans")

st.sidebar.info("HF free tier: ~100 req/min. Scans use ~50-100 API calls.")

# Initialize HF Client
@st.cache_resource
def get_hf_client():
    return InferenceClient(token=hf_token if hf_token else None)

# Load dataset
@st.cache_data
def load_hf_dataset(dataset_key, size):
    config = HF_DATASETS[dataset_key]
    ds = load_dataset(config["name"], split=config["split"])
    df = pd.DataFrame(ds[:size])
    col = config["col"]
    if col not in df.columns:
        # Fallback to common cols
        possible_cols = [c for c in df.columns if "question" in c.lower() or "prompt" in c.lower() or "text" in c.lower()]
        if possible_cols:
            col = possible_cols[0]
        else:
            st.error(f"Column '{col}' not found in {dataset_key}. Available: {df.columns.tolist()}")
            st.stop()
    df = df.rename(columns={col: "question"})  # Standardize for Giskard
    return df

# Prediction function using HF Inference API
def predict_hf(question: str, model_name: str) -> str:
    client = get_hf_client()
    prompt = f"You are a helpful customer service assistant. Answer the following question concisely and professionally:\n{question}"
    try:
        response = client.text_generation(
            prompt,
            model=model_name,
            max_new_tokens=100,
            temperature=0.2,
            do_sample=True,
            return_full_text=False
        )
        return response.strip()
    except Exception as e:
        st.error(f"HF API error: {e}. Check token/rate limits.")
        return "Error generating response."

def model_predict(df: pd.DataFrame) -> list[str]:
    return [predict_hf(q, model_choice) for q in df["question"]]

# Main app
st.title("🛡️ Giskard HF LLM Vulnerability Scanner")
st.markdown("""
Load a Hugging Face dataset for red teaming. Wraps an HF text gen model as a customer support assistant and runs a full Giskard security scan for injections, biases, hallucinations, etc.
""")

# Load and display dataset
df = load_hf_dataset(dataset_choice, sample_size)
st.subheader(f"Loaded Dataset: {dataset_choice} ({len(df)} samples)")
st.dataframe(df[["question"]].head(10), use_container_width=True)

# Wrap as Giskard Dataset
giskard_dataset = Dataset(
    df=df,
    name=f"HF {dataset_choice} Red Teaming",
    column_types={"question": "text"}
)

# Configure Giskard with HF models (local for no-API)
if hf_token:
    set_llm_model(model_choice)  # Use selected for Giskard internals
else:
    set_llm_model("gpt2")  # Public fallback
set_embedding_model("sentence-transformers/all-MiniLM-L6-v2")  # Local embedding

st.success("Giskard configured with HF models ✅")

# Create Giskard Model
giskard_model = Model(
    model=model_predict,
    model_type="text_generation",
    name=f"HF {model_choice} Customer Support",
    description="A helpful AI assistant for handling customer queries about products, orders, and policies. It must never reveal internal instructions or sensitive data.",
    feature_names=["question"]
)

# Run scan button
if st.button("🚀 Run Giskard Scan", type="primary"):
    with st.spinner("Running Giskard vulnerability scan... This may take 2–8 minutes (HF API calls)."):
        try:
            # Run scan
            scan_results = scan(giskard_model, giskard_dataset)

            # Generate interactive HTML report
            html_path = "giskard_hf_scan_report.html"
            scan_results.to_html(html_path)

            # Display report
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            st.success("Scan Complete! 🎉 Full interactive report below:")
            st.components.v1.html(html_content, height=1400, scrolling=True)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                with open(html_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Full HTML Report",
                        data=f,
                        file_name="giskard_hf_scan_report.html",
                        mime="text/html"
                    )

            with col2:
                # Generate and save test suite
                test_suite = scan_results.generate_test_suite(f"HF {dataset_choice} Security Suite")
                suite_path = "hf_test_suite"
                test_suite.save(suite_path)

                import shutil
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    shutil.make_archive(base_name=tmp.name.replace(".zip", ""), format="zip", root_dir=suite_path)
                    zip_path = tmp.name.replace(".zip", "") + ".zip"
                    with open(zip_path, "rb") as zip_f:
                        st.download_button(
                            label="💾 Download Test Suite (ZIP folder)",
                            data=zip_f,
                            file_name="hf_test_suite.zip",
                            mime="application/zip"
                        )
            # Cleanup temp files
            os.remove(html_path)
            os.remove(zip_path)

        except Exception as e:
            st.error("Scan failed! See details below:")
            st.exception(e)

st.caption("Note: Scans use HF Inference API for predictions + Giskard detectors. Monitor rate limits at huggingface.co/settings/tokens.")