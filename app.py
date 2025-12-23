import streamlit as st
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Sidebar for API key (overrides secrets for local testing)
api_key = st.sidebar.text_input("OpenAI API Key:", type="password", help="Enter your key here for local testing. Use secrets on Streamlit Cloud.")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

st.title("RAGAS Evaluation App")
st.markdown("A simple Streamlit app to evaluate RAG outputs using RAGAS metrics. Supports single evaluation with sample datasets for quick testing.")

# Sample datasets
@st.cache_data
def get_samples():
    return [
        {
            "name": "Sample 1: Python Basics",
            "question": "What is Python?",
            "answer": "Python is a high-level programming language known for its simplicity and readability.",
            "contexts": ["Python is a high-level programming language designed for readability.", "It was created by Guido van Rossum in the late 1980s.", "Python supports multiple programming paradigms."],
            "ground_truth": "Python is a high-level programming language."
        },
        {
            "name": "Sample 2: Capital of France",
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris, a major European city famous for its culture and history.",
            "contexts": ["France is a country in Western Europe.", "Its capital city is Paris.", "Paris is known for the Eiffel Tower and Louvre Museum."],
            "ground_truth": "Paris"
        },
        {
            "name": "Sample 3: AI Ethics",
            "question": "What are some key principles of AI ethics?",
            "answer": "Key principles include fairness, transparency, accountability, and privacy in AI systems.",
            "contexts": ["AI ethics emphasizes fairness to avoid bias.", "Transparency ensures users understand AI decisions.", "Accountability holds developers responsible for AI outcomes.", "Privacy protects user data in AI applications."],
            "ground_truth": "Fairness, transparency, accountability, privacy."
        }
    ]

samples = get_samples()

# Sample loading section
st.subheader("Load Sample Dataset")
selected_sample = st.selectbox("Choose a sample:", [s["name"] for s in samples])
if selected_sample:
    sample = next(s for s in samples if s["name"] == selected_sample)
    if st.button(f"Load {selected_sample}"):
        st.session_state.question = sample["question"]
        st.session_state.answer = sample["answer"]
        st.session_state.contexts = "---".join(sample["contexts"])
        st.session_state.ground_truth = sample["ground_truth"]

# Input fields (use session_state for persistence)
if "question" not in st.session_state:
    st.session_state.question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "contexts" not in st.session_state:
    st.session_state.contexts = ""
if "ground_truth" not in st.session_state:
    st.session_state.ground_truth = ""

question = st.text_area("Question:", value=st.session_state.question, height=100, key="q_input")
generated_answer = st.text_area("Generated Answer:", value=st.session_state.answer, height=100, key="a_input")
contexts_input = st.text_area("Retrieved Contexts (separate with '---'):", value=st.session_state.contexts, height=150, key="c_input")
ground_truth = st.text_area("Ground Truth (optional):", value=st.session_state.ground_truth, height=100, key="gt_input")

# Update session_state on change (for persistence)
st.session_state.question = question
st.session_state.answer = generated_answer
st.session_state.contexts = contexts_input
st.session_state.ground_truth = ground_truth

if st.button("Evaluate"):
    if not question or not generated_answer or not contexts_input:
        st.error("Please fill in Question, Generated Answer, and Contexts.")
    else:
        # Parse contexts
        contexts = [ctx.strip() for ctx in contexts_input.split("---") if ctx.strip()]
        
        # Prepare data
        data = {
            "question": [question],
            "answer": [generated_answer],
            "contexts": [contexts],
        }
        has_ground_truth = bool(ground_truth.strip())
        if has_ground_truth:
            data["ground_truths"] = [[ground_truth.strip()]]  # List[List[str]] for RAGAS
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        metrics = [faithfulness, answer_relevancy, context_precision]
        if has_ground_truth:
            metrics.append(context_recall)
        
        try:
            with st.spinner("Evaluating... This may take 30-60 seconds."):
                result = evaluate(dataset, metrics=metrics)
            st.success("Evaluation complete!")
            st.subheader("Results")
            cols = st.columns(len(result))
            for i, (metric, score) in enumerate(result.items()):
                with cols[i]:
                    st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.info("Troubleshoot: Verify OpenAI API key, check input formats, or try a sample dataset.")

st.markdown("---")
st.markdown("""
### Usage Notes:
- **Samples**: Select and load a sample to auto-fill fields for quick testing.
- **Contexts**: Multiple chunks separated by '---' (e.g., from vector DB retrieval).
- **Ground Truth**: Enables `context_recall`; provide as a concise reference answer.
- **API Key**: Sidebar for local dev; set `OPENAI_API_KEY` in Streamlit Cloud secrets for deployment.
- **Metrics**: Faithfulness (no hallucinations), Answer Relevancy, Context Precision (+ Recall with GT).
- **Customization**: Add batch eval via CSV upload or more metrics (e.g., from ragas.metrics).
""")