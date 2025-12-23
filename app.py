import streamlit as st
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

st.title("RAGAS Evaluation App")
st.markdown("A simple Streamlit app to evaluate a single RAG output using RAGAS metrics. Assumes OpenAI API key is set in Streamlit secrets as `OPENAI_API_KEY`.")

# Input fields
question = st.text_area("Enter the Question:", height=100)
generated_answer = st.text_area("Enter the Generated Answer:", height=100)
contexts_input = st.text_area("Enter the Retrieved Contexts (separate multiple contexts with '---'):", height=150)
ground_truth = st.text_area("Enter the Ground Truth (optional, required for context_recall):", height=100)

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
            data["ground_truths"] = [[ground_truth.strip()]]  # ragas expects List[List[str]]
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        metrics = [faithfulness, answer_relevancy, context_precision]
        if has_ground_truth:
            metrics.append(context_recall)
        
        try:
            with st.spinner("Evaluating..."):
                result = evaluate(dataset, metrics=metrics)
            st.success("Evaluation complete!")
            st.write("**Results:**")
            for metric, score in result.items():
                st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.info("Common issues: Check OpenAI API key in secrets.toml, or ensure inputs are valid.")

st.markdown("---")
st.markdown("""
### Usage Notes:
- **Contexts**: Provide one or more retrieved documents, separated by '---'. Each will be treated as a separate context chunk.
- **Ground Truth**: Optional but recommended for full metrics (enables context_recall). Provide as a single string.
- **API Key**: Add `OPENAI_API_KEY = "your_key_here"` to your Streamlit Cloud secrets.
- **Customization**: Edit the code to add more metrics or batch evaluation via CSV upload.
""")