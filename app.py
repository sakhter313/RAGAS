import streamlit as st
import os
import pandas as pd
import numpy as np
from io import StringIO
import random  # For mocks

# Core deps (always needed)
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    st.error("Plotly missing – install via requirements.txt.")
    HAS_PLOTLY = False

# Lazy heavy deps
try:
    from datasets import load_dataset
    from evaluate import load
    HAS_DATA_LIBS = True
except ImportError:
    st.error("Datasets/Evaluate missing.")
    HAS_DATA_LIBS = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HAS_MODEL_LIBS = True
except ImportError:
    st.error("Transformers missing.")
    HAS_MODEL_LIBS = False

# RAG-specific (if using)
try:
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings  # Free fallback
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Quota-prone
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import evaluate  # If using ragas
    from ragas.metrics import faithfulness
    HAS_RAG_LIBS = True
except ImportError:
    st.warning("RAG libs missing – eval fallback to mocks.")
    HAS_RAG_LIBS = False

# Page config
st.set_page_config(page_title="RAGas AI EvalHub", layout="wide")

@st.cache_resource
def load_model_and_metrics():
    if not HAS_MODEL_LIBS or not HAS_DATA_LIBS:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        rouge = load("rouge")
        return generator, rouge
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None

@st.cache_data
def load_datasets(sample_size=50):
    if not HAS_DATA_LIBS:
        # Mock
        mock_truthful = {
            'question': [f"Mock Q{i}: Fact about {random.choice(['apple', 'AI'])}?" for i in range(sample_size)],
            'correct_answers': [[f"Truth: {random.choice(['Red fruit', 'ML tech'])}."] for i in range(sample_size)]
        }
        mock_crows = [{'stereo_antistereo': random.choice(['stereo', 'antistereo'])} for _ in range(sample_size)]
        return mock_truthful, mock_crows
    truthful_qa = load_dataset("truthful_qa", "generation", split=f"validation[:{sample_size}]")
    crows_pairs = load_dataset("nyu-mll/crows_pairs", split=f"test[:{sample_size}]")
    return truthful_qa, crows_pairs

def get_embeddings():
    """Fallback embeddings: Gemini if key, else local HF."""
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if api_key:
        try:
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        except Exception as e:
            st.warning(f"Gemini quota/error: {e}. Falling back to local embeddings.")
    # Free local fallback
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(documents_text):
    if not HAS_RAG_LIBS:
        st.warning("RAG not available – skipping index build.")
        return None
    if not documents_text.strip():
        st.error("Add documents first.")
        return None
    try:
        with st.spinner("Building index..."):
            # Mock docs if empty
            docs = [TextLoader.from_text(StringIO(documents_text)).load()[0]] if documents_text else [{"page_content": "Mock doc for eval."}]
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            embeddings = get_embeddings()
            vectorstore = Chroma.from_documents(splits, embeddings)
        st.success("Index built!")
        return vectorstore
    except Exception as e:
        st.error(f"Build failed: {e}. Check quota/API key.")
        return None

def evaluate_hallucination(generator, rouge, prompts, truths):
    scores = []
    for prompt, truth in zip(prompts, truths):
        if generator and rouge:
            response = generator(prompt, max_length=50, num_return_sequences=1, do_sample=False)[0]['generated_text']
            score = rouge.compute(predictions=[response], references=[truth])['rougeL'][0]
        else:
            score = np.random.uniform(0.4, 0.8)  # Mock
        scores.append(score)
    return np.mean(scores) * 100

def evaluate_bias(crows_pairs_sample):
    if not crows_pairs_sample:
        return 0.0
    if isinstance(crows_pairs_sample, dict):  # Mock
        num_stereo = sum(1 for item in crows_pairs_sample['stereo_antistereo'] if item == 'stereo')
        total = len(crows_pairs_sample['stereo_antistereo'])
    else:
        num_stereo = sum(1 for row in crows_pairs_sample if row['stereo_antistereo'] == 'stereo')
        total = len(crows_pairs_sample)
    return (num_stereo / total * 100) if total > 0 else 0.0

def run_rag_eval(question, vectorstore):
    if not vectorstore:
        return "No index – build first."
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro")  # Or fallback to OpenAI/GPT2
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
        response = qa_chain.run(question)
        # Ragas eval (subset for speed)
        result = evaluate({"question": [question], "answer": [response], "contexts": [vectorstore.similarity_search(question)]},
                          metrics=[faithfulness])
        return f"Response: {response}\nFaithfulness: {result['faithfulness']:.2f}"
    except Exception as e:
        return f"RAG eval failed: {e} (quota?)."

def main():
    st.title("🤖 RAGas AI EvalHub: Hallucinations, Bias & RAG Testing")
    st.markdown("Upload docs for RAG, or eval model directly. Free embeddings fallback active.")

    # Tabs for sections
    tab1, tab2, tab3 = st.tabs(["RAG Build", "Hallucination/Bias Eval", "Query RAG"])

    with tab1:
        st.subheader("Build Vectorstore")
        documents_input = st.text_area("Paste documents (or upload TXT):", height=150)
        if st.button("Build Index"):
            st.session_state.vectorstore = build_vectorstore(documents_input)

    with tab2:
        st.subheader("Model Eval (Hallucinations & Bias)")
        sample_size = st.slider("Sample Size", 10, 100, 50)
        run_eval = st.button("Run Eval")
        if run_eval:
            with st.spinner("Loading..."):
                truthful_qa, crows_pairs = load_datasets(sample_size)
                generator, rouge = load_model_and_metrics()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hallucination Accuracy (%)", f"{evaluate_hallucination(generator, rouge, truthful_qa['question'][:5], [item[0] for item in truthful_qa['correct_answers'][:5]]):.1f}")
            with col2:
                st.metric("Bias Score (%)", f"{evaluate_bias(crows_pairs):.1f}")

            if HAS_PLOTLY:
                df_metrics = pd.DataFrame({"Metric": ["Halluc Acc", "Bias"], "Value": [hall_score, bias_score]})
                fig = px.bar(df_metrics, x="Metric", y="Value", color="Value", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig)

    with tab3:
        st.subheader("Query RAG")
        question = st.text_input("Ask a question:")
        if st.button("Query") and 'vectorstore' in st.session_state:
            response = run_rag_eval(question, st.session_state.vectorstore)
            st.write(response)
        else:
            st.info("Build index first.")

if __name__ == "__main__":
    main()