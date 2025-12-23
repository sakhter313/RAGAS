import streamlit as st
import os
import random
from datasets import Dataset, load_dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

# ==================== CONFIG & PROVIDER SETUP ====================

PROVIDERS = {
    "Groq (Ultra-fast inference – Free tier available)": {
        "llm_class": "ChatGroq",
        "embedding_class": "GoogleGenerativeAIEmbeddings",
        "llm_model": "llama-3.3-70b-versatile",
        "embedding_model": "models/embedding-001",
        "needs_keys": ["GROQ_API_KEY", "GOOGLE_API_KEY"],
        "description": "Generation/Judge: Llama-3.3-70B (Groq). Embeddings: Gemini (free)."
    },
    "Google Gemini (Free tier)": {
        "llm_class": "ChatGoogleGenerativeAI",
        "embedding_class": "GoogleGenerativeAIEmbeddings",
        "llm_model": "gemini-1.5-flash",
        "embedding_model": "models/embedding-001",
        "needs_keys": ["GOOGLE_API_KEY"],
        "description": "Fully Gemini – free & reliable."
    },
    "OpenAI": {
        "llm_class": "ChatOpenAI",
        "embedding_class": "OpenAIEmbeddings",
        "llm_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-3-small",
        "needs_keys": ["OPENAI_API_KEY"],
        "description": "GPT-3.5-turbo + OpenAI embeddings."
    }
}

# Sidebar Provider Selection
st.sidebar.header("Provider Selection")
selected_provider_name = st.sidebar.selectbox(
    "Choose Provider:",
    options=list(PROVIDERS.keys()),
    index=0
)

provider_config = PROVIDERS[selected_provider_name]

# API Key Inputs
api_keys = {}
for env_var in provider_config["needs_keys"]:
    label = env_var.replace("_API_KEY", "").title() + " API Key"
    help_text = {
        "GROQ_API_KEY": "Get at https://console.groq.com/keys",
        "GOOGLE_API_KEY": "Get at https://aistudio.google.com/app/apikey",
        "OPENAI_API_KEY": "Get at https://platform.openai.com/api-keys"
    }.get(env_var, "")
    api_keys[env_var] = st.sidebar.text_input(label, type="password", help=help_text)
    if api_keys[env_var]:
        os.environ[env_var] = api_keys[env_var]

# Validate keys
missing_keys = [k for k in provider_config["needs_keys"] if not api_keys.get(k)]
if missing_keys:
    st.error(f"Please provide the following API key(s): {', '.join([k.replace('_API_KEY', '') for k in missing_keys])}")
    st.stop()

st.sidebar.success(f"{selected_provider_name} configured")
st.sidebar.info(provider_config["description"])

# Dynamic Imports
if "Groq" in selected_provider_name:
    from langchain_groq import ChatGroq
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
elif "Gemini" in selected_provider_name:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
else:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Instantiate classes
ChatLLMClass = globals()[provider_config["llm_class"]]
EmbeddingsClass = globals()[provider_config["embedding_class"]]

llm_model = provider_config["llm_model"]
embedding_model = provider_config["embedding_model"]

ragas_llm = ChatLLMClass(model=llm_model, temperature=0)
embeddings = EmbeddingsClass(model=embedding_model)

# ==================== SAMPLE DATA ====================

@st.cache_data
def get_samples():
    return [
        {
            "name": "Sample 1: Python Basics",
            "question": "What is Python?",
            "documents": [
                "Python is a high-level programming language designed for readability. It was created by Guido van Rossum in the late 1980s.",
                "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
                "It is widely used in web development, data science, automation, and more."
            ],
            "ground_truth": "Python is a high-level programming language."
        },
        {
            "name": "Sample 2: Capital of France",
            "question": "What is the capital of France?",
            "documents": [
                "France is a country in Western Europe with a population of over 67 million.",
                "Its capital city is Paris, which is also the largest city in the country.",
                "Paris is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
            ],
            "ground_truth": "Paris"
        },
        {
            "name": "Sample 3: AI Ethics",
            "question": "What are some key principles of AI ethics?",
            "documents": [
                "AI ethics encompasses principles like fairness to avoid bias in algorithms.",
                "Transparency ensures that AI decisions are explainable to users and stakeholders.",
                "Accountability requires developers to take responsibility for AI system outcomes.",
                "Privacy is crucial to protect user data and prevent misuse in AI applications."
            ],
            "ground_truth": "Fairness, transparency, accountability, privacy."
        }
    ]

samples = get_samples()

# ==================== HELPER FUNCTIONS ====================

def build_vectorstore(documents_text: str):
    docs = [Document(page_content=d.strip()) for d in documents_text.split("---") if d.strip()]
    if not docs:
        return None
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    return Chroma.from_documents(splits, embeddings)

def run_rag(question: str, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatLLMClass(model=llm_model, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    result = qa_chain({"query": question})
    return result["result"], [doc.page_content for doc in result["source_documents"]]

def evaluate_ragas(question, answer, contexts, ground_truth=None):
    contexts_list = [c.strip() for c in contexts.split("---") if c.strip()] if isinstance(contexts, str) else contexts
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts_list],
    }
    if ground_truth and ground_truth.strip():
        data["ground_truths"] = [[ground_truth.strip()]]

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if "ground_truths" in data:
        metrics.append(context_recall)

    return evaluate(dataset, metrics=metrics, llm=ragas_llm)

# ==================== MAIN APP ====================

st.title("RAGAS Evaluation App – Modular & Multi-Provider")
st.markdown("Clean, modular code with Groq (default), Gemini, or OpenAI support.")

# --- Batch Evaluation ---
st.subheader("Batch Evaluation with Hugging Face Dataset")
col1, col2, col3 = st.columns(3)
with col1:
    dataset_name = st.text_input("Dataset Name:", value="squad")
with col2:
    split = st.text_input("Split:", value="validation")
with col3:
    max_samples = 15 if "Groq" in selected_provider_name or "Gemini" in selected_provider_name else 5
    num_samples = st.slider("Samples:", 1, max_samples, 5)

if st.button("Load & Evaluate HF Dataset"):
    try:
        with st.spinner("Running batch RAG + RAGAS evaluation..."):
            hf_dataset = load_dataset(dataset_name, split=split)
            indices = random.sample(range(len(hf_dataset)), num_samples)
            questions = [hf_dataset[i]["question"] for i in indices]
            ground_truths = [
                [hf_dataset[i]["answers"]["text"][0]] if hf_dataset[i]["answers"]["text"] else []
                for i in indices
            ]
            contexts_list = [hf_dataset[i]["context"] for i in indices]

            vectorstore = build_vectorstore("---".join(contexts_list))
            answers = []
            retrieved_contexts = []
            for q in questions:
                ans, ctxs = run_rag(q, vectorstore)
                answers.append(ans)
                retrieved_contexts.append(ctxs)

            result = evaluate_ragas(
                questions, answers, retrieved_contexts,
                ground_truths[0][0] if ground_truths and ground_truths[0] else None
            )  # Simplified – uses first GT as proxy; for full use list

        st.success("Batch complete!")
        st.subheader("Average Results")
        cols = st.columns(len(result))
        for i, (m, s) in enumerate(result.items()):
            with cols[i]:
                st.metric(m.replace("_", " ").title(), f"{s:.4f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Single Evaluation ---
st.subheader("Single Evaluation")

# Sample loader
selected_sample = st.selectbox("Load Sample:", ["None"] + [s["name"] for s in samples])
if selected_sample != "None":
    sample = next(s for s in samples if s["name"] == selected_sample)
    if st.button(f"Load {selected_sample}"):
        st.session_state.question = sample["question"]
        st.session_state.documents = "---".join(sample["documents"])
        st.session_state.ground_truth = sample["ground_truth"]
        st.session_state.vectorstore = None
        st.rerun()

# Document input
documents_input = st.text_area(
    "Documents (--- separated):",
    value=st.session_state.get("documents", ""),
    height=150
)

if st.button("Build Index"):
    if documents_input.strip():
        with st.spinner("Building index..."):
            st.session_state.vectorstore = build_vectorstore(documents_input)
        st.success("Index built!")
    else:
        st.error("Add documents.")

# Session state
defaults = {"question": "", "answer": "", "contexts": "", "ground_truth": "", "documents": documents_input, "vectorstore": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

question = st.text_area("Question:", value=st.session_state.question, height=80)
st.session_state.question = question

col1, col2 = st.columns(2)
with col1:
    if st.session_state.vectorstore and st.button("Generate Answer with RAG"):
        with st.spinner("Generating..."):
            answer, contexts = run_rag(question, st.session_state.vectorstore)
            st.session_state.answer = answer
            st.session_state.contexts = "---".join(contexts)
        st.rerun()

with col2:
    manual = st.checkbox("Manual Mode", value=not bool(st.session_state.vectorstore))
    if manual:
        st.session_state.answer = st.text_area("Answer:", value=st.session_state.answer, height=80)
        st.session_state.contexts = st.text_area("Contexts:", value=st.session_state.contexts, height=120)

ground_truth = st.text_area("Ground Truth (optional):", value=st.session_state.ground_truth, height=80)
st.session_state.ground_truth = ground_truth

if st.button("Evaluate Single"):
    if not question or not st.session_state.answer or not st.session_state.contexts:
        st.error("Fill required fields.")
    else:
        try:
            with st.spinner("Evaluating..."):
                result = evaluate_ragas(
                    question,
                    st.session_state.answer,
                    st.session_state.contexts,
                    ground_truth
                )
            st.success("Done!")
            st.subheader("Results")
            cols = st.columns(len(result))
            for i, (m, s) in enumerate(result.items()):
                with cols[i]:
                    st.metric(m.replace("_", " ").title(), f"{s:.4f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Modular design: Config-driven providers, reusable helpers, clean separation.")