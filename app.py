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
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ==================== SINGLE PLATFORM: GROQ + GEMINI EMBEDDINGS ====================

st.sidebar.header("API Keys (Groq + Gemini Embeddings)")

groq_api_key = st.sidebar.text_input("Groq API Key:", type="password", help="Get free key at https://console.groq.com/keys")
gemini_api_key = st.sidebar.text_input("Google Gemini API Key (for embeddings):", type="password", help="Get free key at https://aistudio.google.com/app/apikey")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Validate both keys
if not groq_api_key or not gemini_api_key:
    st.error("Please provide both Groq API Key and Google Gemini API Key to continue.")
    st.info("Groq powers ultra-fast generation & RAGAS judging (Llama 3.3 70B). Gemini provides free, high-quality embeddings.")
    st.stop()

st.sidebar.success("Groq + Gemini configured – Blazing fast & free tier friendly!")
st.sidebar.info("Generation & Judge: Llama-3.3-70B (Groq) | Embeddings: Gemini")

# Fixed configuration (single platform)
llm_model = "llama-3.3-70b-versatile"
embedding_model = "models/embedding-001"

ragas_llm = ChatGroq(model=llm_model, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

st.title("RAGAS Evaluation App – Groq + Gemini (Single Platform)")
st.markdown("Simplified, high-performance setup: **Groq** for lightning-fast LLM inference + **Gemini** for free embeddings.")

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
    llm = ChatGroq(model=llm_model, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    result = qa_chain({"query": question})
    return result["result"], [doc.page_content for doc in result["source_documents"]]

def evaluate_ragas(question, answer, contexts, ground_truth=None):
    contexts_list = [c.strip() for c in contexts.split("---") if c.strip()] if isinstance(contexts, str) else contexts
    data = {
        "question": [question] if isinstance(question, str) else question,
        "answer": [answer] if isinstance(answer, str) else answer,
        "contexts": [contexts_list] if isinstance(contexts_list[0], str) else contexts_list,
    }
    if ground_truth and ground_truth.strip():
        data["ground_truths"] = [[ground_truth.strip()]]

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if "ground_truths" in data:
        metrics.append(context_recall)

    return evaluate(dataset, metrics=metrics, llm=ragas_llm)

# ==================== MAIN APP ====================

# --- Batch Evaluation ---
st.subheader("Batch Evaluation with Hugging Face Dataset")
st.info("Groq is extremely fast – safely run 10–15 samples!")
col1, col2, col3 = st.columns(3)
with col1:
    dataset_name = st.text_input("Dataset Name:", value="squad")
with col2:
    split = st.text_input("Split:", value="validation")
with col3:
    num_samples = st.slider("Number of Samples:", 1, 15, 8)

if st.button("Load & Evaluate HF Dataset"):
    try:
        with st.spinner("Running batch RAG + RAGAS evaluation (Groq speed!)..."):
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

            result = evaluate_ragas(questions, answers, retrieved_contexts)

        st.success("Batch evaluation complete!")
        st.subheader("Average Results")
        cols = st.columns(len(result))
        for i, (m, s) in enumerate(result.items()):
            with cols[i]:
                st.metric(m.replace("_", " ").title(), f"{s:.4f}")

        if st.checkbox("Show First Sample Details"):
            st.write("**Question:**", questions[0])
            st.write("**Generated Answer:**", answers[0])
            st.write("**Retrieved Contexts:**", "---".join(retrieved_contexts[0]))
            if ground_truths[0]:
                st.write("**Ground Truth:**", ground_truths[0][0])

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Single Evaluation ---
st.subheader("Single Evaluation")

selected_sample = st.selectbox("Load Sample:", ["None"] + [s["name"] for s in samples])
if selected_sample != "None":
    sample = next(s for s in samples if s["name"] == selected_sample)
    if st.button(f"Load {selected_sample}"):
        st.session_state.question = sample["question"]
        st.session_state.documents = "---".join(sample["documents"])
        st.session_state.ground_truth = sample["ground_truth"]
        st.session_state.vectorstore = None
        st.rerun()

documents_input = st.text_area(
    "Documents (--- separated):",
    value=st.session_state.get("documents", ""),
    height=150
)

if st.button("Build Index"):
    if documents_input.strip():
        with st.spinner("Building index with Gemini embeddings..."):
            st.session_state.vectorstore = build_vectorstore(documents_input)
        st.success("Index built!")
    else:
        st.error("Please add at least one document.")

# Session state initialization
for key in ["question", "answer", "contexts", "ground_truth", "documents", "vectorstore"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "vectorstore" else None

question = st.text_area("Question:", value=st.session_state.question, height=80)
st.session_state.question = question

col1, col2 = st.columns(2)
with col1:
    if st.session_state.vectorstore and st.button("Generate Answer with RAG"):
        with st.spinner("Generating with Groq (very fast!)..."):
            answer, contexts = run_rag(question, st.session_state.vectorstore)
            st.session_state.answer = answer
            st.session_state.contexts = "---".join(contexts)
        st.rerun()

with col2:
    manual = st.checkbox("Manual Mode", value=not bool(st.session_state.vectorstore))
    if manual:
        st.session_state.answer = st.text_area("Answer:", value=st.session_state.answer, height=80)
        st.session_state.contexts = st.text_area("Contexts (--- separated):", value=st.session_state.contexts, height=120)

ground_truth = st.text_area("Ground Truth (optional):", value=st.session_state.ground_truth, height=80)
st.session_state.ground_truth = ground_truth

if st.button("Evaluate Single"):
    if not question or not st.session_state.answer or not st.session_state.contexts:
        st.error("Please fill Question, Answer, and Contexts.")
    else:
        try:
            with st.spinner("Running RAGAS evaluation (Groq judging)..."):
                result = evaluate_ragas(
                    question,
                    st.session_state.answer,
                    st.session_state.contexts,
                    ground_truth
                )
            st.success("Evaluation complete!")
            st.subheader("Results")
            cols = st.columns(len(result))
            for i, (m, s) in enumerate(result.items()):
                with cols[i]:
                    st.metric(m.replace("_", " ").title(), f"{s:.4f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Single-platform optimized: Groq (LLM + Judge) + Gemini (Embeddings) → Maximum speed, minimal cost.")