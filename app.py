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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter  # Updated import for v0.3+
from langchain_classic.chains import RetrievalQA  # Legacy chain via classic
from langchain_core.documents import Document
import openai  # For error handling

# Sidebar for API key (overrides secrets for local testing)
api_key = st.sidebar.text_input("OpenAI API Key:", type="password", help="Enter your key here for local testing. Use secrets on Streamlit Cloud.")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

st.title("RAGAS Evaluation App with LangChain RAG & HF Datasets")
st.markdown("Evaluate RAG outputs manually, via LangChain pipeline, or batch on Hugging Face datasets. Supports samples for quick testing. Updated for Dec 2025 compat.")

# Sample datasets (as before)
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
            "expected_answer": "Python is a high-level programming language known for its simplicity and readability.",
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
            "expected_answer": "The capital of France is Paris, a major European city famous for its culture and history.",
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
            "expected_answer": "Key principles include fairness, transparency, accountability, and privacy in AI systems.",
            "ground_truth": "Fairness, transparency, accountability, privacy."
        }
    ]

samples = get_samples()

# HF Dataset Section
st.subheader("Batch Evaluation with Hugging Face Dataset")
st.warning("**Quota Tip**: Batch eval uses multiple OpenAI calls (embeddings + generations + RAGAS). Start with N=1 to avoid quota errors. Upgrade plan if needed: https://platform.openai.com/account/usage")
col1, col2, col3 = st.columns(3)
with col1:
    dataset_name = st.text_input("Dataset Name (e.g., squad):", value="squad")
with col2:
    split = st.text_input("Split (e.g., validation):", value="validation")
with col3:
    num_samples = st.slider("Number of Samples to Evaluate:", min_value=1, max_value=5, value=1)  # Reduced max/default

if st.button("Load & Evaluate HF Dataset"):
    try:
        with st.spinner("Loading dataset and evaluating... This may take 1-3 minutes for small N."):
            # Load HF dataset
            hf_dataset = load_dataset(dataset_name, split=split)
            # Subsample
            subsample = random.sample(range(len(hf_dataset)), num_samples)
            questions = [hf_dataset[i]['question'] for i in subsample]
            ground_truths = []
            for i in subsample:
                answers = hf_dataset[i].get('answers', {}).get('text', [])
                ground_truths.append([answers[0]] if answers else [])
            contexts_list = [hf_dataset[i]['context'] for i in subsample]
            
            # Build index from subsample contexts
            all_docs = [Document(page_content=ctx) for ctx in contexts_list]
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(all_docs)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = Chroma.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduced k to save calls
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
            
            # Generate answers
            answers = []
            retrieved_contexts = []
            for q in questions:
                result = qa_chain({"query": q})
                answers.append(result["result"])
                retrieved = [doc.page_content for doc in result["source_documents"]]
                retrieved_contexts.append(retrieved)
            
            # Prepare RAGAS data
            data = {
                "question": questions,
                "answer": answers,
                "contexts": retrieved_contexts,
            }
            has_gt = any(gt for gt in ground_truths)
            if has_gt:
                data["ground_truths"] = [gt for gt in ground_truths if gt]
            
            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]
            if has_gt:
                metrics.append(context_recall)
            
            result = evaluate(dataset, metrics=metrics)
        
        st.success("Batch evaluation complete!")
        st.subheader("Average Results")
        cols = st.columns(len(result))
        for i, (metric, score) in enumerate(result.items()):
            with cols[i]:
                st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
        
        # Optional: Show first sample details
        if st.checkbox("Show First Sample Details"):
            st.write(f"**Question:** {questions[0]}")
            st.write(f"**Generated Answer:** {answers[0]}")
            st.write(f"**Retrieved Contexts:** {'---'.join(retrieved_contexts[0])}")
            if ground_truths[0]:
                st.write(f"**Ground Truth:** {ground_truths[0][0]}")
    except openai.APIError as e:
        if e.code == "insufficient_quota" or "quota" in str(e).lower():
            st.error(f"OpenAI Quota Exceeded: {str(e).split('message')[0] if 'message' in str(e) else str(e)} Check billing: https://platform.openai.com/account/usage")
            st.info("Workaround: Use Single Evaluation mode or reduce N to 1. Upgrade to paid plan for more usage.")
        else:
            st.error(f"OpenAI API Error: {str(e)}")
    except Exception as e:
        st.error(f"Failed to load/evaluate: {str(e)}")
        st.info("Tips: Use small datasets like 'squad' (validation split). Ensure dataset has 'question', 'context', 'answers' keys.")

# Single Evaluation Section
st.subheader("Single Evaluation (Manual or LangChain RAG)")
# Sample loading
selected_sample = st.selectbox("Choose a sample (optional):", [""] + [s["name"] for s in samples])
if selected_sample:
    sample = next(s for s in samples if s["name"] == selected_sample)
    if st.button(f"Load {selected_sample}"):
        st.session_state.question = sample["question"]
        st.session_state.documents = "---".join(sample["documents"])
        st.session_state.ground_truth = sample["ground_truth"]
        if "documents" in st.session_state and st.session_state.documents:
            st.session_state.vectorstore = build_index(st.session_state.documents)
        st.rerun()

# RAG Pipeline
documents_input = st.text_area(
    "Enter Documents (separate with '---'):", 
    value=st.session_state.get("documents", ""), 
    height=150, 
    key="docs_input"
)
if st.button("Build Index"):
    try:
        if documents_input.strip():
            st.session_state.vectorstore = build_index(documents_input)
            st.success("Index built!")
        else:
            st.error("Provide documents.")
    except openai.APIError as e:
        if e.code == "insufficient_quota" or "quota" in str(e).lower():
            st.error(f"OpenAI Quota Exceeded during indexing: {str(e).split('message')[0] if 'message' in str(e) else str(e)} Check billing: https://platform.openai.com/account/usage")
        else:
            st.error(f"OpenAI API Error during indexing: {str(e)}")

@st.cache_resource
def build_index(_docs_input):
    docs = [Document(page_content=doc.strip()) for doc in _docs_input.split("---") if doc.strip()]
    if not docs:
        return None
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(splits, embeddings)

# Inputs
if "question" not in st.session_state: st.session_state.question = ""
if "answer" not in st.session_state: st.session_state.answer = ""
if "contexts" not in st.session_state: st.session_state.contexts = ""
if "ground_truth" not in st.session_state: st.session_state.ground_truth = ""
if "documents" not in st.session_state: st.session_state.documents = ""
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None

question = st.text_area("Question:", value=st.session_state.question, height=80, key="q_input")
st.session_state.question = question

col1, col2 = st.columns(2)
with col1:
    if st.session_state.vectorstore and st.button("Generate Answer with RAG"):
        try:
            with st.spinner("Generating..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduced k
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
                result = qa_chain({"query": question})
                st.session_state.answer = result["result"]
                st.session_state.contexts = "---".join([doc.page_content for doc in result["source_documents"]])
            st.rerun()
        except openai.APIError as e:
            if e.code == "insufficient_quota" or "quota" in str(e).lower():
                st.error(f"OpenAI Quota Exceeded during generation: {str(e).split('message')[0] if 'message' in str(e) else str(e)} Check billing: https://platform.openai.com/account/usage")
            else:
                st.error(f"OpenAI API Error during generation: {str(e)}")

with col2:
    manual_mode = st.checkbox("Manual Mode", value=not st.session_state.vectorstore)
    if manual_mode:
        generated_answer = st.text_area("Generated Answer:", value=st.session_state.answer, height=80, key="a_input")
        contexts_input = st.text_area("Retrieved Contexts:", value=st.session_state.contexts, height=120, key="c_input")
    else:
        generated_answer = st.session_state.answer
        contexts_input = st.session_state.contexts
        st.info("Auto-populated from RAG.")

st.session_state.answer = generated_answer if manual_mode else st.session_state.answer
st.session_state.contexts = contexts_input if manual_mode else st.session_state.contexts

ground_truth = st.text_area("Ground Truth (optional):", value=st.session_state.ground_truth, height=80, key="gt_input")
st.session_state.ground_truth = ground_truth

if st.button("Evaluate Single"):
    if not question or not generated_answer or not contexts_input:
        st.error("Fill required fields.")
    else:
        try:
            contexts = [ctx.strip() for ctx in contexts_input.split("---") if ctx.strip()]
            data = {"question": [question], "answer": [generated_answer], "contexts": [contexts]}
            has_gt = bool(ground_truth.strip())
            if has_gt:
                data["ground_truths"] = [[ground_truth.strip()]]
            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]
            if has_gt:
                metrics.append(context_recall)
            with st.spinner("Evaluating..."):
                result = evaluate(dataset, metrics=metrics)
            st.success("Complete!")
            st.subheader("Single Results")
            cols = st.columns(len(result))
            for i, (metric, score) in enumerate(result.items()):
                with cols[i]:
                    st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
        except openai.APIError as e:
            if e.code == "insufficient_quota" or "quota" in str(e).lower():
                st.error(f"OpenAI Quota Exceeded during evaluation: {str(e).split('message')[0] if 'message' in str(e) else str(e)} Check billing: https://platform.openai.com/account/usage")
            else:
                st.error(f"OpenAI API Error during evaluation: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("""
### Notes (Dec 23, 2025 Update):
- **Quota Handling**: Added specific catch for insufficient_quota (429-like). Reduced batch max to 5, default 1; k=2 for retrieval. Warnings & links to billing.
- **Compat**: As before; all imports stable.
- **HF Batch**: Improved error handling; try N=1 for testing.
- **Single**: Safer with try-excepts around API ops.
- **Tips**: For free tier limits, stick to manual/single. Paid plans unlock more.
""")