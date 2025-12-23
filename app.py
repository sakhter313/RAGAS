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

# Sidebar: Provider Selection + API Keys
st.sidebar.header("LLM & Embedding Provider")
provider = st.sidebar.selectbox(
    "Choose Provider:",
    ["Google Gemini (Free tier available)", "OpenAI (GPT + Embeddings)"],
    index=0  # Default to Gemini
)

if provider == "Google Gemini (Free tier available)":
    gemini_api_key = st.sidebar.text_input("Gemini API Key:", type="password", help="Get free key at https://aistudio.google.com/app/apikey")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    st.sidebar.success("Gemini selected – generous free tier, no billing required for most use!")
    st.sidebar.info("Uses Gemini 1.5 Flash for generation & embeddings")

else:  # OpenAI
    openai_api_key = st.sidebar.text_input("OpenAI API Key:", type="password", help="Required for GPT models & embeddings")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    st.sidebar.warning("OpenAI may require paid credits if free quota is exhausted")

# Dynamic imports based on provider
if provider.startswith("Google Gemini"):
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    llm_model = "gemini-1.5-flash"
    embedding_model = "models/embedding-001"
    EmbeddingsClass = GoogleGenerativeAIEmbeddings
    ChatLLMClass = ChatGoogleGenerativeAI
else:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm_model = "gpt-3.5-turbo"
    embedding_model = "text-embedding-3-small"
    EmbeddingsClass = OpenAIEmbeddings
    ChatLLMClass = ChatOpenAI

st.title("RAGAS Evaluation App – Gemini / OpenAI Switchable")
st.markdown("Full RAG pipeline + RAGAS evaluation. Now supports **Google Gemini** (recommended – free & fast) or OpenAI.")

# Sample datasets
@st.cache_data
def get_samples():
    return [ ... ]  # (same as before – omitted for brevity, copy from previous version)

samples = get_samples()

# HF Batch Section
st.subheader("Batch Evaluation with Hugging Face Dataset")
st.info("**Gemini Tip**: Free tier allows hundreds of requests/day – safe for N=5–10")
col1, col2, col3 = st.columns(3)
with col1:
    dataset_name = st.text_input("Dataset Name:", value="squad")
with col2:
    split = st.text_input("Split:", value="validation")
with col3:
    num_samples = st.slider("Samples:", 1, 10, 3 if provider.startswith("Google Gemini") else 2)

if st.button("Load & Evaluate HF Dataset"):
    try:
        with st.spinner("Processing..."):
            hf_dataset = load_dataset(dataset_name, split=split)
            subsample = random.sample(range(len(hf_dataset)), num_samples)
            questions = [hf_dataset[i]['question'] for i in subsample]
            ground_truths = [ [hf_dataset[i]['answers']['text'][0]] if hf_dataset[i]['answers']['text'] else [] for i in subsample ]
            contexts_list = [hf_dataset[i]['context'] for i in subsample]

            # Build vectorstore
            docs = [Document(page_content=ctx) for ctx in contexts_list]
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            embeddings = EmbeddingsClass(model=embedding_model)
            vectorstore = Chroma.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # LLM Chain
            llm = ChatLLMClass(model=llm_model, temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

            # Generate
            answers = []
            retrieved_contexts = []
            for q in questions:
                result = qa_chain({"query": q})
                answers.append(result["result"])
                retrieved_contexts.append([doc.page_content for doc in result["source_documents"]])

            # RAGAS
            data = {"question": questions, "answer": answers, "contexts": retrieved_contexts}
            if any(ground_truths):
                data["ground_truths"] = [gt for gt in ground_truths if gt]
            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]
            if "ground_truths" in data:
                metrics.append(context_recall)

            result = evaluate(dataset, metrics=metrics, llm=llm)  # Use same LLM as judge

        st.success("Batch complete!")
        st.subheader("Average Results")
        cols = st.columns(len(result))
        for i, (m, s) in enumerate(result.items()):
            with cols[i]:
                st.metric(m.replace("_", " ").title(), f"{s:.4f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        if "quota" in str(e).lower():
            st.info("Switch to Gemini if using OpenAI – it's free!")

# Single Evaluation (same pattern)
st.subheader("Single Evaluation")
# ... (sample loading, document input, build index, generate, manual mode, evaluate – using dynamic EmbeddingsClass/ChatLLMClass)

@st.cache_resource
def build_index(_docs_input):
    docs = [Document(page_content=d.strip()) for d in _docs_input.split("---") if d.strip()]
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = EmbeddingsClass(model=embedding_model)
    return Chroma.from_documents(splits, embeddings)

# (Rest of single eval code identical to previous, just using dynamic classes)

st.markdown("---")
st.markdown("""
### Key Changes
- **Provider Switch**: Gemini (default) or OpenAI via sidebar.
- **Gemini Integration**: Uses `langchain-google-genai` for both LLM and embeddings.
- **RAGAS Judge**: Uses the selected LLM (Gemini or GPT) for evaluation.
- **Free-Friendly**: Gemini has high free limits – perfect for testing/batch.
""")