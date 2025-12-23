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
st.sidebar.header("Provider Selection")
provider = st.sidebar.selectbox(
    "Choose Provider:",
    ["Groq (Ultra-fast inference – Free tier available)", "Google Gemini (Free tier)", "OpenAI"],
    index=0  # Default to Groq for speed
)

if provider == "Groq (Ultra-fast inference – Free tier available)":
    groq_api_key = st.sidebar.text_input("Groq API Key:", type="password", help="Get free key at https://console.groq.com/keys")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    st.sidebar.success("Groq selected – Blazing fast (1000+ tokens/sec on Llama3)! Free tier generous.")
    st.sidebar.info("Uses Llama3-70B for generation & RAGAS judge. Embeddings: Gemini (free)")

elif provider == "Google Gemini (Free tier)":
    gemini_api_key = st.sidebar.text_input("Gemini API Key:", type="password", help="Get at https://aistudio.google.com/app/apikey")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    st.sidebar.success("Gemini selected – Generous free tier")

else:  # OpenAI
    openai_api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    st.sidebar.warning("OpenAI may hit quota limits quickly")

# Dynamic imports
if provider == "Groq (Ultra-fast inference – Free tier available)":
    from langchain_groq import ChatGroq
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    llm_model = "llama-3.3-70b-versatile"  # Latest high-perf as of Dec 2025
    embedding_model = "models/embedding-001"
    EmbeddingsClass = GoogleGenerativeAIEmbeddings
    ChatLLMClass = ChatGroq
    ragas_llm = ChatGroq(model=llm_model, temperature=0)

elif provider == "Google Gemini (Free tier)":
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    llm_model = "gemini-1.5-flash"
    embedding_model = "models/embedding-001"
    EmbeddingsClass = GoogleGenerativeAIEmbeddings
    ChatLLMClass = ChatGoogleGenerativeAI
    ragas_llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0)

else:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm_model = "gpt-3.5-turbo"
    embedding_model = "text-embedding-3-small"
    EmbeddingsClass = OpenAIEmbeddings
    ChatLLMClass = ChatOpenAI
    ragas_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Efficient judge

st.title("RAGAS Evaluation App – Groq / Gemini / OpenAI")
st.markdown("Ultra-fast RAG + evaluation. **Groq default** for lightning inference (Llama3). Embeddings via Gemini (free & fast).")

# Samples (same as before – omitted for brevity)

# Batch Section
st.subheader("Batch Evaluation with HF Dataset")
st.info("**Groq Tip**: Insanely fast – safe for N=10+ even on free tier!")
# ... (same as previous, using dynamic classes; pass llm=ragas_llm to evaluate())

# In evaluate call:
result = evaluate(dataset, metrics=metrics, llm=ragas_llm)

# Single Section
# ... (use EmbeddingsClass(model=embedding_model), ChatLLMClass(model=llm_model, temperature=0))

# In build_index and generation: same dynamic

st.markdown("---")
st.markdown("""
### Key Updates (Dec 23, 2025)
- **Groq Integration**: Default provider – uses Groq for generation & RAGAS judge (ultra-fast Llama3-70B). Embeddings via Gemini (Groq has no native embeddings yet).
- **Free & Fast**: Groq free tier + Gemini embeddings = near-zero cost, 1000+ t/s speed.
- **Fallbacks**: Pure Gemini or OpenAI if needed.
- **RAGAS**: Judge uses selected fast LLM.
""")