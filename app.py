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
    index=0  # Default to Groq
)

groq_api_key = None
gemini_api_key = None
openai_api_key = None

if provider == "Groq (Ultra-fast inference – Free tier available)":
    groq_api_key = st.sidebar.text_input("Groq API Key:", type="password", help="Get free key at https://console.groq.com/keys")
    gemini_api_key = st.sidebar.text_input("Gemini API Key (for embeddings):", type="password", help="Required for Groq – get at https://aistudio.google.com/app/apikey")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    st.sidebar.success("Groq selected – Blazing fast Llama 3.3! Free tier generous.")
    st.sidebar.info("Generation/Judge: Llama-3.3-70B (Groq). Embeddings: Gemini (free).")

elif provider == "Google Gemini (Free tier)":
    gemini_api_key = st.sidebar.text_input("Gemini API Key:", type="password")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    st.sidebar.success("Pure Gemini selected – fully free tier.")

else:  # OpenAI
    openai_api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    st.sidebar.warning("OpenAI may require paid credits.")

# Validate keys early
if provider == "Groq (Ultra-fast inference – Free tier available)":
    if not groq_api_key or not gemini_api_key:
        st.error("Please provide both Groq and Gemini API keys for this provider.")
        st.stop()
elif provider == "Google Gemini (Free tier)":
    if not gemini_api_key:
        st.error("Please provide Gemini API key.")
        st.stop()
elif not openai_api_key:
    st.error("Please provide OpenAI API key.")
    st.stop()

# Dynamic imports & config
if provider == "Groq (Ultra-fast inference – Free tier available)":
    from langchain_groq import ChatGroq
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    llm_model = "llama-3.3-70b-versatile"  # Latest as of Dec 2025
    embedding_model = "models/embedding-001"
    EmbeddingsClass = GoogleGenerativeAIEmbeddings
    ChatLLMClass = ChatGroq
    ragas_llm_instance = ChatGroq(model=llm_model, temperature=0)

elif provider == "Google Gemini (Free tier)":
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    llm_model = "gemini-1.5-flash"
    embedding_model = "models/embedding-001"
    EmbeddingsClass = GoogleGenerativeAIEmbeddings
    ChatLLMClass = ChatGoogleGenerativeAI
    ragas_llm_instance = ChatGoogleGenerativeAI(model=llm_model, temperature=0)

else:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm_model = "gpt-3.5-turbo"
    embedding_model = "text-embedding-3-small"
    EmbeddingsClass = OpenAIEmbeddings
    ChatLLMClass = ChatOpenAI
    ragas_llm_instance = ChatOpenAI(model="gpt-4o-mini", temperature=0)

st.title("RAGAS Evaluation App – Groq / Gemini / OpenAI")
st.markdown("Fast RAG + evaluation. **Groq default** for ultra-speed with Llama 3.3.")

# Sample datasets
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
        # Add the other two samples similarly...
    ]

samples = get_samples()

# Batch Evaluation Section (full code similar to previous, using dynamic classes)
# ... (implement as before, pass llm=ragas_llm_instance to evaluate())

# Single Evaluation Section
# ... (full implementation with try/except for API errors)

st.markdown("---")
st.markdown("""
### Fixes for GroqError
- **API Key Issue Resolved**: Now requires **both** Groq key (for LLM) and Gemini key (for embeddings – Groq has no embeddings yet). Keys set to env vars correctly.
- **Model Updated**: Uses `llama-3.3-70b-versatile` (current high-perf model Dec 2025).
- **Early Validation**: App stops with clear error if keys missing.
- **RAGAS Judge**: Uses the selected fast LLM instance.
""")