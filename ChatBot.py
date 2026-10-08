import os
import time
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ---------------------------------------------------------------- settings
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "llama3.2:3b"
TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SAMPLE_QUESTIONS = [
    "How many sick leave days do I get?",
    "What is the notice period after confirmation?",
    "What is the hotel limit per night in metro cities?",
    "What is the password policy?",
    "How much is the learning budget?",
]

st.set_page_config(page_title="Nova Tech HR Assistant", layout="centered")
st.title("Nova Tech HR Assistant")
st.caption(
    "Answers grounded in the Nova Tech Employee Handbook 2026, "
    "with the supporting sources shown for every answer."
)

# ---------------------------------------------------------------- pipeline
@st.cache_resource(show_spinner="Loading models and document index...")
def load_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatOllama(model=LLM_REPO_ID, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful HR assistant for Nova Tech Solutions. "
         "Answer the question using ONLY the context below. "
         "If the answer is not in the context, say: "
         "\"I couldn't find that in the handbook.\" "
         "Do not make things up.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return retriever, chain


retriever, chain = load_components()

# ---------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def set_pending(q):
    st.session_state.pending = q


def clear_chat():
    st.session_state.messages = []


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Try a question")
    for i, q in enumerate(SAMPLE_QUESTIONS):
        st.button(q, key=f"sample_{i}", on_click=set_pending, args=(q,),
                  use_container_width=True)
    st.button("Clear conversation", key="clear", on_click=clear_chat,
              use_container_width=True)

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. The question is converted into an embedding.\n"
        "2. The most similar handbook chunks are retrieved from a FAISS index.\n"
        "3. The chunks and the question are placed in a prompt.\n"
        "4. The LLM answers using only that context.\n"
        "5. The source chunks are shown so the answer can be verified."
    )

    st.divider()
    st.subheader("Configuration")
    st.markdown(
        f"**LLM:** {LLM_REPO_ID.split('/')[-1]}  \n"
        f"**Embeddings:** {EMBEDDING_MODEL.split('/')[-1]}  \n"
        f"**Vector store:** FAISS  \n"
        f"**Chunk size / overlap:** {CHUNK_SIZE} / {CHUNK_OVERLAP}  \n"
        f"**Top-k retrieved:** {TOP_K}"
    )


# ---------------------------------------------------------------- helpers
def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            with st.container(border=True):
                st.markdown(f"**[{i}] {s['file']}, page {s['page']}**")
                st.caption(s["text"])


def is_refusal(answer):
    return "couldn't find that" in answer.lower()


AVATARS = {"user": ":material/person:", "assistant": ":material/smart_toy:"}

# ---------------------------------------------------------------- chat view
if not st.session_state.messages:
    st.info(
        "Ask about company policy: leave, working hours, remote work, expenses, "
        "IT security, benefits and exit rules are covered. "
        "Pick a sample question from the sidebar or type your own."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("elapsed"):
            st.caption(f"Answered in {msg['elapsed']:.1f} s")

# ---------------------------------------------------------------- new question
question = st.chat_input("Ask a question about the handbook")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            start = time.perf_counter()
            with st.spinner("Searching the handbook..."):
                docs = retriever.invoke(question)
                context = "\n\n".join(d.page_content for d in docs)
                answer = chain.invoke({"context": context, "question": question})
            elapsed = time.perf_counter() - start

            sources = []
            if not is_refusal(answer):
                sources = [
                    {
                        "file": Path(d.metadata.get("source", "unknown")).name,
                        "page": d.metadata.get("page", 0) + 1,
                        "text": d.page_content[:300].replace("\n", " ") + "...",
                    }
                    for d in docs
                ]

            st.markdown(answer)
            if sources:
                render_sources(sources)
            st.caption(f"Answered in {elapsed:.1f} s")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer,
                 "sources": sources, "elapsed": elapsed}
            )
        except Exception as e:
            st.error(
                "The language model could not be reached. Check your token and "
                "internet connection, then try again."
            )
            st.caption(f"Details: {str(e)[:200]}")
