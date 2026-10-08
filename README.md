# Nova Tech HR Assistant: Document Q&A Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the
Nova Tech Employee Handbook, a fictional sample document. Instead of relying on
what the LLM already knows, the system retrieves the relevant parts of the
document and gives them to the LLM as context. Answers are grounded in the
source, and the supporting passages are shown with every answer.

Built with LangChain, FAISS, Ollama and Streamlit.

## How it works

```
OFFLINE (run once)                       ONLINE (every question)
PDF -> split into chunks                 User question
    -> embeddings (MiniLM)                   -> embedding
    -> FAISS index (saved to disk)           -> FAISS returns top-k similar chunks
                                             -> prompt (context + question)
                                             -> LLM (Llama 3.2 3B, local via Ollama)
                                             -> answer + sources shown in Streamlit
```

1. **Load:** the PDF is read page by page with `PyPDFLoader`.
2. **Chunk:** pages are split into overlapping chunks of 800 characters
   (150 overlap) with `RecursiveCharacterTextSplitter`.
3. **Embed and index:** each chunk is converted to a 384-dimension vector with
   `all-MiniLM-L6-v2` and stored in a FAISS index.
4. **Retrieve:** the question is embedded and the 4 closest chunks are returned.
5. **Generate:** the chunks and question go into a prompt that tells the LLM to
   answer only from the context and to say so when the answer is not there.
6. **Show sources:** the file name, page number and a preview of each chunk used
   are displayed under the answer.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python |
| Orchestration | LangChain (prompt template, retriever, chain built with `\|`) |
| Document loading | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (runs locally) |
| Vector store | FAISS |
| LLM | llama3.2:3b run locally with Ollama (earlier version: Llama 3.1 8B via Hugging Face Inference Providers) |
| Interface | Streamlit |
| Testing | pytest |

## Project structure

```
data/documents/          Source PDF
faiss_index/             Saved vector index (generated, not committed)
rag_chatbot.ipynb        Ingestion, retrieval experiments, RAG chain
ChatBot.py               Streamlit chat interface
tests/test_retriever.py  Retrieval tests
requirements.txt         Dependencies
```

## Setup

1. **Clone the repository and install dependencies**
```
   pip install -r requirements.txt
```
2. **Install Ollama** from https://ollama.com and download the model:
```
   ollama pull llama3.2:3b
```
   Make sure Ollama is running in the background.
3. **Build the index.** Open `rag_chatbot.ipynb` and run the cells through
   "Save the index". This creates the `faiss_index` folder.
4. **Run the app**
```
   python -m streamlit run ChatBot.py
```
5. **Run the tests** (after the index exists)
```
   python -m pytest tests -q
```

Note: the app uses the local Ollama model and needs no token. The notebook's
generation cells were written against a hosted Hugging Face model and need a
Hugging Face token if you run them as they are.

## Retrieval experiments

To choose the chunking settings I wrote 7 test questions, each with a key phrase
that a correct chunk must contain. A question counts as retrieved if that phrase
appears in one of the returned chunks.

| Chunk size | Top-k | Chunks | Score |
|---|---|---|---|
| 400 | 2 | 25 | 7/7 |
| 400 | 4 | 25 | 7/7 |
| 800 | 2 | 13 | 7/7 |
| 800 | 4 | 13 | 7/7 |
| 1200 | 2 | 11 | 6/7 |
| 1200 | 4 | 11 | 7/7 |

**Chosen settings: chunk size 800, overlap 150,
