# Nova Tech HR Assistant: Document Q&A Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the
Nova Tech Employee Handbook, a fictional sample document. Instead of relying on
what the LLM already knows, the system retrieves the relevant parts of the
document and gives them to the LLM as context. Answers are grounded in the
source, and the supporting passages are shown with every answer.

Built with LangChain, FAISS, Hugging Face and Streamlit.

## How it works

```
OFFLINE (run once)                       ONLINE (every question)
PDF -> split into chunks                 User question
    -> embeddings (MiniLM)                   -> embedding
    -> FAISS index (saved to disk)           -> FAISS returns top-k similar chunks
                                             -> prompt (context + question)
                                             -> LLM (Llama 3.1 8B Instruct)
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
| LLM | meta-llama/Llama-3.1-8B-Instruct via Hugging Face Inference Providers |
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
2. **Create a Hugging Face access token** at
   https://huggingface.co/settings/tokens with permission to make calls to
   Inference Providers. Make sure at least one inference provider that serves the
   model is enabled in your account settings.
3. **Build the index.** Open `rag_chatbot.ipynb`, enter your token when asked,
   and run the cells through "Save the index". This creates the `faiss_index`
   folder.
4. **Set your token** in the terminal you will run the app from (Windows):
```
   set HF_TOKEN=your_token_here
```
5. **Run the app**
```
   python -m streamlit run ChatBot.py
```
   If you skip step 4, the app asks for the token in the sidebar.  

6. **Run the tests** (after the index exists)
```
   python -m pytest tests -q
```

The token is never stored in the project. Do not commit it.

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

**Chosen settings: chunk size 800, overlap 150, top-k 4.** It retrieved every
answer and produced readable, self-contained chunks. Larger chunks mix several
topics and dilute the match (1200 with k=2 missed a question). Very small chunks
can split a rule from its context.

Limits of this test: the question set is small, and it measures retrieval, not
the quality of the final generated answer.

## Example questions

| Question | Behaviour |
|---|---|
| How many sick leave days do I get? | Answers 6 and shows the source chunk |
| What is the notice period after confirmation? | 60 days up to Senior Engineer, 90 days for Lead and above |
| What is the hotel limit per night in metro cities? | 6,000 INR |
| What is the CEO's name? | "I couldn't find that in the handbook." (not in the document) |

## Challenges and how I solved them

- **Token not recognised.** `huggingface_hub` reads `HF_TOKEN`, not the variable
  name I first used. I set both and entered the token with `getpass` so it is
  never saved in the notebook.
- **PyTorch failed to load on Windows** (`shm.dll` error). I isolated it by
  testing `import torch` on its own, then repaired the installation.
- **The first LLM was not available.** No enabled provider served
  Qwen2.5-7B-Instruct. I wrote a loop that sent a test message to several
  candidate models, then switched to Llama 3.1 8B Instruct. Because settings live
  in one place, this was a one-line change.
- **The retriever cannot say "I don't know".** Top-k always returns k chunks, even
  when none contain the answer. The prompt therefore instructs the LLM to answer
  only from the context and otherwise say it could not find the answer, and the
  app hides the sources when it declines.

## Limitations and future work

- Each question is answered independently, so there is no conversation memory
  for follow-up questions.
- Only one short document is indexed.
- Answer quality is checked by hand. A larger labelled test set would allow
  measuring it.
- Possible extensions: conversation memory, multiple documents, and a check that
  the answer is supported by the retrieved text before showing it.

## Notes

The handbook is a fictional sample document created for practice. This project
uses LangChain only, without LangGraph.
