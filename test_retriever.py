from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_retriever(k=4):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    return store.as_retriever(search_kwargs={"k": k})


def test_returns_k_chunks():
    results = load_retriever(k=4).invoke("How many sick leave days do I get?")
    assert len(results) == 4


def test_sick_leave_answer_is_retrieved():
    results = load_retriever().invoke("How many sick leave days do I get?")
    text = " ".join(d.page_content for d in results).lower()
    assert "sick leave" in text


def test_notice_period_answer_is_retrieved():
    results = load_retriever().invoke("What is the notice period after confirmation?")
    text = " ".join(d.page_content.replace("\n", " ") for d in results)
    assert "60 days" in text
