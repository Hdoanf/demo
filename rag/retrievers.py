# rag/retrievers.py
# SỬA LẠI IMPORT CHO ĐÚNG

from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from core.config import llm, vector_store
from core.models import documents


# =========================
# 1. Hybrid Retriever
# =========================
def get_hybrid_retriever(k: int = 7):
    """Hybrid Search: BM25 + Vector"""

    if not documents:
        raise ValueError("Chưa có documents. Cần khởi tạo từ main.py")
    
    if vector_store is None:
        raise ValueError("Chưa có vector_store")

    # BM25 (keyword)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Vector (semantic)
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    # Ensemble
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )

    return ensemble


# =========================
# 2. Rerank Retriever
# =========================
def get_rerank_retriever(k: int = 7):
    """Rerank bằng LLM"""

    if llm is None:
        raise ValueError("Chưa có llm")

    base = get_hybrid_retriever(k)

    compressor = LLMChainExtractor.from_llm(llm)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base
    )

    return retriever


# =========================
# 3. Parent Retriever
# =========================
def get_parent_retriever():
    """Giữ context lớn"""

    if vector_store is None:
        raise ValueError("Chưa có vector_store")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    # Thêm documents vào retriever
    parent_docs = parent_splitter.split_documents(documents)
    retriever.add_documents(parent_docs, ids=None)

    return retriever


# =========================
# 4. Self Query Retriever
# =========================
def get_self_query_retriever():
    """Filter theo metadata"""

    if vector_store is None or llm is None:
        raise ValueError("Chưa có vector_store hoặc llm")

    metadata_field_info = [
        AttributeInfo(name="major", description="Ngành học", type="string"),
        AttributeInfo(name="page", description="Số trang", type="integer"),
        AttributeInfo(name="source", description="Tên file", type="string"),
        AttributeInfo(name="original_name", description="Tên gốc", type="string"),
    ]

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents="Tài liệu học tập sinh viên",
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        verbose=True
    )

    return retriever


# =========================
# 5. Advanced Retriever
# =========================
def get_advanced_retriever(k: int = 7):
    """
    Pipeline: Hybrid → Rerank
    """

    hybrid = get_hybrid_retriever(k * 2)

    rerank = ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm),
        base_retriever=hybrid
    )

    return rerank


# =========================
# 6. Hàm chọn retriever
# =========================
def get_retriever(retriever_type: str = "advanced", k: int = 7):
    """
    Chọn retriever theo loại:
    - hybrid: BM25 + Vector
    - rerank: Hybrid + Rerank
    - parent: Parent Document
    - self_query: Self Query
    - advanced: Hybrid + Rerank (mặc định)
    """
    if retriever_type == "hybrid":
        return get_hybrid_retriever(k)
    elif retriever_type == "rerank":
        return get_rerank_retriever(k)
    elif retriever_type == "parent":
        return get_parent_retriever()
    elif retriever_type == "self_query":
        return get_self_query_retriever()
    else:
        return get_advanced_retriever(k)