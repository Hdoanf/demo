try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever

    from langchain.storage import InMemoryStore
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    from langchain_community.retrievers import BM25Retriever
    from langchain.chains.query_constructor.base import AttributeInfo

    print("✅ TẤT CẢ THƯ VIỆN OK")

except Exception as e:
    print("❌ LỖI:", e)