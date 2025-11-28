"""
RAG pipeline orchestration logic powered by LangChain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:  # Optional dependency
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None

from .config import Settings


class RAGPipeline:
    """Encapsulates the Retrieval-Augmented Generation workflow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self.embedding_function = self._init_embeddings()
        self.llm = self._init_llm()
        self.vector_store: Optional[FAISS] = None
        self.retriever = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="result"
        )
        self.qa_chain: Optional[RetrievalQA] = None

    def _init_embeddings(self):
        """Return embedding model (OpenAI if key set, otherwise SentenceTransformer)."""
        if self.settings.openai_api_key:
            return OpenAIEmbeddings(model="text-embedding-3-large")
        return HuggingFaceEmbeddings(model_name=self.settings.embedding_model_name)

    def _init_llm(self):
        """Return configured chat model."""
        provider = self.settings.llm_provider.lower()
        if provider == "groq":
            if not ChatGroq:
                raise RuntimeError(
                    "langchain-groq is required for Groq provider. Install extra and retry."
                )
            if not self.settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY must be set for Groq provider.")
            return ChatGroq(
                groq_api_key=self.settings.groq_api_key,
                model_name=self.settings.llm_model_name,
                temperature=0.1,
            )
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for OpenAI provider.")
        return ChatOpenAI(
            api_key=self.settings.openai_api_key,
            model=self.settings.llm_model_name,
            temperature=0.2,
        )

    def load_documents(self, file_path: Path) -> List[Document]:
        """Load PDF documents using LangChain loader."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into semantically coherent chunks."""
        return self.text_splitter.split_documents(documents)

    def build_vector_store(self, chunks: List[Document]) -> None:
        """Create FAISS index from document chunks."""
        if not chunks:
            raise ValueError("No chunks to index.")
        self.vector_store = FAISS.from_documents(chunks, self.embedding_function)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.settings.top_k}
        )
        self.memory.clear()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for arbitrary text payloads."""
        if not texts:
            return []
        return self.embedding_function.embed_documents(texts)

    def is_ready(self) -> bool:
        """Check if vector store is initialized."""
        return self.qa_chain is not None and self.vector_store is not None

    def query(self, question: str) -> Dict[str, Any]:
        """Execute RetrievalQA query and format response."""
        if not self.is_ready():
            raise RuntimeError("Vector store not initialized. Upload a document first.")
        result = self.qa_chain.invoke({"query": question})
        answer = result.get("result")
        source_documents: List[Document] = result.get("source_documents", [])
        sources = [
            {
                "page_content": doc.page_content[:300],
                "metadata": doc.metadata,
            }
            for doc in source_documents
        ]
        history = [
            {"role": msg.type, "content": msg.content}
            for msg in self.memory.chat_memory.messages
        ]
        return {"answer": answer, "sources": sources, "chat_history": history}


__all__ = ["RAGPipeline"]


