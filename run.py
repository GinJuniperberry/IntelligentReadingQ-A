"""PDF question-answering wrapper with conversation history."""

from typing import Any, Dict, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory

from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline
from querying.query_pipeline import RagPipeline


class PDFQA:
    """A small facade for indexing one PDF and asking follow-up questions."""

    def __init__(self, pdf_source: str, api_key: Optional[str] = None):
        """
        Args:
            pdf_source: Local PDF path or HTTP(S) URL.
            api_key: DashScope API key supplied by the caller.
        """
        self.pdf_source = pdf_source
        self.api_key = api_key
        self.cfg = QaConfig()
        self.file_hash: Optional[str] = None
        self.chat_history: Optional[InMemoryChatMessageHistory] = None
        self.rag: Optional[RagPipeline] = None

    def _build_index(self) -> None:
        """Build the vector index lazily on the first question."""
        if self.file_hash is not None:
            return
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not configured.")

        print("Building index. The first run will call the Embedding API...")
        print(f"PDF source: {self.pdf_source}")

        indexer = IndexingPipeline(self.cfg)
        self.file_hash = indexer.build_from_source(self.pdf_source, self.api_key)
        self.chat_history = InMemoryChatMessageHistory()
        self.rag = RagPipeline(self.cfg)

        print(f"Index ready. file_hash: {self.file_hash}")

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask one question and update the conversation history."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.file_hash is None:
            self._build_index()

        result = self.rag.query(
            dashscope_api_key=self.api_key,
            file_hash=self.file_hash,
            question=question.strip(),
            chat_history=self.chat_history.messages,
            top_k=4,
            recall_k=10,
        )

        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(result["answer"])
        return result
