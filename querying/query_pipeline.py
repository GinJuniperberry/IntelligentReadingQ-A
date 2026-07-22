"""Online RAG query pipeline."""

from typing import Any, Dict

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from config.setting import QaConfig
from indexing.vectorstorage import VectorStoreManager
from .answer import AnswerGenerator
from .query_rewrite import QueryRewriter
from .rerank import RerankPipeline
from .vector_retriever import recall_with_scores


class RagPipeline:
    """Query an existing PDF vector index and generate an answer."""

    def __init__(self, config: QaConfig):
        self.cfg = config
        self._llm = None
        self._embeddings = None
        self._vectorstores = {}

    def _init_llm(self, api_key: str) -> ChatTongyi:
        if self._llm is None:
            self._llm = ChatTongyi(model="qwen3-max", api_key=api_key)
        return self._llm

    def _init_embeddings(self, api_key: str) -> DashScopeEmbeddings:
        if self._embeddings is None:
            self._embeddings = DashScopeEmbeddings(
                model=self.cfg.embedding_model,
                dashscope_api_key=api_key,
            )
        return self._embeddings

    def _load_vectorstore(self, file_hash: str, api_key: str) -> Chroma:
        if file_hash not in self._vectorstores:
            embeddings = self._init_embeddings(api_key)
            manager = VectorStoreManager(self.cfg, embeddings)
            self._vectorstores[file_hash] = manager.load_or_build(file_hash)
        return self._vectorstores[file_hash]

    def query(
        self,
        dashscope_api_key: str,
        file_hash: str,
        question: str,
        chat_history,
        *,
        top_k: int = 4,
        recall_k: int = 10,
    ) -> Dict[str, Any]:
        try:
            llm = self._init_llm(dashscope_api_key)
            vectorstore = self._load_vectorstore(file_hash, dashscope_api_key)

            rewriter = QueryRewriter(llm)
            search_query = rewriter.rewrite(question, chat_history)

            recalled_docs, vec_scores = recall_with_scores(
                search_query,
                vectorstore,
                k=recall_k,
            )

            rerank_pipeline = RerankPipeline(self.cfg, llm)
            evidence_list, context = rerank_pipeline.rerank(
                query=search_query,
                recalled_docs=recalled_docs,
                vec_scores=vec_scores,
                final_top_n=top_k,
            )

            answer_gen = AnswerGenerator(llm)
            answer = answer_gen.generate(search_query, chat_history, context)

            return {
                "answer": answer,
                "context": context,
                "evidence": evidence_list,
                "search_query": search_query,
                "rewritten": search_query != question,
                "doc_count": len(evidence_list),
            }
        except Exception as exc:
            return {
                "error": f"查询失败: {exc}",
                "answer": "抱歉，处理请求时出现错误，请检查 API Key、模型可用性和 PDF 索引状态。",
                "context": "",
                "evidence": [],
                "search_query": question,
                "rewritten": False,
                "doc_count": 0,
            }
