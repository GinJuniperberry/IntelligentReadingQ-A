"""LLM-assisted reranking for retrieved PDF chunks."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.setting import QaConfig
from querying.query_prompt import RERANK_PROMPT


@dataclass
class RetrievedEvidence:
    """Evidence chunk returned after reranking."""

    content: str
    source: str
    page: Optional[int]
    score: float
    vector_score: float
    llm_score: Optional[float] = None


class RerankPipeline:
    """Recall -> lightweight filtering -> LLM batch scoring."""

    def __init__(self, config: QaConfig, llm: ChatTongyi) -> None:
        self.__config = config
        self.__llm = llm
        self.__batch_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RERANK_PROMPT),
                (
                    "human",
                    "问题：{question}\n\n{chunks}\n\n"
                    "请为以上每个候选片段打分，格式为：\n"
                    "候选片段1: X.X\n候选片段2: X.X\n...\n",
                ),
            ]
        )
        self.__batch_chain = self.__batch_prompt | self.__llm | StrOutputParser()

    def rerank(
        self,
        query: str,
        recalled_docs: List[Document],
        vec_scores: List[float],
        final_top_n: int = 4,
    ) -> Tuple[List[RetrievedEvidence], str]:
        pre_docs = self._filter_documents(recalled_docs)
        score_map = {doc.page_content: score for doc, score in zip(recalled_docs, vec_scores)}

        final_docs, llm_scores = self.__llm_rerank_batch(query, pre_docs, top_n=final_top_n)
        if not final_docs:
            fallback = [
                (score, doc)
                for doc, score in zip(recalled_docs, vec_scores)
                if doc in pre_docs
            ]
            fallback.sort(key=lambda item: item[0], reverse=True)
            selected = fallback[:final_top_n]
            final_docs = [doc for _, doc in selected]
            llm_scores = [round(max(0.0, min(1.0, score)) * 10, 1) for score, _ in selected]

        evidence_list = []
        for index, doc in enumerate(final_docs):
            content = doc.page_content
            meta = doc.metadata or {}
            evidence_list.append(
                RetrievedEvidence(
                    content=content,
                    source=meta.get("source", "Unknown"),
                    page=meta.get("page"),
                    score=llm_scores[index],
                    vector_score=score_map.get(content, 0.0),
                    llm_score=llm_scores[index],
                )
            )

        context_lines = []
        for index, evidence in enumerate(evidence_list, 1):
            page = f"第 {evidence.page + 1} 页" if evidence.page is not None else "未知页"
            context_lines.append(
                f"[证据{index} | 来源：{evidence.source} | {page}]\n{evidence.content}"
            )

        return evidence_list, "\n\n".join(context_lines)

    @staticmethod
    def _filter_documents(
        docs: List[Document],
        min_len: int = 1,
        max_len: int = 2000,
    ) -> List[Document]:
        filtered = []
        for doc in docs:
            text = doc.page_content
            if min_len <= len(text) <= max_len:
                filtered.append(doc)

        seen = set()
        kept = []
        for doc in filtered:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                kept.append(doc)
        return kept

    def __llm_rerank_batch(
        self,
        question: str,
        documents: List[Document],
        top_n: int = 4,
    ) -> Tuple[List[Document], List[float]]:
        if not documents:
            return [], []

        chunks_text = []
        for index, doc in enumerate(documents, 1):
            content = doc.page_content[:1000]
            chunks_text.append(f"候选片段{index}:\n{content}")

        try:
            result_text = self.__batch_chain.invoke(
                {
                    "question": question,
                    "chunks": "\n\n".join(chunks_text),
                }
            )
            scores = self.__parse_batch_scores(result_text, len(documents))
            scores = [max(0.0, min(10.0, score)) for score in scores]

            scored = list(zip(scores, documents))
            scored.sort(key=lambda item: item[0], reverse=True)

            docs = [doc for _, doc in scored[:top_n]]
            final_scores = [score for score, _ in scored[:top_n]]
            return docs, final_scores
        except Exception as exc:
            print(f"批量评分失败: {exc}")
            return [], []

    def __parse_batch_scores(self, text: str, expected_count: int) -> List[float]:
        pattern = r"候选片段\s*(\d+)\s*[:：]\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, text)

        if matches:
            score_dict = {int(index): float(score) for index, score in matches}
            return [score_dict.get(index + 1, 0.0) for index in range(expected_count)]

        numbers = re.findall(r"\b(?:10(?:\.0)?|[0-9](?:\.\d+)?)\b", text)
        if len(numbers) >= expected_count:
            return [float(number) for number in numbers[:expected_count]]

        return [0.0] * expected_count
