"""Query rewriting module."""

import logging
from typing import List, Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from querying.query_prompt import REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrite context-dependent questions into standalone retrieval queries."""

    REFERENCE_KEYWORDS = {"那篇", "刚才", "它", "这个", "该", "那", "这", "其"}

    def __init__(self, llm: ChatTongyi) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REWRITE_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "用户问题：{question}\n改写后的查询："),
            ]
        )
        self._chain = self._prompt | llm | StrOutputParser()

    def rewrite(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> str:
        question = (question or "").strip()
        if not question:
            return question

        if not self._needs_rewrite(question):
            logger.debug(f"Question does not need rewriting: {question}")
            return question

        try:
            rewrite = self._chain.invoke(
                {
                    "question": question,
                    "chat_history": chat_history or [],
                }
            ).strip()
            logger.info(f"Query rewritten: '{question}' -> '{rewrite}'")
            return rewrite or question
        except Exception as exc:
            logger.warning(f"Query rewrite failed: {exc}; using original question")
            return question

    def _needs_rewrite(self, question: str) -> bool:
        if len(question) < 5:
            return False
        return any(keyword in question for keyword in self.REFERENCE_KEYWORDS)
