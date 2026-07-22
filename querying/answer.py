# querying/answer.py
# 职责：基于检索到的上下文，生成最终答案（带引用标注）
from typing import List

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .query_prompt import READING_HELPER_PROMPT


class AnswerGenerator:
    """答案生成器：根据问题、对话历史、检索上下文生成答案"""

    def __init__(self, llm: ChatTongyi):
        """
        初始化生成器，预编译 prompt 链
        :param llm: 已配置好通义千问 LLM 实例
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", READING_HELPER_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.__llm = prompt | llm | StrOutputParser()

    def generate(self, question: str, chat_history: List[BaseMessage], context: str) -> str:
        """
        生成答案
        :param question: 用户原始问题（不是改写后的查询）
        :param chat_history: 对话历史对象（需有 .messages 属性，如 ChatMessageHistory）
        :param context: 格式化后的上下文字符串（包含引用编号和片段内容）
        :return: 答案字符串
        """
        # 提取消息列表（兼容不同历史对象）
        return self.__llm.invoke({
            "input": question,
            "chat_history": chat_history,
            "context": context
        })
