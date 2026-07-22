"""
向量检索模块
"""
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document


def recall_with_scores(
    query: str,
    vectorstore: Chroma,
    k: int = 30
) -> Tuple[List[Document], List[float]]:
    """
    向量检索召回
    :param query: 查询字符串
    :param vectorstore: Chroma向量库实例
    :param k: 召回数量
    :return: (文档列表, 分数列表)
    """
    if not query or not vectorstore:
        return [], []

    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

        docs = []
        scores = []
        for doc, score in results:
            doc.page_content = doc.page_content.strip()
            docs.append(doc)
            scores.append(score)

        return docs, scores
    except Exception as e:
        print(f"向量检索失败: {e}")
        return [], []