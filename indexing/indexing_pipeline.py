# indexing/pipeline.py
"""
索引构建流水线
"""
import logging
from typing import Optional
from urllib.parse import urlparse

import requests

from config.setting import QaConfig
from .ingest import PdfIngestor
from .storage import PdfBytesStore
from .vectorstorage import VectorStoreManager
from langchain_community.embeddings import DashScopeEmbeddings

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """索引构建流水线：PDF → 切分 → Embedding → 持久化Chroma"""

    def __init__(self, config: QaConfig):
        self._config = config
        self._store = PdfBytesStore()
        self._ingestor = PdfIngestor(self._config)

    def build_from_source(
            self,
            pdf_source: str,
            dashscope_api_key: str
    ) -> str:
        """
        从PDF源（本地文件或URL）构建索引
        :param pdf_source: PDF文件路径或网络URL
        :param dashscope_api_key: DashScope API Key
        :return: 文件 MD5 哈希
        """
        if not pdf_source:
            raise ValueError("PDF源不能为空")
        if not dashscope_api_key:
            raise ValueError("DashScope API Key不能为空")

        # 1. 计算文件哈希
        file_hash = self._compute_hash(pdf_source)
        logger.info(f"PDF文件哈希: {file_hash}")

        # 2. 加载并切分文档
        chunks = self._ingestor.ingest(pdf_source)

        if not chunks:
            raise ValueError("PDF切分后未生成任何文本块")

        # 3. 构建向量库
        embeddings = DashScopeEmbeddings(
            model=self._config.embedding_model,
            dashscope_api_key=dashscope_api_key
        )

        vector_store = VectorStoreManager(self._config, embeddings)
        vector_store.load_or_build(file_hash, chunks)

        logger.info(f"索引构建完成，file_hash: {file_hash}")
        return file_hash

    def _compute_hash(self, pdf_source: str) -> str:
        """计算PDF源的哈希值"""
        # 判断是否为URL
        try:
            result = urlparse(pdf_source)
            is_url = all([result.scheme in ['http', 'https'], result.netloc])
        except Exception:
            is_url = False

        if is_url:
            # URL: 下载后计算哈希
            response = requests.get(pdf_source, timeout=30)
            response.raise_for_status()
            pdf_bytes = response.content
            return self._store.compute_hash(pdf_bytes)
        else:
            # 本地文件: 流式计算哈希
            return self._store.compute_hash_from_file(pdf_source)
