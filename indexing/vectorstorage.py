# indexing/vectorstore.py
"""
向量库管理模块
"""
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from config.setting import QaConfig

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量库管理器"""

    def __init__(self, config: QaConfig, embeddings: DashScopeEmbeddings):
        """
        初始化向量库管理器
        :param config: 系统配置
        :param embeddings: Embedding模型实例
        """
        self._config = config
        self._embeddings = embeddings

    def load_or_build(
            self,
            file_hash: str,
            chunks: Optional[List[Document]] = None
    ) -> Chroma:
        """
        加载已有向量库，如果不存在则新建
        :param file_hash: PDF文件的MD5哈希
        :param chunks: 切分后的Document列表（新建时必须提供）
        :return: Chroma 向量库实例
        """
        if not file_hash:
            raise ValueError("文件哈希不能为空")

        persist_dir = self._get_persist_dir(file_hash)
        collection_name = self._get_collection_name(file_hash)

        # 创建或加载向量库
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"}
        )

        # 检查是否已有数据
        if self._has_data(vector_store):
            logger.info(f"加载已有向量库: {collection_name}")
            return vector_store

        # 新建向量库
        if not chunks:
            raise ValueError("向量库不存在且未提供chunks，无法建库")

        logger.info(f"新建向量库: {collection_name}")
        return self._build_vector_store(vector_store, chunks)

    def _has_data(self, vector_store: Chroma) -> bool:
        """ 检查向量库是否已有数据 """
        try:
            return vector_store._collection.count() > 0
        except Exception as e:
            logger.warning(f"检查向量库数据失败: {e}")
            return False

    def _build_vector_store(
            self,
            vector_store: Chroma,
            chunks: List[Document]
    ) -> Chroma:
        """构建向量库"""
        if not chunks:
            raise ValueError("文档块列表不能为空")

        try:
            vector_store.add_documents(
                documents=chunks,
                ids=[f"doc-{idx}" for idx in range(1, len(chunks) + 1)]
            )
            logger.info(f"成功添加 {len(chunks)} 个文档块到向量库")
            return vector_store
        except Exception as e:
            logger.error(f"构建向量库失败: {e}")
            raise

    def _get_persist_dir(self, file_hash: str) -> Path:
        """获取向量库持久化目录"""
        persist_dir = Path(self._config.chroma_root_dir) / f"pdf_{file_hash}"
        # parents=True 自动创建所有缺失的父目录。
        # exist_ok=True 如果目录已经存在，不报错。
        persist_dir.mkdir(parents=True, exist_ok=True)
        return persist_dir

    def _get_collection_name(self, file_hash: str) -> str:
        """获取向量库集合名称"""
        return f"{self._config.collection_prefix}_{file_hash}"