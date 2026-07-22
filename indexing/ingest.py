# indexing/ingest.py
"""
PDF文档加载与切分模块
"""
from pathlib import Path
from typing import List
import logging
from urllib.parse import urlparse

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import QaConfig

logger = logging.getLogger(__name__)

class PdfIngestor:
    """PDF文档加载与切分器"""

    # 中文文本分隔符
    SEPARATORS = [
        "\n", "(?<=。)", "(?<=！)", "(?<=？)", "(?<=；)", "(?<=，)", " ", ""
    ]

    def __init__(self, config: QaConfig) -> None:
        """
        初始化PDF加载器
        :param config: 配置信息
        """
        self._config = config

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            is_separator_regex=True,
            separators=self.SEPARATORS,
        )

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档为文本块
        :param documents: 原始 Document 列表
        :return: 切分后的 Document 列表
        """
        if not documents:
            logger.warning("PDF文档为空，无法切分")
            return []

        try:
            chunks = self._splitter.split_documents(documents)
            logger.debug(f"文档切分完成: {len(documents)} 页 → {len(chunks)} 个文本块")
            return chunks
        except Exception as e:
            raise RuntimeError(f"文档切分失败: {e}")

    def ingest(self, pdf_source: str) -> List[Document]:
        """
        从 PDF 文件路径或 URL 加载并切分文档
        :param pdf_source: PDF文件路径或网络 URL
        :return: 切分后的 Document 列表
        """
        if not pdf_source:
            raise ValueError("pdf_source 不能为空")

        # 判断是URL还是本地文件
        if self._is_url(pdf_source):
            logger.info(f"检测到URL，从网络加载PDF: {pdf_source}")
            return self._ingest_from_url(pdf_source)
        else:
            logger.info(f"检测到本地文件，加载PDF: {pdf_source}")
            return self._ingest_from_file(pdf_source)

    def _ingest_from_url(self, url: str) -> List[Document]:
        """
        从URL加载PDF
        :param url: PDF文件的URL
        :return: 切分后的Document列表
        """
        try:
            # PyMuPDFLoader 直接支持 URL
            loader = PyMuPDFLoader(url)
            documents = loader.load()
            return self._split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"从URL加载PDF失败 ({url}): {e}")

    def _is_url(self, source: str) -> bool:
        """
        判断字符串是否为 URL
        :param source: 源字符串
        :return: 是否为 URL
        """
        try:
            result = urlparse(source)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except Exception:
            return False

    def _ingest_from_file(self, file_path: str) -> List[Document]:
        """
        从本地文件加载 PDF
        :param file_path: 文件路径
        :return: 切分后的Document列表
        """
        path = Path(file_path)
        # 验证文件存在
        if not path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")

        try:
            loader = PyMuPDFLoader(str(path))
            documents = loader.load()
            return self._split_documents(documents)
        except Exception as e:
            raise RuntimeError(f"PDF加载失败 ({file_path}): {e}")



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  # 设置为INFO级别，显示INFO及以上日志
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pdf_path = r'E:\python\k-ai-knowledge-2.0\smart-reading\data\files\sample_document.pdf'

    ingestor = PdfIngestor(QaConfig())
    data = ingestor.ingest(pdf_path)
    print(data)