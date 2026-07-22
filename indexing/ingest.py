# indexing/ingest.py
"""
PDF文档加载与切分模块
"""
from pathlib import Path
from typing import List
import logging
import re
from urllib.parse import urlparse

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import QaConfig

logger = logging.getLogger(__name__)

class PdfIngestor:
    """PDF文档加载与切分器"""

    QUESTION_START = re.compile(
        r"^(?:\d+\s*[.、)）]|[一二三四五六七八九十]+\s*[.、)）]|(?:问题|问)\s*[:：])"
    )
    BLOCK_START = re.compile(
        r"^(?:\d+\s*[.、)）]|[一二三四五六七八九十]+\s*[.、)）]|(?:问题|问|答案|答)\s*[:：])"
    )
    ANSWER_START = re.compile(r"^(?:答案|答)\s*[:：]")
    CJK_END = re.compile(r"[\u3400-\u9fff]$")
    CJK_START = re.compile(r"^[\u3400-\u9fff]")
    NO_SPACE_END = re.compile(r"[\u3400-\u9fff，。；：！？、）】》]$")
    SENTENCE_END = ("。", "！", "？", "!", "?", "；", ";")

    # 优先在下一道题之前切分，避免把问题和答案拆到两个 chunk。
    SEPARATORS = [
        r"\n(?=\s*(?:\d+\s*[.、)）]|[一二三四五六七八九十]+\s*[.、)）]|(?:问题|问)\s*[:：]))",
        "\n\n",
        "\n",
        r"(?<=[。！？!?；;])",
        r"(?<=[，,])",
        " ",
        "",
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

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Remove PDF visual line wraps while preserving semantic block boundaries."""
        lines = []
        for raw_line in text.replace("\u00ad", "").splitlines():
            line = re.sub(r"[\t \u3000]+", " ", raw_line).strip()
            if line:
                lines.append(line)

        normalized: List[str] = []
        for line in lines:
            if not normalized:
                normalized.append(line)
                continue

            previous = normalized[-1]
            if cls.BLOCK_START.match(line) or previous.endswith(cls.SENTENCE_END):
                normalized.append(line)
                continue

            separator = ""
            if not (cls.NO_SPACE_END.search(previous) and cls.CJK_START.search(line)):
                separator = " "
            normalized[-1] = f"{previous}{separator}{line}"

        return "\n".join(normalized)

    @classmethod
    def _normalize_documents(cls, documents: List[Document]) -> List[Document]:
        normalized_documents: List[Document] = []

        for document in documents:
            if not document.page_content or not document.page_content.strip():
                continue

            text = cls._normalize_text(document.page_content)
            lines = text.splitlines()

            # PDF loaders return one Document per page. Join an answer at the
            # start of a new page back to the question at the previous page end.
            if normalized_documents and lines and cls.ANSWER_START.match(lines[0]):
                previous = normalized_documents[-1]
                previous_last_line = previous.page_content.rsplit("\n", 1)[-1]
                if cls.QUESTION_START.match(previous_last_line):
                    previous.page_content = f"{previous.page_content}\n{lines.pop(0)}"
                    previous.metadata["page_end"] = (document.metadata or {}).get("page")

            if lines:
                normalized_documents.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata=dict(document.metadata or {}),
                    )
                )

        return normalized_documents

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
            normalized_documents = self._normalize_documents(documents)
            chunks = self._splitter.split_documents(normalized_documents)
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
