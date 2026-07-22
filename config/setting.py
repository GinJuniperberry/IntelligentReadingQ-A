# config/setting.py
"""
配置管理模块
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class QaConfig:
    """QA系统配置"""
    # 文本切分配置
    chunk_size: int = 500
    chunk_overlap: int = 80
    index_version: str = "v3"

    # 图片型 PDF OCR 配置
    ocr_dpi: int = 144
    ocr_min_confidence: float = 0.45

    # 向量库配置
    chroma_root_dir: str = "./chroma_db"

    collection_prefix: str = 'pdf_qa'

    # Embedding配置
    embedding_model: str = "text-embedding-v1"

    # 临时文件目录
    temp_dir: str = "./temp"

    def __post_init__(self):
        """初始化后创建必要目录"""
        for dir_path in [self.chroma_root_dir, self.temp_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
