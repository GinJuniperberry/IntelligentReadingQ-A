# main.py
"""
主程序入口
"""
import logging
import os

from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    try:
        # 初始化配置
        config = QaConfig()

        # 初始化流水线
        pipeline = IndexingPipeline(config)

        # 你的API Key
        api_key = os.environ.get("DASHSCOPE_API_KEY")

        # ========== 示例1: 从本地文件构建 ==========
        # pdf_path = r"E:\python\k-ai-knowledge-2.0\smart-reading\data\files\sample_document.pdf"
        # logger.info(f"开始从本地文件构建索引: {pdf_path}")
        # file_hash = pipeline.build_from_source(pdf_path, api_key)
        # logger.info(f"✅ 索引构建完成，file_hash: {file_hash}")

        # ========== 示例2: 从网络URL构建 ==========
        pdf_url = "https://gitee.com/he-wenlin/k-ai-knowledge-2.0/raw/master/smart-reading/data/files/sample_document.pdf"
        logger.info(f"开始从URL构建索引: {pdf_url}")
        file_hash = pipeline.build_from_source(pdf_url, api_key)
        logger.info(f"✅ 索引构建完成，file_hash: {file_hash}")

    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {e}")
    except ValueError as e:
        logger.error(f"❌ 参数错误: {e}")
    except Exception as e:
        logger.error(f"❌ 构建失败: {e}")
        raise


if __name__ == "__main__":
    main()