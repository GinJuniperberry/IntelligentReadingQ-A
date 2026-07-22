import unittest

from langchain_core.documents import Document

from config.setting import QaConfig
from indexing.ingest import PdfIngestor
from querying.rerank import RerankPipeline


class PdfIngestRegressionTest(unittest.TestCase):
    def test_short_question_and_answer_are_kept_together(self):
        document = Document(
            page_content="1．世界上最美的女人是谁？\n答：世界上最美的女\n人是范冰冰。",
            metadata={"page": 0},
        )

        chunks = PdfIngestor(QaConfig())._split_documents([document])

        self.assertEqual(len(chunks), 1)
        self.assertIn("1．世界上最美的女人是谁？", chunks[0].page_content)
        self.assertIn("答：世界上最美的女人是范冰冰。", chunks[0].page_content)

    def test_short_retrieved_chunk_is_not_filtered_out(self):
        document = Document(page_content="问题？\n答：简短答案。")

        kept = RerankPipeline._filter_documents([document])

        self.assertEqual(kept, [document])

    def test_answer_on_next_page_is_joined_to_its_question(self):
        documents = [
            Document(page_content="9. 跨页问题是什么？", metadata={"page": 0}),
            Document(
                page_content="答：这是下一页开头的答案。\n10. 下一题是什么？\n答：下一题答案。",
                metadata={"page": 1},
            ),
        ]

        chunks = PdfIngestor(QaConfig())._split_documents(documents)
        combined = "\n".join(chunk.page_content for chunk in chunks)

        self.assertIn("9. 跨页问题是什么？\n答：这是下一页开头的答案。", combined)


if __name__ == "__main__":
    unittest.main()
