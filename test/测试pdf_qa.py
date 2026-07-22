import os
import sys
from run import PDFQA

pdf_path = r"E:\python\k-ai-knowledge-2.0\smart-reading\data\files\sample_document.pdf"

api_key = os.environ.get("DASHSCOPE_API_KEY")
qa = PDFQA(pdf_path, api_key=api_key)

# 模拟逐条发送
q1 = "GPT-3的论文链接是什么？"
r1 = qa.ask(q1)
print(f"Q: {q1}\nA: {r1['answer']}\n")

q2 = "它的Transformer层数有多少层？"   # 这里的“它”依赖上一轮历史
r2 = qa.ask(q2)
print(f"Q: {q2}\nA: {r2['answer']}\n")