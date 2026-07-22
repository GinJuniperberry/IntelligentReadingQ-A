from indexing.storage import PdfBytesStore

store = PdfBytesStore()
pdf_path = r"E:\python\k-ai-knowledge-2.0\smart-reading\data\files\sample_document.pdf"

# 先创建字符串，再编码为字节串
text = '你好你好'
print(store.compute_hash(text.encode('utf-8')))

text2 = '你好你好'
print(store.compute_hash(text2.encode('utf-8')))

text3 = '你好你d好'
print(store.compute_hash(text3.encode('utf-8')))

text4 = '你好你 好'
print(store.compute_hash(text4.encode('utf-8')))