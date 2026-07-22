# PDF 智能阅读问答 Demo

这是一个基于 PDF 的 RAG 问答演示项目。系统会先读取 PDF、切分文本、调用 DashScope Embedding 构建 Chroma 向量库，再通过检索、重排和通义千问生成答案。

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

启动后浏览器会打开 Streamlit 页面。也可以手动访问终端里显示的本地地址，通常是：

```text
http://localhost:8501
```

在左侧栏输入 DashScope API Key 后即可提问。Key 仅用于当前 Streamlit 会话，不会写入项目文件。

## 目录结构

```text
smart-reading/
├─ app.py                       # Streamlit 演示入口
├─ run.py                       # PDFQA 封装类
├─ requirements.txt             # 演示项目依赖
├─ config/
│  └─ setting.py                # QA 配置
├─ indexing/                    # PDF 读取、切分、向量索引
├─ querying/                    # 查询改写、召回、重排、回答生成
├─ data/files/sample_document.pdf
└─ test/                        # 原始脚本示例
```

## 上线配置

部署到 Streamlit Community Cloud 或其他平台时，可以由每位演示者在页面左侧输入自己的 Key；也可以在平台环境变量或 secrets 中配置默认 Key：

```text
DASHSCOPE_API_KEY=你的 DashScope API Key
```

Streamlit Community Cloud 的 secrets 可写为：

```toml
DASHSCOPE_API_KEY = "你的 DashScope API Key"
```
