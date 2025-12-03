# CRAG 服务使用文档

OpenAI 风格的纠正性检索增强生成（Corrective-RAG）服务。

## 📋 前置要求

### 1. 创建向量数据库
**必须先运行 Jupyter Notebook 创建向量数据库**，服务才能启动：

```bash
# 在 Jupyter 中运行 langgraph_crag_local_zenking.ipynb
# 确保执行了向量数据库创建的 Cell
```

执行后会生成 `chroma_db_for_crag_local_zenking/` 目录。

### 2. 准备环境服务
- ✅ Embedding 服务运行在 `http://192.168.8.230:50009`
- ✅ Ollama 服务运行在 `http://192.168.8.231:11434`
- ✅ Tavily API Key 已配置

## 🚀 快速开始

### 1. 安装依赖
```powershell
cd d:\ai_works\langgraph\examples\rag\crag_service
pip install -r requirements_crag.txt
```

### 2. 配置环境变量
```powershell
# 复制配置模板
Copy-Item .env.example .env

# 编辑 .env 文件
# CHROMA_DB_DIR=chroma_db_for_crag_local_zenking  # 必须存在
# EMBEDDING_BASE_URL=http://192.168.8.230:50009
# OLLAMA_BASE_URL=http://192.168.8.231:11434
# TAVILY_API_KEY=your-tavily-api-key
```

### 3. 启动服务

**Windows PowerShell:**
```powershell
.\start_crag_service.ps1
```

**Linux/Mac:**
```bash
chmod +x start_crag_service.sh
./start_crag_service.sh
```

**或直接运行:**
```bash
python crag_service.py
```

服务将在 `http://localhost:8000` 启动。

### 4. 验证服务

**健康检查:**
```bash
curl http://localhost:8000/health
```

**运行测试:**
```bash
python test_crag_service.py
```

## 📡 API 使用

### 1. OpenAI 兼容接口

**cURL 示例:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "crag-agent",
    "messages": [
      {"role": "user", "content": "失蜡铸造原理是什么？"}
    ]
  }'
```

**Python 示例:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "crag-agent",
        "messages": [
            {"role": "user", "content": "失蜡铸造原理是什么？"}
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
print(f"执行步骤: {result['metadata']['steps']}")
```

### 2. 响应格式

```json
{
  "id": "chatcmpl-abc12345",
  "object": "chat.completion",
  "created": 1733270400,
  "model": "crag-agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "失蜡铸造是一种..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "metadata": {
    "steps": [
      "retrieve_documents",
      "grade_document_retrieval",
      "generate_answer"
    ],
    "documents_count": 4
  }
}
```

## 🔍 工作流程

服务自动执行以下步骤：

1. **retrieve_documents** - 从向量数据库检索相关文档
2. **grade_document_retrieval** - 评估文档相关性
3. **web_search** (可选) - 若文档不相关，触发网络搜索
4. **generate_answer** - 基于文档生成答案

执行轨迹会在响应的 `metadata.steps` 中返回。

## ⚙️ 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHROMA_DB_DIR` | 向量数据库目录（必须已存在） | `chroma_db_for_crag_local_zenking` |
| `EMBEDDING_BASE_URL` | Embedding 服务地址 | `http://192.168.8.230:50009` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://192.168.8.231:11434` |
| `TAVILY_API_KEY` | Tavily 搜索 API Key | - |
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务端口 | `8000` |

### 模型配置

在 `crag_service.py` 中可修改：

```python
CRAGAgent(
    chroma_db_dir=chroma_db_dir,
    embedding_model="BAAI/bge-large-zh-v1.5",  # Embedding 模型
    ollama_model="qwen3:32b",                   # Ollama 模型
    ...
)
```

## 🧪 测试

### 运行完整测试
```bash
python test_crag_service.py
```

### 测试特定场景

**知识库内问题（不触发 Web 搜索）:**
```python
test_chat_completion("失蜡铸造原理是什么?")
# 预期步骤: retrieve_documents → grade_document_retrieval → generate_answer
```

**知识库外问题（触发 Web 搜索）:**
```python
test_chat_completion("北京今天天气怎么样?")
# 预期步骤: retrieve_documents → grade_document_retrieval → web_search → generate_answer
```

## 📊 性能优化

### 1. 向量检索优化
```python
# 调整检索文档数量
self.retriever = self.vectorstore.as_retriever(k=4)  # 默认 4 个
```

### 2. Web 搜索优化
```python
# 调整搜索结果数量
self.web_search_tool = TavilySearchResults(k=3)  # 默认 3 个
```

### 3. 并发配置
```python
# 修改 uvicorn 启动参数
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    workers=4,  # 增加 worker 数量
)
```

## ❗ 常见问题

### 1. 服务启动失败：向量数据库不存在
```
ValueError: 向量数据库目录不存在或为空
```

**解决方案:**
- 先运行 `langgraph_crag_local_zenking.ipynb` 创建向量数据库
- 确认 `CHROMA_DB_DIR` 配置正确

### 2. Embedding 服务连接失败
```
requests.exceptions.ConnectionError
```

**解决方案:**
- 检查 `EMBEDDING_BASE_URL` 是否正确
- 确认 Embedding 服务已启动

### 3. Ollama 服务连接失败
```
ConnectionError: Ollama service unreachable
```

**解决方案:**
- 检查 `OLLAMA_BASE_URL` 是否正确
- 确认模型已下载: `ollama pull qwen3:32b`

### 4. Tavily API 调用失败
```
TavilyAPIError: Invalid API key
```

**解决方案:**
- 检查 `TAVILY_API_KEY` 是否正确
- 确认 API Key 有效且未过期

## 📝 开发说明

### 添加新的评分器
```python
def _build_graph(self):
    # 添加新的评分逻辑
    custom_grader_prompt = PromptTemplate(...)
    self.custom_grader = custom_grader_prompt | self.llm | JsonOutputParser()
```

### 修改工作流
```python
# 在 _build_graph 中修改图结构
workflow.add_node("custom_step", custom_step_func)
workflow.add_edge("grade_documents", "custom_step")
```

### 自定义响应格式
```python
# 在 chat_completions 函数中修改
response = ChatCompletionResponse(
    ...
    metadata={
        "steps": result["steps"],
        "custom_field": "custom_value",
    }
)
```

## 📚 相关资源

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [CRAG 论文](https://arxiv.org/abs/2401.15884)
- [Tavily Search](https://tavily.com/)
- [Ollama](https://ollama.ai/)

## 🆘 支持

遇到问题？请检查：
1. 向量数据库是否已创建
2. 所有环境变量是否正确配置
3. 依赖服务（Embedding/Ollama）是否正常运行
4. 查看服务日志获取详细错误信息