import os
import uuid
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ==================== Embedding 实现 ====================
class OpenAIStyleEmbeddings(Embeddings):
    """适配 OpenAI /v1/embeddings 风格接口的嵌入实现"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {"input": list(texts), "model": self.model}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "crag-service/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        data = result.get("data")
        if not data:
            raise ValueError(f"Embedding service returned no data: {result}")
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


# ==================== Graph State ====================
class GraphState(TypedDict):
    """图状态定义"""

    question: str
    generation: str
    search: str
    documents: List[Document]
    steps: List[str]


# ==================== API Models ====================
class Message(BaseModel):
    """消息模型"""

    role: str = Field(..., description="角色: user, assistant, system")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天补全请求 (OpenAI 兼容)"""

    model: str = Field(default="crag-agent", description="模型名称")
    messages: List[Message] = Field(..., description="对话消息列表")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False, description="是否流式返回")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class ChatCompletionResponse(BaseModel):
    """聊天补全响应 (OpenAI 兼容)"""

    id: str = Field(..., description="响应ID")
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    choices: List[Dict[str, Any]] = Field(..., description="生成的回复")
    usage: Dict[str, int] = Field(..., description="Token 使用统计")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="执行轨迹等元数据"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    version: str
    model: str


# ==================== CRAG Agent 封装 ====================
class CRAGAgent:
    """纠正性检索增强生成智能体"""

    def __init__(
        self,
        chroma_db_dir: str,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        embedding_base_url: str = "http://localhost:50009",
        ollama_model: str = "qwen3:32b",
        ollama_base_url: str = "http://192.168.8.231:11434",
    ):
        # 加载环境变量
        load_dotenv()

        # 初始化配置
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url

        # 初始化 Embedding
        self.embedding = OpenAIStyleEmbeddings(
            model=embedding_model,
            base_url=embedding_base_url,
        )

        # 加载向量数据库
        self.vectorstore = self._load_vectorstore(chroma_db_dir)
        self.retriever = self.vectorstore.as_retriever(k=4)

        # 初始化 LLM
        self.llm = ChatOllama(
            model=ollama_model,
            format="json",
            temperature=0,
            base_url=ollama_base_url,
        )

        # 初始化 Web 搜索工具
        self.web_search_tool = TavilySearchResults(k=3)

        # 构建 LangGraph
        self.graph = self._build_graph()

    def _load_vectorstore(self, chroma_db_dir: str) -> Chroma:
        """加载向量数据库"""
        if not os.path.exists(chroma_db_dir) or not os.listdir(chroma_db_dir):
            raise ValueError(
                f"向量数据库目录不存在或为空: {chroma_db_dir}\n"
                "请先运行 Jupyter Notebook 创建向量数据库"
            )

        print(f"✓ 加载向量数据库: {chroma_db_dir}")
        return Chroma(
            collection_name="rag_local_markdown_docs",
            embedding_function=self.embedding,
            persist_directory=chroma_db_dir,
        )

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        # 创建 Retrieval Grader
        retrieval_grader_prompt = PromptTemplate(
            template="""你是一名批改小测验的教师。你将收到以下两项内容：
1. 一个问题（QUESTION）
2. 学生提供的一个事实依据（FACT）

你需要对"相关性召回率"（RELEVANCE RECALL）进行评分，评分规则如下：
- 若事实依据（FACT）中的**任意一条表述**与问题（QUESTION）相关，评分即为1。
- 若事实依据（FACT）中的**所有表述**均与问题（QUESTION）无关，评分即为0。

请给出"yes"或"no"的二元评分，以表明该事实依据（文档）是否与问题相关。
请将二元评分以JSON格式呈现，仅包含"score"这一个键，且无需前缀说明或额外解释。

问题：{question}
事实依据：{documents}
""",
            input_variables=["question", "documents"],
        )
        self.retrieval_grader = retrieval_grader_prompt | self.llm | JsonOutputParser()

        # 创建 RAG Chain
        rag_prompt = PromptTemplate(
            template="""你是一个问答任务助手。

使用以下文档来回答问题。
如果你不知道答案，就说你不知道。
使用最多三句话，并保持答案简洁：
问题: {question}
文档: {documents}
答案:
""",
            input_variables=["question", "documents"],
        )
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

        # 定义节点函数
        def retrieve(state):
            """检索文档"""
            question = state["question"]
            documents = self.retriever.invoke(question)
            steps = state.get("steps", [])
            steps.append("retrieve_documents")
            return {"documents": documents, "question": question, "steps": steps}

        def grade_documents(state):
            """评分文档并决定是否需要 web search"""
            question = state["question"]
            documents = state["documents"]
            steps = state.get("steps", [])
            steps.append("grade_document_retrieval")

            filtered_docs = []
            search = "No"

            print("documents 的个数:", len(documents))
            for d in documents:
                score = self.retrieval_grader.invoke(
                    {"question": question, "documents": d.page_content}
                )
                grade = score.get("score", "no")
                print(f"文档评分: {grade}")

                if grade in ["1", "yes", "Yes", 1, True]:
                    filtered_docs.append(d)
                else:
                    search = "Yes"  # 有不相关文档,触发 web search

            return {
                "documents": filtered_docs,
                "question": question,
                "search": search,
                "steps": steps,
            }

        def web_search(state):
            """执行 web 搜索"""
            question = state["question"]
            documents = state.get("documents", [])
            steps = state.get("steps", [])
            steps.append("web_search")

            web_results = self.web_search_tool.invoke({"query": question})
            documents.extend(
                [
                    Document(page_content=d["content"], metadata={"url": d["url"]})
                    for d in web_results
                ]
            )
            return {"documents": documents, "question": question, "steps": steps}

        def generate(state):
            """生成最终答案"""
            question = state["question"]
            documents = state["documents"]
            generation = self.rag_chain.invoke(
                {"documents": documents, "question": question}
            )
            steps = state.get("steps", [])
            steps.append("generate_answer")
            return {
                "documents": documents,
                "question": question,
                "generation": generation,
                "steps": steps,
            }

        # ✅ 修复:决策函数只返回下一步的节点名
        def decide_to_generate(state):
            """决定是 web search 还是直接生成答案"""
            search = state.get("search", "No")
            if search == "Yes":
                return "search"  # 需要 web search
            else:
                return "generate"  # 直接生成答案

        # ✅ 修复:构建图时确保流程单向
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("web_search", web_search)

        # 设置边
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # ✅ 关键:条件边只有两个出口,不会循环回 retrieve
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "search": "web_search",  # 不相关 → web search
                "generate": "generate",  # 相关 → 生成答案
            },
        )

        # Web search 后直接生成,不再回到 retrieve
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def invoke(self, question: str) -> Dict[str, Any]:
        """执行 CRAG 查询"""
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state_dict = self.graph.invoke({"question": question, "steps": []}, config)
        return {
            "response": state_dict["generation"],
            "steps": state_dict["steps"],
            "documents": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in state_dict.get("documents", [])
            ],
        }
