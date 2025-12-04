"""
OpenAI 风格的 CRAG 服务
提供兼容 OpenAI API 的接口用于纠正性检索增强生成
"""

import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from CRAGAgent import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CRAGAgent,
    HealthResponse,
)

# ==================== FastAPI 服务 ====================
app = FastAPI(
    title="CRAG Service",
    description="OpenAI 风格的纠正性检索增强生成服务",
    version="1.0.0",
)


# 初始化全局 Agent (延迟加载)
agent: Optional[CRAGAgent] = None


def get_agent() -> CRAGAgent:
    """获取或初始化 Agent"""
    global agent
    if agent is None:
        agent = CRAGAgent()
    return agent

@app.get("/")
async def root():
    """重定向到 API 文档"""
    return RedirectResponse(url="/docs")
    

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy", version="1.0.0", model="crag-agent-qwen3:32b"
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天补全接口
    
    示例请求:
    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "crag-agent",
        "messages": [{"role": "user", "content": "失蜡铸造原理是什么？"}],
        "stream": false
      }'
    ```
    """
    try:
        # 提取用户问题 (取最后一条 user 消息)
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        question = user_messages[-1].content

        # 调用 CRAG Agent
        crag_agent = get_agent()
        result = crag_agent.invoke(question)

        # 构建 OpenAI 格式响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["response"],
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": len(question.split()),
                "completion_tokens": len(result["response"].split()),
                "total_tokens": len(question.split()) + len(result["response"].split()),
            },
            metadata={
                "steps": result["steps"],
                "documents_count": len(result["documents"]),
            },
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: ChatCompletionRequest):
    """兼容 /v1/completions 接口"""
    return await chat_completions(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
