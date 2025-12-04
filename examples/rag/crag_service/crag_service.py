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

import json
from fastapi.responses import StreamingResponse
from loguru import logger

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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天补全接口

    支持流式和非流式输出
    """
    try:
        # 提取用户问题
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        question = user_messages[-1].content
        crag_agent = get_agent()

        #  流式响应
        if request.stream:
            logger.info("Starting streaming response for question: {}", question)
            return StreamingResponse(
                stream_openai_response(crag_agent, question, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
                },
            )

        #  非流式响应（原有逻辑）
        result = crag_agent.invoke(question)

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


async def stream_openai_response(agent: CRAGAgent, question: str, model: str):
    """
    生成 OpenAI 流式格式的响应

    格式示例:
    data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"你好"},"index":0}]}
    data: [DONE]
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(datetime.now().timestamp())

    # 发送初始步骤信息（可选）
    steps_data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(steps_data)}\n\n"

    # 流式发送生成内容
    async for event in agent.astream(question):
        if event["type"] == "step":
            # 可选：发送步骤信息作为注释
            step_comment = f": step={event['content']}\n\n"
            yield step_comment

        elif event["type"] == "chunk":
            chunk_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event["content"]},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        elif event["type"] == "metadata":
            # 发送元数据（作为注释）
            metadata_comment = f": metadata={json.dumps(event['content'])}\n\n"
            yield metadata_comment

        elif event["type"] == "done":
            # 发送结束标志
            done_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
