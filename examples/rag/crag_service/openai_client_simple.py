"""
OpenAI SDK 调用 CRAG 服务的快速示例
"""

from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",  # CRAG 服务不需要真实 key
)

# ==================== 示例 1: 非流式调用 ====================
print("示例 1: 非流式调用")
print("-" * 60)

response = client.chat.completions.create(
    model="crag-agent",
    messages=[
        {"role": "user", "content": "失蜡铸造原理是什么？"}
    ],
    stream=False,
)

print(response.choices[0].message.content)
print()

# ==================== 示例 2: 流式调用 ====================
print("示例 2: 流式调用")
print("-" * 60)

stream = client.chat.completions.create(
    model="crag-agent",
    messages=[
        {"role": "user", "content": "首饰雕蜡工艺的主要步骤有哪些？"}
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)

print("\n")

# ==================== 示例 3: 多个问题 ====================
print("示例 3: 测试多个问题")
print("-" * 60)

questions = [
    "什么是脱蜡过程？",
    "首饰制作需要哪些工具？",
]

for i, question in enumerate(questions, 1):
    print(f"\n问题 {i}: {question}")
    print("回答: ", end='')
    
    stream = client.chat.completions.create(
        model="crag-agent",
        messages=[{"role": "user", "content": question}],
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    
    print()
