"""
使用 OpenAI SDK 调用 CRAG 服务的完整示例
支持流式和非流式调用
"""

from openai import OpenAI
import sys

# ==================== 配置 ====================
CRAG_SERVICE_URL = "http://localhost:8000/v1"  # CRAG 服务地址
API_KEY = "dummy-key"  # CRAG 服务不需要真实 API Key，但 SDK 需要设置


def test_non_streaming():
    """测试非流式调用"""
    print("\n" + "=" * 60)
    print("测试 1: 非流式调用")
    print("=" * 60)
    
    # 初始化客户端
    client = OpenAI(
        base_url=CRAG_SERVICE_URL,
        api_key=API_KEY,
    )
    
    # 调用 CRAG 服务
    question = "失蜡铸造原理是什么？"
    print(f"\n📝 问题: {question}")
    print("\n等待响应...\n")
    
    try:
        response = client.chat.completions.create(
            model="crag-agent",
            messages=[
                {"role": "user", "content": question}
            ],
            stream=False,  # 非流式
        )
        
        # 打印响应
        print("✅ 响应:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
        # 打印元数据
        print(f"\n📊 响应ID: {response.id}")
        print(f"📊 模型: {response.model}")
        print(f"📊 创建时间: {response.created}")
        
        if hasattr(response, 'usage'):
            print(f"📊 Token 使用:")
            print(f"   - Prompt: {response.usage.prompt_tokens}")
            print(f"   - Completion: {response.usage.completion_tokens}")
            print(f"   - Total: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return False


def test_streaming():
    """测试流式调用"""
    print("\n" + "=" * 60)
    print("测试 2: 流式调用")
    print("=" * 60)
    
    # 初始化客户端
    client = OpenAI(
        base_url=CRAG_SERVICE_URL,
        api_key=API_KEY,
    )
    
    # 调用 CRAG 服务（流式）
    question = "首饰雕蜡工艺的主要步骤有哪些？"
    print(f"\n📝 问题: {question}")
    print("\n✅ 流式响应:")
    print("-" * 60)
    
    try:
        stream = client.chat.completions.create(
            model="crag-agent",
            messages=[
                {"role": "user", "content": question}
            ],
            stream=True,  # 流式
        )
        
        # 实时打印流式输出
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
        
        print("\n" + "-" * 60)
        print(f"\n📊 完整响应长度: {len(full_response)} 字符")
        
        return True
        
    except Exception as e:
        print(f"❌ 流式调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_questions_streaming():
    """测试多个问题的流式调用"""
    print("\n" + "=" * 60)
    print("测试 3: 批量流式调用")
    print("=" * 60)
    
    client = OpenAI(
        base_url=CRAG_SERVICE_URL,
        api_key=API_KEY,
    )
    
    questions = [
        "什么是脱蜡过程？",
        "首饰制作需要哪些工具？",
        "北京今天天气怎么样？",  # 这个问题会触发 web search
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 60}")
        print(f"问题 {i}: {question}")
        print('─' * 60)
        
        try:
            stream = client.chat.completions.create(
                model="crag-agent",
                messages=[{"role": "user", "content": question}],
                stream=True,
            )
            
            print("回答: ", end='')
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end='', flush=True)
            
            print()  # 换行
            
        except Exception as e:
            print(f"❌ 问题 {i} 调用失败: {e}")


def test_with_context():
    """测试带上下文的多轮对话"""
    print("\n" + "=" * 60)
    print("测试 4: 多轮对话（带上下文）")
    print("=" * 60)
    
    client = OpenAI(
        base_url=CRAG_SERVICE_URL,
        api_key=API_KEY,
    )
    
    # 模拟多轮对话
    conversation = [
        {"role": "user", "content": "失蜡铸造是什么？"},
    ]
    
    print("\n第一轮:")
    print(f"用户: {conversation[0]['content']}")
    print("助手: ", end='')
    
    try:
        # 第一轮
        stream = client.chat.completions.create(
            model="crag-agent",
            messages=conversation,
            stream=True,
        )
        
        assistant_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                assistant_response += content
        
        print()
        
        # 添加助手响应到对话历史
        conversation.append({"role": "assistant", "content": assistant_response})
        
        # 第二轮（注意：CRAG 服务是无状态的，每次都是独立查询）
        conversation.append({"role": "user", "content": "它的主要步骤是什么？"})
        
        print("\n第二轮:")
        print(f"用户: {conversation[-1]['content']}")
        print("助手: ", end='')
        
        stream = client.chat.completions.create(
            model="crag-agent",
            messages=conversation,  # 传递完整对话历史
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
        
        print()
        
    except Exception as e:
        print(f"\n❌ 多轮对话失败: {e}")


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试 5: 错误处理")
    print("=" * 60)
    
    client = OpenAI(
        base_url=CRAG_SERVICE_URL,
        api_key=API_KEY,
    )
    
    # 测试空消息
    print("\n测试 5.1: 空消息")
    try:
        response = client.chat.completions.create(
            model="crag-agent",
            messages=[],  # 空消息列表
            stream=False,
        )
        print("❌ 应该抛出错误但没有")
    except Exception as e:
        print(f"✅ 正确捕获错误: {type(e).__name__}")
    
    # 测试无效的消息格式
    print("\n测试 5.2: 只有 system 消息（没有 user 消息）")
    try:
        response = client.chat.completions.create(
            model="crag-agent",
            messages=[{"role": "system", "content": "你是一个助手"}],
            stream=False,
        )
        print("❌ 应该抛出错误但没有")
    except Exception as e:
        print(f"✅ 正确捕获错误: {type(e).__name__}")


def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 60)
    print("交互模式")
    print("=" * 60)
    print("输入问题，按 Ctrl+C 或输入 'quit' 退出\n")
    
    client = OpenAI(
        base_url=CRAG_SERVICE_URL,
        api_key=API_KEY,
    )
    
    try:
        while True:
            question = input("\n💬 你的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not question:
                print("⚠️  请输入问题")
                continue
            
            print("\n🤖 助手: ", end='')
            
            try:
                stream = client.chat.completions.create(
                    model="crag-agent",
                    messages=[{"role": "user", "content": question}],
                    stream=True,
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end='', flush=True)
                
                print()
                
            except Exception as e:
                print(f"\n❌ 调用失败: {e}")
    
    except KeyboardInterrupt:
        print("\n\n👋 再见！")


def main():
    """主函数"""
    print("=" * 60)
    print("CRAG 服务 OpenAI SDK 调用示例")
    print("=" * 60)
    print(f"服务地址: {CRAG_SERVICE_URL}")
    print(f"目标模型: crag-agent")
    print("=" * 60)
    
    # 检查服务是否可用
    print("\n检查服务状态...")
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ CRAG 服务运行正常")
            health_data = response.json()
            print(f"   版本: {health_data.get('version')}")
            print(f"   模型: {health_data.get('model')}")
        else:
            print("⚠️  服务响应异常")
            return
    except Exception as e:
        print(f"❌ 无法连接到 CRAG 服务: {e}")
        print("请确保服务已启动: python crag_service.py")
        return
    
    # 运行测试
    tests = [
        ("非流式调用", test_non_streaming),
        ("流式调用", test_streaming),
        ("批量流式调用", test_multiple_questions_streaming),
        ("多轮对话", test_with_context),
        ("错误处理", test_error_handling),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} - {test_name}")
    
    # 询问是否进入交互模式
    print("\n" + "=" * 60)
    choice = input("是否进入交互模式？(y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        interactive_mode()


if __name__ == "__main__":
    main()
