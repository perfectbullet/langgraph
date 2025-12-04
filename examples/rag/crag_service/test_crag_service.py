"""
CRAG 服务测试脚本
用于测试 OpenAI 风格的 CRAG API 服务
"""

import requests


def test_health():
    """测试健康检查接口"""
    print("\n" + "=" * 60)
    print("🏥 测试健康检查")
    print("=" * 60)

    try:
        response = requests.get("http://localhost:8000/health")
        response.raise_for_status()
        result = response.json()

        print(f"✅ 服务状态: {result['status']}")
        print(f"✅ 服务版本: {result['version']}")
        print(f"✅ 使用模型: {result['model']}")
        return True
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


def test_chat_completion(question: str, verbose: bool = True):
    """测试聊天补全接口"""
    if verbose:
        print("\n" + "=" * 60)
        print(f"💬 测试聊天补全")
        print("=" * 60)
        print(f"📝 问题: {question}")

    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "crag-agent",
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()

        if verbose:
            print(f"\n✅ 回复ID: {result['id']}")
            print(f"✅ 创建时间: {result['created']}")
            print(f"✅ 使用模型: {result['model']}")
            print(f"\n📖 答案:")
            print("-" * 60)
            print(result["choices"][0]["message"]["content"])
            print("-" * 60)
            print(f"\n🔍 执行轨迹: {' → '.join(result['metadata']['steps'])}")
            print(f"📚 使用文档数: {result['metadata']['documents_count']}")
            print(f"📊 Token 统计: {result['usage']}")

        return result

    except requests.exceptions.Timeout:
        print(f"❌ 请求超时")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        if hasattr(e.response, "text"):
            print(f"错误详情: {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return None


def test_knowledge_base_question():
    """测试知识库内的问题"""
    print("\n" + "=" * 60)
    print("📚 测试场景 1: 知识库问题")
    print("=" * 60)

    questions = [
        "失蜡铸造原理是什么?",
        # "首饰雕蜡工艺的主要步骤有哪些?",
        # "什么是脱蜡过程?"
    ]

    for question in questions:
        result = test_chat_completion(question)
        if result and result["metadata"]["steps"]:
            if "web_search" in result["metadata"]["steps"]:
                print("⚠️  注意: 触发了 Web 搜索，可能知识库中没有相关信息")


def test_web_search_question():
    """测试需要 Web 搜索的问题"""
    print("\n" + "=" * 60)
    print("🌐 测试场景 2: Web 搜索问题")
    print("=" * 60)

    questions = [
        "今天北京天气怎么样?",
        # "最新的人工智能技术发展如何?",
    ]

    for question in questions:
        result = test_chat_completion(question)
        if result and result["metadata"]["steps"]:
            if "web_search" in result["metadata"]["steps"]:
                print("✅ 成功触发 Web 搜索")
            else:
                print("⚠️  注意: 未触发 Web 搜索")


def test_curl_example():
    """展示 cURL 调用示例"""
    print("\n" + "=" * 60)
    print("📋 cURL 调用示例")
    print("=" * 60)

    curl_command = """
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "crag-agent",
    "messages": [
      {"role": "user", "content": "失蜡铸造原理是什么？"}
    ],
    "stream": false
  }'
"""
    print(curl_command)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🚀" * 30)
    print("CRAG 服务测试套件")
    print("🚀" * 30)

    # 1. 健康检查
    if not test_health():
        print("\n❌ 服务未启动，请先启动服务:")
        print("   python crag_service.py")
        return

    # 2. 测试知识库问题
    test_knowledge_base_question()

    # 3. 测试 Web 搜索问题
    test_web_search_question()

    # 4. 显示 cURL 示例
    # test_curl_example()

    print("\n" + "✅" * 30)
    print("测试完成!")
    print("✅" * 30 + "\n")


if __name__ == "__main__":
    # 运行所有测试
    run_all_tests()

    # 或者单独测试某个功能
    # test_health()
    # test_chat_completion("失蜡铸造原理是什么?")
