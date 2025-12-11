# GitHub Copilot Instructions for LangGraph

这份文档旨在帮助 GitHub Copilot 更好地理解 LangGraph 项目的结构、开发规范和最佳实践。

## 项目概述

LangGraph 是一个用于构建、管理和部署长时运行的状态化代理(agents)的低级编排框架。它被 Klarna、Replit、Elastic 等公司使用。

### 核心特性
- **状态化代理**: 构建具有持久状态的多角色应用
- **低级编排**: 提供对代理工作流的精细控制
- **检查点机制**: 支持状态持久化和恢复
- **多语言支持**: Python 和 JavaScript/TypeScript SDK

## 项目结构

这是一个 **monorepo** 项目,所有库都位于 `libs/` 目录下:

```
langgraph/
├── libs/
│   ├── checkpoint/           # 检查点基础接口
│   ├── checkpoint-postgres/  # Postgres 检查点实现
│   ├── checkpoint-sqlite/    # SQLite 检查点实现
│   ├── cli/                  # 命令行工具
│   ├── langgraph/           # 核心框架(状态化多角色代理)
│   ├── prebuilt/            # 高级 API(创建和运行代理和工具)
│   ├── sdk-js/              # JavaScript/TypeScript SDK
│   └── sdk-py/              # Python SDK
├── docs/                     # 文档
├── examples/                 # 示例代码
└── AGENTS.md                # 开发指南
```

### 依赖关系图

```
checkpoint
├── checkpoint-postgres
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph

sdk-py
├── langgraph
└── cli

sdk-js (独立)
```

**重要**: 对某个库的更改可能会影响所有依赖它的下游库。

## 开发工作流

### 修改代码前的检查清单

当修改任何库的代码时,在创建 Pull Request 之前,必须在该库目录下运行:

```bash
make format   # 运行代码格式化工具
make lint     # 运行 linter
make test     # 执行测试套件
```

### 运行特定测试

```bash
TEST=path/to/test.py make test
```

可以在 `TEST` 变量中传递其他 pytest 参数。

### Python 环境要求

- **Python 版本**: >= 3.10 (支持 3.10, 3.11, 3.12, 3.13)
- **依赖管理**: 使用 `pyproject.toml`
- **构建系统**: hatchling

## 代码规范

### 通用原则

1. **向后兼容**: 您的更改不能破坏现有 API,除非是关键 bug 或安全修复
2. **作用域隔离**: 更改应尽可能隔离,通常不应影响多个包
3. **测试覆盖**: Bug 修复必须包含失败的单元测试(修复前失败,修复后通过)
4. **查重**: 在创建新 issue 或 PR 前,检查是否已存在类似的

### Pull Request 流程

1. 遵循 "fork and pull request" 工作流
2. 填写 PR 模板,注明相关 issue 并标记相关维护者
3. 确保通过格式化、linting 和测试检查
4. 如需反馈,请标记维护者

### Bug 修复

- 在提出修复前先创建 issue,确保方案正确解决根本问题
- 必须包含相关的单元测试

### 新功能

- 在开发前先在 [forum.langchain.com](https://forum.langchain.com/) 开启讨论
- 维护者会帮助确定必要的更改范围

## 代码示例

### 创建一个简单的 Agent

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# 运行 agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

### 构建自定义工作流

对于需要自定义架构、长期记忆和复杂任务处理的场景,参考 [LangGraph 基础教程](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)。

## 文档规范

LangGraph 文档遵循 [Diataxis 框架](https://diataxis.fr),包含四种类型:

1. **Tutorials (教程)**: 通过实践活动引导用户学习
2. **How-to guides (操作指南)**: 解决特定问题的步骤
3. **References (参考文档)**: API 和技术规范
4. **Conceptual guides (概念指南)**: 解释概念和原理

## 核心概念

### 状态化代理 (Stateful Agents)
- 代理可以维护跨多次交互的状态
- 支持长时运行的任务和对话

### 检查点 (Checkpoints)
- 状态持久化机制
- 支持多种后端(内存、SQLite、Postgres)
- 允许暂停和恢复代理执行

### 图结构 (Graph Structure)
- 代理工作流表示为有向图
- 节点代表操作或决策点
- 边定义执行流程

### 多角色系统 (Multi-Actor)
- 支持多个代理协作
- 每个代理可以有独立的状态和行为

## 常用命令速查

| 命令 | 用途 |
|------|------|
| `make format` | 格式化代码 |
| `make lint` | 运行 linter |
| `make test` | 运行所有测试 |
| `TEST=path/to/test.py make test` | 运行特定测试 |
| `pip install -U langgraph` | 安装/更新 LangGraph |

## 学习资源

- **快速开始**: [Quickstart](https://langchain-ai.github.io/langgraph/agents/agents/)
- **基础教程**: [LangGraph basics](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)
- **API 文档**: [Latest docs](https://langchain-ai.github.io/langgraph/)
- **示例代码**: `examples/` 目录包含丰富的示例
- **社区讨论**: [LangChain Forum](https://forum.langchain.com/)

## 贡献指南要点

1. **Fork and PR**: 使用 fork 和 pull request 工作流
2. **填写模板**: PR 必须填写完整的模板
3. **通过检查**: 确保通过所有自动化检查
4. **向后兼容**: 避免破坏性更改
5. **作用域控制**: 每次只修改一个包
6. **测试驱动**: Bug 修复需要测试,新功能需要讨论

## Copilot 使用建议

当使用 GitHub Copilot 在此项目中工作时:

1. **遵循项目结构**: 将代码放在正确的 `libs/` 子目录中
2. **注意依赖关系**: 修改基础库时考虑下游影响
3. **编写测试**: 为新代码生成相应的测试用例
4. **保持一致性**: 遵循现有代码的风格和模式
5. **文档同步**: 修改 API 时更新相关文档
6. **类型提示**: Python 代码应包含完整的类型注解
7. **异步支持**: 考虑异步场景的支持

---

此文档会随着项目发展而更新。如有疑问,请查阅 `AGENTS.md` 和 `CONTRIBUTING.md`。
