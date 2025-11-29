from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from pathlib import Path
import os

# ✅ 方式1: 读取单个 Markdown 文件
def load_single_markdown(file_path: str):
    """加载单个 Markdown 文件"""
    loader = UnstructuredMarkdownLoader(file_path)
    docs = loader.load()
    return docs

# ✅ 方式2: 读取多个 Markdown 文件
def load_multiple_markdowns(file_paths: list):
    """加载多个 Markdown 文件"""
    docs_list = []
    for file_path in file_paths:
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            docs_list.extend(docs)
            print(f"✅ 成功加载: {file_path}")
        except Exception as e:
            print(f"❌ 加载失败 {file_path}: {e}")
    return docs_list

# ✅ 方式3: 读取整个目录下的所有 Markdown 文件
def load_markdown_directory(directory_path: str):
    """加载目录下所有 Markdown 文件"""
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.md",  # 匹配所有 .md 文件
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    docs = loader.load()
    return docs


# === 使用示例 ===

# 示例1: 加载单个文件
print("=" * 60)
print("📄 示例1: 加载单个 Markdown 文件")
print("=" * 60)

single_file = "d:/ai_works/documents/example.md"  # 修改为您的文件路径
if os.path.exists(single_file):
    docs_list = load_single_markdown(single_file)
    print(f"加载了 {len(docs_list)} 个文档")
else:
    print(f"⚠️ 文件不存在: {single_file}")


# 示例2: 加载多个指定的文件
print("\n" + "=" * 60)
print("📄 示例2: 加载多个 Markdown 文件")
print("=" * 60)

md_files = [
    "d:/ai_works/documents/file1.md",
    "d:/ai_works/documents/file2.md",
    "d:/ai_works/documents/file3.md",
]

# 过滤存在的文件
existing_files = [f for f in md_files if os.path.exists(f)]
if existing_files:
    docs_list = load_multiple_markdowns(existing_files)
    print(f"✅ 总共加载了 {len(docs_list)} 个文档")
else:
    print("⚠️ 没有找到任何文件")


# 示例3: 加载整个目录
print("\n" + "=" * 60)
print("📁 示例3: 加载目录下所有 Markdown 文件")
print("=" * 60)

docs_directory = "d:/ai_works/documents"  # 修改为您的目录路径
if os.path.exists(docs_directory):
    docs_list = load_markdown_directory(docs_directory)
    print(f"✅ 从目录加载了 {len(docs_list)} 个文档")
else:
    print(f"⚠️ 目录不存在: {docs_directory}")


# === 文档切分和向量化 ===

print("\n" + "=" * 60)
print("✂️ 文档切分")
print("=" * 60)

# 使用 RecursiveCharacterTextSplitter 切分文档
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, 
    chunk_overlap=50
)

doc_splits = text_splitter.split_documents(docs_list)
print(f'✅ 切分后的文档块数量: {len(doc_splits)}')

# 打印前几个切分的示例
for i, split in enumerate(doc_splits[:3], 1):
    print(f"\n--- 文档块 {i} ---")
    print(f"内容: {split.page_content[:200]}...")
    print(f"元数据: {split.metadata}")


# === 创建向量存储 ===

print("\n" + "=" * 60)
print("🔍 创建向量存储")
print("=" * 60)

# 使用智谱 AI Embeddings
embeddings = ZhipuAIEmbeddings(
    model="embedding-2",  # 或使用其他模型
    api_key="your_zhipu_api_key"  # 替换为您的 API Key
)

# 创建 Chroma 向量数据库
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    collection_name="local_markdown_docs",
    persist_directory="./chroma_db"  # 持久化存储路径
)

print("✅ 向量存储创建成功！")

# 测试检索
query = "什么是 LangChain?"
results = vectorstore.similarity_search(query, k=3)
print(f"\n查询: {query}")
print(f"找到 {len(results)} 个相关结果")