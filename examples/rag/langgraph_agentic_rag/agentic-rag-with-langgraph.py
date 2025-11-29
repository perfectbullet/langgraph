from typing import Literal

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek


from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import O
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import os

# Step 1: Set OpenAI API Key
os.environ["DEEPSEEK_API_KEY"] = "sk-0511c57af3604877b63cf32ea9ae7f01"


# Define the tool for context retrieval
@tool
def retrieve_context(query: str):
    """Search for relevant documents."""
    text_path = 'data/03_第一章.md'

    absolute_path = os.path.abspath(text_path)
    print(f"Loading documents from: {absolute_path}")
    # Load documents
    loader = UnstructuredMarkdownLoader(text_path)
    docs = loader.load()

    # Split documents with larger chunk size to reduce total chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)

    # Limit to first 60 chunks to avoid API limit
    doc_splits = doc_splits[:60]

    # Create VectorStore
    embedding = ZhipuAIEmbeddings(api_key="569d1fc0e3734ddea956bb63fe9fef75.ASLL7ikeolDpsZkT")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="python_docs",
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever()
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])


tools = [retrieve_context]
tool_node = ToolNode(tools)

# OpenAI LLM model
model = ChatDeepSeek(model="deepseek-chat", temperature=0).bind_tools(tools)

# Function to decide whether to continue or stop the workflow
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, go to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, finish the workflow
    return END


# Function that invokes the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}  # Returns as a list to add to the state


# Define the workflow with LangGraph
workflow = StateGraph(MessagesState)

# Add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Connect nodes
workflow.add_edge(START, "agent")  # Initial entry
workflow.add_conditional_edges("agent", should_continue)  # Decision after the "agent" node
workflow.add_edge("tools", "agent")  # Cycle between tools and agent

# Configure memory to persist the state
checkpointer = MemorySaver()

# Compile the graph into a LangChain Runnable application
app = workflow.compile(checkpointer=checkpointer)

# Execute the workflow
final_state = app.invoke(
    {"messages": [HumanMessage(content="Explain what a list is in Python")]},
    config={"configurable": {"thread_id": 42}}
)

# Show the final response
print(final_state["messages"][-1].content)