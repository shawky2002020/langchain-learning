# LangChain Gemini Learning

A collection of projects and experiments for learning **LangChain**, **Google Gemini**, and **LangGraph**.

## 🚀 Practiced Concepts

This repository explores key concepts in building agentic workflows:
- **LangGraph Fundamentals**: Building graphs with `StateGraph`, `Nodes`, `Edges`, and `Conditional Edges`.
- **Tool Use**: Binding custom tools to Gemini models using `@tool` and `bind_tools`.
- **State Management**: Managing conversation history and application state (`AgentState`).
- **RAG (Retrieval-Augmented Generation)**: Integrating vector stores (`Chroma`) and embeddings for document Q&A.

## 🤖 Implemented Agents

### 1. [Memory Agent](agents/Memory_Agent.py)
A conversational agent that maintains context and history across turns. It demonstrates how to manage message state effectively using `AIMessage` and `HumanMessage`.

### 2. [ReAct Agent](agents/ReAct.py)
An agent implementing the **ReAct** (Reasoning + Acting) pattern. It uses a set of calculator tools (`add`, `subtract`, `multiply`) to solve multi-step problems by reasoning about which tool to use next.

### 3. [Drafter Agent](agents/Drafter.py)
A document creation assistant that can:
- **Write/Update content**: Uses an `update` tool to modify text.
- **Save files**: Uses a `save` tool to persist the document to disk.
- **Demonstrates**: Tool calling for side effects and multi-turn workflows.

### 4. [RAG Agent](agents/RAG_Agent.py)
A specialized agent for Question Answering over documents.
- **Features**: Loads PDFs using `PyPDFLoader`, chunks text, creates embeddings with `GoogleGenerativeAIEmbeddings`, and stores them in `Chroma`.
- **Usage**: Retrieves relevant context to answer user queries about specific documents (e.g., Stock Market Performance).

## 📂 Key Directories

- `agents/`: Python scripts containing the agent implementations.
- `graphs/`: Jupyter notebooks exploring specific graph patterns (Loops, Multiple Inputs, Sequencing).
- `main.py`: A simple entry point for testing basic Gemini connectivity.