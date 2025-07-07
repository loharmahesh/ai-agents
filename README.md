# 🔬 AI Research Assistant using OpenAI SDK agent

A powerful AI-driven research assistant that uses multi-agent collaboration to perform deep research on any topic. It generates a structured research plan, gathers facts via web search, and compiles a well-structured research report — all through OpenAI’s Open SDK and Azure OpenAI models.

## 🌐 Live Interface

This app runs via **Streamlit** and uses **Async Azure OpenAI**, **Open SDK agents**, and **WebSearchTool** to deliver an automated research workflow.

---

## 🚀 Features

- ✅ Automatic **Query Rewriting** for improved web search
- ✅ **Web Search Agent** that collects facts using `save_important_fact`
- ✅ **Research Editor Agent** that compiles Markdown-based report
- ✅ **Triage Agent** that coordinates all agents
- ✅ 🧠 **Memory** via Streamlit session state
- ✅ 📥 Downloadable final report in `.md` format
- ✅ ⏱️ Real-time progress tracking with UI indicators

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **OpenAgents SDK (Open SDK)**
- **Azure OpenAI (via `AsyncAzureOpenAI`)**
- **Pydantic**
- **dotenv** for secure configuration
