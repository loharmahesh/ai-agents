import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import asyncio
import streamlit as st
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    handoff,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

# Load environment variables
load_dotenv()
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

openai_client = AsyncAzureOpenAI(
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_version=api_version,
)

# ---------- SESSION STATE INIT ----------
for key, default in {
    "collected_facts": [],
    "research_started": False,
    "research_done": False,
    "report_result": None,
    "research_progress": 0,
}.items():
    st.session_state.setdefault(key, default)

# ---------- SETUP ----------
set_default_openai_api("chat_completions")
set_default_openai_client(openai_client)
set_tracing_disabled(True)

# ---------- CUSTOM TOOL ----------
@function_tool
def save_important_fact(fact: str, source: str = None) -> str:
    if fact in [entry["fact"] for entry in st.session_state.collected_facts]:
        return "⚠️ Fact already saved."
    
    st.session_state.collected_facts.append({
        "fact": fact,
        "source": source or "Not specified",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    return f"✅ Fact saved: {fact}"

# ---------- AGENTS ----------
query_rewriting_agent = Agent(
    name="Query Rewriting Agent",
    instructions="Generate 5–8 diverse, precise search queries. Respond in a bullet list.",
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client),
)

research_agent = Agent(
    name="Research Agent",
    instructions="""
        For each query:
        1. Search the web.
        2. Save key insights using save_important_fact().
        3. Format output in Markdown.
    """,
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client),
    tools=[WebSearchTool(), save_important_fact],
)

class ResearchReport(BaseModel):
    title: str
    outline: List[str]
    report: str
    sources: List[str]
    word_count: int = Field(..., gt=0, description="Total word count must be positive")

editor_agent = Agent(
    name="Editor Agent",
    instructions="Compile a well-structured research report, ensuring clarity and completeness.",
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client),
    output_type=ResearchReport,
)

class ResearchPlan(BaseModel):
    topic: str
    search_queries: List[str]
    focus_areas: List[str]

triage_agent = Agent(
    name="Triage Agent",
    instructions="Generate structured search queries and focus areas based on the topic.",
    handoffs=[handoff(query_rewriting_agent), handoff(research_agent), handoff(editor_agent)],
    model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client),
    output_type=ResearchPlan,
)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")
st.markdown("""
<style>
    .main-title { font-size: 2.5em; font-weight: 700; color: #4CAF50; }
    .subtitle { font-size: 1.2em; color: #666; margin-top: -10px; }
</style>
<div class="main-title">🔬 AI Research Assistant</div>
<div class="subtitle">Conduct in-depth research using smart agents and web search</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 📝 Research Input")
    user_topic = st.text_input("Enter a topic for research:")
    start_button = st.button("Start Research", type="primary", disabled=not user_topic)
    st.markdown("🔔 **Status**")
    st.progress(st.session_state.research_progress)
    st.caption("💡 Tip: Be specific with your topic for better results!")

# ---------- ASYNC RESEARCH FUNCTION ----------
async def run_research(topic: str):
    st.session_state.update({"research_started": True, "research_done": False, "collected_facts": [], "report_result": None, "research_progress": 0})

    st.markdown("### 🧠 Generating Research Plan...")
    plan_result = await Runner.run(triage_agent, f"Create a research plan for: {topic}")
    research_plan = plan_result.final_output
    st.session_state.research_progress = 25

    with st.expander("📋 Research Plan", expanded=True):
        st.markdown(f"**🔎 Topic:** `{research_plan.topic}`")
        st.markdown("<ul>" + "".join([f"<li>🔍 {q}</li>" for q in research_plan.search_queries]) + "</ul>", unsafe_allow_html=True)
        st.markdown("<ul>" + "".join([f"<li>🎯 {a}</li>" for a in research_plan.focus_areas]) + "</ul>", unsafe_allow_html=True)

    st.markdown("### 🔍 Conducting Research...")
    await asyncio.sleep(1.5)
    st.session_state.research_progress = 50

    st.markdown("### 📝 Compiling Research Report...")
    editor_result = await Runner.run(editor_agent, plan_result.to_input_list())
    st.session_state.update({"report_result": editor_result.final_output, "research_done": True, "research_progress": 100})

# ---------- RUN RESEARCH ----------
if start_button:
    with st.spinner(f"Running research for: {user_topic}"):
        asyncio.run(run_research(user_topic))

# ---------- DISPLAY REPORT ----------
if st.session_state.research_done and st.session_state.report_result:
    report = st.session_state.report_result
    st.markdown(f"<h2>📖 {report.title}</h2>", unsafe_allow_html=True)
    st.success(f"📝 Word Count: {report.word_count}")

    with st.expander("🧾 Outline", expanded=True):
        st.markdown("\n".join(f"- ✅ {item}" for item in report.outline))

    st.markdown("### 📄 Report Content")
    st.markdown(report.report)

    with st.expander("🔗 Sources", expanded=True):
        st.markdown("\n".join(f"- 🔗 {src}" for src in report.sources))

    st.download_button(
        label="⬇️ Download Report",
        data=report.report,
        file_name=f"{report.title.replace(' ', '_')}.md",
        mime="text/markdown"
    )
