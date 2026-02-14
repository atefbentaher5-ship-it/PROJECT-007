import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from llama_cpp import Llama
import os
import json

# ===================== CONFIG =====================
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MEMORY_FILE = "memory.json"

MAX_HISTORY = 4
MAX_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

SYSTEM_PROMPT = (
    "You are a scientific, skeptical assistant. "
    "You use evidence, logic, and clear explanations. "
    "If you are unsure, say so."
)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=4,
        n_batch=64,
        verbose=False
    )

llm = load_model()

# ===================== MEMORY =====================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

# ===================== WEB SEARCH =====================
def web_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            body = r.get("body", "")
            link = r.get("href", "")
            results.append(f"{title}\n{body}\nSource: {link}\n")
    return "\n".join(results)

def read_url(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[:150])
    except Exception as e:
        return f"Error reading webpage: {e}"

# ===================== PROMPT BUILDER =====================
def build_prompt(memory, user_input, web_context=""):
    history = ""
    for m in memory[-MAX_HISTORY:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history += f"{role}: {m['content']}\n"

    prompt = SYSTEM_PROMPT + "\n\n"

    if web_context.strip():
        prompt += "Web search results (use these sources for factual accuracy):\n"
        prompt += web_context[:2000] + "\n\n"

    prompt += history
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def generate(prompt):
    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=["User:"]
    )
    return output["choices"][0]["text"].strip()

# ===================== STREAMLIT GUI =====================
st.set_page_config(page_title="PROJECT007 AI", page_icon="🤖", layout="wide")

st.title("🤖 PROJECT007 — Mistral AI with Real Web Search")
st.write("Local Mistral-7B model + DuckDuckGo web search + persistent memory.")

# Load memory into session
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

# Sidebar controls
st.sidebar.header("⚙️ Settings")
use_web = st.sidebar.checkbox("Enable Web Search", value=True)
max_results = st.sidebar.slider("Web results", 3, 10, 5)

if st.sidebar.button("🧹 Clear Memory"):
    st.session_state.memory = []
    save_memory([])
    st.sidebar.success("Memory cleared.")

# Chat UI
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message
    st.session_state.memory.append({"role": "user", "content": user_input})

    web_context = ""
    if use_web:
        with st.spinner("Searching the web..."):
            web_context = web_search(user_input, max_results=max_results)

    prompt = build_prompt(st.session_state.memory, user_input, web_context)

    with st.spinner("Thinking..."):
        reply = generate(prompt)

    # Add assistant reply
    st.session_state.memory.append({"role": "assistant", "content": reply})
    st.session_state.memory = st.session_state.memory[-MAX_HISTORY:]

    save_memory(st.session_state.memory)

# Display messages
for m in st.session_state.memory:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.write(m["content"])
    else:
        with st.chat_message("assistant"):
            st.write(m["content"])
