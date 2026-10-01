"""Streamlit chat UI for the Bonum website assistant."""
from __future__ import annotations

import logging
import os

import streamlit as st

# On Streamlit Cloud settings come from Secrets, not .env. They must be copied into the
# environment BEFORE src.config is imported, because config reads them at import time.
for _name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "RELEVANCE_THRESHOLD"):
    try:
        if not os.getenv(_name) and _name in st.secrets:
            os.environ[_name] = str(st.secrets[_name])
    except Exception:  # no secrets file locally
        pass

from src import config, prompts  # noqa: E402
from src.chatbot import BonumChatbot, LLMError  # noqa: E402
from src.retriever import VectorStoreError  # noqa: E402

log = logging.getLogger(__name__)

st.set_page_config(page_title="Bonum AI Assistant", page_icon="💬", layout="centered")

PURPLE = "#6b3fa0"
st.markdown(
    f"""
<style>
.block-container {{ max-width: 780px; padding-top: 5rem; }}
.bonum-header {{ display:flex; align-items:center; gap:16px; padding:16px 20px; border-radius:14px;
  background:linear-gradient(135deg,{PURPLE},#9b6fd1); color:#fff; margin-bottom:12px; }}
.bonum-header img {{ height:46px; background:#fff; border-radius:8px; padding:4px 8px; }}
.bonum-header h1 {{ margin:0; font-size:1.4rem; color:#fff; padding:0; }}
.bonum-header p {{ margin:2px 0 0 0; font-size:0.9rem; opacity:0.9; }}
.stButton > button {{ border-radius:20px; border:1px solid {PURPLE}; color:{PURPLE}; }}
.stButton > button:hover {{ background:{PURPLE}; color:#fff; border-color:{PURPLE}; }}
.sources {{ font-size:0.85rem; color:#555; margin-top:6px; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>
<div class="bonum-header"><img src="{config.LOGO_URL}" alt="Bonum eDesigns"><div>
<h1>Bonum AI Assistant</h1><p>Answers based on the official Bonum eDesigns website</p></div></div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Bonum knowledge base...")
def get_bot() -> BonumChatbot:
    return BonumChatbot()


try:
    bot = get_bot()
except (VectorStoreError, LLMError) as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    log.exception("Startup failure")
    st.error("The assistant could not start. Please check the setup instructions in the README.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": prompts.WELCOME_MESSAGE, "sources": []}]


def render(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", "\\$"))
        if message.get("sources"):
            links = "  \n".join(f"- [{s['title']}]({s['url']})" for s in message["sources"])
            st.markdown(f"**Sources**  \n{links}")


with st.sidebar:
    st.markdown("### About")
    st.write("This assistant answers only from content published on bonum.in.")
    st.markdown(f"[Visit bonum.in]({config.BASE_URL}) · [Contact Bonum]({config.CONTACT_URL})")
    if st.button("Clear conversation", use_container_width=True):
        del st.session_state["messages"]
        st.rerun()

for message in st.session_state.messages:
    render(message)

pending = None
if len(st.session_state.messages) == 1:
    st.caption("Try asking:")
    columns = st.columns(2)
    for index, suggestion in enumerate(prompts.SUGGESTED_QUESTIONS):
        if columns[index % 2].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            pending = suggestion

question = st.chat_input("Ask about Bonum's services, projects or contact details...") or pending
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[1:]]
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    render(st.session_state.messages[-1])
    with st.chat_message("assistant"):
        with st.spinner("Searching Bonum's website..."):
            result = bot.answer(question, history)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.text,
            "sources": [{"title": s.title, "url": s.url} for s in result.sources],
        }
    )
    st.rerun()

st.caption("AI-generated from bonum.in content. For quotations or custom requirements, please contact Bonum.")