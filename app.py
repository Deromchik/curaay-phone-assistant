# ============================================
# EXAI Prompt Tester — Streamlit Application
# ============================================
from git_info import render_git_sync_indicator
from preset_loader import PRESET_KEYS, load_preset
from llm_logger import register_llm_exchange
from llm_client import collect_stream
from default_prompt import answer_generator
from browser_prompt import answer_generator_browser
import asyncio
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

# Allow imports from repository root (default_prompt, browser_prompt, llm_client)
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_DEFAULT_MODEL = os.getenv(
    "OPENROUTER_DEFAULT_MODEL", "google/gemini-2.5-flash"
)

DEFAULT_USER_QUESTION = (
    "Open Redmine ticket 31275 and list the AL objects it references."
)


def _base_config() -> dict[str, Any]:
    return {
        "prompt_type": "default",
        "language": "uk",
        "database_occurances": [],
        "knowledge_base": [],
        "examples": [],
        "special_context_chunks": [],
        "file_chunks": [],
        "learning_video_answer_text": "",
        "expert_id": "",
        "answer_model": OPENROUTER_DEFAULT_MODEL,
        "is_brief_mode": False,
        "is_expert_specific": False,
        "image_data": False,
        "screenshot_mode": False,
        "is_voice_mode": True,
    }


def apply_preset_to_session(prompt_type: str) -> None:
    preset = load_preset(prompt_type)
    for key in PRESET_KEYS:
        if key in preset:
            st.session_state[key] = deepcopy(preset[key])


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "llm_logs" not in st.session_state:
        st.session_state.llm_logs = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = DEFAULT_USER_QUESTION
    if "presets_initialized" not in st.session_state:
        for key, value in _base_config().items():
            st.session_state[key] = value
        apply_preset_to_session("default")
        st.session_state.presets_initialized = True


def sync_form_widgets_from_session() -> None:
    st.session_state.cfg_language = st.session_state.language
    st.session_state.cfg_answer_model = (
        st.session_state.answer_model or OPENROUTER_DEFAULT_MODEL
    )
    st.session_state.cfg_expert_id = st.session_state.expert_id or ""
    st.session_state.cfg_learning_video = st.session_state.learning_video_answer_text
    st.session_state.cfg_brief_mode = st.session_state.is_brief_mode
    st.session_state.cfg_expert_specific = st.session_state.is_expert_specific
    st.session_state.cfg_image_data = st.session_state.image_data
    st.session_state.cfg_screenshot_mode = st.session_state.screenshot_mode
    st.session_state.cfg_voice_mode = st.session_state.is_voice_mode
    st.session_state.cfg_database_occurances = json.dumps(
        st.session_state.database_occurances, ensure_ascii=False, indent=2
    )
    st.session_state.cfg_knowledge_base = json.dumps(
        st.session_state.knowledge_base, ensure_ascii=False, indent=2
    )
    st.session_state.cfg_examples = json.dumps(
        st.session_state.examples, ensure_ascii=False, indent=2
    )
    st.session_state.cfg_special_context_chunks = json.dumps(
        st.session_state.special_context_chunks, ensure_ascii=False, indent=2
    )
    st.session_state.cfg_file_chunks = json.dumps(
        st.session_state.file_chunks, ensure_ascii=False, indent=2
    )


def on_prompt_type_change() -> None:
    if st.session_state.conversation_started:
        return
    apply_preset_to_session(st.session_state.cfg_prompt_type)
    sync_form_widgets_from_session()


def parse_json_field(raw: str, field_name: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field_name}: {exc}") from exc


def validate_config_from_form(
    database_occurances_str: str,
    knowledge_base_str: str,
    examples_str: str,
    special_context_chunks_str: str,
    file_chunks_str: str,
) -> dict[str, Any]:
    database_occurances = parse_json_field(
        database_occurances_str, "Database occurances"
    )
    knowledge_base = parse_json_field(knowledge_base_str, "Knowledge base")
    examples = parse_json_field(examples_str, "Examples")
    special_context_chunks = parse_json_field(
        special_context_chunks_str, "Special context chunks"
    )
    file_chunks = parse_json_field(file_chunks_str, "File chunks")

    for name, value in [
        ("Database occurances", database_occurances),
        ("Examples", examples),
        ("Special context chunks", special_context_chunks),
        ("File chunks", file_chunks),
    ]:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a JSON array")

    if not isinstance(knowledge_base, (list, dict)):
        raise ValueError("Knowledge base must be a JSON array or object")

    return {
        "database_occurances": database_occurances,
        "knowledge_base": knowledge_base,
        "examples": examples,
        "special_context_chunks": special_context_chunks,
        "file_chunks": file_chunks,
    }


def save_config_to_session(
    prompt_type: str,
    language: str,
    learning_video_answer_text: str,
    expert_id: str,
    answer_model: str,
    is_brief_mode: bool,
    is_expert_specific: bool,
    image_data: bool,
    screenshot_mode: bool,
    is_voice_mode: bool,
    parsed: dict[str, Any],
) -> None:
    st.session_state.prompt_type = prompt_type
    st.session_state.language = language
    st.session_state.learning_video_answer_text = learning_video_answer_text
    st.session_state.expert_id = expert_id or None
    st.session_state.answer_model = answer_model.strip() or OPENROUTER_DEFAULT_MODEL
    st.session_state.is_brief_mode = is_brief_mode
    st.session_state.is_expert_specific = is_expert_specific
    st.session_state.image_data = image_data
    st.session_state.screenshot_mode = screenshot_mode
    st.session_state.is_voice_mode = is_voice_mode
    st.session_state.database_occurances = parsed["database_occurances"]
    st.session_state.knowledge_base = parsed["knowledge_base"]
    st.session_state.examples = parsed["examples"]
    st.session_state.special_context_chunks = parsed["special_context_chunks"]
    st.session_state.file_chunks = parsed["file_chunks"]


async def generate_answer(user_question: str, conversation_history: list) -> str:
    common_kwargs = {
        "user_question": user_question,
        "conversation_history": conversation_history,
        "database_occurances": st.session_state.database_occurances,
        "knowledge_base": st.session_state.knowledge_base,
        "language": st.session_state.language,
        "isStream": True,
        "is_brief_mode": st.session_state.is_brief_mode,
        "is_expert_specific": st.session_state.is_expert_specific,
        "learning_video_answer_text": st.session_state.learning_video_answer_text,
        "special_context_chunks": st.session_state.special_context_chunks,
        "image_data": st.session_state.image_data,
        "expert_id": st.session_state.expert_id,
        "file_chunks": st.session_state.file_chunks,
        "screenshot_mode": st.session_state.screenshot_mode,
        "is_voice_mode": st.session_state.is_voice_mode,
        "answer_model": st.session_state.answer_model,
        "examples": st.session_state.examples,
    }

    generator = (
        "answer_generator_browser"
        if st.session_state.prompt_type == "browser"
        else "answer_generator"
    )

    if st.session_state.prompt_type == "browser":
        stream = await answer_generator_browser(**common_kwargs)
    else:
        stream = await answer_generator(**common_kwargs)

    response = await collect_stream(stream)

    register_llm_exchange(
        st.session_state.llm_logs,
        assistant_response=response,
        generator=generator,
        inputs={
            "prompt_type": st.session_state.prompt_type,
            **common_kwargs,
        },
        turn=len(st.session_state.messages) + 1,
        user_question=user_question,
        conversation_history=conversation_history,
    )

    return response


def run_answer_generator(user_question: str, conversation_history: list) -> str:
    return asyncio.run(generate_answer(user_question, conversation_history))


def get_download_json() -> str:
    payload = {
        "config": {
            key: st.session_state.get(key)
            for key in PRESET_KEYS
        },
        "user_question": st.session_state.user_question,
        "messages": st.session_state.messages,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_llm_logs_json() -> str:
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "prompt_type": st.session_state.prompt_type,
        "llm_calls": st.session_state.llm_logs,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_conversation_from_json(json_str: str) -> bool:
    try:
        loaded = json.loads(json_str)
        if isinstance(loaded, list):
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in loaded
                if m.get("role") in ("user", "assistant")
            ]
        elif isinstance(loaded, dict):
            config = loaded.get("config", {})
            for key in PRESET_KEYS:
                if key in config:
                    st.session_state[key] = config[key]
            if "user_question" in loaded:
                st.session_state.user_question = loaded["user_question"]
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in loaded.get("messages", [])
                if m.get("role") in ("user", "assistant")
            ]
        else:
            st.error("Invalid format: expected a JSON object or array.")
            return False

        st.session_state.conversation_started = True
        return True
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return False
    except Exception as exc:
        st.error(f"Error loading conversation: {exc}")
        return False


def reset_conversation() -> None:
    prompt_type = st.session_state.prompt_type
    st.session_state.messages = []
    st.session_state.conversation_started = False
    st.session_state.llm_logs = []
    st.session_state.user_question = DEFAULT_USER_QUESTION
    apply_preset_to_session(prompt_type)
    sync_form_widgets_from_session()
    st.session_state.user_question_input = DEFAULT_USER_QUESTION


def main() -> None:
    st.set_page_config(
        page_title="EXAI Prompt Tester",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%); }
        .chat-message {
            padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem;
            color: #1a1a2e; font-size: 1rem; line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .user-message {
            background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
            border-left: 4px solid #4a90a4;
        }
        .assistant-message {
            background: linear-gradient(135deg, #e8f4f8 0%, #d4e8f0 100%);
            border-left: 4px solid #2d6a7a;
        }
        .main-header {
            color: #1a1a2e; font-size: 2.2rem; font-weight: 700;
            text-align: center; padding: 1.2rem 0; margin-bottom: 1rem;
            border-bottom: 3px solid #2d6a7a;
        }
        .sub-header {
            color: #2d4a5a; font-size: 1rem; text-align: center; margin-bottom: 1.5rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    global OPENROUTER_API_KEY, OPENROUTER_DEFAULT_MODEL
    try:
        if hasattr(st, "secrets"):
            if "OPENROUTER_API_KEY" in st.secrets:
                OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
                os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
            if "OPENROUTER_DEFAULT_MODEL" in st.secrets:
                OPENROUTER_DEFAULT_MODEL = st.secrets["OPENROUTER_DEFAULT_MODEL"]
                os.environ["OPENROUTER_DEFAULT_MODEL"] = OPENROUTER_DEFAULT_MODEL
    except Exception:
        pass

    if not OPENROUTER_API_KEY:
        st.error(
            "OpenRouter API key is not configured. Set OPENROUTER_API_KEY in "
            "environment variables or `.streamlit/secrets.toml`."
        )
        st.info(
            """Example `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "sk-or-..."
OPENROUTER_DEFAULT_MODEL = "google/gemini-2.5-flash"
```"""
        )
        st.stop()

    init_session_state()
    disabled = st.session_state.conversation_started

    col_chat, col_side = st.columns([2, 1])

    with col_side:
        render_git_sync_indicator(APP_DIR)

        st.markdown("---")
        st.markdown("### Prompt Configuration")

        prompt_type = st.radio(
            "Prompt type",
            options=["default", "browser"],
            format_func=lambda x: "Default prompt" if x == "default" else "Browser prompt",
            index=0 if st.session_state.prompt_type == "default" else 1,
            disabled=disabled,
            key="cfg_prompt_type",
            on_change=on_prompt_type_change,
        )

        language_options = ["uk", "en", "de"]
        language_labels = {"uk": "Ukrainian", "en": "English", "de": "German"}
        
        language = st.selectbox(
            "Language",
            options=language_options,
            format_func=lambda x: language_labels.get(x, x),
            index=language_options.index(st.session_state.language) if st.session_state.language in language_options else 0,
            disabled=disabled,
            key="cfg_language",
        )

        answer_model = st.text_input(
            "Answer model (OpenRouter slug)",
            value=st.session_state.answer_model or OPENROUTER_DEFAULT_MODEL,
            disabled=disabled,
            key="cfg_answer_model",
            help="Used when Expert specific mode is off.",
        )

        expert_id = st.text_input(
            "Expert ID (optional)",
            value=st.session_state.expert_id or "",
            disabled=disabled,
            key="cfg_expert_id",
        )

        learning_video_answer_text = st.text_area(
            "Learning video answer text",
            value=st.session_state.learning_video_answer_text,
            height=80,
            disabled=disabled,
            key="cfg_learning_video",
        )

        st.markdown("**Mode flags**")
        is_brief_mode = st.checkbox(
            "Brief mode",
            value=st.session_state.is_brief_mode,
            disabled=disabled,
            key="cfg_brief_mode",
        )
        is_voice_mode = st.checkbox(
            "Voice mode",
            value=st.session_state.is_voice_mode,
            disabled=disabled,
            key="cfg_voice_mode",
        )
        is_expert_specific = st.checkbox(
            "Expert specific",
            value=st.session_state.is_expert_specific,
            disabled=disabled,
            key="cfg_expert_specific",
        )
        image_data = st.checkbox(
            "Image data",
            value=st.session_state.image_data,
            disabled=disabled,
            key="cfg_image_data",
        )
        screenshot_mode = st.checkbox(
            "Screenshot mode",
            value=st.session_state.screenshot_mode,
            disabled=disabled,
            key="cfg_screenshot_mode",
        )

        st.markdown("**Context inputs (JSON)**")
        database_occurances_str = st.text_area(
            "Database occurances",
            value=json.dumps(
                st.session_state.database_occurances, ensure_ascii=False, indent=2
            ),
            height=120,
            disabled=disabled,
            key="cfg_database_occurances",
        )
        knowledge_base_str = st.text_area(
            "Knowledge base",
            value=json.dumps(
                st.session_state.knowledge_base, ensure_ascii=False, indent=2
            ),
            height=100,
            disabled=disabled,
            key="cfg_knowledge_base",
        )
        examples_str = st.text_area(
            "Examples",
            value=json.dumps(st.session_state.examples,
                             ensure_ascii=False, indent=2),
            height=80,
            disabled=disabled,
            key="cfg_examples",
        )
        special_context_chunks_str = st.text_area(
            "Special context chunks",
            value=json.dumps(
                st.session_state.special_context_chunks, ensure_ascii=False, indent=2
            ),
            height=100,
            disabled=disabled,
            key="cfg_special_context_chunks",
        )
        file_chunks_str = st.text_area(
            "File chunks",
            value=json.dumps(st.session_state.file_chunks,
                             ensure_ascii=False, indent=2),
            height=80,
            disabled=disabled,
            key="cfg_file_chunks",
        )

        st.markdown("---")
        st.markdown("### Export")
        st.download_button(
            label="Download LLM logs JSON",
            data=get_llm_logs_json(),
            file_name=(
                f"{st.session_state.prompt_type}_llm_logs_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ),
            mime="application/json",
            use_container_width=True,
            disabled=not st.session_state.llm_logs,
            help="Full LLM request/response log for every turn in this session.",
        )
        if st.session_state.messages:
            st.download_button(
                label="Download conversation JSON",
                data=get_download_json(),
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        st.markdown("---")

        st.markdown("### Load Conversation")
        uploaded_file = st.file_uploader("Upload JSON file", type=[
                                         "json"], key="file_upload")
        if uploaded_file is not None and st.button("Load from file", use_container_width=True):
            content = uploaded_file.read().decode("utf-8")
            if load_conversation_from_json(content):
                st.success("Conversation loaded!")
                st.rerun()

        paste_json = st.text_area(
            "Or paste conversation JSON", height=120, key="paste_json"
        )
        if st.button("Load from pasted JSON", use_container_width=True):
            if paste_json.strip():
                if load_conversation_from_json(paste_json):
                    st.success("Conversation loaded!")
                    st.rerun()
            else:
                st.warning("Please paste JSON first.")

        st.markdown("---")
        if st.session_state.conversation_started and st.button(
            "Reset Conversation", use_container_width=True
        ):
            reset_conversation()
            st.rerun()

    with col_chat:
        st.markdown(
            '<div class="main-header">EXAI Prompt Tester</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Test default_prompt and browser_prompt via OpenRouter</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.conversation_started:
            log_count = len(st.session_state.llm_logs)
            st.caption(f"LLM calls logged this session: {log_count}")
            st.download_button(
                label="⬇️ Download LLM logs JSON",
                data=get_llm_logs_json(),
                file_name=(
                    f"{st.session_state.prompt_type}_llm_logs_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                ),
                mime="application/json",
                use_container_width=True,
                disabled=log_count == 0,
                key="chat_download_llm_logs",
            )

        if not st.session_state.conversation_started:
            user_question = st.text_area(
                "User question",
                value=st.session_state.user_question,
                height=100,
                disabled=disabled,
                key="user_question_input",
                help="Question sent to the LLM as user_question (separate from chat follow-ups).",
            )
            st.session_state.user_question = user_question

            if st.button("Start Conversation", use_container_width=True):
                try:
                    parsed = validate_config_from_form(
                        database_occurances_str,
                        knowledge_base_str,
                        examples_str,
                        special_context_chunks_str,
                        file_chunks_str,
                    )
                    save_config_to_session(
                        prompt_type,
                        language,
                        learning_video_answer_text,
                        expert_id,
                        answer_model,
                        is_brief_mode,
                        is_expert_specific,
                        image_data,
                        screenshot_mode,
                        is_voice_mode,
                        parsed,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

                question = st.session_state.user_question.strip()
                if not question:
                    st.warning("Please enter a user question.")
                    st.stop()

                st.session_state.messages.append(
                    {"role": "user", "content": question}
                )

                conversation_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                with st.spinner("Generating answer..."):
                    try:
                        response = run_answer_generator(
                            question, conversation_history)
                    except Exception as exc:
                        st.error(f"API error: {exc}")
                        st.session_state.messages.pop()
                        st.stop()

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.session_state.conversation_started = True
                st.rerun()

        if st.session_state.messages:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f"<strong>User:</strong><br>{msg['content']}</div>",
                        unsafe_allow_html=True,
                    )
                elif msg["role"] == "assistant":
                    st.markdown(
                        f'<div class="chat-message assistant-message">'
                        f"<strong>Assistant:</strong><br>{msg['content']}</div>",
                        unsafe_allow_html=True,
                    )

        if st.session_state.conversation_started:
            user_input = st.chat_input("Type your message...")
            if user_input:
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )

                conversation_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                with st.spinner("Thinking..."):
                    try:
                        response = run_answer_generator(
                            user_input, conversation_history
                        )
                    except Exception as exc:
                        st.error(f"API error: {exc}")
                        st.session_state.messages.pop()
                        st.stop()

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()


if __name__ == "__main__":
    main()
