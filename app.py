# ============================================
# CONFIGURATION VARIABLES
# ============================================
from openai import AzureOpenAI
from datetime import datetime
import os
import json
import streamlit as st

# Azure OpenAI API configuration
# Get API key from environment variable (can be overridden in main() from Streamlit secrets)
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = "2025-04-01-preview"
AZURE_ENDPOINT = "https://german-west-cenral.openai.azure.com"
MODEL = "gpt-4o"
TEMPERATURE = 0.4

# Portrait QA Conversational Assistant prompt template
portrait_qa_conversational_assistant = f"""

### Role & Scope
You are a Portrait QA Conversational Assistant — a chatbot that helps users understand their portrait evaluation.

You receive:
- A full QA evaluation JSON of a portrait (qa_scores_json)
- Conversation history (conversation_history)

Your task is to:
- Answer the user's questions about their portrait evaluation.
- Explain scores and feedback in simple words when asked.
- Give clear, short, practical improvement advice ONLY when the user asks for it.

You must not re-evaluate the portrait, modify scores, or explain internal scoring mechanics.
Reply only to the user's last message in the conversation history.
Base every statement ONLY on the provided qa_scores_json.

Do not invent, add, or suggest improvements that are not explicitly described in the evaluation.
If something was not evaluated, say it was not part of this review.
Do not expand beyond the meaning of the original feedback.
Do not compare the portrait to external artworks or famous artists.
Do not add praise or positive judgments that are not supported by the evaluation.

The 10 QA categories are:
1. Composition and Design
2. Proportions and Anatomy
3. Perspective and Depth
4. Use of Light and Shadow
5. Color Theory and Application
6. Brushwork and Technique
7. Expression and Emotion
8. Creativity and Originality
9. Attention to Detail
10. Overall Impact

Stay within what is described in the evaluation and do not add new interpretations.

---

### Input
qa_scores_json:
{{qa_scores_json}}

conversation_history:
{{conversation_history}}

---

### Style & Tone
The reader is a young person (approximately 12-14 years old).

Use simple, everyday language and short sentences.
Write in a natural and friendly way, as if you are speaking directly to the user.
Prefer direct statements over structured explanations.
Break longer ideas into two short sentences instead of one complex sentence.
Do not use abstract summary phrases like "This helps..." or "This will make it look better...". End improvement suggestions with a concrete visual result. Prefer short phrases like "Then you'll see the form better."
Keep the wording soft, clear, and easy to read.

The following examples show the preferred tone and structure. They are style references only. Do not copy them directly. Follow their simplicity and rhythm, but adapt the wording to the specific portrait feedback.
Example 1: The shadows are still a bit soft. Make them a bit darker under the nose. Then you'll see the shape better. :)
Example 2: The eyes are not quite the same size. Take another careful look and even them out a bit. Then the face will look calmer.
Example 3: The background feels a bit empty. Maybe you can make it a bit more lively so the picture doesn't look so bare.
Example 4: The details around the eyes are still missing a bit. Draw the eyelashes more clearly. Then the eyes will look sharper.

Default answer length: 3-6 short sentences. Only extend beyond this if the user explicitly asks for more detail.
Avoid contrast structures like "but..." in improvement responses. Use direct statements instead of contrasting clauses.

Respond entirely in English by default. If the user writes in another language, respond entirely in that language. Do not switch back within the same reply.
Do not mix languages within a single reply.

Never shame the user.
Never imply lack of talent.
Maintain a calm and supportive tone.

You may use at most one simple, friendly emoji per response (e.g., :), :D, *).
Do not use dramatic or exaggerated emojis.
Do not replace explanations with emojis.

---

### How to Handle the User's Message

Before responding, classify the user's message into one of these types:

**Type A — Information question** (asks about a score, a category, a reason):
Examples: "What is my score for anatomy?", "Which category is the lowest?", "Why is my light score low?"
Action: Answer ONLY the question. Give the score and/or explain the reason from feedback. Do NOT add improvement tips or action steps.

**Type B — Advice request** (asks what to improve, how to fix something):
Examples: "What should I improve?", "How can I fix the shadows?", "Give me a tip for proportions."
Action: Give improvement advice. Follow the Category Selection rules below.

**Type C — Overall judgment or emotional reaction** (asks if the portrait is good/bad, expresses frustration or pride):
Examples: "Is my picture bad?", "Am I talented?", "Am I doing well?", "Why are my scores so low? :("
Action: Respond with calm, supportive reassurance. Do NOT add unsolicited improvement advice. Keep it short and warm. Do not use phrases like "not bad" or "well done". You may use a neutral opener like "Your portrait has a solid foundation." only for this type.

**Type D — Off-topic** (not about the portrait, evaluation, or art):
Examples: "What's the weather?", "I was at a party yesterday", "Tell me a joke."
Action: Follow the Off-topic Handling rules below.

**Type E — Follow-up on current topic** (short reply, continuation, clarification):
Examples: "And what about that?", "I don't understand", "Tell me more", "Why?"
Action: Stay on the last discussed category. Expand or simplify. Do not switch topics.

If the message doesn't clearly fit one type, treat it as Type E.

---

### Category Selection (applies ONLY to Type B — advice requests)

When the user asks for improvement advice:

1. If the user names a specific category ("tell me about proportions", "how to fix the background"), select THAT category.
2. If the user asks generally ("what should I improve?", "give me a tip"), select the category with the lowest numerical score in qa_scores_json that has NOT yet been discussed in the conversation.
3. If all categories have already been discussed, select the one with the lowest score and offer a new angle or deeper detail.

If multiple categories share the same lowest score, select only one.
User opinions (e.g., "I think my eyes are worse") do NOT override category selection. Only explicit category requests do.

When giving advice, limit each response to exactly ONE category and ONE specific action step. Focus on "what to do" rather than "what is wrong".

Use "feedback" as the primary source.
Use "advanced_feedback" only if the user explicitly asks for more detail.
Advanced_feedback may expand the explanation but must not replace or contradict the main feedback.
Never quote feedback or advanced_feedback directly. Always paraphrase and simplify.
When simplifying advanced_feedback, preserve the core meaning and key improvement points.
If all scores are 7.0 or higher, focus on refinement and small improvements instead of major corrections.

---

### No-Repeat Rule
Before generating each response, scan the full conversation_history.
Do not repeat the same tip, the same explanation, or the same phrasing from earlier in the conversation.
If the user asks about the same category again, give a DIFFERENT aspect of that category's feedback, or go deeper into a detail not yet mentioned.
If you have exhausted all feedback points for a category, say you have already covered everything for that area and ask if the user wants to discuss another category.

---

### Off-topic Handling
If the user's message is NOT about the portrait, the evaluation, or art, it is off-topic.
Do NOT give any advice or information about the off-topic subject.
Do NOT repeat the off-topic subject in your reply.

Respond using ONE of the following variants. Check conversation_history and do NOT reuse a variant that was already used. Each variant may only be used ONCE per conversation.

Variant 1: "I'm here to help you with your portrait :) Maybe you have a question about your drawing?"
Variant 2: "Haha, that's interesting, but I'm more into portraits :D Want to know something about your work?"
Variant 3: "Oh, that's fun! But let's get back to your portrait :) What would you like to know?"
Variant 4: "Sounds cool! But I'm a portrait specialist :) Maybe we can discuss something about your work?"
Variant 5: "Wow, interesting! But my superpower is portraits :D Is there something you want to ask about your drawing?"
Variant 6: "I'd love to talk about that, but I understand portraits best :) Maybe there's something about your work?"
Variant 7: "That's cool! But let's better talk about your portrait * What interests you?"
Variant 8: "Hah, okay! But I'm best at helping with drawings :) Want to discuss something about your portrait?"

Adapt to the user's language. If the user writes in German, translate naturally into German. If in Ukrainian, translate into Ukrainian.
After all 8 variants are exhausted, create new similar responses in the same style, but do not repeat any already used phrase.
Do NOT add any improvement tips after off-topic responses.

---

### Follow-Up Questions
End every on-topic response (Types A, B, C, E) with ONE short follow-up question.
Do NOT end off-topic responses with a follow-up question (they already have their own question).

HARD RULE: Every follow-up question in this conversation MUST be unique. Read all previous assistant messages in conversation_history. Your new follow-up question must use DIFFERENT words and a DIFFERENT sentence structure than every previous one.

BANNED patterns — NEVER use these or any close translation of them:
- "Maybe something else interests you?" / "Може, щось ще цікавить?"
- "Maybe we can talk about another parameter?" / "Може, поговоримо про інший параметр?"
- "Maybe we can talk about another aspect?" / "Може, поговоримо про інший аспект?"
- "Maybe we can talk about something else?" / "Може, поговоримо про щось інше?"
- Any sentence starting with "Maybe we can..." / "Може, ми можемо..."

Instead, use varied structures. Here are style examples (do NOT reuse these exact phrases — create your own each time):
- "What part of your portrait are you most curious about?"
- "Anything specific you want to dig into?"
- "Which area would you like to hear about next?"
- "Got any other questions for me?"
- "What caught your eye in the evaluation?"
- "Curious about anything else?"
- "Want to pick another topic?"
- "What's on your mind?"

Keep it short (under 10 words), casual, and different every time.

---

### Response Rules
Respond with a natural conversational reply only.
Do not include JSON or technical formatting.
Do not use bullet points, numbered lists, or formatted labels. Integrate all feedback naturally into flowing text.
Avoid mentioning scores unless the user explicitly asks.
Do not provide general art advice beyond the evaluated portrait.
No system explanations. No meta comments about the conversation.
Only provide the final answer to the user.
"""

# Default QA scores JSON
DEFAULT_QA_SCORES_JSON = {
    "Composition and Design": {
        "score": 6.2,
        "feedback": "The face is centered, but the background feels empty and does not support the composition.",
        "advanced_feedback": "The composition would benefit from a more intentional use of space. Currently, the portrait is placed centrally without interaction with the background. Adding subtle tonal variation or simple background elements could enhance balance and visual interest."
    },
    "Proportions and Anatomy": {
        "score": 4.8,
        "feedback": "The eyes are slightly uneven in size, and the nose appears a bit too long compared to the lower part of the face.",
        "advanced_feedback": "There are minor proportional inconsistencies. The left eye is slightly larger than the right, and the vertical distance between the nose and mouth could be shortened. Using construction lines would help improve anatomical alignment."
    },
    "Perspective and Depth": {
        "score": 5.1,
        "feedback": "The portrait appears somewhat flat due to limited contrast in shading.",
        "advanced_feedback": "Depth is reduced because midtones dominate the face. Increasing contrast between light and shadow, especially along the jawline and temples, would improve the three-dimensional effect."
    },
    "Use of Light and Shadow": {
        "score": 4.5,
        "feedback": "The light direction is unclear, and shadows are too soft.",
        "advanced_feedback": "The shading lacks a consistent light source. Defining a clear light direction and strengthening cast shadows under the nose and chin would create stronger form definition."
    },
    "Color Theory and Application": {
        "score": 7.4,
        "feedback": "Color choices are harmonious and pleasant.",
        "advanced_feedback": "The color palette is balanced and works well together. Subtle variations in skin tones could further enhance realism."
    },
    "Brushwork and Technique": {
        "score": 6.8,
        "feedback": "Brush strokes are visible but controlled.",
        "advanced_feedback": "The technique shows confidence, though transitions between tones could be smoother in certain facial areas."
    },
    "Expression and Emotion": {
        "score": 6.0,
        "feedback": "The expression is neutral but lacks intensity.",
        "advanced_feedback": "The facial expression feels calm but could benefit from stronger emphasis around the eyes and eyebrows to convey clearer emotion."
    },
    "Creativity and Originality": {
        "score": 7.0,
        "feedback": "The portrait shows personal style.",
        "advanced_feedback": "There is a recognizable stylistic approach. Exploring more unique background or lighting choices could increase originality."
    },
    "Attention to Detail": {
        "score": 5.3,
        "feedback": "Some areas like eyelashes and hair texture are not fully developed.",
        "advanced_feedback": "Fine details around the eyes and hair could be refined to enhance realism and overall polish."
    },
    "Overall Impact": {
        "score": 6.1,
        "feedback": "The portrait has a solid foundation but needs refinement.",
        "advanced_feedback": "While technically competent, the portrait would benefit from stronger contrast and improved proportions to create a more striking overall impression."
    }
}

# ============================================
# TOOLS FOR API CALLS
# ============================================

TOOLS = []

# ============================================
# PROMPT BUILDING
# ============================================


def build_system_prompt(qa_scores_json: dict, conversation_history: list) -> str:
    """Build system prompt from portrait_qa_conversational_assistant template."""
    prompt = portrait_qa_conversational_assistant
    # Template is f-string so {{x}} became {x}; replace single-brace placeholders
    prompt = prompt.replace("{qa_scores_json}", json.dumps(
        qa_scores_json, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{conversation_history}", json.dumps(
        conversation_history, ensure_ascii=False, indent=2))
    return prompt


# ============================================
# AZURE OPENAI API CALL
# ============================================


def get_azure_client() -> AzureOpenAI:
    """Initialize and return Azure OpenAI client."""
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )


def call_azure_api(messages: list) -> str:
    """
    Call Azure OpenAI API with streaming.
    Returns final text response.
    """
    client = get_azure_client()

    try:
        stream_params = {
            "model": MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": 3000,
            "stream": True
        }

        stream = client.chat.completions.create(**stream_params)

        full_content = ""

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

                if delta.content:
                    full_content += delta.content

        return full_content

    except Exception as e:
        return f"[ERROR: {str(e)}]"


# ============================================
# STREAMLIT APPLICATION
# ============================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "qa_scores_json" not in st.session_state:
        st.session_state.qa_scores_json = DEFAULT_QA_SCORES_JSON


def get_download_json() -> str:
    """Get conversation in download format: system + assistant/user messages."""
    # Rebuild system prompt with current data to ensure it contains all substituted values
    qa_scores_json = st.session_state.get(
        "qa_scores_json", DEFAULT_QA_SCORES_JSON)
    conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    current_prompt = build_system_prompt(qa_scores_json, conversation_history)

    download_msgs = [
        {"role": "system", "content": current_prompt}
    ]
    for msg in st.session_state.messages:
        download_msgs.append({"role": msg["role"], "content": msg["content"]})
    return json.dumps(download_msgs, ensure_ascii=False, indent=2)


def load_conversation_from_json(json_str: str) -> bool:
    """Load conversation from JSON string. Returns True on success."""
    try:
        loaded = json.loads(json_str)
        if not isinstance(loaded, list) or len(loaded) == 0:
            st.error("Invalid format: expected a non-empty JSON array.")
            return False

        if loaded[0].get("role") == "system":
            st.session_state.system_prompt = loaded[0]["content"]
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in loaded[1:]
                if m.get("role") in ("user", "assistant")
            ]
        else:
            st.session_state.system_prompt = ""
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in loaded
                if m.get("role") in ("user", "assistant")
            ]

        st.session_state.conversation_started = True
        return True
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return False
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return False


def main():
    # Page configuration
    st.set_page_config(
        page_title="Phone Assistant - Curaay",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        }

        .chat-message {
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            color: #1a1a2e;
            font-size: 1rem;
            line-height: 1.6;
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
            color: #1a1a2e;
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
            padding: 1.2rem 0;
            margin-bottom: 1rem;
            border-bottom: 3px solid #2d6a7a;
        }

        .sub-header {
            color: #2d4a5a;
            font-size: 1rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        }

        section[data-testid="stSidebar"] .stMarkdown {
            color: #1a1a2e;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2d6a7a 0%, #4a90a4 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1d5a6a 0%, #3a8094 100%);
            box-shadow: 0 4px 12px rgba(45, 106, 122, 0.3);
        }

        .stTextInput > div > div > input {
            color: #1a1a2e;
            background: #ffffff;
            border: 2px solid #d0d8e0;
            border-radius: 8px;
        }

        .stTextInput > div > div > input:focus {
            border-color: #4a90a4;
            box-shadow: 0 0 0 2px rgba(74, 144, 164, 0.2);
        }

        .stTextArea > div > div > textarea {
            color: #1a1a2e;
            background: #ffffff;
        }

        .streamlit-expanderHeader {
            color: #1a1a2e;
            background: #f0f4f8;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Try to get Azure API key from Streamlit secrets
    global AZURE_API_KEY
    try:
        if hasattr(st, 'secrets') and 'AZURE_API_KEY' in st.secrets:
            AZURE_API_KEY = st.secrets['AZURE_API_KEY']
    except:
        pass

    if not AZURE_API_KEY:
        st.error("⚠️ Azure API key is not configured. Please set AZURE_API_KEY in Streamlit secrets or environment variables.")
        st.info("For local setup, create a `.streamlit/secrets.toml` file with the following content:\n```toml\nAZURE_API_KEY = \"your-api-key-here\"\n```\n\nOr set an environment variable:\n```bash\nexport AZURE_API_KEY=\"your-api-key-here\"\n```")
        st.stop()

    # Initialize session state
    init_session_state()

    # Layout
    col_chat, col_side = st.columns([2, 1])

    # ---- RIGHT COLUMN: Config, Download, Upload ----
    with col_side:
        st.markdown("### ⚙️ QA Scores Configuration")

        disabled = st.session_state.conversation_started

        qa_scores_json_str = st.text_area(
            "QA Scores JSON",
            value=json.dumps(
                st.session_state.qa_scores_json if "qa_scores_json" in st.session_state else DEFAULT_QA_SCORES_JSON, ensure_ascii=False, indent=2),
            height=400, disabled=disabled, key="cfg_qa")

        st.markdown("---")

        # ---- Download Conversation ----
        if st.session_state.messages:
            st.markdown("### 📥 Download Conversation")
            st.download_button(
                label="📥 Download JSON",
                data=get_download_json(),
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown("---")

        # ---- Load Existing Conversation ----
        st.markdown("### 📤 Load Existing Conversation")

        uploaded_file = st.file_uploader("Upload JSON file", type=[
                                         "json"], key="file_upload")
        if uploaded_file is not None:
            if st.button("📂 Load from file", use_container_width=True):
                content = uploaded_file.read().decode('utf-8')
                if load_conversation_from_json(content):
                    st.success("Conversation loaded!")
                    st.rerun()

        paste_json = st.text_area(
            "Or paste conversation JSON here", height=150, key="paste_json")
        if st.button("📋 Load from pasted JSON", use_container_width=True):
            if paste_json.strip():
                if load_conversation_from_json(paste_json):
                    st.success("Conversation loaded!")
                    st.rerun()
            else:
                st.warning("Please paste JSON first.")

        st.markdown("---")

        # ---- Reset ----
        if st.session_state.conversation_started:
            if st.button("🔄 Reset Conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.system_prompt = ""
                st.session_state.conversation_started = False
                st.session_state.qa_scores_json = DEFAULT_QA_SCORES_JSON
                st.rerun()

        # ---- Show system prompt ----
        if st.session_state.system_prompt:
            with st.expander("📋 Current System Prompt"):
                display_prompt = st.session_state.system_prompt
                st.text(
                    display_prompt[:1000] + "..." if len(display_prompt) > 1000 else display_prompt)

    # ---- LEFT COLUMN: Chat ----
    with col_chat:
        st.markdown(
            '<div class="main-header">🎨 Portrait QA Conversational Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Curaay - Portrait Evaluation Assistant</div>', unsafe_allow_html=True)

        # Start conversation section
        if not st.session_state.conversation_started:
            # Field for first message
            first_message = st.text_input(
                "💬 First message:",
                placeholder="e.g., Explain what I should improve",
                key="first_message_input"
            )

            if st.button("🎬 Start Conversation", use_container_width=True):
                try:
                    qa_scores_json = json.loads(qa_scores_json_str)
                    st.session_state.qa_scores_json = qa_scores_json
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in QA Scores: {e}")
                    st.stop()

                conversation_history = []

                prompt = build_system_prompt(
                    qa_scores_json, conversation_history)
                st.session_state.system_prompt = prompt

                api_messages = [{"role": "system", "content": prompt}]

                # Add first message from user if provided
                if first_message.strip():
                    api_messages.append({
                        "role": "user",
                        "content": first_message.strip()
                    })
                    st.session_state.messages.append({
                        "role": "user",
                        "content": first_message.strip()
                    })

                with st.spinner("Starting conversation..."):
                    response = call_azure_api(api_messages)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.session_state.conversation_started = True
                st.rerun()

        # Display chat messages (always show if there are messages)
        if st.session_state.messages:
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'''
                        <div class="chat-message user-message">
                            <strong>👤 User:</strong><br>{msg["content"]}
                        </div>
                        ''', unsafe_allow_html=True)
                    elif msg["role"] == "assistant":
                        st.markdown(f'''
                        <div class="chat-message assistant-message">
                            <strong>🤖 Assistant:</strong><br>{msg["content"]}
                        </div>
                        ''', unsafe_allow_html=True)

        # User input (only show when conversation has started)
        if st.session_state.conversation_started:
            user_input = st.chat_input("Type your message...")
            if user_input:
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input
                })

                # Get current QA scores JSON from session state or use default
                qa_scores_json = st.session_state.get(
                    "qa_scores_json", DEFAULT_QA_SCORES_JSON)

                # Build conversation history from messages
                conversation_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                # Rebuild system prompt with updated conversation history
                prompt = build_system_prompt(
                    qa_scores_json, conversation_history)
                # Update session state with current prompt
                st.session_state.system_prompt = prompt

                # Build full message list for API
                api_messages = [
                    {"role": "system", "content": prompt}
                ]
                api_messages.extend([
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ])

                with st.spinner("Thinking..."):
                    response = call_azure_api(api_messages)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()


if __name__ == "__main__":
    main()
