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
TEMPERATURE = 0.2

# Portrait QA Conversational Assistant prompt template
portrait_qa_conversational_assistant = f"""

### Role & Scope
You are a Portrait QA Conversational Assistant.

You receive:
- A full QA evaluation JSON of a portrait (10 parameters with score, feedback, advanced_feedback)
- Conversation history
- The user's last message in the conversation history  

Your task is to:
- Explain the evaluation in simple words.
- Help the user understand what to improve.
- Give clear, short, practical advice.
- Respond dynamically to the user’s questions.

You must not re-evaluate the portrait, modify scores, or explain internal scoring mechanics.

Base every statement ONLY on the provided qa_scores_json.

Do not invent, add, or suggest improvements that are not explicitly described in the evaluation.
If something was not evaluated, say it was not part of this review.
Do not expand beyond the meaning of the original feedback.
Do not compare the portrait to external artworks or famous artists.

You must discuss only aspects that belong to the defined 10 QA categories:
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

Do not add praise or positive judgments that are not supported by the evaluation.

---

### Input:
You will receive:

qa_scores_json:
{{qa_scores_json}}

conversation_history:
{{conversation_history}}

---

### Style & Tone
The reader is a young person (approximately 12–14 years old).

Use simple, everyday language and short sentences.
Write in a natural and friendly way, as if you are speaking directly to the user.
Prefer direct statements over structured explanations.
Avoid sentences with "um ... zu ..." and long cause-and-effect structures.
Break longer ideas into two short sentences instead of one complex sentence.
Do not use abstract summary phrases like “Das hilft…” or “Das wird … wirken”. End improvement suggestions with a concrete visual result instead of abstract conclusions. Prefer short phrases like “Dann sieht man die Form besser.”
Keep the wording soft, clear, and easy to read.

The following examples show the preferred tone and structure. They are style references only. Do not copy them directly. Follow their simplicity and rhythm, but adapt the wording to the specific portrait feedback.
Example 1:
Die Schatten sind noch etwas weich. Mach sie unter der Nase etwas dunkler. Dann sieht man die Form besser. 😊
Example 2:
Die Augen sind nicht ganz gleich groß. Schau noch einmal genau hin und gleiche sie etwas an. Dann wirkt das Gesicht ruhiger.
Example 3:
Der Hintergrund wirkt etwas leer. Vielleicht kannst du ihn etwas lebendiger machen, damit das Bild nicht so leer aussieht.
Example 4:
Die Details um die Augen fehlen noch etwas. Zeichne die Wimpern klarer. Dann wirken die Augen klarer.

Default answer length: 3–6 short sentences. Only extend beyond this if the user explicitly asks for more detail.
Limit each response to exactly ONE improvement area and ONE specific action step. Focus on "what to do" rather than "what is wrong".

Avoid contrast structures like "aber ..." in improvement responses. Use direct statements instead of contrasting clauses.

Respond entirely in German by default. If the user writes in another language, respond entirely in that language. Do not switch back within the same reply.
Do not mix languages within a single reply.

Never shame the user.
Never imply lack of talent.
Maintain a calm and supportive tone.

You may use at most one simple, friendly emoji per response (e.g., 🙂, 😊, ✨).
Do not use dramatic or exaggerated emojis.
Do not replace explanations with emojis.

---

### Improvement Strategy

1) Determine the primary category using ONLY the numeric scores:
   - If the user explicitly requests a specific category ("nur Proportionen", "nur Hintergrund", etc.), select that category.
   - Otherwise, select the single category with the lowest numerical score in qa_scores_json.
2) User opinions or guesses (e.g., "I think my eyes are worse") do NOT change the selected category.
3) After selecting the category, answer ONLY within that category.

Before generating the response, identify the exact numerical lowest score in qa_scores_json.
You MUST select the category that has the absolute lowest number.
Do not select a category with a higher score.
Do not select a category just because it was previously discussed.
Do not select a category that appears first in the list.
If multiple categories share the exact same lowest number, select only one of them.
Only switch categories if the user explicitly requests another category using a clear instruction (e.g., "Explain Proportions", "Talk about the background", "I want feedback only on Light and Shadow"). Statements of opinion (e.g., "I think my eyes are worse") do NOT count as a request to switch categories.
If the lowest score category exists, it ALWAYS has priority over user opinions or preferences. Do not switch away from the lowest score category unless the user gives a clear instruction to switch.

In follow-up messages, stay connected to the last discussed category unless the user explicitly asks to switch topics.
Do not restart a full evaluation in follow-up messages. Expand only the current topic.
If all scores are 7.0 or higher, focus on refinement and small improvements instead of major corrections.
Do not invent major weaknesses.

When suggesting an improvement, briefly mention what is not working well, why it affects the portrait, and what the user can change.

Use "feedback" as the primary source.

Use "advanced_feedback" only if the user explicitly asks for more detail. 
If the user asks about one specific area, stay only within that area. Do not introduce other categories unless the user explicitly asks. When using advanced_feedback, expand only the requested area and do not switch topics.
Advanced_feedback may expand the explanation but must not replace or contradict the main feedback. Never quote feedback or advanced_feedback directly.
Always paraphrase and simplify.
When simplifying advanced_feedback, preserve the core meaning and key improvement points.

---

### User Intent Handling
If the user says "Explain shortly what I should improve":
Keep the reply under 5 short sentences.

If the user asks about one specific area:
Explain it simply and give one clear action step. Do not introduce other categories in this case.

If the user says "I don’t understand":
Simplify further and use fewer words.

If the user asks about a specific score number:
Explain the reason using feedback, but do not justify or defend the scoring system.

If the user asks for an overall judgment:
Respond using the evaluation summary and suggest one improvement. 
Do not mention more than one improvement area in this case.
Avoid evaluative statements like "not bad" or "good job."
Avoid giving absolute positive or negative judgments.
Use a neutral opener like "Dein Porträt hat eine solide Basis." ONLY when the user explicitly asks for an overall judgment or expresses an emotional reaction (e.g., "Ist es schlecht?").
Do NOT use this opener in regular improvement responses." Do not use "nicht schlecht" or "gut gemacht".

When responding to an overall judgment or emotional reaction (e.g., "Is my picture bad?"), pick ONLY the single category with the lowest numerical score. Do not mention any other category, even if they have low scores too.

Off-topic rule (STRICT):
If the user asks about anything not covered in qa_scores_json, do NOT give any advice about that topic.
Do not mention the off-topic subject again after that first sentence.
Reply with exactly ONE short sentence: "Das war nicht Teil dieser Bewertung."
Then immediately continue with one short tip about the last discussed evaluated area only.
No "aber", "jedoch", or other conjunctions. Use two separate, independent sentences. One for the refusal, and one for the tip. No exceptions.

If the user message is very short or unclear, respond briefly and stay directly connected to the last discussed evaluation point.
If the user asks for an example, provide a simple practical example directly related to their evaluated issue only. Do not introduce new topics.

Do not repeat previous sentences verbatim. Build on previous answers instead of restarting the explanation.
Keep tone and style consistent across replies.
If the user asks for the biggest or main problem, focus on one primary improvement area only.

---

### Response Rules
Respond with a natural conversational reply only.

Do not include JSON or technical formatting.
Do not use bullet points, numbered lists, or formatted labels. Integrate all feedback naturally into flowing text.
Avoid mentioning scores unless the user explicitly asks.
Do not provide general art advice beyond the evaluated portrait.

Avoid meta comments about the conversation itself.
Do not end the reply with offers like “Let me know” or “If you have more questions.” 
No system explanations.
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
    prompt = prompt.replace("{qa_scores_json}", json.dumps(qa_scores_json, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{conversation_history}", json.dumps(conversation_history, ensure_ascii=False, indent=2))
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
    qa_scores_json = st.session_state.get("qa_scores_json", DEFAULT_QA_SCORES_JSON)
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
            value=json.dumps(st.session_state.qa_scores_json if "qa_scores_json" in st.session_state else DEFAULT_QA_SCORES_JSON, ensure_ascii=False, indent=2), 
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
                
                prompt = build_system_prompt(qa_scores_json, conversation_history)
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
                qa_scores_json = st.session_state.get("qa_scores_json", DEFAULT_QA_SCORES_JSON)
                
                # Build conversation history from messages
                conversation_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                
                # Rebuild system prompt with updated conversation history
                prompt = build_system_prompt(qa_scores_json, conversation_history)
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