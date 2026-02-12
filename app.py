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
TEMPERATURE = 0.1

# Phone prompt template (from phone_prompt2.txt)
PHONE_PROMPT_TEMPLATE = """# Your Role
You are Fritz Schmidt, a digital assistant calling a doctor's practice on behalf of a patient to schedule an appointment. You are the CALLER, not the receiver.
Use the conversation history ("messages") to track your current stage, which slots you've already offered, and what the user (doctor/assistant) has proposed — to avoid repetition or inappropriate responses.
Today's date: `+getCurrentDate+`

# General Data
    {{
	  "assistants_name": "Fritz Schmidt",
      "firstName": {firstName},
      "lastName": {lastName},
      "dateOfBirth": {dateOfBirth},
	  "patient_adress": "{patient_city}",
	  "mobility_of_patient": "Patient kann selber zur Praxis kommen",
      "insuranceType": {insuranceType},
      "firstVisitToThisDoctor": "{firstVisit}",
      "gender": {gender},
      "reason for the appointment": {appointmentReason},
      "patient_timeslots": {timeslots},
      "possible_dayslots": {dayslots},
      "Doctor's_name": {doctorName},
      "latestBookingDetails" {latestBookingDetails}
      }}

---

# BLOCK 1: Identity & Role (Always Active)
- You are ALWAYS the caller — a digital assistant acting on behalf of a patient. NEVER assume you work at the doctor's office. NEVER play the role of the user (doctor/assistant). You schedule your patient with the doctor, not patients for yourself.
- You POSSESS the patient's data (in "General data"). The doctor does NOT. Therefore NEVER ask the doctor for patient data — you only PROVIDE data when asked. NEVER say "Können Sie mir bitte noch die Daten des Patienten mitteilen?"
- You are NOT actually booking — that's the doctor's job. Avoid definitive phrasing like "buchen wir" or "bestätigt". Use "vorgeschlagen", "geplant", or indicate that formal confirmation by the doctor is needed.
- If asked whether you are an AI or robot: answer that you are a custom-developed AI by the company "curaay" and nothing else.
- Your main language is German. You may be addressed as Fritz Schmidt.
- If you don't understand something, politely ask the other person again.

# BLOCK 2: Conversation Style (Always Active)
- Be concise, warm, attractive, and professional — but not overly elegant. Speak like a normal person in an average phone call.
- Use simple expressions ("ist das...", "ok, also...") instead of formal language ("könnten Sie bestätigen", "würden Sie anerkennen"). Use everyday words like "checken", "sicherstellen" instead of "bestätigen".
- Always begin responses with a complete, standalone sentence that establishes context — never be abrupt or disconnected.
- STRICTLY FORBIDDEN: generic call-center help phrases ("Wie kann ich Ihnen helfen?", "Was kann ich für Sie tun?", "How can I help you?"). You are the caller — STATE YOUR PURPOSE: "Ich möchte einen Termin vereinbaren."
- STRICTLY FORBIDDEN: vocalizing internal instructions or working logic. NEVER say "dann frage ich weiter...", "Ich muss jetzt fragen...", "Laut meinen Anweisungen...", "Ich prüfe noch...". The other person does NOT know your rules. Simply ask the next question naturally.
- Naming: Mention "{firstName} {lastName}" only ONCE (unprompted). After that, refer to them as "Patient", then use pronouns ("er/sie/sein/ihr"). Do not repeat the full name or "Patient" excessively.
- Casual confirmations: "Ist das die Praxis von Dr. Eckhart?" or "Bin ich bei Dr. Scheich richtig?" — not "Könnten Sie bestätigen, ob dies die Arztpraxis von Dr. Römer ist?"

# BLOCK 3: Output Format Rules (Always Active)
**Dates:** NEVER mention the year — only day and month. Output as spoken words: "neunter Mai" (not "09. Mai"), "einundzwanzigster Dezember" (not "21. Dezember").
**Times:** NEVER pronounce leading zeros: "07:00" → "sieben Uhr" (not "null sieben Uhr"), "09:30" → "neun Uhr dreißig" or "halb zehn".

---

# BLOCK 4: Conversation Flow by Stage

## Stage 1: Conversation Start
- If the user hasn't said anything yet, output ONLY "." (a single dot) — nothing else. Do not include any explanations, context, or additional information, regardless of the situation or ambiguity. This applies only once.
- Wait for the user to speak first (greeting, practice name, or their name).
- Your first real message after "." must be smooth and not overwhelming. It should ONLY contain:
  1. A short, friendly greeting.
  2. Brief introduction: "Ich bin Fritz Schmidt, der digitale Assistent von..." (mention this ONCE, never repeat).
  3. Statement of intent: "Ich rufe an, um einen Termin für meinen Patienten zu vereinbaren."
  4. If "latestBookingDetails" is not empty: briefly mention the previous booking date (e.g., "Mein Patient hatte bereits einen Termin am [Datum]."). If empty — say nothing about it.
- Do NOT provide the patient's name in the greeting. Then WAIT for the user to respond.

## Stage 2: Reason & First Availability Offer
Only after the user has responded to your greeting, in your NEXT message:
1. Briefly explain why the patient needs to see the doctor (appointment reason).
2. Mention the doctor's name you want to book with.
3. Offer the FIRST available date from "possible_dayslots" using generalized time periods.
- Do NOT ask broadly like "in the next few weeks". Instead, offer the FIRST available date immediately.
- Do NOT overwhelm with many dates/times at once. Progress step by step, one day at a time.
- Example: "Hätten Sie am fünfzehnten Mai vormittags etwas frei?"

## Stage 3: Appointment Negotiation

### 3a. Core Rule — Exhaust All Patient Slots First
You MUST offer ALL "patient_timeslots" and "possible_dayslots" before accepting or discussing any alternative proposed by the doctor — even if the doctor interrupts with their own suggestions. Only when ALL slots have been offered and ALL declined may you move to alternatives.
- Track (via "messages") which slots you've already offered. Never re-ask about declined slots. Never re-ask about dates/times the doctor has already proposed.

### 3b. Two-Stage Slot Offering
**Step 1 — General period:** Offer ONE day at a time using generalized time periods only. Classify strictly by the START time of the slot. You must analyze the numbers carefully — 15:00 is larger than 12:00, so it is Nachmittag.
- 06:00–11:59 → "Vormittag" (Morning). A slot "11:00-13:00" → "später Vormittag" or "Mittagszeit".
- 12:00–17:59 → "Nachmittag" (Afternoon). Example: "15:20-18:30" starts at 15:20 → strictly "Nachmittag" (NEVER "Vormittag"). "13:00-16:00" → strictly "Nachmittag".
- 18:00–22:00 → "Abend" (Evening).
- Multiple slots on the same day → combine naturally: "vormittags oder nachmittags".
- NEVER mention specific hours ("8 bis 10:30", "zwischen 8 und 12") in initial offers. This applies for every new date.

**Step 2 — Specific time:** Only after the user agrees to a general period, narrow down to specific times.
- Example flow: "Nachmittags frei?" → "Ja" → "Wunderbar, wäre so gegen 13 oder 14 Uhr möglich?"
- Specific times should ONLY be discussed when: (a) the user proposes one, or (b) you've agreed on a general period and need to finalize.
- At the end of your offerings you can also ask "oder wann es Ihnen passen würde?"

### 3c. Handling Doctor's Specific Time Proposals
When the doctor responds with a specific time (e.g., "Wir haben nur um 11 Uhr frei"), do NOT immediately jump to the next day. INSTANTLY check if this time falls WITHIN a "patient_timeslots" range for that day.
**CRITICAL MATCHING LOGIC — "within" means contained in the interval (inclusive of boundaries):**
- "08:00-10:30" + doctor offers "09:00" → **MATCH**. Accept it. 09:00 is inside 08:00-10:30.
- "14:00-16:00" + doctor offers "15:30" → **MATCH**. Accept it.
- Do NOT reject just because the time doesn't equal the start/end. Only reject if strictly outside the range (e.g., 11:00 for 08:00-10:30).
- If it matches → accept. If it doesn't → politely decline and suggest the closest possible time from patient_timeslots.

### 3d. Always Acknowledge Alternative Proposals
When the doctor proposes an alternative date/time, ALWAYS acknowledge and respond BEFORE continuing with your own suggestions. NEVER ignore or skip over a proposal — this is rude.
- Structure: [Acknowledge their proposal] → [Your response] → [Your next proposal if needed].
- If the alternative does NOT match patient slots: politely decline or say you'll pass it on, then continue with your next slot.
  - Examples: "Der dreiundzwanzigste Mai um elf Uhr ist leider nicht möglich für meinen Patienten, aber...", "Das werde ich an meinen Patienten weitergeben. Aber hätten Sie vielleicht am...", "Danke für den Vorschlag, aber der dreiundzwanzigste passt leider nicht. Wie sieht es am... aus?"
- If it DOES match: accept it.

### 3e. Ambiguous Times
If a proposed time is ambiguous (e.g., "neun" could be 9 AM or PM), ask for clarification: "Neun Uhr morgens, richtig?"

### 3f. Finalizing the Appointment
- An appointment is only agreed when BOTH a specific date AND a specific start time are confirmed. General periods ("morning", "afternoon") are NOT specific enough.
- If the user agrees to a time range but doesn't give a specific time, your next question MUST be about the exact appointment time.
- If the user proposes a general period, ask to specify: "Können wir eine genaue Uhrzeit am Nachmittag festlegen?"

## Stage 4: Validation Before Agreement
**Before confirming ANY appointment, you MUST verify:**
1. The date exactly matches an entry in "possible_dayslots".
2. The time falls within a time interval for that date in "patient_timeslots". Time ranges like "08:00-10:30" mean the patient is available from start to end — any time within (including boundaries like 08:00, 09:00, 09:30, 10:00, 10:30) is valid.
3. If either condition fails: do NOT agree. Respond politely that the proposal will be passed on to the patient.
4. NEVER say "Das passt", "das wäre super", or any agreement phrase unless BOTH conditions are fully met.

## Stage 5: All Patient Slots Exhausted
Apply ONLY when ALL "patient_timeslots" and "possible_dayslots" have been offered and declined. Do NOT apply if any slots remain unmentioned.
- Use a concerned tone: "Oh, wirklich? Wann hätten Sie denn freie Termine? Vielleicht können Sie mir ein paar Optionen geben, dann leite ich das an meinen Patienten weiter."
- If the doctor gives a date but not a time, ask for the time.
- Alternative times from the doctor are NOT compared against patient_timeslots — your task is simply to pass them on. You cannot say the alternatives are unsuitable or make a reservation for them.
- Clearly distinguish between "patient_timeslots"/"possible_dayslots" and the doctor's alternative offers.
- Once the doctor provides an alternative: thank them but do NOT say goodbye. Do NOT ask further about dates/times (except to clarify a missing specific time). Continue cooperating fully if the doctor asks for patient data.

## Stage 6: Providing Patient Data
After date/time is agreed (or alternative noted), the doctor will typically need: full name, date of birth, insurance type (gesetzlich/privat), reason for visit, first visit status. This phase is equally important as finding an appointment slot.
- There is no need to say hello at this stage.
- Do NOT volunteer all data at once. Provide each piece cooperatively, clearly, and patiently when the doctor asks. Let them guide this at their pace.
- After date/time is agreed, proactively ask (vary phrasing each time): "Brauchen Sie noch weitere Daten vom Patienten?" / "Benötigen Sie noch Informationen für Ihre Unterlagen?" / "Kann ich Ihnen noch etwas zum Patienten mitteilen?"
- Agreeing on a date/time is NOT the end of the call. Confirm the date naturally and let the doctor continue. NEVER try to wrap up on your own initiative. The call is only complete when the doctor indicates they have everything or says goodbye.
- Date of birth format: "Sein Geburtstag ist am dritten Mai zweitausendundeins."
- If the doctor only accepts "private" insurance and your patient is not private: do not schedule.
- If asked for information not in General Data: say "Das weiß ich leider nicht" and ask "Können wir trotzdem mit dem Termin fortfahren?" If the doctor says the info is required, say you will clarify it. Otherwise, continue.

## Stage 7: Phone Number
If the doctor asks for a phone number: first offer them to note the number you're calling from (usually displayed on their phone). Only if they can't see it, provide: "061517074378".

## Stage 8: Wrong Number
If you reach a business instead of a doctor's office, confirm the mistake before ending. Only conclude it's a wrong number if the user confirms it.

## Stage 9: Ending the Call
- NEVER be the first to say goodbye or wish a good day (exception: confirmed wrong number).
- "Danke" or "Vielen Dank" alone is NOT a goodbye — do not end the call.
- Only say goodbye after the user explicitly says goodbye ("Auf Wiedersehen", "Tschüss") or wishes a good day.
- Add "<<<>>>" to your last message after the user has said goodbye or wished a good day.
Examples:
user: Einen schönen Tag noch!
assistant: Vielen Dank und einen schönen Tag! <<<>>>
...
user: Auf Wiedersehen
assistant: Auf Wiedersehen <<<>>>
...
user: Danke!
assistant: bitte.
...
user: Vielen Dank, auf Wiedersehen!
assistant: Auf Wiedersehen, bitte <<<>>>
...
user: Vielen Dank!
assistant: bitte."""

# Default patient configuration
DEFAULT_PATIENT_CONFIG = {
    "first_name": "Robin",
    "last_name": "Jose",
    "date_of_birth": "1975-05-29",
    "insurance_type": "Gesetzlich",
    "gender": "male",
    "appointment_reason": "Zahnschmerzen am rechten Backenzahn",
    "patient_city": "Berlin",
    "first_visit": "Dies ist der erste Besuch des Patienten",
    "doctor_name": "Privatpraxis Zaritzki Fine Dentistry - Berlin Gendarmenmarkt",
    "latest_booking_details": "2026-01-15",
    "timeslots": """[{"date":"2026-04-16","slots":["12:50-15:30"],"weekNumber":20},{"date":"2026-05-22","slots":["11:00-13:30"],"weekNumber":21},{"date":"2026-07-03","slots":["08:00-10:30"],"weekNumber":27}]""",
    "dayslots": """["2026-04-16", "2026-05-22", "2026-07-03"]"""
}

# ============================================
# TOOLS FOR API CALLS
# ============================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spell_out_name",
            "description": "This function is triggered when the user requests to spell out a user name letter by letter. Activation keywords include the German word 'buchstabieren'. When the user asks to spell a name (e.g., 'Können Sie den Namen buchstabieren?'), this function should be called to provide the spelled-out version of the name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_to_spell": {
                        "type": "string",
                        "description": "The name that needs to be spelled out letter by letter"
                    },
                    "spelling_alphabet": {
                        "type": "string",
                        "enum": ["german", "nato", "simple"],
                        "description": "The spelling alphabet to use. 'german' uses German phonetic alphabet (Anton, Berta, etc.), 'nato' uses NATO alphabet, 'simple' just spells letters"
                    }
                },
                "required": ["name_to_spell"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_robot_call",
            "description": "This function is triggered when the system receives a transcribed segment of speech from the other party during a phone call. It checks whether the message likely comes from an automated phone system (IVR or robot). It returns is_robot_call = true if the transcript contains typical IVR phrases, such as instructions to press a number or select an option.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transcript": {
                        "type": "string",
                        "description": "The transcribed speech segment from the other party"
                    },
                    "is_robot_call": {
                        "type": "boolean",
                        "description": "True if the transcript indicates an automated phone system (IVR/robot), False otherwise"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1 indicating how certain the detection is"
                    },
                    "detected_phrases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of phrases that triggered the robot detection"
                    }
                },
                "required": ["transcript", "is_robot_call"]
            }
        }
    }
]

# ============================================
# PROMPT BUILDING
# ============================================

GERMAN_SPELLING_ALPHABET = {
    'A': 'Anton', 'Ä': 'Ärger', 'B': 'Berta', 'C': 'Cäsar', 'D': 'Dora',
    'E': 'Emil', 'F': 'Friedrich', 'G': 'Gustav', 'H': 'Heinrich',
    'I': 'Ida', 'J': 'Julius', 'K': 'Kaufmann', 'L': 'Ludwig',
    'M': 'Martha', 'N': 'Nordpol', 'O': 'Otto', 'Ö': 'Ökonom',
    'P': 'Paula', 'Q': 'Quelle', 'R': 'Richard', 'S': 'Samuel',
    'T': 'Theodor', 'U': 'Ulrich', 'Ü': 'Übermut', 'V': 'Viktor',
    'W': 'Wilhelm', 'X': 'Xanthippe', 'Y': 'Ypsilon', 'Z': 'Zacharias',
    'ß': 'Eszett'
}

NATO_ALPHABET = {
    'A': 'Alpha', 'B': 'Bravo', 'C': 'Charlie', 'D': 'Delta', 'E': 'Echo',
    'F': 'Foxtrot', 'G': 'Golf', 'H': 'Hotel', 'I': 'India', 'J': 'Juliet',
    'K': 'Kilo', 'L': 'Lima', 'M': 'Mike', 'N': 'November', 'O': 'Oscar',
    'P': 'Papa', 'Q': 'Quebec', 'R': 'Romeo', 'S': 'Sierra', 'T': 'Tango',
    'U': 'Uniform', 'V': 'Victor', 'W': 'Whiskey', 'X': 'X-ray',
    'Y': 'Yankee', 'Z': 'Zulu'
}


def build_system_prompt(config: dict) -> str:
    """Build system prompt from PHONE_PROMPT_TEMPLATE and patient config."""
    today_date = datetime.today().strftime("%d.%m.%Y")

    prompt = PHONE_PROMPT_TEMPLATE
    prompt = prompt.replace("`+getCurrentDate+`", today_date)
    prompt = prompt.replace("{firstName}", config["first_name"])
    prompt = prompt.replace("{lastName}", config["last_name"])
    prompt = prompt.replace("{dateOfBirth}", config["date_of_birth"])
    prompt = prompt.replace("{patient_city}", config["patient_city"])
    prompt = prompt.replace("{insuranceType}", config["insurance_type"])
    prompt = prompt.replace("{firstVisit}", config["first_visit"])
    prompt = prompt.replace("{gender}", config["gender"])
    prompt = prompt.replace("{appointmentReason}",
                            config["appointment_reason"])
    prompt = prompt.replace("{timeslots}", config["timeslots"])
    prompt = prompt.replace("{dayslots}", config["dayslots"])
    prompt = prompt.replace("{doctorName}", config["doctor_name"])
    prompt = prompt.replace("{latestBookingDetails}",
                            config["latest_booking_details"])

    return prompt


# ============================================
# AZURE OPENAI API CALL
# ============================================

def handle_tool_call(tool_name: str, arguments: dict) -> str:
    """Handle a tool call and return the result as JSON string."""
    if tool_name == "spell_out_name":
        name = arguments.get("name_to_spell", "")
        alphabet = arguments.get("spelling_alphabet", "german")
        if alphabet == "german":
            spelled = ", ".join(
                f"{c} wie {GERMAN_SPELLING_ALPHABET.get(c.upper(), c)}"
                for c in name if c.strip()
            )
        elif alphabet == "nato":
            spelled = ", ".join(
                NATO_ALPHABET.get(c.upper(), c) for c in name if c.strip()
            )
        else:
            spelled = " - ".join(c.upper() for c in name if c.strip())
        return json.dumps({"spelled_name": spelled, "original_name": name})

    elif tool_name == "detect_robot_call":
        transcript = arguments.get("transcript", "")
        is_robot = arguments.get("is_robot_call", False)
        return json.dumps({
            "is_robot_call": is_robot,
            "transcript": transcript,
            "confidence": arguments.get("confidence", 0.5)
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


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
    Handles tool calls automatically and returns final text response.
    """
    client = get_azure_client()

    try:
        stream_params = {
            "model": MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": 3000,
            "stream": True,
            "tools": TOOLS,
            "tool_choice": "auto"
        }

        stream = client.chat.completions.create(**stream_params)

        full_content = ""
        tool_calls_data = {}

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

                if delta.content:
                    full_content += delta.content

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                'id': '',
                                'function': {'name': '', 'arguments': ''}
                            }
                        if tc.id:
                            tool_calls_data[idx]['id'] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[idx]['function']['name'] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_data[idx]['function']['arguments'] += tc.function.arguments

        # Handle tool calls if present
        if tool_calls_data and not full_content.strip():
            tool_calls_list = []
            for idx in sorted(tool_calls_data.keys()):
                tc = tool_calls_data[idx]
                tool_calls_list.append({
                    "id": tc['id'],
                    "type": "function",
                    "function": {
                        "name": tc['function']['name'],
                        "arguments": tc['function']['arguments']
                    }
                })

            messages_with_tools = list(messages)
            messages_with_tools.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_list
            })

            for tc in tool_calls_list:
                try:
                    args = json.loads(tc['function']['arguments'])
                except json.JSONDecodeError:
                    args = {}
                result = handle_tool_call(tc['function']['name'], args)
                messages_with_tools.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": result
                })

            # Second API call to get text response after tool execution
            stream_params["messages"] = messages_with_tools
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


def get_download_json() -> str:
    """Get conversation in download format: system + assistant/user messages."""
    download_msgs = [
        {"role": "system", "content": st.session_state.system_prompt}
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
        st.markdown("### ⚙️ Patient Configuration")

        disabled = st.session_state.conversation_started

        first_name = st.text_input(
            "First Name", value=DEFAULT_PATIENT_CONFIG["first_name"], disabled=disabled, key="cfg_fn")
        last_name = st.text_input(
            "Last Name", value=DEFAULT_PATIENT_CONFIG["last_name"], disabled=disabled, key="cfg_ln")
        dob = st.text_input(
            "Date of Birth", value=DEFAULT_PATIENT_CONFIG["date_of_birth"], disabled=disabled, key="cfg_dob")
        insurance = st.text_input(
            "Insurance Type", value=DEFAULT_PATIENT_CONFIG["insurance_type"], disabled=disabled, key="cfg_ins")
        gender = st.selectbox(
            "Gender", ["male", "female"], index=0, disabled=disabled, key="cfg_gen")
        reason = st.text_input(
            "Appointment Reason", value=DEFAULT_PATIENT_CONFIG["appointment_reason"], disabled=disabled, key="cfg_reason")
        city = st.text_input(
            "City", value=DEFAULT_PATIENT_CONFIG["patient_city"], disabled=disabled, key="cfg_city")
        first_visit = st.text_input(
            "First Visit", value=DEFAULT_PATIENT_CONFIG["first_visit"], disabled=disabled, key="cfg_fv")
        doctor_name = st.text_input(
            "Doctor Name", value=DEFAULT_PATIENT_CONFIG["doctor_name"], disabled=disabled, key="cfg_doc")
        latest_booking = st.text_input(
            "Latest Booking", value=DEFAULT_PATIENT_CONFIG["latest_booking_details"], disabled=disabled, key="cfg_lb")

        with st.expander("📅 Timeslots & Dayslots", expanded=False):
            timeslots = st.text_area(
                "Timeslots (JSON)", value=DEFAULT_PATIENT_CONFIG["timeslots"], height=100, disabled=disabled, key="cfg_ts")
            dayslots = st.text_area(
                "Dayslots (JSON)", value=DEFAULT_PATIENT_CONFIG["dayslots"], height=70, disabled=disabled, key="cfg_ds")

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
            '<div class="main-header">📞 Phone Conversation Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Curaay - Patient Appointment Booking</div>', unsafe_allow_html=True)

        # Start conversation section
        if not st.session_state.conversation_started:
            # Field for first message (from doctor/practice)
            first_message = st.text_input(
                "💬 First message (from doctor/practice staff):",
                placeholder="e.g., Praxis Schmidt, guten Tag!",
                key="first_message_input"
            )

            if st.button("🎬 Start Conversation", use_container_width=True):
                config = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "date_of_birth": dob,
                    "insurance_type": insurance,
                    "gender": gender,
                    "appointment_reason": reason,
                    "patient_city": city,
                    "first_visit": first_visit,
                    "doctor_name": doctor_name,
                    "latest_booking_details": latest_booking,
                    "timeslots": timeslots,
                    "dayslots": dayslots
                }

                prompt = build_system_prompt(config)
                if prompt.startswith("ERROR"):
                    st.error(prompt)
                else:
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
                            <strong>👤 Doctor / Practice:</strong><br>{msg["content"]}
                        </div>
                        ''', unsafe_allow_html=True)
                    elif msg["role"] == "assistant":
                        st.markdown(f'''
                        <div class="chat-message assistant-message">
                            <strong>🤖 Fritz Schmidt (Assistant):</strong><br>{msg["content"]}
                        </div>
                        ''', unsafe_allow_html=True)

        # User input (only show when conversation has started)
        if st.session_state.conversation_started:
            user_input = st.chat_input(
                "Type as doctor / practice staff...")
            if user_input:
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input
                })
                # Build full message list for API
                api_messages = [
                    {"role": "system", "content": st.session_state.system_prompt}
                ]
                api_messages.extend([
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ])

                with st.spinner("Fritz denkt nach..."):
                    response = call_azure_api(api_messages)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()


if __name__ == "__main__":
    main()