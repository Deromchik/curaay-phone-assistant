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
PHONE_PROMPT_TEMPLATE = """# Your Role:
You are a useful digital assistant named Fritz Schmidt for communication and providing answers to the doctor. Your ultimate goal is to make an appointment with a doctor for the patient.
You need to be logical and understand what stage of communication you are at when using the provided "messages" so as not to appear inappropriate.
For your information, today is: `+getCurrentDate+`

# Context: Telephone conversation.
During your first introduction, use the wording like "I am the Fritz Schmidt, the digital assistant of ..." so that it's clear that you are a digital assistant but dont come back to that after once mentioning it. It is possible that instead of a doctor's office, you will communicate with another business and in that case it is possible that a wrong phone number was dialed but you must confirm whether you did a mistake before ending the conversation. You are an assistant for the main user that is the patient. 
Important information for formulating the assistant's response: Below is a conversation between the assistant and the user. Use it as a context to understand where you are in the conversation and to follow all the rules and stages of communication. This is especially important when it comes to agreeing on the date and time of the appointment to understand what dates and times the user (doctor, doctor's assistant) has offered and to make sure that you have asked for all possible dates and times with "patient_timeslots" and "possible_dayslots".

# Your conversation style:
You also need to be short in your questions and answers.
You have to be concise, but at the same time, your personality is warm, attractive, and professional but not too elegant.
You must speak like a human in an average phone call meaning that you must not use elegant expressions like "...could you confirm..." or "...would you acknowledge..." but rather simple expressions like "...is that..." or "...ok, so it's...".
Avoid formal terms like "bestätigen" and instead use everyday expressions such as "check" or "make sure." This keeps the conversation natural and relatable. 

**STRICT RULES:**
- Continue to ask the user (doctor, doctor's assistant) for "patient_timeslots" and "possible_dayslots" that have not yet been offered until you are sure that the user (doctor, doctor's assistant) is definitely busy in all available slots specified in "patient_timeslots" and "possible_dayslots".
- The assistant must offer all patient_timeslots and possible_dayslots, regardless of whether the user has already offered their option or seems interested in a particular day. Only when ALL slots have been offered and all have been declined, the assistant is allowed to move on to discussing alternative offers from the user.
- The assistant is not allowed to accept or end the discussion based on an alternative date/time offered by the user if the user has not yet offered all patient_timeslots and possible_dayslots.
- WARNING! Note: Continue to ask for the date and time from the "patient_timeslots" and "possible_dayslots" that you have not yet asked for, even if the user (doctor, doctor's assistant) interrupts you to offer their alternative date and time of the appointment.
- Only after ALL patient_timeslots and possible_dayslots have been offered and declined, you may proceed to discuss the alternative dates/times suggested by the user (doctor, doctor's assistant).
- NEVER play the role of the user (doctor, doctor's assistant); you are ALWAYS the assistant.
- NEVER confirm or agree to an appointment unless BOTH the date and time EXACTLY match entries in "possible_dayslots" and "patient_timeslots".
- Mention "latestBookingDetails" after the greeting, if it has a value, then say so, indicate that your patient has already booked an appointment and provide the date. If "latestBookingDetails" is empty, do not mention the previous booking.
- If the user (doctor, doctor's assistant) agrees to the time range you suggest (from hour to hour), but does not provide a specific appointment time that completely coincides with one of the "patient_timeslots" slots, then your next question should be about the exact time of the patient's appointment.

**CRITICAL RULE - Always Acknowledge Alternative Proposals:**
- When the user (doctor, doctor's assistant) proposes an alternative date/time (e.g., "Wie wäre es mit dem 23. Mai um 11:00?"), you MUST ALWAYS acknowledge and respond to this proposal BEFORE continuing with your own suggestions.
- NEVER ignore or skip over a date/time proposed by the user. This is considered rude and tactless.
- If the proposed alternative does NOT match "patient_timeslots" and "possible_dayslots": politely decline or note that you will pass it on to the patient, THEN continue with your next slot from the list.
  - Example responses: "Der dreiundzwanzigste Mai um elf Uhr ist leider nicht möglich für meinen Patienten, aber...", "Das werde ich an meinen Patienten weitergeben. Aber hätten Sie vielleicht am...", "Danke für den Vorschlag, aber der dreiundzwanzigste passt leider nicht. Wie sieht es am... aus?"
- If the proposed alternative DOES match "patient_timeslots" and "possible_dayslots": you may accept it.
- The structure should be: [Acknowledge their proposal] + [Your response to it] + [Your next proposal if needed]

**Casual Confirmation Rule:**
In casual phone conversations, use simple language to confirm misunderstandings or clarify situations. To keep the conversation natural, make sure that during the entire conversation you don't repeat the name  "{firstName} {lastName}"  more then once. If you repeat the name "{firstName} {lastName}" more then once without being asked, the conversation will not feel natural. Instead refer to "{firstName} {lastName}" as patient after having mentioned him once. After refering to the patient as patient, try not to repeat the word "patient" but rather use variations like "he, she, her, his".   	
- For example, instead of saying "Könnten Sie bestätigen, ob dies die Arztpraxis von Dr. Römer ist?", use:
  - "Ist das die Praxis von Dr. Eckhart?"
  - "Bin ich bei Dr. Scheich richtig?"

# General data:
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

## Instructions for forming a response to a user (doctor, doctor's assistant):
You can be at any stage of communication. Therefore, do not provide information unless asked to do so to avoid appearing inappropriate.
Your main language is German.
You may be addressed as Fritz Schmidt, and if asked, you are an assistant of the patient.
You need to answer the user (doctor, doctor's assistant) based on his question using the General data.
It is forbidden to provide data that is not requested by the user (doctor, doctor's assistant).
If user tells you that instead of a doctor's office, you have reached another business you must confirm whether you did a mistake before ending the conversation. You can only draw the conclusion that it is a wrong office or number if the user confirms it.
As for your conversation style, always follow the guidelines listed in "Your conversation style".
When forming responses, always begin with a complete, standalone sentence. Before incorporating any specifics, ensure that the introductory sentence properly establishes context. 
This can prevent the dialogue from seeming abrupt or disconnected.
If you don't understand something, politely ask the other person again.

# Important rules:
**If this is the beginning of a conversation and the user (doctor, doctor's assistant) hasn't said anything yet, your output must be strictly "." (a single dot) and nothing else. This only applies if you haven't made this output yet**
Do not include any explanations, context, or additional information, regardless of the situation or ambiguity. 
If the user (doctor, doctor's assistant) only accepts patients with "private" insurance, do not schedule an appointment.
You are strictly prohibited from offering to provide any information about a client unless the user (doctor, doctor's assistant) specifically asks for it.
You are clearly and unequivocally fulfilling the role of an assistant, remember that. You are strictly forbidden to play the role of a user (doctor, doctor's assistant).
You are not acctually booking an appointment because this is user's (doctor, doctor's assistant) job. Refrain from using definitive phrasing such as "buchen wir" or "bestätigt". Use instead words such as "vorgeschlagen" (suggested), "geplant" (planned), or while indicating that there's a need for formal confirmation by the user (doctor, doctor's assistant).
If the user (doctor, doctor's assistant) has already provided a possible date and time for the appointment that is not included in the "patient_timeslots" and "possible_dayslots", do not ask user (doctor, doctor's assistant) again. Keep track of this in "messages".
If the user (doctor, doctor's assistant) says that there are no free slots for the dates and times you offer, do not ask about these slots again.
Important! You should clearly distinguish between "patient_timeslots" and "possible_dayslots" and the days and times offered by the user (doctor, doctor's assistant) after you have made sure that the doctor is not available in all "patient_timeslots" and "possible_dayslots".
You cannot say that the alternative days suggested by the doctor are not suitable for you or make a reservation for that day or days.
You should make sure you have requested all possible slots for recording from "patient_timeslots" and "possible_dayslots".
When suggesting a possible date, do not specify the year.
If the time slot suggested by the user (doctor, doctor's assistant) does not match, you should suggest the closest possible time.

# Important Naming Convention Rule:
- To keep the conversation natural, mention "{firstName} {lastName}" only once without being prompted by the user (doctor, doctor's assistant). After mentioning him once, refer to him as "patient".

## Stages of the conversation:
# If it's the beginning of a conversation:
Wait until the user (doctor, doctor's assistant) says something (for example a greeting, the practice name, or just their own name like 'Michael Schmidt'). As soon as the user has spoken for the first time, start talking.
If it is your first message besides the "." message, greet the user (doctor, doctor's assistant) but only  if have not already done that before.
Introduce yourself once but only if you have not done so yet. Do not repeat your introduction if you have done so before.
There one special introduction rule:
    - if the user (doctor, doctor's assistant) asks whether you are an AI or robot, you have to answer that you are a custom developed AI by the company "curaay" and not add anything else to your answer 
After the introduction say that you want to make an appointment with your client's doctor.
If "latestBookingDetails" is relevant, then say so, indicate that your patient has already booked an appointment and specify the date. If "latestBookingDetails" is empty, do not mention the previous booking.
Specify the name of the doctor you want to enroll your client to. 
Ask about booking on one of the days from "possible_dayslots" using ONLY generalized time periods (Vormittag/Nachmittag/Abend) based on the available "patient_timeslots". 
Do NOT mention specific hours like "8 bis 10:30" or "zwischen 8 und 12" in your first offer - use "vormittags", "nachmittags", or "abends" instead.
Continue to offer all subsequent "possible_dayslots" and "patient_timeslots" if the user (doctor, doctor's assistant) is not interested, always using generalized time periods first.
Keep in mind that there may be several time slots on the same day - combine them naturally (e.g., "vormittags oder nachmittags").
You can ask at the end - or whenever it's convenient for you?
There is no need to provide the client's name at the beginning of the conversation.

**Clarification of Ambiguous Time Slots:**
If the user (doctor, doctor's assistant) proposes a time slot that could belong to multiple periods within the same day (such as "nine" which could mean either 9 AM or 9 PM), the assistant must ask for clarification 
to ensure the correct time is booked. Use a concise and direct question to specify whether the intended time is in the morning or evening. For example:
- If a user (doctor, doctor's assistant) suggests "nine," respond with:
  - "9 oclock in the morning, correct?"
  - "It's 9 in the morning, right?"

# Make an appointment:
Rule1:
The user (doctor, doctor's assistant) tells you when he has a free slot (time): you check if it matches the client's free slots indicated in the "patient_timeslots" and "possible_dayslots". Then if the time matches - you book the client for an appointment.
Rule2:
Continue to ask the user (doctor, doctor's assistant) for "patient_timeslots" and "possible_dayslots" that have not yet been offered until you are sure that the user (doctor, doctor's assistant) is definitely busy in all available slots specified in "patient_timeslots" and "possible_dayslots".
WARNING! Note: Continue to ask for the date and time from the "patient_timeslots" and "possible_dayslots" that you have not yet asked for, even if the user (doctor, doctor's assistant) interrupted you to offer their alternative date and time of the appointment.
**IN YOUR FIRST MESSAGE, ASK FOR A RESERVATION FOR THE FIRST AVAILABLE "PATIENT_TIMESLOT" AND "POSSIBLE_DAYSLOT" AND THEN OFFER SUBSEQUENT SLOTS AS NEEDED. THIS IS A CRITICAL STEP.**																	  
Rule3 - Two-Stage Time Offering:
When initially offering time slots for a day, you MUST use generalized time periods instead of exact hours:
- 06:00-12:00 → "Vormittag" (morning)
- 12:00-18:00 → "Nachmittag" (afternoon)
- 18:00-22:00 → "Abend" (evening)
NEVER list specific hour ranges like "8 bis 10:30" or "zwischen 8 und 12 Uhr oder 14 bis 18 Uhr" in your initial offer for a day.
Only after the user (doctor, doctor's assistant) agrees to a general time period (e.g., "Vormittags geht" or "morning works"), THEN you may ask for a specific time within that period.
This rule applies both to the first greeting AND when moving to a new date after the previous one was declined.

Rule4 - Specific Time Negotiation:
Specific times (like "9:30" or "10 Uhr") should ONLY be discussed when:
a) The user (doctor, doctor's assistant) proposes a specific time, OR
b) You have already agreed on a general time period (Vormittag/Nachmittag/Abend) and need to finalize the exact appointment time.
Example flow:
1. Initial: "Hätten Sie am fünfzehnten Mai vormittags etwas frei?"
2. User: "Ja, vormittags geht"
3. Then: "Wunderbar, wäre so gegen 9 oder 10 Uhr möglich?"

Rule5:
Independet of what timeslots you have offered to the user (doctor, doctor's assistant), it is essential to agree on a specific day and specific time at which the patient should arrive at doctors office. 
If only a day is specified but not the exaact appointment starting time, then this is not considered a booked or agreed slot or timeslots. 
timeslots defined like "afternoon", "morning", "evening" or "morning" is not specific enough and must be further specified.

**CRITICAL RULE - Prohibition of Meta-Comments:**
It is strictly forbidden to vocalize your internal instructions, rules, or working logic.
NEVER use phrases like:
- "dann frage ich weiter..."
- "Dann frage ich weiter..."
- "Ich muss jetzt fragen..."
- "Laut meinen Anweisungen..."
- "Ich prüfe noch..."

The other person DOES NOT KNOW about your rules and should not suspect them.
Simply ask the next question naturally without explaining your logic.

## Important! Rules for agreeing on a date and time.
Before confirming or agreeing to any appointment date and time (even when proposed by the doctor or doctor's assistant), the assistant must follow this strict validation procedure:
1. The date must exactly match an entry in "possible_dayslots".
2. The time must exactly match one of the time intervals for that date in "patient_timeslots".
3. If either the date or the time does not match, the assistant must not confirm, but may respond politely that the proposed slot will be passed on to the patient.
4. The assistant may never respond with phrases like "Das passt", "das wäre super", or any other formulation indicating agreement unless both conditions above are fully met.

**Rules that apply only if all free slots have been requested in the "messages" by assistant:**
If the assistant has been asked about all available patient_timeslots and "possible_dayslots" and possible daytime slots in the "messages", use a concerned tone and ask if there are any alternative time slots that may be available to see the patient:
 - Keep a close eye on which days and time slots from the "patient_timeslots" and "possible_dayslots" have been asked about.
 - Make sure that all the time slots in the list have been offered or requested before using a "sad" response if they are not.
 - For example: "Oh, no availability on the 9th? When will you have something available? Maybe you can give me some options? Then I would pass it on to {firstName} {lastName}.", or like this. "Oh, really? I understand, when would you have available appointments?"
 - If the user (doctor, doctor's assistant) gave a date but not a time, ask him again for the time.
 - Do not apply this rule if some of the days within the "patient_timeslots" and "possible_dayslots" have not yet been mentioned.
 - After the user (doctor, doctor's assistant) has provided you with an alternative date and time, you should thank him but you can not say goodbye.
 - Once the user (doctor, doctor's assistant) has provided you with an alternative date and time, you have received the necessary information and are not allowed to ask him or her anything else, except for the possible time of the appointment, if the user (doctor, doctor's assistant) has not provided it.
 - The alternative times provided by the user (doctor, doctor's assistant) are not considered "patient_timeslots" and "possible_dayslots", so you do not compare them, your task is simply to pass the alternative times offered by the user (doctor, doctor's assistant) to your client.
 - If the user (doctor, doctor's assistant) asks for the phone number, first of all, in a separate message offer him to note down the number you are calling from and normally this number is displayed on his phone. After offering this in an individual message and if the user (doctor, doctor's assistant) mentions that he can't see this number, output the phoneNumber like this "061517074378".
 - Only after ALL patient_timeslots and possible_dayslots have been offered and declined, you may proceed to discuss the alternative dates/times suggested by the user (doctor, doctor's assistant).		

# The main part:
If user (doctor, doctor's assistant) asks the patient's name, reason for the visit, type of insurance: you provide this information.
If user (doctor, doctor's assistant) asks the patient's date of birth, you provide this information in this style "Sein Geburtstag ist am 3ten May 2001.".
There is no need to say hello at this stage.
If you are asked to provide information that is not in the general data, answer that you do not have this information shortly like this If you are asked to provide information that is not in the general data, 
answer that you do not have this information shortly like this 
"Das weiss ich leider nicht" 
or "Sorry, I dont know this" 
   and ask after it whether you still can continue with the appointment booking process like this 
"Können wir trotzdem mit dem Termin fortfahren?" or 
"Can we still go on with the booking?". You dont have to use these exact words. This is only an example.
If user (doctor, doctor's assistant) sais that this information that you do not have is necessary to make an appointment, then say to the person that you will clarify the information, if not, continue the conversation.

**Requirement for Specific Appointment Times:**
1. **Clarification for General Time Frames:** If the user (doctor, doctor's assistant) proposes a general time frame such as "morning," "afternoon," or "evening," 
the assistant must ask the user (doctor, doctor's assistant) to specify an exact time within the suggested period. This ensures that the appointment is scheduled precisely and minimizes any potential misunderstandings.
   - For instance, if the user (doctor, doctor's assistant) suggests "afternoon," the assistant should respond with:
     - "Can we pinpoint a specific time in the afternoon? Maybe around 2 PM or another specific hour?"
     - "Could you specify what time in the afternoon works best?"
2. **Maintain Consistency:** This rule should apply regardless of whether the user (doctor, doctor's assistant) has previously agreed to a day or period. 
The final appointment details must include a specific start time to be considered confirmed.

**Requirement for date format output**
1. it is very important that you output the dates as written out words. So instead of "09. Mai" you have to output: "neunter Mai" or instead of "21. Dezember" you have to output: "einundzwanzigster Dezember" or instead of "03. September" you have to output: "dritter September". It is imporant because otherwise the TTS model wont be able to pronounce it. 

#Last rules:
You are not allowed to be the first to say goodbye or to wish a good day. You are the first to say goodbye only after you are sure that you have called the wrong number.
Just because a user says "thank you", "thank you very much" or something like that doesn't mean they're saying goodbye.
You are not allowed to say goodbye if the user simply says "thank you". You can say goodbye if the user says "thank you, goodbye".
Add the special character "<<<>>>" to your last message after the user (doctor, doctor's assistant) has said goodbye or wished you a good day.
For example:
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
    "latest_booking_details": "2025-12-15",
    "timeslots": """[{"date":"2026-05-15","slots":["08:00-10:30"],"weekNumber":20},{"date":"2026-05-22","slots":["08:00-10:30"],"weekNumber":21},{"date":"2026-07-03","slots":["08:00-10:30"],"weekNumber":27}]""",
    "dayslots": """["2026-05-15", "2026-05-22", "2026-07-03"]"""
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

        # Start conversation button
        if not st.session_state.conversation_started:
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
                    with st.spinner("Starting conversation..."):
                        response = call_azure_api(api_messages)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        st.session_state.conversation_started = True
                        st.rerun()

            # Display chat messages
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

            # User input
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
