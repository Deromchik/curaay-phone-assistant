import json
from typing import Any


def _format_database_occurances(
    database_occurances: list,
    learning_video_answer_text: str,
    file_chunks: list,
    has_learning_video: bool,
) -> str:
    extras: list[Any] = list(database_occurances)
    if learning_video_answer_text:
        extras.append({"elearning_video_source_content": learning_video_answer_text})
    if file_chunks:
        extras.append({"file_database": file_chunks})
    label = (
        "(content from training videos for Microsoft Dynamics 365 Business Central modules from m+m)"
        if has_learning_video
        else ""
    )
    return f"-DATABASE_OCCURANCES {label} - ```{extras}```"


def build_user_prompt(
    *,
    user_question: str,
    conversation_history: list,
    database_occurances: list,
    knowledge_base: Any,
    examples: list,
    learning_video_answer_text: str,
    special_context_chunks: list,
    file_chunks: list,
    has_learning_video: bool,
    is_expert_specific: bool,
    image_data: bool = False,
    include_image_data: bool = False,
    special_chunks_when_image_only: bool = False,
) -> str:
    """Build the user prompt block shared by default and browser generators."""
    history = conversation_history if is_expert_specific else conversation_history[-5:]

    lines = [
        "You will now receive all necessary context.",
        "",
        _format_database_occurances(
            database_occurances,
            learning_video_answer_text,
            file_chunks,
            has_learning_video,
        ),
        "",
        f"-CONVERSATION_HISTORY - ```{history}```",
        "",
        f"-USER_QUESTION - ```{user_question}```",
        "",
        f"-KNOWLEDGE_BASE - ```{json.dumps(knowledge_base, ensure_ascii=False)}```",
        "",
        f"-EXAMPLES - ```{json.dumps(examples, ensure_ascii=False)}```",
    ]

    if include_image_data:
        lines.extend([
            "",
            f"-IMAGE_DATA - ```{image_data}``` (If True, you MUST use SPECIAL_CONTEXT_CHUNKS as the primary source)",
        ])

    show_special_chunks = special_context_chunks and (
        not special_chunks_when_image_only or image_data
    )
    if show_special_chunks:
        lines.extend([
            "",
            "-SPECIAL_CONTEXT_CHUNKS (the priority is the most important context) -",
            f"```{json.dumps(special_context_chunks, ensure_ascii=False)}```",
        ])

    return "\n".join(lines)
