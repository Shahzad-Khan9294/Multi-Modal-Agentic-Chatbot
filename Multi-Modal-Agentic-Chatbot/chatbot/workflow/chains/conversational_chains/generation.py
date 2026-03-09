# generation.py (chains) — COMPLETE UPDATED VERSION (STREAM ANSWER ONLY + KEEP LONG PROMPT + USED_CHUNKS VIA POST CHAIN)

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate

from ...utils import get_langchain_vllm_model, get_langchain_vllm_model_sr


# -----------------------------
# Chunk / citation formatting
# -----------------------------
class Citation(BaseModel):
    """Represents a numbered chunk passed to the LLM."""
    chunk_id: int = Field(..., description="1-based chunk index")
    text: str = Field(..., description="Chunk text content")
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (id/type/heading/etc.) for traceability",
    )


def format_chunks_numbered(chunks: Union[List[Document], List[str], str, None]) -> str:
    """
    Converts a list of Document chunks (or strings) into a numbered chunk block:

    ### Chunk 1:
    ''' ... '''

    ### Chunk 2:
    ''' ... '''
    """
    if chunks is None:
        return ""

    if isinstance(chunks, str):
        return chunks.strip()

    if not chunks:
        return ""

    lines: List[str] = []
    chunk_index = 1

    for item in chunks:
        if isinstance(item, Document):
            txt = (item.page_content or "").strip()
            meta = item.metadata or {}
        else:
            txt = (str(item) or "").strip()
            meta = {}

        if not txt:
            continue

        _ = Citation(chunk_id=chunk_index, text=txt, meta=meta)
        lines.append(f"### Chunk {chunk_index}:\n'''{txt}'''")
        chunk_index += 1

    return "\n\n".join(lines)


# -----------------------------
# Pydantic schemas
# -----------------------------
class GenerationOutput(BaseModel):
    """
    Kept for internal representation (answer + used_chunks).
    NOTE: We will NOT stream this JSON in /chat_stream.
    """
    answer: str = Field(..., description="Final answer to the user.")
    used_chunks: List[int] = Field(
        default_factory=list,
        description="1-based chunk indices that were actually used to answer.",
    )

class UsedChunksOutput(BaseModel):
    used_chunk_numbers: List[int] = Field(
        default_factory=list,
        description="1-based chunk indices used to construct the answer."
    )

# -----------------------------
# 1) STREAMING CHAIN (answer-only)
# -----------------------------
def get_generation_chain(model: str, max_tokens: int = 1000):
    """
    Streams ONLY the final answer text (no JSON, no used_chunks in output).
    Keeps your full long grounding prompt, but changes OUTPUT instructions for streaming.
    """
    llm = get_langchain_vllm_model(model=model, max_tokens=max_tokens)

    system_template = """
You are a TAG AI assistant. Your task is to answer the user’s query by strictly using the information available in the provided CHUNKS and CHAT HISTORY.

**RESPONSE PRIORITY ORDER**:
  1. First, check the provided CHUNKS for information relevant to the query.
  2. If the CHUNKS are empty or do not fully answer the query, check the CHAT HISTORY.
  3. Use CHAT HISTORY to supplement, reference, or transform previously stated information.
  4. Do not use any knowledge outside of the CHUNKS and CHAT HISTORY.

**GROUNDING RULES**:
  - Use only information explicitly present in the CHUNKS or CHAT HISTORY.
  - You may summarize, rephrase, combine, or restructure information.
  - Do NOT invent, assume, or add new facts, details, or external knowledge.
  - Do NOT fall back to general world knowledge or model training data.

**CHUNKS INTERPRETATION**:
  - The CHUNKS may be structured (lists, fields, tables, metadata).
  - Convert structured information into clear, natural language statements when answering.
  - Some CHUNK entries may include numerical fields such as scores, aggregated scores, or updated scores that are associated with a specific risk subject identified through keywords or relevant subject labels.
  - When such score-related information is present, interpret it only as quantitative indicators tied to the mentioned risk subject and its timeline.

**CHAT HISTORY HANDLING**:
  - Before responding, review the CHAT HISTORY to identify what has already been covered.
  - Avoid repeating information unless it improves clarity or is required for the task.
  - If the user asks for summaries, findings, emails, or transformations, use CHAT HISTORY as a valid source when CHUNKS are not provided.

**TASK HANDLING**:
  - Treat requests such as summarization, findings, comparison, explanation, rephrasing, or email writing as transformations of existing information.
  - Perform these tasks using only the available CHUNKS and/or CHAT HISTORY.

**INTENT CHECK**:
  - Before including any numerical or score-based information, determine whether it is necessary to directly answer the user question.
  - Exclude numerical data (new_aggregated_score) if it does not materially contribute to answering the query.

**CONDITIONAL CI Score REPORTING RULE**:
  - Use CI Score information only from given information for ci score related question.
  - For each record, include **ALL** available data present in CI Score or 'note' field in your output response.

**ANSWER STYLE**:
  - If the question is simple, respond briefly and clearly.
  - If explanation or analysis is required, provide a concise but complete response.
  - Be natural, focused, and easy to read.
  - Match the user input language.

**INSUFFICIENT INFORMATION RULE**:
  - Only state that information is unavailable if neither the CHUNKS nor CHAT HISTORY contains relevant information, even after interpretation.
  - Do not add unsupported details.

**OUTPUT FORMAT (STREAM MODE)**:
  - Output ONLY the final answer text.
  - Do NOT output JSON.
  - Do NOT include used_chunks.
  - Do NOT wrap output in markdown.
  - Do NOT use triple backticks.
""".strip()

    human_template = """
### Provided Chunks:
{chunks_block}

### Chat History:
'''{chat_history}'''

### User Question:
{question}
""".strip()

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(template=system_template),
                additional_kwargs={},
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["chunks_block", "chat_history", "question"],
                    template=human_template,
                ),
                additional_kwargs={},
            ),
        ]
    )

    def _preprocess_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        context = inputs.get("context", None)
        chunks_block = format_chunks_numbered(context)
        return {
            "chunks_block": chunks_block,
            "chat_history": inputs.get("chat_history", ""),
            "question": inputs.get("question", ""),
        }

    return _preprocess_inputs | prompt | llm | StrOutputParser()


# -----------------------------
# 2) POST-STEP CHAIN (used_chunks only, Pydantic)
# -----------------------------
def get_used_chunks_chain(model: str, max_tokens: int = 200):
    """
    After the answer is generated, compute used_chunks without streaming them.
    Returns a UsedChunksOutput object (validated).
    """
    llm = get_langchain_vllm_model_sr(model=model, max_tokens=max_tokens)
    parser = PydanticOutputParser(pydantic_object=UsedChunksOutput)

    system_template = """
You are a strict attribution agent.

Your job:
Given the CHUNKS and the final ANSWER, output which chunk numbers were actually used to produce the answer.

DEFINITION OF "USED":
A chunk is USED only if it contains explicit information that directly supports at least one concrete statement in the ANSWER.
- "Directly supports" means the chunk includes the same fact/claim, a clearly equivalent paraphrase, or a unique detail that appears in the answer.
- If the chunk is only topically related, provides background, or could have been used but is not clearly reflected in the answer, it is NOT USED.

INCLUSION RULES (must satisfy at least one):
Include chunk N only if at least one of these is true:
1) The answer contains a specific fact/number/name/date/decision/outcome AND chunk N contains that same fact.
2) The answer makes a specific comparison or causal statement AND chunk N contains the supporting reasoning in-text.
3) The answer uses a specific label/term (e.g., a statute section, doctrine name, tariff type, country list) AND chunk N contains that same label/term in a supporting context.

EXCLUSION RULES (very important):
- Do NOT include chunks that only talk about the same general topic.
- Do NOT include chunks that are "related reading" or "additional context" unless the answer clearly uses their details.
- Do NOT include multiple chunks that say the same thing unless the answer clearly relies on different unique details from each.
- If two chunks repeat the same information and the answer does not require both, prefer the one that most directly matches the answer wording.

CONSERVATIVENESS:
Be conservative. If you are unsure whether a chunk was used, EXCLUDE it.

OUTPUT RULES:
- Return ONLY valid JSON matching the schema.
- Do NOT wrap output in markdown.
- Do NOT use triple backticks.
- Do NOT include extra keys.
- used_chunk_numbers must contain ONLY integers (1-based chunk indices), sorted ascending, unique.
- If none are used, return exactly: {{"used_chunk_numbers": []}}
""".strip()

    human_template = """
### User Question:
{question}

### Provided Chunks:
{chunks_block}

### Final Answer:
{answer}

Task:
List only chunk numbers that directly support specific statements in the answer (not just related).

Return output strictly following these instructions:
{format_instructions}
""".strip()

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate(prompt=PromptTemplate(template=system_template)),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["chunks_block", "question", "answer", "format_instructions"],
                    template=human_template,
                ),
            ),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    def _preprocess_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        context = inputs.get("context", None)
        chunks_block = format_chunks_numbered(context)
        return {
            "chunks_block": chunks_block,
            "answer": inputs.get("answer", ""),
            "question": inputs.get("question", ""),
        }

    return _preprocess_inputs | prompt | llm | parser