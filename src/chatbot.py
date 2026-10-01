"""RAG orchestration: guardrails -> query rewrite -> retrieval -> grounded LLM answer."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from . import config, prompts
from .retriever import Hit, Retriever

log = logging.getLogger(__name__)

Messages = list[dict[str, str]]

_LEAK_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"(system|hidden|initial|internal|secret)\s+(prompt|instructions?|message)",
        r"ignore\s+(all\s+|your\s+|the\s+|any\s+)?(previous\s+|prior\s+|above\s+)?(instructions?|rules|prompt)",
        r"\bapi[\s_-]?keys?\b",
        r"environment\s+variables?",
        r"\b(reveal|show|print|repeat)\b.*\b(prompt|instructions)\b",
        r"\bjailbreak\b|\bdeveloper mode\b",
    )
]
_GREETING = re.compile(r"^\s*(hi+|hello+|hey+|hola|namaste|good\s+(morning|afternoon|evening))\W*$", re.I)
_THANKS = re.compile(r"^\s*(thanks|thank\s+you|thx|ty)\b[\w\s]*\W*$", re.I)


class LLMError(Exception):
    """Raised with a message that is safe to show to end users."""


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    # answered | not_found | out_of_scope | refused | info | error
    status: str = "answered"
    top_score: float | None = None


class LLM(Protocol):
    def complete(self, messages: Messages, temperature: float = 0.1, max_tokens: int = 500) -> str: ...


class LLMClient:
    """Thin OpenAI wrapper that converts every failure into a friendly LLMError."""

    def __init__(self) -> None:
        api_key = config.get_api_key()
        if not api_key:
            raise LLMError(
                "The OpenAI API key is missing. Add OPENAI_API_KEY to your .env file "
                "(or Streamlit secrets) and restart the app."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=config.OPENAI_BASE_URL or None,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=2,
        )

    def complete(self, messages: Messages, temperature: float = 0.1, max_tokens: int = 500) -> str:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise LLMError("The AI service rejected the API key. Please check OPENAI_API_KEY.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("The AI service is busy or the quota is exhausted. Please try again shortly.") from exc
        except openai.APITimeoutError as exc:
            raise LLMError("The AI service took too long to respond. Please try again.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("I couldn't reach the AI service. Please check the internet connection.") from exc
        except openai.OpenAIError as exc:
            raise LLMError("The AI service returned an error. Please try again.") from exc
        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LLMError("The AI service returned an empty response. Please try again.")
        return text


def _clean_history(history: list[dict]) -> Messages:
    cleaned: Messages = []
    for message in history[-config.HISTORY_MESSAGES:]:
        content = (message.get("content") or "").strip()
        if message.get("role") in ("user", "assistant") and content:
            cleaned.append({"role": message["role"], "content": content[:800]})
    return cleaned


def _format_context(hits: list[Hit]) -> str:
    parts: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        block = f"[{number}] Source: {hit.chunk['url']}\n{hit.chunk['text']}"
        if parts and used + len(block) > config.MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _sources(hits: list[Hit], limit: int = 3) -> list[Source]:
    unique: dict[str, Source] = {}
    for hit in hits:
        url = hit.chunk["url"]
        if url not in unique:
            unique[url] = Source(title=hit.chunk["page_name"], url=url)
        if len(unique) >= limit:
            break
    return list(unique.values())


class BonumChatbot:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None) -> None:
        self.retriever = retriever if retriever is not None else Retriever()
        self.llm = llm if llm is not None else LLMClient()

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        """Never raises: every failure becomes a friendly Answer."""
        try:
            return self._answer(question, history or [])
        except LLMError as exc:
            return Answer(str(exc), status="error")
        except Exception:
            log.exception("Unexpected error while answering")
            return Answer(prompts.ERROR_MESSAGE, status="error")

    def _answer(self, question: str, history: list[dict]) -> Answer:
        question = (question or "").strip()[: config.MAX_QUESTION_CHARS]
        if not question:
            return Answer(prompts.EMPTY_MESSAGE, status="info")
        if any(pattern.search(question) for pattern in _LEAK_PATTERNS):
            return Answer(prompts.REFUSAL_MESSAGE, status="refused")
        if _GREETING.match(question):
            return Answer(prompts.GREETING_MESSAGE, status="info")
        if _THANKS.match(question):
            return Answer(prompts.THANKS_MESSAGE, status="info")

        recent = _clean_history(history)
        standalone = self._standalone_question(question, recent)
        hits = self.retriever.search(standalone)
        if not hits:
            return Answer(prompts.NO_MATCH_MESSAGE, status="out_of_scope", top_score=None)

        interpretation = f"\n(Interpreted as: {standalone})" if standalone != question else ""
        messages: Messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}, *recent]
        messages.append(
            {
                "role": "user",
                "content": prompts.USER_TEMPLATE.format(
                    context=_format_context(hits), question=question, interpretation=interpretation
                ),
            }
        )
        raw = self.llm.complete(messages, temperature=0.1, max_tokens=500).strip()
        return self._parse(raw, hits)

    def _standalone_question(self, question: str, recent: Messages) -> str:
        if not recent:
            return question
        transcript = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in recent[-4:])
        try:
            rewritten = self.llm.complete(
                [
                    {"role": "system", "content": prompts.REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"},
                ],
                temperature=0,
                max_tokens=80,
            )
        except LLMError:
            return question
        rewritten = " ".join(rewritten.split()).strip("\"' ")
        return rewritten if 0 < len(rewritten) <= 300 else question

    @staticmethod
    def _parse(raw: str, hits: list[Hit]) -> Answer:
        top = hits[0].score if hits else None
        if raw.startswith("[OUT_OF_SCOPE]"):
            return Answer(prompts.OUT_OF_SCOPE_MESSAGE, status="out_of_scope", top_score=top)
        if raw.startswith("[NOT_FOUND]"):
            body = raw[len("[NOT_FOUND]"):].strip() or prompts.NOT_FOUND_MESSAGE
            return Answer(body, status="not_found", top_score=top)
        text = raw.replace("[NOT_FOUND]", "").replace("[OUT_OF_SCOPE]", "").strip()
        return Answer(text, sources=_sources(hits), status="answered", top_score=top)
