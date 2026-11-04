import os

from dotenv import load_dotenv

from src.utils.tokens import TokenCounter

load_dotenv()  # the one place ``.env`` is read; everything that needs a key or a URL imports this module first

# ===================================================================================================
# Token budget: the whole GPT prompt must fit within 4K tokens, so the limit is split into fixed
# slices. Counts are in tokens of the ``o200k_base`` encoding, an approximation for ``gpt-6-luna``
# that errs on the safe side.
# ===================================================================================================

LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
"""Chat model used for NPC replies and for history summaries. ``OPENAI_BASE_URL`` in ``.env`` points the OpenAI SDK at
the gateway that serves it."""

RESPONSE_RESERVE_TOKENS = 700
"""Tokens left free for the model's reply."""

PERSONA_TOKENS = 300
"""Maximum size of a persona's system prompt."""

AMBIENT_TOKENS = 100
"""Maximum size of the ambient note (local time and weather) gathered when a conversation starts."""

TOOL_RESULT_TOKENS = 400
"""Maximum size of a tool result injected into the prompt."""

SUMMARY_TOKENS = 600
"""Maximum size of the rolling summary returned by ``GET_CONTEXT``."""

RECENT_TOKENS = 4096 - RESPONSE_RESERVE_TOKENS - PERSONA_TOKENS - AMBIENT_TOKENS - TOOL_RESULT_TOKENS - SUMMARY_TOKENS
"""Tokens available for the verbatim recent messages, including the player's newest message: what is left of the
4096-token prompt limit once every other slice has its share."""

MESSAGE_OVERHEAD_TOKENS = 4
"""Tokens the chat format adds per message on top of its content (role and separators)."""

TOKEN_COUNTER = TokenCounter("o200k_base", MESSAGE_OVERHEAD_TOKENS)
"""Process-wide counter in the ``o200k_base`` encoding; built once because loading the vocabulary is slow."""
