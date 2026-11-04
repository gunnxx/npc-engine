import re

import tiktoken


class TokenCounter:
    """Counts and trims text with a fixed tiktoken encoding."""

    def __init__(self, encoding_name: str, message_overhead: int):
        """Constructor."""
        self._encoder = tiktoken.get_encoding(encoding_name)
        self._message_overhead = message_overhead

    # ==============
    # Public methods
    # ==============

    def count(self, text: str) -> int:
        """Count the tokens of a plain string.

        Args:
            text: Text to measure.

        Returns:
            Number of tokens in the counter's encoding.
        """
        return len(self._encoder.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """Count the tokens of a list of chat messages including the per-message overhead.

        Args:
            messages: Chat messages. ``content`` may be ``None`` on an assistant message that carries
                ``tool_calls``; the name and arguments of each tool call are counted instead.

        Returns:
            Total number of tokens the messages occupy in a prompt.
        """
        total = 0
        for message in messages:
            total += self.count(message.get("content") or "") + self._message_overhead
            for call in message.get("tool_calls") or []:
                total += self.count(call["function"]["name"] + call["function"]["arguments"])
        return total

    def truncate(self, text: str, limit: int) -> str:
        """Cut a text at a sentence boundary so it fits within a token limit.

        Args:
            text: Text to shorten.
            limit: Maximum number of tokens.

        Returns:
            The original text, or its leading sentences that fit within ``limit``.
        """
        if self.count(text) <= limit:
            return text
        kept = ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            candidate = f"{kept} {sentence}".strip()
            if self.count(candidate) > limit:
                break
            kept = candidate
        return kept or text[: limit * 3]
