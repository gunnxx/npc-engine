HTTP_TIMEOUT_SECONDS = 10
"""Timeout for each outbound HTTP request a tool makes."""


class Tool:
    """One capability the MCP server offers through ``TOOL_CALL``."""

    name = ""
    """Identifier used in ``TOOL_CALL`` requests and in the function-calling schema."""

    description = ""
    """What the tool does and when to use it; GPT reads this when deciding whether to call it."""

    parameters = {"type": "object", "properties": {}, "required": []}
    """JSON schema of the keyword arguments ``run`` accepts."""

    ambient = False
    """``True`` for tools gathered once at the start of a conversation rather than on a player's request."""

    # ==============
    # Public methods
    # ==============

    def run(self, **kwargs: object) -> dict:
        """Execute the tool on the server.

        Args:
            **kwargs: Arguments matching ``parameters``.

        Returns:
            JSON-serialisable result, or a dictionary with an ``error`` key when the tool found nothing.
        """
        raise NotImplementedError("Subclasses must implement the run() method.")

    def render(self, result: dict) -> str:
        """Turn a result into the in-character text the model reads.

        Args:
            result: What ``run`` returned.

        Returns:
            One short paragraph.
        """
        raise NotImplementedError("Subclasses must implement the render() method.")

    def schema(self) -> dict:
        """Describe the tool in OpenAI's function-calling format.

        Returns:
            A ``{"type": "function", "function": {...}}`` dictionary.
        """
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }
