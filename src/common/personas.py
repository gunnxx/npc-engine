import json
from pathlib import Path

from pydantic import BaseModel

from src.common.budget import PERSONA_TOKENS, TOKEN_COUNTER

class Persona(BaseModel):
    """Everything the client needs to know about one non-player character."""

    id: str
    """Short identifier, also the MCP session id. Comes from the file name."""

    name: str
    """Display name used in the chat and in logs."""

    description: str
    """One-line description shown in help text."""

    system_prompt: str
    """Full instruction given to GPT as the system message; must fit :data:`src.common.budget.PERSONA_TOKENS`."""


def load_personas(directory: Path = Path(__file__).resolve().parents[2] / "personas") -> dict[str, Persona]:
    """Read and validate every ``*.json`` file in a directory.

    Args:
        directory: Folder holding one JSON object per character, the repository's ``personas/`` by default.

    Returns:
        Personas keyed by id, in file-name order.

    Raises:
        ValueError: If the directory holds no JSON files or a system prompt exceeds the persona token slice.
        pydantic.ValidationError: If a file lacks a required field.
    """
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No persona files (*.json) found in {directory}.")
    personas = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            persona = Persona(id=path.stem, **json.load(handle))
        tokens = TOKEN_COUNTER.count(persona.system_prompt)
        if tokens > PERSONA_TOKENS:
            raise ValueError(f"Persona {persona.id!r} has a {tokens}-token system prompt; the limit is {PERSONA_TOKENS}.")
        personas[persona.id] = persona
    return personas
