from pathlib import Path

from pydantic import ValidationError
import pytest

from src.common.budget import PERSONA_TOKENS, TOKEN_COUNTER
from src.common.personas import Persona, load_personas


def test_default_file_loads_and_fits_budget():
    """The shipped personas validate, carry their file stem as id and stay under the persona slice."""
    personas = load_personas()
    assert list(personas) == ["adam", "mira"]
    assert all(isinstance(p, Persona) and p.id == key for key, p in personas.items())
    assert all(TOKEN_COUNTER.count(p.system_prompt) <= PERSONA_TOKENS for p in personas.values())


def test_missing_field_is_rejected(tmp_path: Path):
    """A file without a system prompt fails validation."""
    (tmp_path / "bob.json").write_text('{"name": "Bob", "description": "A guard."}')
    with pytest.raises(ValidationError):
        load_personas(tmp_path)


def test_oversized_prompt_is_rejected(tmp_path: Path):
    """A system prompt over the persona slice is refused at load time."""
    prompt = "lore " * (PERSONA_TOKENS + 5)
    (tmp_path / "bob.json").write_text(f'{{"name": "Bob", "description": "A guard.", "system_prompt": "{prompt}"}}')
    with pytest.raises(ValueError, match="limit"):
        load_personas(tmp_path)


def test_empty_directory_is_rejected(tmp_path: Path):
    """A directory without JSON files is an error rather than a silent empty registry."""
    with pytest.raises(ValueError, match="No persona files"):
        load_personas(tmp_path)
