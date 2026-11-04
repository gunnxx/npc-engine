import pytest

from src.common.budget import RESPONSE_RESERVE_TOKENS
from src.client import nodes, rule_based_graph
from src.common.personas import load_personas
from tests.fakes import FakeMcp, fake_llm

PERSONAS = load_personas()
"""Shipped personas, keyed by id."""


@pytest.mark.parametrize(
    ("text", "factual", "query"),
    [
        ("What are the different genres in gaming?", True, "different genres in gaming"),
        ("Who was Leif Erikson?", True, "Leif Erikson"),
        ("Tell me about the northern lights", True, "northern lights"),
        ("How many moons does Jupiter have?", True, "moons does Jupiter have"),
        ("How are you today, Adam?", False, ""),
        ("What should I do about the wolves?", False, ""),
        ("What do you think of my plan?", False, ""),
        ("I lost my sword in the river.", False, ""),
    ],
)
def test_factual_routing(text: str, factual: bool, query: str):
    """Factual questions are detected and reduced to a search query; personal ones are not."""
    assert nodes.is_factual_question(text) is factual
    if factual:
        assert nodes.extract_query(text) == query


def test_graph_routes_factual_question_through_tool():
    """A factual question runs the tool node and its result lands in the system prompt."""
    mcp, llm = FakeMcp(summary="Kael arrived with a rusty sword."), fake_llm()
    graph = rule_based_graph.build_graph(mcp, llm, PERSONAS["adam"])
    state = graph.invoke({"user_input": "What are the genres in gaming?"})
    assert state["needs_tool"] and mcp.calls == [("search_wikipedia", {"query": "genres in gaming"})]
    system = state["prompt"][0]["content"]
    assert "Lorebook entry" in system and "Kael arrived" in system and PERSONAS["adam"].system_prompt in system
    assert state["prompt"][-1] == {"role": "user", "content": "What are the genres in gaming?"}
    assert mcp.added == [("user", "What are the genres in gaming?"), ("assistant", "Ah, traveller.")]
    assert len(llm.calls) == 1 and "tools" not in llm.calls[0]
    assert llm.calls[0]["max_tokens"] == RESPONSE_RESERVE_TOKENS


def test_graph_skips_tool_for_roleplay():
    """A non-factual message goes straight to prompt assembly with the recent turns included."""
    recent = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Welcome."}]
    mcp = FakeMcp(recent=recent)
    state = rule_based_graph.build_graph(mcp, fake_llm(), PERSONAS["adam"]).invoke({"user_input": "Should I go east?"})
    assert not state["needs_tool"] and mcp.calls == []
    assert state["prompt"][1:3] == recent
    assert "Lorebook entry" not in state["prompt"][0]["content"]
    assert state["prompt_tokens"] <= 4096 - RESPONSE_RESERVE_TOKENS


def test_second_npc_uses_its_own_persona():
    """The persona registry drives the system prompt, so another NPC needs no new code."""
    state = rule_based_graph.build_graph(FakeMcp(), fake_llm(), PERSONAS["mira"]).invoke({"user_input": "Hello there"})
    assert state["prompt"][0]["content"].startswith("You are Mira")


def test_ambient_note_rides_in_every_system_prompt():
    """The ambient note gathered at start is part of the system message, after the summary."""
    mcp = FakeMcp(summary="Kael arrived.")
    ambient = "The player's setting. For the player it is Tuesday 23:40 in Singapore, night on a weekday."
    graph = rule_based_graph.build_graph(mcp, fake_llm(), PERSONAS["adam"], ambient)
    system = graph.invoke({"user_input": "Hello"})["prompt"][0]["content"]
    assert system.index("Kael arrived.") < system.index(ambient)
