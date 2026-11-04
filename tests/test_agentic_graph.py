import json

import pytest

from src.common.budget import TOKEN_COUNTER, TOOL_RESULT_TOKENS
from src.client import agentic_graph, nodes
from tests.fakes import FakeMcp, fake_llm, tool_call
from src.common.personas import load_personas
from src.tools import LOOKUP_TOOLS

ADAM = load_personas()["adam"]
"""Persona every test in this module speaks as."""


def test_tool_call_triggers_lookup_and_second_pass():
    """When GPT calls the tool, the client runs it and calls GPT again with the tool exchange appended."""
    mcp = FakeMcp(summary="Kael arrived with a rusty sword.")
    llm = fake_llm(tool_calls=[tool_call("video game genres")])
    state = agentic_graph.build_graph(mcp, llm, ADAM).invoke({"user_input": "What are the genres in gaming?"})
    assert state["needs_tool"] and state["tool_args"] == {"query": "video game genres"} and mcp.calls == [("search_wikipedia", {"query": "video game genres"})]
    assert [call["tool_choice"] for call in llm.calls] == ["auto", "auto"]
    assert all(call["tools"] == [tool.schema() for tool in LOOKUP_TOOLS] for call in llm.calls)
    first, second = (call["messages"] for call in llm.calls)
    assert first[-1] == {"role": "user", "content": "What are the genres in gaming?"}
    assert "Lorebook entry" not in first[0]["content"] and "Kael arrived" in first[0]["content"]
    assert second[:-2] == first
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1"
    assert "Lorebook entry" in second[-1]["content"] and "Genres classify" in second[-1]["content"]
    assert state["reply"] == "Ah, traveller." and state["prompt_tokens"] > 0
    assert mcp.added == [("user", "What are the genres in gaming?"), ("assistant", "Ah, traveller.")]


def test_plain_answer_skips_tool():
    """When GPT answers in text, the turn ends after one call and the recent turns were in the prompt."""
    recent = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Welcome."}]
    mcp, llm = FakeMcp(recent=recent), fake_llm()
    state = agentic_graph.build_graph(mcp, llm, ADAM).invoke({"user_input": "Should I go east?"})
    assert not state["needs_tool"] and mcp.calls == [] and len(llm.calls) == 1
    assert state["prompt"][1:3] == recent and state["reply"] == "Ah, traveller."


@pytest.mark.parametrize(
    "calls",
    [
        [tool_call("   ")],
        [tool_call(raw='{"query": "unterminated')],
        [tool_call("x", name="other_tool")],
        [tool_call(raw="[1, 2]")],
    ],
    ids=["blank query", "broken json", "unknown tool", "not an object"],
)
def test_malformed_tool_calls_are_ignored(calls: list):
    """Anything other than a well-formed call to the Wikipedia tool means no lookup."""
    assert nodes.pick_tool_call([agentic_call(c) for c in calls]) == (None, "", {})


def agentic_call(call) -> dict:
    """Convert a fake SDK tool call into the dict shape the graph stores.

    Args:
        call: Object from :func:`tests.fakes.tool_call`.

    Returns:
        Dict with ``id``, ``type`` and ``function``.
    """
    return {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}


def test_schemas_fit_in_tool_slice():
    """On the first pass the schemas take the place of the absent tool result, so together they must fit that slice."""
    assert TOKEN_COUNTER.count(json.dumps([tool.schema() for tool in LOOKUP_TOOLS])) <= TOOL_RESULT_TOKENS
