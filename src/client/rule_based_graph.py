from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.client.core import DialogueState, McpClient
from src.client.nodes import RuleBasedNodes
from src.common.personas import Persona


def build_graph(mcp: McpClient, llm: OpenAI, persona: Persona, ambient: str = ""):
    """Wire the rule-based graph.

    Args:
        mcp: Client bound to the persona's session on the MCP server.
        llm: OpenAI client used by the LLM node.
        persona: Character the graph speaks as.
        ambient: Note from :func:`src.client.core.gather_ambient`, added to every system prompt.

    Returns:
        Compiled graph; invoke it with ``{"user_input": ...}``.
    """
    nodes = RuleBasedNodes(mcp, llm, persona, ambient)
    graph = StateGraph(DialogueState)
    graph.add_node("input", nodes.input)
    graph.add_node("check_context", nodes.check_context)
    graph.add_node("tool_decision", nodes.tool_decision)
    graph.add_node("tool_call", nodes.tool_call)
    graph.add_node("prompt_assembly", nodes.prompt_assembly)
    graph.add_node("llm", nodes.llm)
    graph.add_node("output", nodes.output)
    graph.add_edge(START, "input")
    graph.add_edge("input", "check_context")
    graph.add_edge("check_context", "tool_decision")
    graph.add_conditional_edges("tool_decision", nodes.route_after_decision, ["tool_call", "prompt_assembly"])
    graph.add_edge("tool_call", "prompt_assembly")
    graph.add_edge("prompt_assembly", "llm")
    graph.add_edge("llm", "output")
    graph.add_edge("output", END)
    return graph.compile()
