from src.tools.base import Tool
from src.tools.clock import ClockTool
from src.tools.weather import WeatherTool
from src.tools.wikipedia import WikipediaTool

TOOLS: dict[str, Tool] = {tool.name: tool for tool in (WikipediaTool(), ClockTool(), WeatherTool())}
"""Every tool the server exposes, keyed by name."""

LOOKUP_TOOLS = [tool for tool in TOOLS.values() if not tool.ambient]
"""Tools a player's question may trigger; the agentic graph advertises exactly these to GPT."""

__all__ = ["LOOKUP_TOOLS", "TOOLS", "Tool"]
