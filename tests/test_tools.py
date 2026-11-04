import pytest

from src.common.budget import AMBIENT_TOKENS, TOKEN_COUNTER
from src.client.core import gather_ambient
from src.tools import LOOKUP_TOOLS, TOOLS, Tool
from src.tools.clock import ClockTool, part_of_day
from src.tools.weather import WeatherTool
from src.tools.wikipedia import WikipediaTool
from tests.fakes import FakeMcp


def test_registry_splits_lookup_and_ambient_tools():
    """Wikipedia is the lookup tool; clock and weather are ambient; every tool has a complete schema."""
    assert [t.name for t in LOOKUP_TOOLS] == ["search_wikipedia"]
    assert {t.name for t in TOOLS.values() if t.ambient} == {"weather_now", "clock_now"}
    for tool in TOOLS.values():
        schema = tool.schema()["function"]
        assert schema["name"] == tool.name and schema["description"] and schema["parameters"]["type"] == "object"


def test_base_tool_is_abstract():
    """The base class refuses to run or render."""
    with pytest.raises(NotImplementedError):
        Tool().run()
    with pytest.raises(NotImplementedError):
        Tool().render({})


@pytest.mark.parametrize(("hour", "label"), [(0, "the small hours"), (7, "early morning"), (10, "morning"),
                                             (13, "midday"), (16, "afternoon"), (20, "evening"), (23, "night")])
def test_part_of_day(hour: int, label: str):
    """Hours map to the expected coarse labels."""
    assert part_of_day(hour) == label


def test_clock_runs_and_renders():
    """The clock reports the fixed location and renders a sentence a model can use."""
    result = ClockTool().run()
    assert result["location"] == "Singapore" and len(result["time"]) == 5
    sentence = ClockTool().render({"location": "Singapore", "weekday": "Tuesday", "time": "23:40",
                                   "part_of_day": "night", "is_weekend": False})
    assert sentence == "For the player it is Tuesday 23:40 in Singapore, night on a weekday."


def test_weather_renders_day_and_night():
    """Rendering mentions temperature, condition and whether it is light out."""
    base = {"location": "Singapore", "temperature_c": 27.2, "humidity_pct": 91, "condition": "an overcast sky"}
    assert WeatherTool().render({**base, "is_day": True}).endswith("in daylight.")
    assert "27°C with an overcast sky and 91% humidity, after dark." in WeatherTool().render({**base, "is_day": False})


def test_wikipedia_render_handles_missing_article():
    """An article becomes a Lorebook entry; an error becomes an admission."""
    assert WikipediaTool().render({"title": "T", "extract": "E", "url": "u"}).startswith('Lorebook entry on "T":')
    assert "no entry" in WikipediaTool().render({"error": "nothing"})


def test_gather_ambient_skips_failed_tools_and_fits_budget():
    """A tool that errors is left out; the rest form one note under the ambient slice."""
    mcp = FakeMcp()
    note = gather_ambient(mcp)
    assert note.startswith("The player's real-world setting") and "Tuesday 23:40" in note and "°C" not in note
    assert TOKEN_COUNTER.count(note) <= AMBIENT_TOKENS
    assert sorted(name for name, _ in mcp.calls) == ["clock_now", "weather_now"]
