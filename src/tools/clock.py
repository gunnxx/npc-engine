from datetime import datetime
from zoneinfo import ZoneInfo

from src.tools.base import Tool
from src.tools.location import LOCATION_NAME, TIMEZONE


def part_of_day(hour: int) -> str:
    """Name the part of the day an hour falls in.

    Args:
        hour: Hour of the day, 0 to 23.

    Returns:
        ``the small hours`` before 5, ``early morning`` before 8, ``morning`` before 12, ``midday`` before 14,
        ``afternoon`` before 18, ``evening`` before 22, otherwise ``night``.
    """
    bounds = ((5, "the small hours"), (8, "early morning"), (12, "morning"), (14, "midday"), (18, "afternoon"), (22, "evening"))
    for bound, label in bounds:
        if hour < bound:
            return label
    return "night"


class ClockTool(Tool):
    """Report the current local time at the fixed location."""

    name = "clock_now"
    """Tool identifier."""

    description = f"Current local date and time in {LOCATION_NAME}, where the player is."
    """Shown in the schema; the tool is ambient so GPT never calls it itself."""

    ambient = True
    """Gathered once when a conversation starts."""

    # ==============
    # Public methods
    # ==============

    def run(self) -> dict:
        """Read the clock.

        Returns:
            ``location``, ``iso``, ``weekday``, ``time`` (24-hour ``HH:MM``), ``part_of_day`` and ``is_weekend``.
        """
        now = datetime.now(ZoneInfo(TIMEZONE))
        return {
            "location": LOCATION_NAME,
            "iso": now.isoformat(timespec="minutes"),
            "weekday": now.strftime("%A"),
            "time": now.strftime("%H:%M"),
            "part_of_day": part_of_day(now.hour),
            "is_weekend": now.weekday() >= 5,
        }

    def render(self, result: dict) -> str:
        """Describe the time in one sentence.

        Args:
            result: What :meth:`run` returned.

        Returns:
            For example ``"For the player it is Tuesday 23:40 in Singapore, night on a weekday."``.
        """
        day_kind = "a weekend" if result["is_weekend"] else "a weekday"
        return (
            f"For the player it is {result['weekday']} {result['time']} in {result['location']}, "
            f"{result['part_of_day']} on {day_kind}."
        )
