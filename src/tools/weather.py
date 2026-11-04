import requests

from src.tools.base import HTTP_TIMEOUT_SECONDS, Tool
from src.tools.location import LOCATION_NAME, TIMEZONE


class WeatherTool(Tool):
    """Fetch the current weather at the fixed location."""

    name = "weather_now"
    """Tool identifier."""

    description = f"Current weather in {LOCATION_NAME}, where the player is."
    """Shown in the schema; the tool is ambient so GPT never calls it itself."""

    ambient = True
    """Gathered once when a conversation starts."""

    # ==============
    # Public methods
    # ==============

    def run(self) -> dict:
        """Fetch the current conditions.

        Returns:
            ``location``, ``temperature_c``, ``humidity_pct``, ``condition``, ``is_day`` and ``observed_at``.
        """
        phrases = {  # WMO weather interpretation codes, as Open-Meteo reports them
            0: "clear skies",
            1: "mostly clear skies",
            2: "partly cloudy skies",
            3: "an overcast sky",
            45: "fog",
            48: "freezing fog",
            51: "light drizzle",
            53: "drizzle",
            55: "heavy drizzle",
            61: "light rain",
            63: "rain",
            65: "heavy rain",
            80: "light showers",
            81: "showers",
            82: "violent showers",
            95: "a thunderstorm",
            96: "a thunderstorm with hail",
            99: "a thunderstorm with heavy hail",
        }
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 1.3521,
                "longitude": 103.8198,
                "current": "temperature_2m,relative_humidity_2m,weather_code,is_day",
                "timezone": TIMEZONE,
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        current = response.json()["current"]
        return {
            "location": LOCATION_NAME,
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "condition": phrases.get(current["weather_code"], "unsettled weather"),
            "is_day": bool(current["is_day"]),
            "observed_at": current["time"],
        }

    def render(self, result: dict) -> str:
        """Describe the weather in one sentence.

        Args:
            result: What :meth:`run` returned.

        Returns:
            For example ``"Outside, in Singapore, it is 27°C under an overcast sky, after dark."``.
        """
        light = "in daylight" if result["is_day"] else "after dark"
        return (
            f"Outside, in {result['location']}, it is {result['temperature_c']:.0f}°C with {result['condition']} "
            f"and {result['humidity_pct']:.0f}% humidity, {light}."
        )
