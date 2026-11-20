from datetime import datetime

import requests

WEATHER_API = "https://api.open-meteo.com/v1/forecast"
GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"
IP_ADDRESS_API = "https://ipwho.is/"


def get_current_date():
    """returns the current date and time in the format"""
    return datetime.now().strftime("%m-%d-%Y %H:%M:%S")


def get_weather(city: str) -> str:
    """
    Returns the current weather for the given city.
    """

    try:
        geo = requests.get(f"{GEOCODE_API}?name={city}&count=1", timeout=10).json()

        if not geo.get("results"):
            return f"Could not find {city}"

        location = geo["results"][0]

        weather = requests.get(
            f"{WEATHER_API}"
            f"?latitude={location['latitude']}"
            f"&longitude={location['longitude']}"
            "&current=temperature_2m",
            timeout=10,
        ).json()

        current = weather["current"]

        return f"{city}: {current['temperature_2m']}°C"

    except requests.RequestException as e:
        return f"Error while fetching weather data: {e}"


def get_my_location() -> str:
    """
    call this tool whenever the user asks where they live,
    where they are currently located, or what city they are in.
    Returns only the user's city name.
    """
    try:
        data = requests.get(IP_ADDRESS_API, timeout=10).json()
        return data.get("city", "Unknown city")
    except requests.RequestException as e:
        return f"Error while fetching location data: {e}"


TOOLS = [
    {
        "type:": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time in the format",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get the weather for",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_location",
            "description": "call this tool whenever the user asks where they live, where they are currently located, or what city they are in. Returns only the user's city name.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

FUNCTION_NAME_TO_FUNCTION = {
    "get_current_date": get_current_date,
    "get_weather": get_weather,
    "get_my_location": get_my_location,
}
