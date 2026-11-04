import argparse
import logging
import os

from openai import OpenAI
import uvicorn

from src.client import DEFAULT_MODE, GRAPH_BUILDERS, Chat
from src.common.personas import Persona, load_personas
from src.ui.app import create_app

logger = logging.getLogger(__name__)


def run_browser(personas: dict[str, Persona], persona: str, mode: str, server: str):
    """Serve the browser chat with one character on ``127.0.0.1:8001`` until interrupted.

    Args:
        personas: Loaded characters.
        persona: Id of the character to talk to.
        mode: Routing mode, a key of :data:`src.client.GRAPH_BUILDERS`.
        server: Base URL of the MCP server.
    """
    chat = Chat(personas[persona], mode, OpenAI(), server)
    print(f"Chat UI with {chat.persona.name} at http://127.0.0.1:8001")
    uvicorn.run(create_app(chat), host="127.0.0.1", port=8001, log_level="warning")


def run_terminal(personas: dict[str, Persona], persona: str, mode: str, server: str):
    """Run an interactive chat loop with one character.

    Args:
        personas: Loaded characters.
        persona: Id of the character to talk to.
        mode: Routing mode, a key of :data:`src.client.GRAPH_BUILDERS`.
        server: Base URL of the MCP server.
    """
    chat = Chat(personas[persona], mode, OpenAI(), server)
    print(f"Talking to {chat.persona.name}, {chat.persona.description[0].lower() + chat.persona.description[1:]}")
    print(f"Routing mode: {chat.mode}. Type /reset to clear memory, /quit to leave.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            chat.reset()
            print(f"{chat.persona.name}'s memory has been cleared.")
            continue
        print(f"{chat.persona.name}: {chat.send(user_input)['reply']}\n")


def main():
    """Parse the command line and start the chosen front end."""
    parser = argparse.ArgumentParser(description="Chat with an NPC through the MCP server.")
    parser.add_argument("--terminal", action="store_true", help="chat in the terminal instead of the browser")
    parser.add_argument("--persona", default="adam", help="character to talk to, a file stem in personas/")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=sorted(GRAPH_BUILDERS), help="who decides on tool use")
    parser.add_argument("--server", default=os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000"), help="base URL of the MCP server")
    parser.add_argument("--verbose", action="store_true", help="show routing and context logs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    personas = load_personas()
    if args.persona not in personas:
        parser.error(f"unknown persona {args.persona!r}; choose from {', '.join(personas)}")
    if args.terminal:
        run_terminal(personas, args.persona, args.mode, args.server)
    else:
        run_browser(personas, args.persona, args.mode, args.server)


if __name__ == "__main__":
    main()
