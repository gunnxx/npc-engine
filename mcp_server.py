import logging

import uvicorn

from src.server.app import app


def main():
    """Configure logging and serve the app on ``127.0.0.1:8000`` until interrupted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
