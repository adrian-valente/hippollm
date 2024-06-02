import os
from pathlib import Path
import webbrowser

import click
from rich import print

from .app import create_app

__all__ = ["app"]

path = os.path.abspath(os.path.dirname(__file__))


@click.command("start", short_help="Start the app")
@click.argument("arg", type=str)
@click.option("--path", help="Location of storage file")
@click.option("--emb", help="Embedding model")
def start(arg, path, emb):

    if arg == "start":
        # lsof -i:9200
        # lsof -i:5000
        # kill -9 <PID>

        app = create_app(Path(path), emb)

        print("🎉 Starting the app.")
        webbrowser.open(os.path.join("file://" + os.path.dirname(__file__), "web/app.html"))
        app.run()
        
    elif arg == "open":
        print("😎 Opening web.")
        webbrowser.open(os.path.join("file://" + path, "web/app.html"))
        
if __name__ == "__main__":
    start()