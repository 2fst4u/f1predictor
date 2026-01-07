import webbrowser
import threading
import sys
import os
from app import app

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    # If there are arguments, they might be passed by the OS or launcher,
    # but we ignore them and just start the app.

    print("Starting F1 Predictor Web App...")

    # Schedule browser to open after a short delay to allow server to start
    threading.Timer(1.5, open_browser).start()

    try:
        app.run(port=5000, debug=False)
    except Exception as e:
        print(f"Error starting app: {e}")
        input("Press Enter to exit...")
