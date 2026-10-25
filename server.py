"""Compatibility entry point. All server code lives in app_v2.py."""
from app_v2 import app

if __name__ == "__main__":
    app.run(debug=False, port=5001)
