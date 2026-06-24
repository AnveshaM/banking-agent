"""
voice/token_server.py — Minimal token server for the browser frontend.

The browser needs a LiveKit participant token to connect to the room.
This tiny HTTP server generates one on demand.

Run with:
  python voice/token_server.py

Then open voice/frontend.html in your browser.
The "click to call" button will fetch a token from http://localhost:8080/token.
"""
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from livekit import api as lk_api


LIVEKIT_URL    = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
TOKEN_SERVER_PORT  = int(os.getenv("TOKEN_SERVER_PORT", "8080"))


class TokenHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/token":
            self.send_response(404)
            self.end_headers()
            return

        # Generate a unique participant identity for each browser tab
        identity = f"caller-{uuid.uuid4().hex[:6]}"

        token = (
            lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name("Web Caller")
            .with_grants(
                lk_api.VideoGrants(
                    room_join=True,
                    room="bank-support-demo",
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )

        payload = json.dumps({
            "token": token,
            "url": LIVEKIT_URL,
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Quieten the server logs — only print errors
        if args[1] not in ("200", "304"):
            super().log_message(format, *args)


if __name__ == "__main__":
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        print("ERROR: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET must be set in .env")
        sys.exit(1)

    print(f"Token server running at http://localhost:{TOKEN_SERVER_PORT}")
    print(f"Open voice/frontend.html in your browser to start a call")
    print(f"LiveKit URL: {LIVEKIT_URL}\n")

    server = HTTPServer(("0.0.0.0", TOKEN_SERVER_PORT), TokenHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nToken server stopped")
