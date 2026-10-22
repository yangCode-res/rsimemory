#!/usr/bin/env python3
"""Loopback experiment login helper; credentials never leave the server."""
import http.cookiejar
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import build_opener, HTTPCookieProcessor, Request

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:3000"
PROFILE = json.loads((ROOT / 'deploy/langfuse/observability.json').read_text())
ENTRY = PROFILE.get('review_entry', {}).get('default_path', '/overview')
if ENTRY not in ('/overview',):
    ENTRY = '/overview'
TARGET = "http://localhost:3001" + ENTRY
sys.path.insert(0, str(ROOT))


def login():
    env = dict(line.split("=", 1) for line in
               (ROOT / "observability/langfuse/.env").read_text().splitlines()
               if line and not line.startswith("#") and "=" in line)
    jar = http.cookiejar.CookieJar()
    client = build_opener(HTTPCookieProcessor(jar))
    with client.open(BASE + "/api/auth/csrf", timeout=20) as response:
        csrf = json.load(response)["csrfToken"]
    body = urlencode({"csrfToken": csrf,
        "email": env["LANGFUSE_INIT_USER_EMAIL"],
        "password": env["LANGFUSE_INIT_USER_PASSWORD"],
        "callbackUrl": TARGET, "json": "true"}).encode()
    request = Request(BASE + "/api/auth/callback/credentials", data=body,
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    with client.open(request, timeout=30) as response:
        result = json.load(response)
    cookies = [cookie for cookie in jar if "session-token" in cookie.name]
    if not cookies or "error=" in result.get("url", ""):
        raise RuntimeError("Default experiment account login failed")
    return cookies


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        if host not in ("localhost", "127.0.0.1"):
            self.send_error(403); return
        if self.path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.send_error(403); return
        if self.path in ("/overview", "/api/overview"):
            from research_dashboard import HTML, snapshot
            try:
                body = HTML.encode() if self.path == "/overview" else json.dumps(snapshot(ROOT), ensure_ascii=False, allow_nan=False).encode()
            except Exception:
                self.send_error(503, "Overview temporarily unavailable"); return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8" if self.path == "/overview" else "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if self.path != "/":
            self.send_error(404); return
        try:
            cookies = login()
        except Exception:
            self.send_error(502, "Experiment login failed; check server configuration")
            return
        self.send_response(303)
        for cookie in cookies:
            self.send_header("Set-Cookie", cookie.name + "=" + cookie.value +
                             "; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000")
        self.send_header("Location", TARGET)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 3001), Handler).serve_forever()
