#!/usr/bin/env python3
"""CHIMERA Dashboard Local Web Server."""

import http.server
import mimetypes
import os
import socketserver
import sys
import traceback
from urllib.parse import unquote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Origins permitted for CORS — scoped to localhost dev only (M-097).
ALLOWED_ORIGINS = (
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:8008',
    'http://127.0.0.1:8008',
    'http://localhost:8088',
    'http://127.0.0.1:8088',
    'http://localhost:8880',
    'http://127.0.0.1:8880',
)

# Static-asset extensions eligible for caching (M-110).
STATIC_EXTS = ('.js', '.css', '.png', '.jpg', '.jpeg', '.svg', '.gif', '.woff', '.woff2')


class SafeFileHandler(http.server.BaseHTTPRequestHandler):
    # Suppress default version-disclosing Server header (M-109).
    server_version = 'CHIMERA'
    sys_version = ''

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _send_security_headers(self, ctype=None, content_length=None, path=''):
        """Send the response line plus all security + CORS headers."""
        self.send_response(200)
        if ctype:
            self.send_header('Content-Type', ctype)
        if content_length is not None:
            self.send_header('Content-Length', str(content_length))

        # M-099 — security headers on every response.
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header(
            'Content-Security-Policy',
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'",
        )

        # M-097 — scoped CORS (localhost only).
        origin = self.headers.get('Origin', '')
        if origin in ALLOWED_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
        else:
            self.send_header('Access-Control-Allow-Origin', 'null')

        # M-110 — cache static assets, disable cache for everything else.
        if path.endswith(STATIC_EXTS):
            self.send_header('Cache-Control', 'public, max-age=3600')
        else:
            self.send_header('Cache-Control', 'no-cache')

        # M-109 — Server header is set via class attributes (server_version/
        # sys_version) so every send_response emits "CHIMERA" with no version.

    def _send_error_response(self, code, message):
        """Send a simple error response with security headers."""
        body = f"{code} {message}".encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        # M-099 — security headers on error responses too.
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header(
            'Content-Security-Policy',
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'",
        )
        # M-097 — scoped CORS.
        origin = self.headers.get('Origin', '')
        if origin in ALLOWED_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
        else:
            self.send_header('Access-Control-Allow-Origin', 'null')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _resolve_path(self):
        """Resolve the request path to a filesystem path.

        Returns (full_path, clean_path) on success or (None, None) after
        sending an error response on failure.
        """
        raw_path = self.path.split('?', 1)[0].split('#', 1)[0]

        # Redirect /docs to /docs/
        if raw_path == '/docs':
            self.send_response(301)
            self.send_header('Location', '/docs/')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            return None, None

        # M-103 — decode URL encoding BEFORE any traversal check so that
        # %2e%2e / %2e%2f cannot bypass the filter.
        decoded = unquote(raw_path)

        if '..' in decoded:
            self._send_error_response(403, "Forbidden")
            return None, None

        # Strip the leading slash to obtain the relative file path, then
        # normalize and re-check.  URL paths naturally start with '/', so
        # the isabs check must run on the *relative* portion, not the raw
        # URL path.
        rel_file = decoded.lstrip('/') if decoded not in ('', '/') else ''
        norm = os.path.normpath(rel_file)
        if '..' in norm or os.path.isabs(norm):
            self._send_error_response(403, "Forbidden")
            return None, None

        if decoded in ('', '/', '/docs/'):
            rel_file = 'docs/index.html'
        else:
            rel_file = norm

        full_path = os.path.join(BASE_DIR, rel_file)

        # Containment check: ensure resolved path stays within BASE_DIR.
        real_base = os.path.realpath(BASE_DIR)
        real_full = os.path.realpath(full_path)
        if not real_full.startswith(real_base + os.sep) and real_full != real_base:
            self._send_error_response(403, "Forbidden")
            return None, None

        # If not found directly, check if it lives inside docs/.
        if not os.path.exists(full_path):
            alt_path = os.path.join(BASE_DIR, 'docs', rel_file)
            if os.path.exists(alt_path):
                full_path = alt_path

        if os.path.isdir(full_path):
            index_path = os.path.join(full_path, 'index.html')
            if os.path.exists(index_path):
                full_path = index_path

        if not os.path.exists(full_path) or os.path.isdir(full_path):
            self._send_error_response(404, "Not Found")
            return None, None

        return full_path, decoded

    def _guess_ctype(self, full_path):
        ctype, _ = mimetypes.guess_type(full_path)
        if not ctype:
            if full_path.endswith('.js'):
                ctype = 'application/javascript'
            elif full_path.endswith('.css'):
                ctype = 'text/css'
            elif full_path.endswith('.json'):
                ctype = 'application/json'
            elif full_path.endswith('.md'):
                ctype = 'text/markdown; charset=utf-8'
            elif full_path.endswith('.svg'):
                ctype = 'image/svg+xml'
            else:
                ctype = 'application/octet-stream'
        return ctype

    # ------------------------------------------------------------------
    # HTTP method handlers
    # ------------------------------------------------------------------
    def do_GET(self):
        full_path, clean_path = self._resolve_path()
        if not full_path:
            return

        ctype = self._guess_ctype(full_path)

        try:
            with open(full_path, 'rb') as f:
                content = f.read()
            self._send_security_headers(ctype=ctype, content_length=len(content), path=full_path)
            self.end_headers()
            self.wfile.write(content)
        except Exception:
            # M-098 — log full traceback server-side, send generic message.
            traceback.print_exc()
            self._send_error_response(500, "Internal Server Error")

    def do_HEAD(self):
        """M-108 — respond with headers only, no body."""
        full_path, clean_path = self._resolve_path()
        if not full_path:
            return

        ctype = self._guess_ctype(full_path)

        try:
            content_length = os.path.getsize(full_path)
            self._send_security_headers(ctype=ctype, content_length=content_length, path=full_path)
            self.end_headers()
        except Exception:
            traceback.print_exc()
            self._send_error_response(500, "Internal Server Error")


def run(preferred_port=8000):
    ports = [preferred_port, 8008, 8088, 3000, 8880]
    httpd = None
    port = None
    for p in ports:
        try:
            httpd = socketserver.TCPServer(('127.0.0.1', p), SafeFileHandler)
            port = p
            break
        except OSError:
            continue

    if not httpd:
        print("Error: Could not bind to any port.")
        sys.exit(1)

    print("=" * 60)
    print("  CHIMERA Case Viewer Local Server Running")
    print(f"  URL: http://localhost:{port}/docs/")
    print("=" * 60)
    sys.stdout.flush()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()

if __name__ == '__main__':
    p = 8000
    if len(sys.argv) > 1:
        try:
            p = int(sys.argv[1])
        except ValueError:
            pass
    run(p)
