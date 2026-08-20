#!/usr/bin/env python3
"""CHIMERA Dashboard Local Web Server."""

import http.server
import mimetypes
import os
import socketserver
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SafeFileHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        clean_path = self.path.split('?', 1)[0].split('#', 1)[0]
        
        # Redirect /dashboard to /dashboard/
        if clean_path == '/dashboard':
            self.send_response(301)
            self.send_header('Location', '/dashboard/')
            self.end_headers()
            return
            
        if clean_path in ('', '/', '/dashboard/'):
            rel_file = 'dashboard/index.html'
        else:
            rel_file = clean_path.lstrip('/')

        # Reject path traversal sequences early
        if '..' in rel_file:
            self.send_response(403)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write("403 Forbidden".encode('utf-8'))
            return

        full_path = os.path.join(BASE_DIR, rel_file)

        # Containment check: ensure resolved path stays within BASE_DIR
        real_base = os.path.realpath(BASE_DIR)
        real_full = os.path.realpath(full_path)
        if not real_full.startswith(real_base + os.sep) and real_full != real_base:
            self.send_response(403)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write("403 Forbidden".encode('utf-8'))
            return

        # If not found directly, check if it lives inside dashboard/
        if not os.path.exists(full_path):
            alt_path = os.path.join(BASE_DIR, 'dashboard', rel_file)
            if os.path.exists(alt_path):
                full_path = alt_path

        if os.path.isdir(full_path):
            index_path = os.path.join(full_path, 'index.html')
            if os.path.exists(index_path):
                full_path = index_path

        if not os.path.exists(full_path) or os.path.isdir(full_path):
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"404 Not Found: {clean_path}".encode('utf-8'))
            return

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

        try:
            with open(full_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"500 Internal Error: {e}".encode('utf-8'))

    def do_HEAD(self):
        self.do_GET()

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
    print(f"  URL: http://localhost:{port}/dashboard/")
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
