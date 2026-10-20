from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "rankedsurfers-downloaded"

class UnityHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def guess_type(self, path):
        if path.endswith(".wasm.unityweb"):
            return "application/wasm"

        if path.endswith(".framework.js.unityweb"):
            return "application/javascript"

        if path.endswith(".data.unityweb"):
            return "application/octet-stream"

        return super().guess_type(path)

    def end_headers(self):
        path = self.path.split("?")[0]

        if path.endswith(".unityweb"):
            self.send_header("Content-Encoding", "gzip")

        self.send_header(
            "Cross-Origin-Opener-Policy", "same-origin"
        )
        self.send_header(
            "Cross-Origin-Embedder-Policy", "require-corp"
        )

        super().end_headers()


server = ThreadingHTTPServer(("127.0.0.1", 8000), UnityHandler)

print("Unity server running at http://localhost:8000")

server.serve_forever()
