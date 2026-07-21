"""Helpers for displaying large local HTML products inside Jupyter / VS Code notebooks."""

from __future__ import annotations

import http.server
import os
import socket
import threading
from functools import partial
from typing import Any
from urllib.parse import quote

__all__ = ["display_html_file"]

# Reuse one background server per directory so repeated cells do not bind new ports.
_SERVERS: dict[str, tuple[threading.Thread, int]] = {}
_LOCK = threading.Lock()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_directory_server(directory: str) -> int:
    """Start (or reuse) a localhost static file server rooted at ``directory``."""
    root = os.path.abspath(directory)
    with _LOCK:
        cached = _SERVERS.get(root)
        if cached is not None:
            return cached[1]

        port = _pick_free_port()
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=root)

        class QuietHandler(handler):  # type: ignore[valid-type,misc]
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), QuietHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _SERVERS[root] = (thread, port)
        return port


def display_html_file(
    path: str,
    *,
    height: int = 900,
    width: str | int = "100%",
) -> Any:
    """
    Embed a local HTML file in a notebook output cell via a localhost iframe.

    ``IPython.display.HTML(open(path).read())`` does **not** work well for HISPEC
    stellar-RV previews: they are full HTML documents (often hundreds of MB with
    Plotly), and notebook frontends sanitize / drop nested ``<html>`` / ``<script>``
    content. Serving the file over ``http://127.0.0.1`` and embedding an iframe
    keeps scripts executable without stuffing the file into the notebook JSON.

    Parameters
    ----------
    path : str
        Absolute or relative path to a ``.html`` file on disk.
    height : int, optional
        IFrame height in pixels. Default 900.
    width : str or int, optional
        IFrame width (CSS). Default ``'100%'``.

    Returns
    -------
    IPython.display.IFrame
        The display object (also sent to the notebook frontend via ``display``).
    """
    from IPython.display import IFrame, display

    html_path = os.path.abspath(path)
    if not os.path.isfile(html_path):
        raise FileNotFoundError(f"HTML preview not found: {html_path}")

    directory, filename = os.path.split(html_path)
    port = _ensure_directory_server(directory)
    url = f"http://127.0.0.1:{port}/{quote(filename)}"
    frame = IFrame(src=url, width=width, height=height)
    display(frame)
    return frame
