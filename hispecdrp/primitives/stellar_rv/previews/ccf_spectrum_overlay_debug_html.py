"""Unified spectrum + CCF overlay debug HTML (mask lines vs. shifted stellar template)."""
from __future__ import annotations

import json
import pathlib
from typing import Literal

__all__ = ["build_ccf_spectrum_overlay_debug_html"]

_PKG = pathlib.Path(__file__).resolve().parent
_OVERLAY_PAGE: str = (_PKG / "ccf_spectrum_overlay_unified.html.in").read_text(encoding="utf-8")


def build_ccf_spectrum_overlay_debug_html(
    *,
    title: str,
    payload_json: str,
    v_init_m_s: float,
    v_min_m_s: float,
    v_max_m_s: float,
    v_step_m_s: float,
    zb: float,
    c_ms: float,
    overlay_kind: Literal["mask", "template"],
    spectrum_help_html: str,
) -> str:
    """
    Single interactive page for ``ccf_line_mask`` (mask rectangles) and ``ccf_stellar_template``
    (Doppler-shifted template trace). ``spectrum_help_html`` is inserted inside the help
    paragraph (trusted HTML fragment from this package only).
    """
    if overlay_kind not in ("mask", "template"):
        raise ValueError(overlay_kind)
    return (
        _OVERLAY_PAGE.replace("__TITLE__", title)
        .replace("__PAYLOAD_JSON__", payload_json)
        .replace("__V_INIT__", f"{float(v_init_m_s):.6g}")
        .replace("__V_MIN__", f"{float(v_min_m_s):.1f}")
        .replace("__V_MAX__", f"{float(v_max_m_s):.1f}")
        .replace("__V_STEP__", f"{float(v_step_m_s):.1f}")
        .replace("__ZB__", f"{float(zb):.6g}")
        .replace("__C_MS__", f"{float(c_ms):.12g}")
        .replace("__OVERLAY_KIND__", json.dumps(overlay_kind))
        .replace("__SPECTRUM_HELP__", spectrum_help_html)
    )
