"""Plotly stellar-RV previews and spectrum+CCF overlay debug HTML."""

from .ccf_spectrum_overlay_debug_html import build_ccf_spectrum_overlay_debug_html
from .stellar_rv_preview import (
    C_M_S,
    collect_mask_overlay_payload_for_primitive,
    collect_template_overlay_payload_for_primitive,
    create_stellar_rv_preview,
    maybe_write_ccf_overlay_debug_html,
    write_mask_overlay_debug_pages,
    write_template_overlay_debug_pages,
)

__all__ = [
    "C_M_S",
    "build_ccf_spectrum_overlay_debug_html",
    "collect_mask_overlay_payload_for_primitive",
    "collect_template_overlay_payload_for_primitive",
    "create_stellar_rv_preview",
    "maybe_write_ccf_overlay_debug_html",
    "write_mask_overlay_debug_pages",
    "write_template_overlay_debug_pages",
]
