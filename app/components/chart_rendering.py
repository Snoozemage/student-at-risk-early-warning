"""Render dashboard charts without Streamlit's temporary media-file endpoint."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st


def build_embedded_png_html(
    image_bytes: bytes,
    *,
    alt: str,
    caption: str | None = None,
) -> str:
    """Build accessible HTML for an inline PNG chart."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    caption_html = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return (
        '<figure class="dashboard-chart" style="margin:0">'
        f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}" '
        'style="display:block;width:100%;height:auto">'
        f"{caption_html}</figure>"
    )


def render_png_bytes(
    image_bytes: bytes,
    *,
    alt: str,
    caption: str | None = None,
) -> None:
    """Render chart bytes inline so the browser does not fetch a transient media URL."""
    st.html(build_embedded_png_html(image_bytes, alt=alt, caption=caption))


def render_png_file(
    path: Path,
    *,
    alt: str,
    caption: str | None = None,
) -> bool:
    """Render an existing PNG file; return False when it cannot be read."""
    try:
        image_bytes = path.read_bytes()
    except OSError:
        return False
    render_png_bytes(image_bytes, alt=alt, caption=caption)
    return True


def render_matplotlib_figure(
    figure: plt.Figure,
    *,
    alt: str,
    caption: str | None = None,
) -> None:
    """Serialize a Matplotlib figure to an inline PNG and close the figure."""
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", bbox_inches="tight")
        render_png_bytes(buffer.getvalue(), alt=alt, caption=caption)
    finally:
        plt.close(figure)
