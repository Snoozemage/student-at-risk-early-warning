"""Regression tests for inline dashboard chart rendering."""

import base64
import io
from importlib.metadata import version

import matplotlib.pyplot as plt
from packaging.version import Version
from PIL import Image

from app.components.chart_rendering import render_matplotlib_figure, render_png_file


def test_starlette_version_avoids_file_response_disconnect_regression():
    # Starlette 1.8.0 introduced disconnect-aware FileResponse behavior that
    # reproduced an invalid ASGI response sequence in Streamlit's static route.
    assert Version(version("starlette")) < Version("1.8.0")


def test_png_file_is_embedded_without_a_transient_media_url(tmp_path, monkeypatch):
    image_path = tmp_path / "chart.png"
    image = Image.new("RGB", (4, 3), color="navy")
    image.save(image_path, format="PNG")
    captured = []
    monkeypatch.setattr("app.components.chart_rendering.st.html", captured.append)

    assert render_png_file(image_path, alt="Risk tiers", caption="Observed counts")

    assert len(captured) == 1
    html = captured[0]
    assert 'src="data:image/png;base64,' in html
    assert 'alt="Risk tiers"' in html
    assert "Observed counts" in html
    encoded = html.split("data:image/png;base64,", 1)[1].split('"', 1)[0]
    rendered_bytes = base64.b64decode(encoded)
    assert rendered_bytes == image_path.read_bytes()
    assert "/media/" not in html


def test_missing_png_file_is_reported_without_rendering(tmp_path, monkeypatch):
    rendered = []
    monkeypatch.setattr("app.components.chart_rendering.st.html", rendered.append)

    assert not render_png_file(tmp_path / "missing.png", alt="Unavailable chart")
    assert rendered == []


def test_matplotlib_chart_is_serialized_and_closed(monkeypatch):
    figure, axis = plt.subplots()
    axis.bar(["Tier 1", "Tier 2"], [2, 3])
    captured = []
    monkeypatch.setattr("app.components.chart_rendering.st.html", captured.append)

    render_matplotlib_figure(figure, alt="Priority distribution")

    assert len(captured) == 1
    html = captured[0]
    encoded = html.split("data:image/png;base64,", 1)[1].split('"', 1)[0]
    image_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(image_bytes)) as rendered:
        assert rendered.format == "PNG"
        assert rendered.width > 1 and rendered.height > 1
    assert not plt.fignum_exists(figure.number)
