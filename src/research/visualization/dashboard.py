"""
CropXAI Interactive Dataset Analytics Dashboard
================================================

A Plotly Dash application for interactive dataset exploration.

Tabs:
  1. Class Distribution
  2. Temporal Explorer
  3. Spectral Viewer
  4. Metadata Browser
  5. Sample Grid
  6. Split Analysis

Launch with: ``cropxai explore``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_dashboard(cfg: Any, dataset_stats: Any = None, dataset_metadata: Any = None) -> Any:
    """
    Build and return the Dash application object.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root CropXAI config.
    dataset_stats : DatasetStatistics, optional
        Pre-computed dataset statistics.
    dataset_metadata : DatasetMetadata, optional
        Pre-read dataset metadata.

    Returns
    -------
    dash.Dash
        The configured Dash application.
    """
    try:
        import dash
        from dash import dcc, html, Input, Output, dash_table
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError(
            "Dash and Plotly are required for the dataset dashboard. "
            "Install with: pip install dash plotly"
        )

    app = dash.Dash(
        __name__,
        title="CropXAI — Dataset Explorer",
        suppress_callback_exceptions=True,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={"fontFamily": "Inter, sans-serif", "backgroundColor": "#0f172a", "color": "#e2e8f0", "minHeight": "100vh"},
        children=[
            # Header
            html.Div(
                style={"background": "linear-gradient(90deg, #1e3a5f 0%, #0f172a 100%)", "padding": "1.5rem 2rem"},
                children=[
                    html.H1("CropXAI — Dataset Explorer", style={"color": "#38bdf8", "margin": 0, "fontSize": "1.8rem"}),
                    html.P(
                        f"Dataset: {cfg.dataset.display_name}  |  "
                        f"Classes: {cfg.dataset.labels.num_classes}  |  "
                        f"Bands: {cfg.dataset.bands.num_bands}",
                        style={"color": "#94a3b8", "margin": "0.25rem 0 0"},
                    ),
                ],
            ),

            # Tabs
            dcc.Tabs(
                id="main-tabs",
                value="class-dist",
                style={"backgroundColor": "#1e293b"},
                colors={"border": "#334155", "primary": "#38bdf8", "background": "#1e293b"},
                children=[
                    dcc.Tab(label="📊 Class Distribution", value="class-dist"),
                    dcc.Tab(label="🕐 Temporal Explorer", value="temporal"),
                    dcc.Tab(label="🌈 Spectral Viewer", value="spectral"),
                    dcc.Tab(label="📋 Metadata Browser", value="metadata"),
                    dcc.Tab(label="🖼 Sample Grid", value="samples"),
                    dcc.Tab(label="✂ Split Analysis", value="splits"),
                ],
            ),

            # Content
            html.Div(id="tab-content", style={"padding": "1.5rem"}),
        ],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
    def render_tab(tab: str) -> Any:
        if tab == "class-dist":
            return _render_class_distribution(dataset_stats, cfg)
        elif tab == "temporal":
            return _render_temporal(dataset_stats, cfg)
        elif tab == "spectral":
            return _render_spectral(dataset_stats, cfg)
        elif tab == "metadata":
            return _render_metadata(dataset_metadata, cfg)
        elif tab == "samples":
            return _render_samples_placeholder()
        elif tab == "splits":
            return _render_splits(cfg)
        return html.P("Select a tab above.", style={"color": "#94a3b8"})

    return app


def _render_class_distribution(stats: Any, cfg: Any) -> Any:
    """Render the class distribution tab."""
    from dash import html, dcc
    import plotly.graph_objects as go

    if stats is None or not stats.class_stats:
        return html.Div([
            html.H3("Class Distribution", style={"color": "#38bdf8"}),
            html.P("Run `cropxai analyze` to compute class statistics.", style={"color": "#94a3b8"}),
        ])

    names = [c.class_name for c in stats.class_stats]
    fracs = [c.pixel_fraction * 100 for c in stats.class_stats]

    fig = go.Figure(go.Bar(
        x=fracs, y=names, orientation="h",
        marker_color="#38bdf8",
        text=[f"{f:.1f}%" for f in fracs],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font_color="#e2e8f0",
        title="Pixel-Level Class Distribution",
        xaxis_title="Percentage of Pixels (%)",
        yaxis={"autorange": "reversed"},
        height=max(300, len(names) * 40),
    )

    return html.Div([
        html.H3("Class Distribution", style={"color": "#38bdf8"}),
        html.P(f"Imbalance Ratio: {stats.imbalance_ratio:.2f}x", style={"color": "#f59e0b"}),
        dcc.Graph(figure=fig),
    ])


def _render_spectral(stats: Any, cfg: Any) -> Any:
    """Render spectral statistics."""
    from dash import html, dcc
    import plotly.graph_objects as go

    if stats is None or not stats.band_stats:
        return html.Div([
            html.H3("Spectral Statistics", style={"color": "#38bdf8"}),
            html.P("No band statistics available. Run `cropxai analyze`.", style={"color": "#94a3b8"}),
        ])

    bands = [b.band_name for b in stats.band_stats]
    means = [b.mean for b in stats.band_stats]
    stds = [b.std for b in stats.band_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean", x=bands, y=means, marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="Std", x=bands, y=stds, marker_color="#f59e0b"))
    fig.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", font_color="#e2e8f0",
        title="Per-Band Statistics", barmode="group",
    )
    return html.Div([
        html.H3("Spectral Viewer", style={"color": "#38bdf8"}),
        dcc.Graph(figure=fig),
    ])


def _render_temporal(stats: Any, cfg: Any) -> Any:
    """Render temporal information."""
    from dash import html, dcc
    import plotly.graph_objects as go

    dates = stats.temporal_dates if stats else []
    gaps = stats.temporal_gaps_days if stats else []

    children = [html.H3("Temporal Explorer", style={"color": "#38bdf8"})]

    if dates:
        children.append(html.P(f"Acquisition dates: {len(dates)}", style={"color": "#94a3b8"}))
        children.append(html.P(f"Range: {dates[0]} → {dates[-1]}", style={"color": "#94a3b8"}))
        if gaps:
            fig = go.Figure(go.Bar(x=list(range(len(gaps))), y=gaps, marker_color="#10b981"))
            fig.update_layout(
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", font_color="#e2e8f0",
                title="Temporal Gaps (days)", xaxis_title="Gap Index", yaxis_title="Days",
            )
            children.append(dcc.Graph(figure=fig))
    else:
        children.append(html.P("No temporal data available.", style={"color": "#94a3b8"}))

    return html.Div(children)


def _render_metadata(metadata: Any, cfg: Any) -> Any:
    """Render dataset metadata browser."""
    from dash import html

    rows = [
        ("Dataset Name", cfg.dataset.display_name),
        ("Version", cfg.dataset.version),
        ("Total Files", str(getattr(metadata, "total_files", "N/A"))),
        ("Total Size", f"{getattr(metadata, 'total_size_gb', 'N/A')} GB"),
        ("CRS", ", ".join(str(c) for c in getattr(metadata, "unique_crs", [])) or "N/A"),
        ("Resolution", ", ".join(f"{r}m" for r in getattr(metadata, "unique_resolutions", [])) or "N/A"),
        ("Num Bands", str(cfg.dataset.bands.num_bands)),
        ("Patch Size", f"{cfg.dataset.spatial.patch_size}px"),
        ("Temporal", str(cfg.dataset.temporal.enabled)),
    ]

    table_rows = [
        html.Tr([html.Td(k, style={"fontWeight": "bold", "padding": "0.5rem 1rem", "color": "#38bdf8"}),
                 html.Td(v, style={"padding": "0.5rem 1rem"})])
        for k, v in rows
    ]

    return html.Div([
        html.H3("Metadata Browser", style={"color": "#38bdf8"}),
        html.Table(table_rows, style={"borderCollapse": "collapse", "width": "100%", "backgroundColor": "#1e293b"}),
    ])


def _render_samples_placeholder() -> Any:
    from dash import html
    return html.Div([
        html.H3("Sample Grid", style={"color": "#38bdf8"}),
        html.P("Sample visualization requires dataset files to be loaded.", style={"color": "#94a3b8"}),
    ])


def _render_splits(cfg: Any) -> Any:
    from dash import html
    import plotly.graph_objects as go
    from dash import dcc

    splits = {"train": 0.70, "val": 0.15, "test": 0.15}
    fig = go.Figure(go.Pie(
        labels=list(splits.keys()),
        values=list(splits.values()),
        marker_colors=["#3b82f6", "#f59e0b", "#10b981"],
        hole=0.4,
    ))
    fig.update_layout(paper_bgcolor="#1e293b", font_color="#e2e8f0", title="Dataset Splits")
    return html.Div([
        html.H3("Split Analysis", style={"color": "#38bdf8"}),
        dcc.Graph(figure=fig),
    ])


def run_dashboard(cfg: Any, dataset_stats: Any = None, dataset_metadata: Any = None) -> None:
    """Launch the dashboard application."""
    dash_cfg = cfg.visualization.dashboard
    app = build_dashboard(cfg, dataset_stats, dataset_metadata)
    logger.info(
        "Launching CropXAI Dashboard at http://%s:%d",
        dash_cfg.host,
        dash_cfg.port,
    )
    app.run(
        host=dash_cfg.host,
        port=dash_cfg.port,
        debug=dash_cfg.get("debug", False),
    )


__all__ = ["build_dashboard", "run_dashboard"]
