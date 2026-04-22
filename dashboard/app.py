import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import threading
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from lob.simulation import Simulation, SimulationConfig, StreamingBuffer
from analytics.metrics import run_full_analytics

GOLD         = "#F0C800"
GOLD_DIM     = "#A08800"
GOLD_FAINT   = "#3A2F00"
RED          = "#C0392B"
RED_DIM      = "#7B241C"
GREEN        = "#27AE60"
WHITE        = "#E8E8E8"
GREY         = "#888888"
GREY_DIM     = "#444444"
BG           = "#080808"
BG_CARD      = "#0F0F0F"
BG_PANEL     = "#141414"
BG_HEADER    = "#0A0A0A"
FONT         = "JetBrains Mono, Fira Code, Consolas, monospace"
GOLD_T10     = "rgba(240,200,0,0.04)"
GOLD_T20     = "rgba(240,200,0,0.08)"
RED_T10      = "rgba(192,57,43,0.08)"
RED_T20      = "rgba(192,57,43,0.13)"
GREEN_T20    = "rgba(39,174,96,0.13)"
GREY_GRID    = "rgba(68,68,68,0.4)"
GOLD_GRID    = "rgba(160,136,0,0.2)"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

body, html {{
    margin: 0; padding: 0;
    background: {BG} !important;
    color: {GOLD} !important;
    font-family: {FONT} !important;
    font-size: 11px;
    overflow-x: hidden;
}}

body::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
}}

.container-fluid {{ padding: 0 12px !important; }}

.panel {{
    background: {BG_CARD};
    border: 1px solid {GOLD_DIM}55;
    margin-bottom: 8px;
    position: relative;
}}

.panel::before {{
    content: '';
    position: absolute;
    top: -1px; left: -1px;
    width: 12px; height: 12px;
    border-top: 2px solid {GOLD};
    border-left: 2px solid {GOLD};
}}

.panel::after {{
    content: '';
    position: absolute;
    bottom: -1px; right: -1px;
    width: 12px; height: 12px;
    border-bottom: 2px solid {GOLD};
    border-right: 2px solid {GOLD};
}}

.panel-header {{
    background: {BG_HEADER};
    border-bottom: 1px solid {GOLD_DIM}44;
    padding: 5px 10px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: {GOLD};
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.panel-header::before {{
    content: '▮';
    color: {GOLD};
    font-size: 7px;
}}

.panel-body {{ padding: 6px 8px; }}

.metric-block {{
    background: {BG_PANEL};
    border: 1px solid {GOLD_DIM}33;
    border-left: 2px solid {GOLD};
    padding: 8px 10px;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}}

.metric-block::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, {GOLD}44, transparent);
}}

.metric-label {{
    font-size: 8px;
    color: {GREY};
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}}

.metric-value {{
    font-size: 18px;
    font-weight: 700;
    color: {GOLD};
    text-shadow: 0 0 12px {GOLD}66;
    letter-spacing: 1px;
    line-height: 1;
}}

.metric-sub {{
    font-size: 8px;
    color: {GREY_DIM};
    margin-top: 3px;
    letter-spacing: 1px;
}}

.ticker-wrap {{
    background: {BG_HEADER};
    border-top: 1px solid {GOLD_DIM}44;
    border-bottom: 1px solid {GOLD_DIM}44;
    padding: 5px 0;
    overflow: hidden;
    white-space: nowrap;
    margin-bottom: 8px;
}}

@keyframes ticker-scroll {{
    0%   {{ transform: translateX(100vw); }}
    100% {{ transform: translateX(-100%); }}
}}

.ticker-inner {{
    display: inline-block;
    animation: ticker-scroll 30s linear infinite;
    font-size: 10px;
    letter-spacing: 1.5px;
}}

@keyframes pulse-dot {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.2; }}
}}

.live-dot {{
    display: inline-block;
    width: 6px; height: 6px;
    background: {GOLD};
    border-radius: 50%;
    animation: pulse-dot 1.2s ease-in-out infinite;
    box-shadow: 0 0 6px {GOLD};
    margin-right: 6px;
}}

.live-dot.red {{
    background: {RED};
    box-shadow: 0 0 6px {RED};
}}

.live-dot.green {{
    background: {GREEN};
    box-shadow: 0 0 6px {GREEN};
}}

@keyframes progress-glow {{
    0%, 100% {{ box-shadow: 0 0 6px {GOLD}88; }}
    50%       {{ box-shadow: 0 0 14px {GOLD}cc; }}
}}

.progress-outer {{
    background: {BG_PANEL};
    border: 1px solid {GOLD_DIM}33;
    height: 4px;
    margin-bottom: 8px;
    position: relative;
}}

.progress-inner {{
    height: 100%;
    background: linear-gradient(90deg, {GOLD_DIM}, {GOLD});
    animation: progress-glow 1.5s ease-in-out infinite;
    transition: width 0.4s ease;
}}

.tape-row {{
    font-size: 10px;
    padding: 3px 6px;
    border-left: 2px solid;
    margin-bottom: 2px;
    letter-spacing: 0.5px;
    font-family: {FONT};
    display: flex;
    justify-content: space-between;
}}

.run-btn {{
    background: transparent !important;
    border: 1px solid {GOLD} !important;
    color: {GOLD} !important;
    font-family: {FONT} !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
    position: relative;
    overflow: hidden;
}}

.run-btn::before {{
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, {GOLD}22, transparent);
    transition: left 0.4s ease;
}}

.run-btn:hover::before {{ left: 100%; }}
.run-btn:hover {{
    background: {GOLD_FAINT} !important;
    box-shadow: 0 0 20px {GOLD}44 !important;
}}

.slider-label {{
    font-size: 9px;
    color: {GREY};
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

.rc-slider-rail     {{ background: {GREY_DIM} !important; height: 2px !important; }}
.rc-slider-track    {{ background: {GOLD} !important; height: 2px !important; }}
.rc-slider-mark-text {{
    color: {GREY} !important;
    font-size: 9px !important;
    font-family: {FONT} !important;
}}
.rc-slider-mark-text-active {{
    color: {GOLD} !important;
}}
.rc-slider-handle   {{
    border: 2px solid {GOLD} !important;
    background: {BG} !important;
    width: 10px !important; height: 10px !important;
    margin-top: -4px !important;
    box-shadow: 0 0 6px {GOLD}88 !important;
}}
.rc-slider-tooltip-inner {{
    background-color: {BG} !important;
    border: 1px solid {GOLD_DIM} !important;
    color: {GOLD} !important;
    font-family: {FONT} !important;
    font-size: 9px !important;
    box-shadow: none !important;
}}
.rc-slider-tooltip-arrow {{
    border-top-color: {GOLD_DIM} !important;
}}

.status-line {{
    font-size: 10px;
    color: {GREY};
    letter-spacing: 1.5px;
    padding: 6px 0;
}}

.section-divider {{
    border: none;
    border-top: 1px solid {GOLD_DIM}22;
    margin: 6px 0;
}}

::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {GOLD_DIM}; border-radius: 2px; }}
"""

buffer = StreamingBuffer()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="LOB // TERMINAL",
)
server = app.server

app.index_string = app.index_string.replace(
    "</head>", f"<style>{CSS}</style></head>"
)


def panel(header, children, body_style=None):
    return html.Div([
        html.Div(header, className="panel-header"),
        html.Div(children, className="panel-body",
                 style=body_style or {}),
    ], className="panel")


def empty_fig(label="NO DATA"):
    fig = go.Figure()
    fig.add_annotation(
        text=label, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(color=GREY_DIM, size=10, family=FONT),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_PANEL,
        font=dict(family=FONT, color=GOLD, size=9),
        margin=dict(l=36, r=8, t=16, b=28),
        xaxis=dict(gridcolor=GREY_GRID, zeroline=False,
                   showline=False, color=GREY),
        yaxis=dict(gridcolor=GREY_GRID, zeroline=False,
                   showline=False, color=GREY),
    )
    return fig


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_PANEL,
        font=dict(family=FONT, color=GOLD, size=9),
        margin=dict(l=42, r=8, t=16, b=32),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8),
                    bordercolor=GOLD_GRID, borderwidth=1),
        hoverlabel=dict(bgcolor=BG_CARD, font_family=FONT,
                        font_size=9, bordercolor=GOLD_DIM),
    )
    fig.update_xaxes(gridcolor=GREY_GRID, zeroline=False,
                     showline=True, linecolor=GOLD_GRID, color=GREY)
    fig.update_yaxes(gridcolor=GREY_GRID, zeroline=False,
                     showline=True, linecolor=GOLD_GRID, color=GREY)
    return fig


app.layout = html.Div([

    # ── MASTHEAD ──────────────────────────────────────────────
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("LOB", style={
                        "color": GOLD, "fontWeight": "700",
                        "fontSize": "22px", "letterSpacing": "4px",
                    }),
                    html.Span(" // ", style={"color": GREY_DIM, "fontSize": "18px"}),
                    html.Span("TERMINAL", style={
                        "color": WHITE, "fontWeight": "300",
                        "fontSize": "22px", "letterSpacing": "8px",
                    }),
                    html.Span("  ──  LIMIT ORDER BOOK MICROSTRUCTURE SIMULATOR", style={
                        "color": GREY_DIM, "fontSize": "9px",
                        "letterSpacing": "3px", "marginLeft": "12px",
                        "verticalAlign": "middle",
                    }),
                ], style={"padding": "14px 0 6px 0", "fontFamily": FONT}),
            ], width=9),
            dbc.Col([
                html.Div([
                    html.Span(className="live-dot", id="live-indicator"),
                    html.Span("SYSTEM READY", id="sys-status",
                             style={"fontSize": "9px", "letterSpacing": "2px",
                                    "color": GREY}),
                ], style={"textAlign": "right", "paddingTop": "18px"}),
            ], width=3),
        ]),
    ], style={"borderBottom": f"1px solid {GOLD_DIM}44",
              "marginBottom": "8px", "padding": "0 4px"}),

    # ── TICKER ────────────────────────────────────────────────
    html.Div([
        html.Div(id="ticker-content", className="ticker-inner",
                children="LOB//TERMINAL  ●  AWAITING MARKET DATA  ●  CONFIGURE AND EXECUTE"),
    ], className="ticker-wrap"),

    # ── PROGRESS ──────────────────────────────────────────────
    html.Div([
        html.Div(id="progress-fill", className="progress-inner",
                style={"width": "0%"}),
    ], className="progress-outer"),

    # ── CONTROLS ──────────────────────────────────────────────
    panel("SIMULATION PARAMETERS", dbc.Row([
        dbc.Col([
            html.Div("DURATION (S)", className="slider-label"),
            dcc.Slider(id="duration", min=60, max=3600, step=60, value=300,
                      marks=None,
                      tooltip={"always_visible": False}),
        ], width=3),
        dbc.Col([
            html.Div("SPREAD WIDTH", className="slider-label"),
            dcc.Slider(id="mm-spread", min=0.01, max=0.50, step=0.01, value=0.10,
                      marks=None,
                      tooltip={"always_visible": False}),
        ], width=2),
        dbc.Col([
            html.Div("MARKET MAKERS", className="slider-label"),
            dcc.Slider(id="n-mm", min=1, max=10, step=1, value=3,
                      marks=None,
                      tooltip={"always_visible": False}),
        ], width=2),
        dbc.Col([
            html.Div("NOISE TRADERS", className="slider-label"),
            dcc.Slider(id="n-noise", min=1, max=20, step=1, value=10,
                      marks=None,
                      tooltip={"always_visible": False}),
        ], width=2),
        dbc.Col([
            html.Div("INFORMED TRADERS", className="slider-label"),
            dcc.Slider(id="n-informed", min=0, max=10, step=1, value=2,
                      marks=None,
                      tooltip={"always_visible": False}),
        ], width=2),
        dbc.Col([
            html.Button("▶  EXECUTE", id="run-btn", className="run-btn w-100"),
            html.Div(id="status-text", className="status-line"),
        ], width=1),
    ], className="g-3")),

    # ── METRICS ROW ───────────────────────────────────────────
    dbc.Row([
        dbc.Col(html.Div([
            html.Div("TOTAL TRADES", className="metric-label"),
            html.Div("——", id="m-trades", className="metric-value"),
            html.Div("EXECUTED", className="metric-sub"),
        ], className="metric-block"), width=2),
        dbc.Col(html.Div([
            html.Div("VWAP", className="metric-label"),
            html.Div("——", id="m-vwap", className="metric-value"),
            html.Div("VOL-WEIGHTED AVG PRICE", className="metric-sub"),
        ], className="metric-block"), width=2),
        dbc.Col(html.Div([
            html.Div("MEAN SPREAD", className="metric-label"),
            html.Div("——", id="m-spread", className="metric-value"),
            html.Div("BID-ASK", className="metric-sub"),
        ], className="metric-block"), width=2),
        dbc.Col(html.Div([
            html.Div("ANN. VOLATILITY", className="metric-label"),
            html.Div("——", id="m-vol", className="metric-value"),
            html.Div("REALIZED", className="metric-sub"),
        ], className="metric-block"), width=2),
        dbc.Col(html.Div([
            html.Div("OFI CORRELATION", className="metric-label"),
            html.Div("——", id="m-ofi", className="metric-value"),
            html.Div("CONT ET AL. 2013", className="metric-sub"),
        ], className="metric-block"), width=2),
        dbc.Col(html.Div([
            html.Div("MARKET QUALITY", className="metric-label"),
            html.Div("——", id="m-quality", className="metric-value"),
            html.Div("COMPOSITE SCORE", className="metric-sub"),
        ], className="metric-block"), width=2),
    ], className="g-2 mb-2"),

    # ── ROW 1: PRICE + DEPTH ──────────────────────────────────
    dbc.Row([
        dbc.Col([
            panel("MID PRICE  ·  SPREAD  ·  BOLLINGER BANDS  ·  VOLUME",
                  dcc.Graph(id="price-chart", figure=empty_fig(),
                            style={"height": "300px"},
                            config={"displayModeBar": False})),
        ], width=8),
        dbc.Col([
            panel("LIVE ORDER BOOK DEPTH",
                  dcc.Graph(id="depth-chart", figure=empty_fig(),
                            style={"height": "300px"},
                            config={"displayModeBar": False})),
        ], width=4),
    ], className="g-2"),

    # ── ROW 2: HEATMAP + TAPE ─────────────────────────────────
    dbc.Row([
        dbc.Col([
            panel("LIQUIDITY HEATMAP  ·  PRICE LEVEL DEPTH OVER TIME",
                  dcc.Graph(id="heatmap-chart", figure=empty_fig(),
                            style={"height": "260px"},
                            config={"displayModeBar": False})),
        ], width=8),
        dbc.Col([
            panel("TRADE TAPE  ·  LIVE FEED",
                  dcc.Graph(id="trade-tape", figure=empty_fig(),
                          style={"height": "240px"},
                          config={"displayModeBar":False})),
        ], width=4),
    ], className="g-2"),

    # ── ROW 3: OFI + VOL ──────────────────────────────────────
    dbc.Row([
        dbc.Col([
            panel("ORDER FLOW IMBALANCE  ·  CONT, KUKANOV & STOIKOV (2013)",
                  dcc.Graph(id="ofi-chart", figure=empty_fig(),
                            style={"height": "240px"},
                            config={"displayModeBar": False})),
        ], width=6),
        dbc.Col([
            panel("REALIZED VOLATILITY  ·  ROLLING 20-PERIOD WINDOW",
                  dcc.Graph(id="vol-chart", figure=empty_fig(),
                            style={"height": "240px"},
                            config={"displayModeBar": False})),
        ], width=6),
    ], className="g-2"),

    # ── ROW 4: 3D IMPACT + INVENTORY ─────────────────────────
    dbc.Row([
        dbc.Col([
            panel("3D PRICE IMPACT SURFACE  ·  SIZE × TIME × IMPACT",
                  dcc.Graph(id="impact-3d", figure=empty_fig(),
                            style={"height": "360px"},
                            config={"displayModeBar": False})),
        ], width=6),
        dbc.Col([
            panel("AGENT INVENTORY TRACKER  ·  REAL-TIME POSITIONS",
                  dcc.Graph(id="inventory-chart", figure=empty_fig(),
                            style={"height": "360px"},
                            config={"displayModeBar": False})),
        ], width=6),
    ], className="g-2"),

    # ── ROW 5: PROFILE + PNL ──────────────────────────────────
    dbc.Row([
        dbc.Col([
            panel("MARKET PROFILE  ·  VOLUME AT PRICE",
                  dcc.Graph(id="profile-chart", figure=empty_fig(),
                            style={"height": "300px"},
                            config={"displayModeBar": False})),
        ], width=4),
        dbc.Col([
            panel("AGENT PnL LEADERBOARD  ·  MARK-TO-MARKET",
                  dcc.Graph(id="pnl-chart", figure=empty_fig(),
                            style={"height": "300px"},
                            config={"displayModeBar": False})),
        ], width=8),
    ], className="g-2 mb-4"),

    dcc.Interval(id="poll", interval=600, n_intervals=0),

], style={"backgroundColor": BG, "minHeight": "100vh",
          "padding": "0 16px", "fontFamily": FONT})


@app.callback(
    Output("status-text", "children"),
    Output("live-indicator", "className"),
    Output("sys-status", "children"),
    Input("run-btn", "n_clicks"),
    State("duration", "value"),
    State("n-mm", "value"),
    State("n-noise", "value"),
    State("n-informed", "value"),
    State("mm-spread", "value"),
    prevent_initial_call=True,
)
def start_sim(n_clicks, duration, n_mm, n_noise, n_informed, mm_spread):
    if buffer.running:
        return "ALREADY RUNNING", "live-dot", "RUNNING"
    buffer.reset()
    config = SimulationConfig(
        duration=float(duration),
        n_market_makers=int(n_mm),
        n_noise_traders=int(n_noise),
        n_informed_traders=int(n_informed),
        mm_spread=float(mm_spread),
    )
    def _run():
        sim = Simulation(config=config, buffer=buffer)
        print(f"DEBUG: buffer id in sim = {id(sim.buffer)}global buffer id = {id(buffer)}")
        print(f"DEBUG: starting sim run")
        sim.run()
        print(f"DEBUG: sim run complete, snapshots={len(buffer.snapshots)}, trades={len(buffer.trades)}")
    threading.Thread(target=_run, daemon=True).start()
    return "STREAMING LIVE DATA", "live-dot green", "LIVE"


@app.callback(
    Output("price-chart",     "figure"),
    Output("depth-chart",     "figure"),
    Output("heatmap-chart",   "figure"),
    Output("trade-tape",      "figure"),
    Output("ofi-chart",       "figure"),
    Output("vol-chart",       "figure"),
    Output("impact-3d",       "figure"),
    Output("inventory-chart", "figure"),
    Output("profile-chart",   "figure"),
    Output("pnl-chart",       "figure"),
    Output("m-trades",   "children"),
    Output("m-vwap",     "children"),
    Output("m-spread",   "children"),
    Output("m-vol",      "children"),
    Output("m-ofi",      "children"),
    Output("m-quality",  "children"),
    Output("status-text",    "children"),
    Output("progress-fill",  "style"),
    Output("ticker-content", "children"),
    Input("poll", "n_intervals"),
)
def refresh(n):
    snapshots, trades, agent_states = buffer.read()
    all_trades = trades
    trades = trades[-5000:]
    pct = buffer.progress * 100
    prog_style = {
        "width": f"{pct:.1f}%",
        "height": "100%",
        "background": f"linear-gradient(90deg, {GOLD_DIM}, {GOLD})",
        "transition": "width 0.4s ease",
    }
    blank = empty_fig()
    blanks = [blank] * 10
    dashes = ["——"] * 6

    if not snapshots or len(snapshots) < 5:
        status = "STREAMING..." if buffer.running else "AWAITING EXECUTION"
        ticker = "LOB // TERMINAL  ●  AWAITING MARKET DATA  ●  CONFIGURE PARAMETERS AND EXECUTE SIMULATION"
        return blanks + dashes + [status, prog_style, ticker]

    times  = [s.time       for s in snapshots if s.mid_price]
    mids   = [s.mid_price  for s in snapshots if s.mid_price]
    sprs   = [s.spread     for s in snapshots if s.spread]
    b_deps = [s.bid_depth  for s in snapshots]
    a_deps = [s.ask_depth  for s in snapshots]

    if not mids:
        return blanks + dashes + ["NO PRICE DATA", prog_style, "NO DATA"]

    last_p   = mids[-1]
    first_p  = mids[0]
    chg      = last_p - first_p
    pct_chg  = chg / first_p * 100 if first_p else 0
    arrow    = "▲" if chg >= 0 else "▼"
    chg_col  = GREEN if chg >= 0 else RED
    last_spr = sprs[-1] if sprs else 0

    ticker_str = (
        f"LOB//SIM  ●  "
        f"LAST {last_p:.4f}  {arrow} {abs(pct_chg):.2f}%  ●  "
        f"SPREAD {last_spr:.4f}  ●  "
        f"TRADES {len(trades):,}  ●  "
        f"BID DEPTH {b_deps[-1]:,}  ●  "
        f"ASK DEPTH {a_deps[-1]:,}  ●  "
        f"PROGRESS {pct:.0f}%  ●  "
        f"LOB//TERMINAL  ●  MICROSTRUCTURE ANALYTICS ENGINE"
    )

    # ── PRICE CHART ──────────────────────────────────────────
    price_fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.02,
    )
    price_fig.add_trace(go.Scatter(
        x=times, y=mids, mode="lines", name="MID",
        line=dict(color=GOLD, width=1.5),
    ), row=1, col=1)

    if len(mids) > 20:
        arr = np.array(mids)
        rm  = np.convolve(arr, np.ones(20)/20, mode="valid")
        rs  = np.array([arr[i:i+20].std() for i in range(len(arr)-19)])
        tr  = times[19:]
        price_fig.add_trace(go.Scatter(
            x=tr, y=(rm+2*rs).tolist(), mode="lines",
            line=dict(color="rgba(192,57,43,0.4)", width=0.8, dash="dot"),
            showlegend=False, name="BB+2σ",
        ), row=1, col=1)
        price_fig.add_trace(go.Scatter(
            x=tr, y=(rm-2*rs).tolist(), mode="lines",
            line=dict(color="rgba(192,57,43,0.4)", width=0.8, dash="dot"),
            fill="tonexty", fillcolor=RED_T10,
            showlegend=False, name="BB-2σ",
        ), row=1, col=1)
        price_fig.add_trace(go.Scatter(
            x=tr, y=rm.tolist(), mode="lines",
            line=dict(color="rgba(232,232,232,0.27)", width=0.8, dash="dash"),
            showlegend=False, name="MA20",
        ), row=1, col=1)

    price_fig.add_trace(go.Scatter(
        x=times, y=sprs, mode="lines", name="SPREAD",
        line=dict(color=RED, width=1),
        fill="tozeroy", fillcolor=RED_T20,
    ), row=2, col=1)

    if trades:
        t_times = list(range(max(0, len(trades)-800), len(trades)))
        t_qtys  = [t.quantity  for t in trades[-800:]]
        t_cols  = [GREEN if t.aggressor_order_id > t.passive_order_id
                   else RED for t in trades[-800:]]
        price_fig.add_trace(go.Bar(
            x=t_times, y=t_qtys,
            marker_color=t_cols, name="VOL",
            marker_line_width=0,
        ), row=3, col=1)

    price_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_PANEL,
        font=dict(family=FONT, color=GOLD, size=9),
        margin=dict(l=42, r=8, t=8, b=24),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8),
                    orientation="h", y=1.08),
        showlegend=True,
    )
    price_fig.update_xaxes(gridcolor=GREY_GRID, zeroline=False, color=GREY)
    price_fig.update_yaxes(gridcolor=GREY_GRID, zeroline=False, color=GREY)

    # ── DEPTH CHART ───────────────────────────────────────────
    depth_fig = go.Figure()
    depth     = snapshots[-1].depth_snapshot
    if depth.get("bids"):
        bp, bq = zip(*depth["bids"])
        cum_b  = np.cumsum(list(reversed(bq)))
        depth_fig.add_trace(go.Scatter(
            x=list(reversed(bp)), y=cum_b.tolist(),
            mode="lines", fill="tozeroy",
            fillcolor=GREEN_T20, line=dict(color=GREEN, width=2),
            name="BID",
        ))
    if depth.get("asks"):
        ap, aq = zip(*depth["asks"])
        cum_a  = np.cumsum(list(aq))
        depth_fig.add_trace(go.Scatter(
            x=list(ap), y=cum_a.tolist(),
            mode="lines", fill="tozeroy",
            fillcolor=RED_T20, line=dict(color=RED, width=2),
            name="ASK",
        ))
    style_fig(depth_fig)

    # ── HEATMAP ───────────────────────────────────────────────
    heatmap_fig = empty_fig("BUILDING HEATMAP...")
    if len(snapshots) > 15:
        all_prices = set()
        for s in snapshots:
            for p, _ in s.depth_snapshot.get("bids", []):
                all_prices.add(round(p, 1))
            for p, _ in s.depth_snapshot.get("asks", []):
                all_prices.add(round(p, 1))
        if all_prices:
            plevels = sorted(all_prices)[-50:]
            samp    = snapshots[::max(1, len(snapshots)//80)]
            z_mat   = []
            for s in samp:
                bm = {round(p,1): q for p,q in s.depth_snapshot.get("bids",[])}
                am = {round(p,1): q for p,q in s.depth_snapshot.get("asks",[])}
                z_mat.append([bm.get(p,0)+am.get(p,0) for p in plevels])
            if z_mat and any(any(r) for r in z_mat):
                heatmap_fig = go.Figure(go.Heatmap(
                    z=list(map(list, zip(*z_mat))),
                    x=[s.time for s in samp],
                    y=plevels,
                    colorscale=[
                        [0.0,  BG_PANEL],
                        [0.25, "rgba(160,136,0,0.53)"],
                        [0.6,  GOLD],
                        [1.0,  WHITE],
                    ],
                    showscale=True,
                    colorbar=dict(
                        tickfont=dict(color=GREY, size=8, family=FONT),
                        bgcolor=BG_CARD,
                        bordercolor=GOLD_GRID,
                        thickness=8,
                    ),
                ))
                if mids:
                    heatmap_fig.add_trace(go.Scatter(
                        x=[s.time for s in snapshots if s.mid_price],
                        y=mids, mode="lines", name="MID",
                        line=dict(color=RED, width=1.5),
                    ))
                style_fig(heatmap_fig)

    # ── TRADE TAPE ────────────────────────────────────────────
    tape_fig = empty_fig()
    if trades:
        recent=trades[-40:]
        colors =[GREEN if t.aggressor_order_id > t.passive_order_id else RED for t in recent]
        tape_fig=go.Figure(go.Bar(
            x=list(range(len(recent))),
            y=[t.quantity for t in recent],
            marker_color=colors, 
            marker_line_width=0,
            text=[f"{t.price:.2f}" for t in recent],
            textposition="outside", 
            textfont=dict(size=7, color=GOLD, family=FONT),
        ))
        style_fig(tape_fig)
        tape_fig.update_layout(showlegend=False,
                                margin=dict(l=20, r=8, t=8, b=20))
    tape = tape_fig

    # ── OFI ───────────────────────────────────────────────────
    ofi_fig = empty_fig()
    if len(snapshots) > 5:
        bd   = np.diff([s.bid_depth for s in snapshots])
        ad   = np.diff([s.ask_depth for s in snapshots])
        ofi  = bd - ad
        otimes = times[1:len(ofi)+1]
        ofi_fig = go.Figure()
        ofi_fig.add_trace(go.Bar(
            x=otimes[:len(ofi)], y=ofi.tolist(),
            marker_color=[GREEN if v >= 0 else RED for v in ofi],
            name="OFI", marker_line_width=0,
        ))
        if len(ofi) > 10:
            sm = np.convolve(ofi, np.ones(10)/10, mode="valid")
            ofi_fig.add_trace(go.Scatter(
                x=otimes[9:len(sm)+9], y=sm.tolist(),
                mode="lines", name="MA10",
                line=dict(color=GOLD, width=1.5),
            ))
        style_fig(ofi_fig)

    # ── VOLATILITY ────────────────────────────────────────────
    vol_fig = empty_fig()
    if len(mids) > 25:
        lr   = np.diff(np.log(np.array(mids)))
        w    = 20
        rvol = [lr[i-w:i].std() for i in range(w, len(lr))]
        vtimes = times[w+1:len(rvol)+w+1]
        mean_v = np.mean(rvol)
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(
            x=vtimes[:len(rvol)], y=rvol,
            marker_color=[RED if v > 1.5*mean_v else "rgba(240,200,0,0.6)"
                         for v in rvol],
            name="RVOL", marker_line_width=0,
        ))
        vol_fig.add_hline(y=mean_v, line_dash="dot",
                         line_color="rgba(232,232,232,0.33)", line_width=1,
                         annotation_text="μ",
                         annotation_font=dict(color=WHITE, size=9))
        style_fig(vol_fig)

    # ── 3D IMPACT ─────────────────────────────────────────────
    impact_fig = empty_fig("NEED MORE TRADES")
    if len(trades) > 100:
        try:
            tsz  = np.array([t.quantity  for t in trades])
            tpr  = np.array([t.price     for t in trades])
            timp = np.abs(np.diff(tpr, prepend=tpr[0]))
            tidx = np.arange(len(trades), dtype=float)
            tb = np.linspace(0, len(trades)-1, 15)
            sb = np.linspace(tsz.min(), tsz.max(), 15)
            Z    = np.zeros((14, 14))
            for i in range(14):
                for j in range(14):
                    mask = (
                    (tidx>=tb[i]) & (tidx<tb[i+1]) &
                    (tsz>=sb[j])&(tsz<sb[j+1])
                    )
                    if mask.sum() > 0:
                        Z[i, j] = timp[mask].mean()
            if Z.max()>0:
                impact_fig = go.Figure(go.Surface(
                x=sb[:-1].tolist(), 
                y=tb[:-1].tolist(), 
                z=Z.tolist(),
                colorscale=[
                    [0.0, BG_PANEL],
                    [0.4, GOLD_DIM],
                    [0.7, GOLD],
                    [1.0, WHITE],
                ],
                showscale=False, opacity=0.9,
                ))
                impact_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family=FONT, color=GOLD, size=9),
                    margin=dict(l=0, r=0, t=8, b=0),
                    scene=dict(
                        bgcolor=BG_PANEL,
                        xaxis=dict(title=dict(text="SIZE", font=dict(size=8)), gridcolor=GREY_DIM, color=GREY),
                        yaxis=dict(title=dict(text="TIME", font=dict(size=8)), gridcolor=GREY_DIM, color=GREY),
                        zaxis=dict(title=dict(text="IMPACT", font=dict(size=8)), gridcolor=GREY_DIM, color=GREY),
                    ),
                )
        except Exception as e:
            print(f"3D error: {e}")
            impact_fig = empty_fig("3D ERROR")
    # ── INVENTORY ─────────────────────────────────────────────
    inv_fig = empty_fig()
    if agent_states:
        names   = list(agent_states[-1]["agents"].keys())
        samp    = agent_states[::max(1, len(agent_states)//200)]
        mm_i = n_i = inf_i = 0
        mm_cols  = [GOLD, GOLD_DIM, "rgba(240,200,0,0.53)"]
        n_cols   = ["rgba(136,136,136,0.53)"] * 15
        inf_cols = [RED, RED_DIM]
        inv_fig  = go.Figure()
        for name in names:
            if "MarketMaker" in name:
                col = mm_cols[mm_i % len(mm_cols)]; mm_i += 1
            elif "NoiseTrader" in name:
                col = n_cols[n_i % len(n_cols)]; n_i += 1
            else:
                col = inf_cols[inf_i % len(inf_cols)]; inf_i += 1
            inv_fig.add_trace(go.Scatter(
                x=[s["time"] for s in samp],
                y=[s["agents"].get(name, {}).get("inventory", 0)
                   for s in samp],
                mode="lines",
                name=name.replace("_", " "),
                line=dict(color=col, width=1),
            ))
        inv_fig.add_hline(y=0, line_dash="dot",
                         line_color="rgba(232,232,232,0.2)", line_width=1)
        style_fig(inv_fig)

    # ── MARKET PROFILE ────────────────────────────────────────
    profile_fig = empty_fig()
    if trades:
        pv = {}
        for t in trades:
            b = round(t.price, 1)
            pv[b] = pv.get(b, 0) + t.quantity
        if pv:
            pvs  = sorted(pv.items())
            ps   = [p for p,_ in pvs]
            vs   = [v for _,v in pvs]
            mv   = max(vs)
            cols = [WHITE if v==mv else GOLD if v>mv*0.7
                    else GOLD_DIM for v in vs]
            profile_fig = go.Figure(go.Bar(
                x=vs, y=ps, orientation="h",
                marker_color=cols, marker_line_width=0,
                name="VOL@PRICE",
            ))
            style_fig(profile_fig)

    # ── PNL ───────────────────────────────────────────────────
    pnl_fig = empty_fig()
    if agent_states:
        last = agent_states[-1]["agents"]
        sd   = sorted(last.items(), key=lambda x: x[1]["pnl"], reverse=True)
        ns   = [x[0] for x in sd]
        ps   = [x[1]["pnl"] for x in sd]
        pnl_fig = go.Figure(go.Bar(
            x=ns, y=ps,
            marker_color=[GREEN if p>=0 else RED for p in ps],
            marker_line_width=0,
            text=[f"{p:+.0f}" for p in ps],
            textposition="outside",
            textfont=dict(size=8, color=GOLD, family=FONT),
        ))
        style_fig(pnl_fig)
        pnl_fig.update_layout(xaxis_tickangle=-40,
                              xaxis_tickfont=dict(size=7))

    # ── METRICS ───────────────────────────────────────────────
    n_trades = f"{len(all_trades):,}"
    if trades:
        notional = sum(t.price * t.quantity for t in trades)
        volume   = sum(t.quantity for t in trades)
        vwap_v   = f"{notional/volume:.2f}" if volume else "——"
    else:
        vwap_v = "——"

    mean_spr = f"{np.mean(sprs):.4f}" if sprs else "——"

    if len(mids) > 25:
        lr     = np.diff(np.log(np.array(mids)))
        ann_v  = f"{lr.std()*np.sqrt(252):.4f}"
    else:
        ann_v  = "——"

    if len(snapshots) > 10 and len(mids) > 2:
        bd_a = np.diff([s.bid_depth for s in snapshots])
        ad_a = np.diff([s.ask_depth for s in snapshots])
        ofi_a = bd_a - ad_a
        rets  = np.diff(np.array(mids))
        ml    = min(len(ofi_a), len(rets))
        if ml > 2:
            c = np.corrcoef(ofi_a[:ml], rets[:ml])[0,1]
            ofi_v = f"{c:.4f}" if not np.isnan(c) else "——"
        else:
            ofi_v = "——"
    else:
        ofi_v = "——"

    if sprs and len(mids) > 5:
        ss = max(0, 1 - np.mean(sprs)/2)
        vs = max(0, 1 - np.diff(np.log(np.array(mids))).std()*100)
        ds = min(1, (b_deps[-1]+a_deps[-1])/2000)
        q  = (ss*0.4 + vs*0.3 + ds*0.3)*100
        q_v = f"{q:.0f}/100"
    else:
        q_v = "——"

    status = "STREAMING LIVE" if buffer.running else "COMPLETE"

    return (
        price_fig, depth_fig, heatmap_fig, tape,
        ofi_fig, vol_fig, impact_fig, inv_fig,
        profile_fig, pnl_fig,
        n_trades, vwap_v, mean_spr, ann_v, ofi_v, q_v,
        status, prog_style, ticker_str,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)