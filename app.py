# true_beacon_radio.py  ·  v1.3.5 · 2025-04-30
# ------------------------------------------------
# Brand-locked chart builder (web-only edition):
# • Canvas presets, data-labels, highlight bands, trend-lines
# • Axis-title & tick-label toggles + edits
# • Color randomiser driving all series
# • Resolution preview (no padding)
# • Data normalization (index to base)
# • Pure-SVG export via Kaleido + PNG
# • Standard Streamlit download buttons

import io, re, random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

# ── Brand template & palette ──────────────────────────────────────────────
PALETTE = GOLD, GOLD_LIGHT, GRAY, WHITE = (
    "#987F2F", "#E5C96A", "#B6B6B6", "#FFFFFF"
)
TRANS = "rgba(0,0,0,0)"
pio.templates["truebeacon"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Red Hat Display, sans-serif", size=14, color=GRAY),
        colorway=list(PALETTE),
        plot_bgcolor=TRANS, paper_bgcolor=TRANS,
        xaxis=dict(showgrid=False, showline=True, linecolor=GRAY,
                   zeroline=True, zerolinecolor=GRAY),
        yaxis=dict(showgrid=False, showline=True, linecolor=GRAY,
                   zeroline=True, zerolinecolor=GRAY),
        margin=dict(l=0, r=0, t=0, b=0, pad=0)
    )
)
pio.templates.default = "truebeacon"

slug  = lambda s: re.sub(r"[^0-9A-Za-z]+", "_", s.lower()).strip("_") or "chart"
stamp = lambda: datetime.now().strftime("%Y%m%d-%H%M")

# ── Helpers ────────────────────────────────────────────────────────────────
def parse_manual(txt: str) -> pd.DataFrame:
    delim = next((d for d in ("\t",";","|",",") if d in txt), ",")
    return pd.read_csv(io.StringIO(txt), delimiter=delim)

def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "X": list(range(1,6)),
        "Series 1":[10,12,9,14,8],
        "Series 2":[5,7,6,9,4]
    })

def drawdown_fig(df, x, y, title, palette):
    cum        = (1 + df[y]).cumprod()
    running_max= cum.cummax()
    dd         = cum / running_max - 1
    fig        = px.area(
        x=df[x], y=dd, title=title,
        labels={"y":"Draw-down"},
        color_discrete_sequence=palette
    )
    fig.update_yaxes(tickformat=".0%")
    return fig

def combo_fig(df, x, b, l, title, palette):
    fig = go.Figure()
    fig.add_bar(x=df[x], y=df[b], name=b, marker_color=palette[0])
    fig.add_scatter(
        x=df[x], y=df[l], mode="lines+markers", name=l,
        yaxis="y2", line=dict(color=palette[1])
    )
    fig.update_layout(
        title=title,
        yaxis2=dict(overlaying="y", side="right",
                    showgrid=False, showline=True, linecolor=GRAY)
    )
    return fig

# ── Streamlit scaffold ────────────────────────────────────────────────────
st.set_page_config(page_title="Radio", layout="wide", initial_sidebar_state="collapsed")
st.title("Radio")

# ① DATA INPUT & NORMALIZATION =============================================
tabs = st.tabs(["Upload file","Manual paste"])
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

with tabs[0]:
    up = st.file_uploader("CSV / Excel")
    if st.button("Download sample CSV"):
        st.download_button(
            "sample.csv",
            sample_df().to_csv(index=False).encode(),
            "sample.csv","text/csv"
        )
    if up:
        st.session_state.df = (
            pd.read_csv(up) if up.name.endswith(".csv")
            else pd.read_excel(up)
        )
        st.success("Loaded")
        st.dataframe(st.session_state.df.head(), use_container_width=True)

with tabs[1]:
    txt = st.text_area(
        "Paste data (tab/comma/semicolon/pipe separated)", height=160
    )
    if st.button("Parse paste"):
        try:
            st.session_state.preview = parse_manual(txt)
        except Exception as e:
            st.error(e)
    if "preview" in st.session_state:
        prev = st.session_state.preview
        st.dataframe(prev, use_container_width=True)
        with st.form("rename"):
            cols = [
                st.text_input("", c, key=f"h{i}")
                for i, c in enumerate(prev.columns)
            ]
            if st.form_submit_button("Use this data"):
                prev.columns = cols
                st.session_state.df = prev.copy()
                st.success("Data ready")
                del st.session_state.preview

df = st.session_state.df.copy()
if df.empty:
    st.stop()

# Ask about indexing
if st.checkbox("Normalize numeric series to base 100"):
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    sel = st.multiselect("Select series to normalize", numeric_cols, default=numeric_cols)
    if st.button("Apply normalization"):
        for c in sel:
            first = df[c].iloc[0]
            if first != 0:
                df[c] = df[c] / first * 100
        st.success(f"Indexed {', '.join(sel)} to 100")

cols = list(df.columns)

# ② CHART SELECTION =========================================================
CHARTS = [
    "Line","Scatter","Area","Bar","Grouped Bar","Stacked Bar",
    "Stacked Bar (h)","Histogram","Box","Pie","Donut","Donut + Pie",
    "Radar","Waterfall","Draw-down","Line-Bar Combo"
]
chart = st.selectbox("Chart type", CHARTS)
xcol  = st.selectbox("X axis", cols)

single = {
    "Pie","Donut","Donut + Pie","Histogram","Box",
    "Waterfall","Draw-down"
}
if chart in single:
    ycols = [st.selectbox("Value / Y",[c for c in cols if c!=xcol])]
elif chart=="Radar":
    ycols = st.multiselect(
        "Series",[c for c in cols if c!=xcol],
        default=[c for c in cols if c!=xcol][:1]
    )
elif chart=="Line-Bar Combo":
    bar_y  = st.selectbox("Bar series",[c for c in cols if c!=xcol])
    line_y = st.selectbox(
        "Line series",[c for c in cols if c not in (xcol,bar_y)]
    )
else:
    ycols = st.multiselect(
        "Series",[c for c in cols if c!=xcol],
        default=[c for c in cols if c!=xcol][:2]
    )

# ③ LAYOUT & STYLE =========================================================
PRESETS = {
    "Slide-below 1650×640":(1650,640),
    "Slide-side 815×640": (815,640),
    "Hero 1920×480":     (1920,480),
    "Social 1080×800":   (1080,800),
    "Square 1080×1080":  (1080,1080),
    "Custom…":           None
}
sel = st.selectbox("Canvas preset", list(PRESETS.keys()))
if sel=="Custom…":
    W = st.number_input("Width",400,4000,800,50)
    H = st.number_input("Height",300,3000,600,50)
else:
    W,H = PRESETS[sel]

# Resolution preview (no padding)
scale = min(300/W, 300/H, 1)
pw,ph = int(W*scale), int(H*scale)
st.markdown(
    f"<div style='width:{pw}px;height:{ph}px;"
    f"border:2px dashed {GRAY};margin:8px auto;'></div>"
    f"<div style='text-align:center;'>{W}×{H}px</div>",
    unsafe_allow_html=True
)

# Color randomiser
if "palette" not in st.session_state:
    st.session_state.palette = list(PALETTE)
col1, col2 = st.columns(2)
with col1:
    if st.button("Randomize colors"):
        random.shuffle(st.session_state.palette)
with col2:
    if st.button("Reset colors"):
        st.session_state.palette = list(PALETTE)

# Axis titles & tick-label toggles
show_x_title = st.checkbox("Show X-axis title", True)
x_title_text = st.text_input("X-title text", xcol) if show_x_title else ""
show_x_ticks = st.checkbox("Show X-tick labels", True)

show_y_title = st.checkbox("Show Y-axis title", True)
y_title_text = st.text_input("Y-title text", ycols[0] if ycols else "") if show_y_title else ""
show_y_ticks = st.checkbox("Show Y-tick labels", True)

# Basic style
show_title = st.checkbox("Show chart title", True)
title      = st.text_input("Chart title", "") if show_title else ""
legend     = st.checkbox("Show legend", True)
font_sz    = st.slider("Font size",8,32,14)
grid       = st.checkbox("Gridlines")
yzero      = st.checkbox("Y starts at 0")

# Highlight band
st.caption("Optional highlight band (gold)")
bx0,bx1 = st.text_input("X-start"), st.text_input("X-end")
by0,by1 = st.text_input("Y-start"), st.text_input("Y-end")
op      = st.slider("Opacity",0.05,0.5,0.12,0.01)
vband   = (bx0,bx1,op) if bx0 and bx1 else None
hband   = None
if by0 and by1:
    try:
        hband = (float(by0), float(by1), op)
    except:
        st.error("Y highlight must be numeric")

# Trend-lines
numeric   = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
trend_cfg = []
for i in (1,2):
    on = st.checkbox(f"Enable trend {i}")
    if on and numeric:
        s   = st.selectbox(f"Series {i}", numeric, key=f"s{i}")
        xs  = st.selectbox(f"Start X {i}", df[xcol], key=f"xs{i}")
        xe  = st.selectbox(f"End X {i}", df[xcol], key=f"xe{i}")
        col = st.selectbox(f"Colour {i}", st.session_state.palette,
                           key=f"c{i}", index=i-1)
        sty = st.selectbox(f"Style {i}", ["solid","dash"], key=f"st{i}")
        trend_cfg.append((s,xs,xe,sty,col))

# Data labels
show_labels=False; label_mode="value"
if chart in {"Bar","Grouped Bar","Stacked Bar","Stacked Bar (h)","Pie","Donut","Radar"}:
    show_labels = st.checkbox("Show data labels")
    if chart in {"Pie","Donut"} and show_labels:
        label_mode = st.radio("Label type", ["percent","value"], horizontal=True)

# ④ GENERATE & EXPORT =====================================================
if st.button("Generate"):
    try:
        palette = st.session_state.palette

        # Build figure
        if chart=="Line":
            fig = px.line(df, x=xcol, y=ycols, title=title,
                          color_discrete_sequence=palette)
        elif chart=="Scatter":
            fig = px.scatter(df, x=xcol, y=ycols, title=title,
                             color_discrete_sequence=palette)
        elif chart=="Area":
            fig = px.area(df, x=xcol, y=ycols, title=title,
                          color_discrete_sequence=palette)
        elif chart=="Bar":
            fig = px.bar(df, x=xcol, y=ycols, title=title,
                         text_auto=show_labels,
                         color_discrete_sequence=palette)
        elif chart=="Grouped Bar":
            fig = px.bar(df, x=xcol, y=ycols, barmode="group",
                         title=title, text_auto=show_labels,
                         color_discrete_sequence=palette)
        elif chart=="Stacked Bar":
            fig = px.bar(df, x=xcol, y=ycols, barmode="stack",
                         title=title, text_auto=show_labels,
                         color_discrete_sequence=palette)
        elif chart=="Stacked Bar (h)":
            fig = px.bar(df, x=ycols[0], y=xcol, orientation="h",
                         barmode="stack", title=title,
                         text_auto=show_labels,
                         color_discrete_sequence=palette)
        elif chart=="Histogram":
            fig = px.histogram(df, x=xcol, y=ycols[0], title=title,
                               color_discrete_sequence=palette)
        elif chart=="Box":
            fig = px.box(df, x=xcol, y=ycols[0], title=title,
                         color_discrete_sequence=palette)
        elif chart=="Pie":
            fig = px.pie(df, names=xcol, values=ycols[0], title=title,
                         color_discrete_sequence=palette)
            txt = "percent+label" if label_mode=="percent" else "value+label"
            fig.update_traces(textinfo=txt)
        elif chart=="Donut":
            fig = px.pie(df, names=xcol, values=ycols[0],
                         hole=.4, title=title,
                         color_discrete_sequence=palette)
            txt = "percent+label" if label_mode=="percent" else "value+label"
            fig.update_traces(textinfo=txt)
        elif chart=="Donut + Pie":
            txt = "percent+label" if label_mode=="percent" else "value+label"
            fig = make_subplots(rows=1, cols=2, specs=[[{"type":"domain"}]*2])
            half = len(palette)//2 or 1
            p1, p2 = palette[:half], palette[half:]
            fig.add_trace(go.Pie(
                labels=df[xcol], values=df[ycols[0]],
                hole=.4, textinfo=txt, marker=dict(colors=p1)
            ), 1, 1)
            fig.add_trace(go.Pie(
                labels=df[xcol], values=df[ycols[0]],
                textinfo=txt, marker=dict(colors=p2)
            ), 1, 2)
            fig.update_layout(title=title)
        elif chart=="Radar":
            fig = go.Figure()
            for idx, c in enumerate(ycols):
                fig.add_trace(go.Scatterpolar(
                    r=df[c], theta=df[xcol], fill="toself",
                    name=c, line=dict(color=palette[idx%len(palette)]),
                    text=df[c] if show_labels else None,
                    textposition="top center"
                ))
            fig.update_polars(
                radialaxis=dict(showticklabels=True,
                                tickformat=".0%", gridcolor=GRAY),
                angularaxis=dict(rotation=90)
            )
            fig.update_layout(title=title)
        elif chart=="Waterfall":
            fig = go.Figure(go.Waterfall(
                x=df[xcol], y=df[ycols[0]],
                measure=["relative"]*(len(df)-1)+["total"],
                textposition=("outside" if show_labels else "none"),
                marker=dict(color=palette[0])
            ))
            fig.update_layout(title=title)
        elif chart=="Draw-down":
            fig = drawdown_fig(df, xcol, ycols[0], title, palette)
        elif chart=="Line-Bar Combo":
            fig = combo_fig(df, xcol, bar_y, line_y, title, palette)
        else:
            st.stop()

        # Axis titles & ticks
        fig.update_xaxes(
            title_text=(x_title_text if show_x_title else ""),
            showticklabels=show_x_ticks
        )
        fig.update_yaxes(
            title_text=(y_title_text if show_y_title else ""),
            showticklabels=show_y_ticks
        )

        # Highlight bands
        if vband:
            x0,x1,o = vband
            fig.add_vrect(x0=x0, x1=x1,
                          fillcolor=GOLD, opacity=o, line_width=0)
        if hband:
            y0,y1,o = hband
            fig.add_hrect(y0=y0, y1=y1,
                          fillcolor=GOLD, opacity=o, line_width=0)

        # Trend-lines
        for ser,xs,xe,sty,col in trend_cfg:
            i0 = df.index[df[xcol]==xs][0]
            i1 = df.index[df[xcol]==xe][-1]
            rng = np.arange(i0, i1+1)
            y   = df.loc[rng, ser]
            m,b = np.polyfit(rng, y, 1)
            fig.add_scatter(x=df.loc[rng, xcol], y=m*rng+b,
                            mode="lines", name=f"{ser} trend",
                            line=dict(color=col, dash=sty))

        # Final layout tweaks (margin already zero)
        fig.update_layout(
            width=W, height=H,
            showlegend=legend,
            font=dict(size=font_sz)
        )
        if yzero:
            fig.update_yaxes(rangemode="tozero")
        if grid:
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

        # Render
        st.plotly_chart(fig, use_container_width=False,
            config={"modeBarButtonsToAdd":["drawrect","eraseshape"],
                    "displaylogo":False})

        # EXPORT PNG + PURE SVG
        png_bytes = fig.to_image(format="png", width=W, height=H, scale=2)
        svg_bytes = pio.to_image(fig, format="svg", width=W, height=H, scale=1)
        base      = f"{slug(title or chart)}_{W}x{H}_{stamp()}"

        st.download_button("Download PNG", png_bytes,
                           f"{base}.png","image/png")
        st.download_button("Download SVG", svg_bytes,
                           f"{base}.svg","image/svg+xml")

    except Exception as e:
        st.error(e)
