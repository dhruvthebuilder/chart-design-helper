# true_beacon_radio.py
# ------------------------------------------------
# Brand-locked chart builder with trend-line colour picker,
# data-labels, extra canvas presets, inline helper text.

import io, csv, re
from datetime import datetime
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
import base64, streamlit.components.v1 as components

# ── Brand palette & template ───────────────────────────────────────────────
PALETTE = GOLD, GOLD_LIGHT, GRAY, WHITE = (
    "#987F2F", "#FFECB8", "#B6B6B6", "#FFFFFF"
)
TRANS = "rgba(0,0,0,0)"

pio.templates["truebeacon"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Red Hat Display, sans-serif", size=14, color=GRAY),
        colorway=list(PALETTE) + ["#8A8A8A"],   # guarantee >3 series
        plot_bgcolor=TRANS, paper_bgcolor=TRANS,
        xaxis=dict(showgrid=False, showline=True, linecolor=GRAY,
                   zeroline=True, zerolinecolor=GRAY),
        yaxis=dict(showgrid=False, showline=True, linecolor=GRAY,
                   zeroline=True, zerolinecolor=GRAY)))
pio.templates.default = "truebeacon"

slug  = lambda s: re.sub(r"[^0-9A-Za-z]+", "_", s.lower()).strip("_") or "chart"
stamp = lambda: datetime.now().strftime("%Y%m%d-%H%M")

# ── Helpers ────────────────────────────────────────────────────────────────
def parse_manual(txt:str)->pd.DataFrame:
    delim = next((d for d in ("\t",";","|",",") if d in txt), ",")
    return pd.read_csv(io.StringIO(txt), delimiter=delim)

def sample_df()->pd.DataFrame:
    return pd.DataFrame({"X":range(1,6),
                         "Series 1":[10,12,9,14,8],
                         "Series 2":[5,7,6,9,4]})

def drawdown_fig(df,x,y,title):
    dd=(1+df[y]).cumprod()/(1+df[y]).cumprod().cummax()-1
    fig=px.area(x=df[x],y=dd,title=title,labels={"y":"Draw-down"})
    fig.update_yaxes(tickformat=".0%");return fig

def combo_fig(df,x,b,l,title):
    fig=go.Figure()
    fig.add_bar(x=df[x],y=df[b],name=b)
    fig.add_scatter(x=df[x],y=df[l],name=l,mode="lines+markers",yaxis="y2")
    fig.update_layout(title=title,
        yaxis2=dict(overlaying="y",side="right",showgrid=False,
                    showline=True,linecolor=GRAY));return fig

# ── UI scaffold ────────────────────────────────────────────────────────────
st.set_page_config(page_title="True Beacon — Radio", layout="wide")
st.title("True Beacon — Radio")

#DATA  ────────────────────────────────────────────────────────────────
tabs=st.tabs(["Upload", "Manual paste"])
if "df" not in st.session_state: st.session_state.df=pd.DataFrame()

with tabs[0]:
    st.caption("Please referesh before generating a new graph. Upload a CSV or Excel file.")
    up=st.file_uploader(" ",label_visibility="collapsed")
    if st.button("Download sample CSV"):
        st.download_button("sample.csv",
            sample_df().to_csv(index=False).encode(),"sample.csv","text/csv")
    if up:
        st.session_state.df=(pd.read_csv(up)
            if up.name.endswith(".csv") else pd.read_excel(up))
        st.success("File loaded.")
        st.dataframe(st.session_state.df.head(),use_container_width=True)

with tabs[1]:
    st.caption("Paste a range copied from Excel / Sheets.")
    txt=st.text_area(" ",height=160,label_visibility="collapsed")
    if st.button("Parse paste"):
        try: st.session_state.preview=parse_manual(txt)
        except Exception as e: st.error(e)
    if "preview" in st.session_state:
        prev=st.session_state.preview
        st.dataframe(prev,use_container_width=True)
        with st.form("rename"):
            st.caption("Rename columns (optional) then confirm.")
            cols=[st.text_input("",c,key=f"hdr{i}") for i,c in enumerate(prev.columns)]
            if st.form_submit_button("Use data"):
                prev.columns=cols; st.session_state.df=prev.copy()
                st.success("Data accepted."); del st.session_state.preview

df=st.session_state.df
if df.empty: st.stop()
cols=list(df.columns)

#CHART  ────────────────────────────────────────────────────────────────
st.markdown("### Chart")
CHARTS=["Line","Scatter","Area","Bar","Grouped Bar","Stacked Bar",
        "Stacked Bar (h)","Histogram","Box","Pie","Donut","Donut + Pie","Radar",
        "Heatmap","Waterfall","Draw-down","Line-Bar Combo"]
chart=st.selectbox("Chart type",CHARTS)
xcol =st.selectbox("X axis",cols)

single={"Pie","Donut","Donut + Pie","Histogram","Box","Heatmap","Waterfall","Draw-down"}
if chart in single:
    ycols=[st.selectbox("Values", [c for c in cols if c!=xcol])]
elif chart=="Radar":
    ycols=st.multiselect("Series",[c for c in cols if c!=xcol],
                         default=[c for c in cols if c!=xcol][:1])
elif chart=="Line-Bar Combo":
    bar_y=st.selectbox("Bar series",[c for c in cols if c!=xcol])
    line_y=st.selectbox("Line series",[c for c in cols if c not in (xcol,bar_y)])
else:
    ycols=st.multiselect("Series",[c for c in cols if c!=xcol],
                         default=[c for c in cols if c!=xcol][:2])

# ③ LAYOUT & OPTIONS  ────────────────────────────────────────────────────
st.markdown("### Layout & options")
PRESETS={"Slide – below 1650×640":(1650,640),
         "Slide – side 815×640":(815,640),
         "Hero 1920×480":(1920,480),
         "Social 1080×800":(1080,800),
         "Square 1080×1080":(1080,1080),
         "Custom…":None}
sel=st.selectbox("Canvas preset",PRESETS.keys())
W,H=(st.number_input("Width",400,4000,800,50),
     st.number_input("Height",300,3000,600,50)) if sel=="Custom…" else PRESETS[sel]

colA,colB=st.columns(2)
with colA:
    show_title=st.checkbox("Title",True)
    title     =st.text_input(" ",label_visibility="collapsed") if show_title else ""
    legend    =st.checkbox("Legend",True)
    font_sz   =st.slider("Font pt",8,32,14)
    grid      =st.checkbox("Gridlines")
    yzero     =st.checkbox("Y starts at 0")

with colB:
    # highlight options
    st.caption("Highlight band (brand gold)")
    bx0,bx1=st.text_input("X-start"),st.text_input("X-end")
    by0,by1=st.text_input("Y-start"),st.text_input("Y-end")
    op=st.slider("Opacity",0.05,0.5,0.12,0.01)
    vband=(bx0,bx1,op) if bx0 and bx1 else None
    hband=None
    if by0 and by1:
        try: hband=(float(by0),float(by1),op)
        except: st.error("Y highlight must be numeric")

st.markdown("#### Trend-lines")
numeric=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
trend_cfg=[]
if numeric:
    for i,col in enumerate(("1","2")):
        on=st.checkbox(f"Enable trend-line {col}")
        if on:
            s = st.selectbox(f"Series {col}",numeric,key=f"t{col}s")
            xs= st.selectbox(f"Start X {col}",df[xcol],key=f"t{col}xs")
            xe= st.selectbox(f"End X {col}",df[xcol],key=f"t{col}xe")
            colour = st.selectbox(f"Colour {col}",PALETTE,key=f"t{col}c",
                                  index=i if i<2 else 0)
            style  = st.selectbox(f"Style {col}",["solid","dash"],key=f"t{col}st")
            trend_cfg.append((s,xs,xe,style,colour))

# Data-label toggle
show_labels=False
label_mode="value"
if chart in {"Bar","Grouped Bar","Stacked Bar","Stacked Bar (h)","Pie","Donut","Radar"}:
    show_labels=st.checkbox("Show data-labels")
    if chart in {"Pie","Donut"} and show_labels:
        label_mode=st.radio("Label",["percent","value"],horizontal=True)

# ④ GENERATE  ──────────────────────────────────────────────────────────────
if st.button("Generate"):
    try:
        fig=None
        if chart=="Line":fig=px.line(df,x=xcol,y=ycols,title=title)
        elif chart=="Scatter":fig=px.scatter(df,x=xcol,y=ycols,title=title)
        elif chart=="Area":fig=px.area(df,x=xcol,y=ycols,title=title)
        elif chart=="Bar":fig=px.bar(df,x=xcol,y=ycols,title=title,
                                     text_auto=True if show_labels else None)
        elif chart=="Grouped Bar":fig=px.bar(df,x=xcol,y=ycols,barmode="group",
                                             title=title,text_auto=show_labels)
        elif chart=="Stacked Bar":fig=px.bar(df,x=xcol,y=ycols,barmode="stack",
                                             title=title,text_auto=show_labels)
        elif chart=="Stacked Bar (h)":fig=px.bar(df,x=ycols[0],y=xcol,orientation="h",
                                                 barmode="stack",title=title,
                                                 text_auto=show_labels)
        elif chart=="Histogram":fig=px.histogram(df,x=xcol,y=ycols[0],title=title)
        elif chart=="Box":fig=px.box(df,x=xcol,y=ycols[0],title=title)
        elif chart=="Pie":
            fig=px.pie(df,names=xcol,values=ycols[0],title=title,
                       hole=0,textinfo="percent+label" if label_mode=="percent"
                       else "value+label")
        elif chart=="Donut":
            fig=px.pie(df,names=xcol,values=ycols[0],hole=.4,title=title,
                       textinfo="percent+label" if label_mode=="percent"
                       else "value+label")
        elif chart=="Donut + Pie":
            fig=make_subplots(rows=1,cols=2,specs=[[{'type':'domain'}]*2])
            fig.add_trace(go.Pie(labels=df[xcol],values=df[ycols[0]],hole=.4,
                                 textinfo="percent+label" if label_mode=="percent"
                                 else "value+label"),1,1)
            fig.add_trace(go.Pie(labels=df[xcol],values=df[ycols[0]],
                                 textinfo="percent+label" if label_mode=="percent"
                                 else "value+label"),1,2)
            fig.update_layout(title=title)
        elif chart=="Radar":
            fig=go.Figure()
            for c in ycols:
                fig.add_trace(go.Scatterpolar(r=df[c],theta=df[xcol],
                                              fill="toself",name=c,
                                              text=df[c] if show_labels else None,
                                              textposition="top center"))
            fig.update_polars(radialaxis=dict(showticklabels=True,
                                              tickformat=".0%",
                                              gridcolor=GRAY),
                              angularaxis=dict(rotation=90))
            fig.update_layout(title=title)
        elif chart=="Heatmap":
            fig=px.density_heatmap(df,x=xcol,y=ycols[0],title=title,
                                   color_continuous_scale=px.colors.sequential.Greys)
        elif chart=="Waterfall":
            fig=go.Figure(go.Waterfall(x=df[xcol],y=df[ycols[0]],
                        measure=["relative"]*(len(df)-1)+["total"],
                        textposition="outside" if show_labels else "none"))
            fig.update_layout(title=title)
        elif chart=="Draw-down":fig=drawdown_fig(df,xcol,ycols[0],title)
        elif chart=="Line-Bar Combo":fig=combo_fig(df,xcol,bar_y,line_y,title)
        else: st.stop()

        # highlight
        if vband: vx0,vx1,op=vband; fig.add_vrect(x0=vx0,x1=vx1,fillcolor=GOLD,opacity=op,line_width=0)
        if hband: vy0,vy1,op=hband; fig.add_hrect(y0=vy0,y1=vy1,fillcolor=GOLD,opacity=op,line_width=0)

        # trend-lines
        for ser,xs,xe,style,col in trend_cfg:
            i0=df.index[df[xcol]==xs][0]; i1=df.index[df[xcol]==xe][-1]
            rng=np.arange(i0,i1+1); y=df.loc[rng,ser]
            m,b=np.polyfit(rng,y,1); trend=m*rng+b
            fig.add_scatter(x=df.loc[rng,xcol],y=trend,mode="lines",
                            name=f"{ser} trend",line=dict(color=col,dash=style))

        fig.update_layout(width=W,height=H,showlegend=legend,font=dict(size=font_sz))
        if yzero: fig.update_yaxes(rangemode="tozero")
        if grid: fig.update_xaxes(showgrid=True); fig.update_yaxes(showgrid=True)

        st.plotly_chart(fig,use_container_width=False,
            config={"modeBarButtonsToAdd":["drawrect","eraseshape"],"displaylogo":False})
        png=fig.to_image(format="png",width=W,height=H,scale=2); svg=fig.to_image(format="svg",width=W,height=H,scale=2)
        base=f"{slug(title or chart)}_{W}x{H}_{stamp()}"
        st.download_button("PNG",png,f"{base}.png","image/png")
        st.download_button("SVG",svg,f"{base}.svg","image/svg+xml")
    except Exception as e:
        st.error(e)
