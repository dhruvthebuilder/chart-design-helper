# true_beacon_radio.py  ·  v3.2.3 · 2025-10-23
# ------------------------------------------------
# Professional chart builder with manual date control - COMPLETE VERSION
# Clean interface, custom color control, advanced typography
# FIXED: Grouped bar chart functionality and None value handling
# RESTORED: Full tabbed interface with Upload, Paste, and Sample options
# FIXED: Stacked bar charts using proper Altair implementation
# PRESERVED: All export functionality including SVG, PNG, JSON, HTML

import io, re, random
from datetime import datetime
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
import altair as alt
import pycountry

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit.dataframe_util')

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chart Builder", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Configuration ─────────────────────────────────────────────────────────
FONT_FAMILIES = {
    "Red Hat Display": "Red Hat Display, sans-serif",
    "Inter": "Inter, sans-serif", 
    "Helvetica": "Helvetica, Arial, sans-serif",
    "Georgia": "Georgia, serif",
    "Roboto": "Roboto, sans-serif",
    "SF Pro": "-apple-system, BlinkMacSystemFont, sans-serif",
    "IBM Plex Sans": "IBM Plex Sans, sans-serif"
}

# Brand defaults
DEFAULT_COLORS = ["#987F2F", "#E5C96A", "#B6B6B6", "#FFFFFF"]
TRANS = "rgba(0,0,0,0)"

# Initialize session state
if "colors" not in st.session_state:
    st.session_state.colors = DEFAULT_COLORS.copy()

if "font_family" not in st.session_state:
    st.session_state.font_family = "Red Hat Display"

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "types_configured" not in st.session_state:
    st.session_state.types_configured = False

if "column_types" not in st.session_state:
    st.session_state.column_types = {}

if "preview" not in st.session_state:
    st.session_state.preview = pd.DataFrame()

# ── Core Functions ────────────────────────────────────────────────────────
def create_plotly_template(colors, font_family):
    """Create a custom Plotly template"""
    return go.layout.Template(
        layout=go.Layout(
            font=dict(
                family=FONT_FAMILIES.get(font_family, FONT_FAMILIES["Red Hat Display"]), 
                size=14, 
                color=colors[2] if len(colors) > 2 else "#B6B6B6"
            ),
            colorway=colors,
            plot_bgcolor=TRANS, 
            paper_bgcolor=TRANS,
            xaxis=dict(
                showgrid=False, 
                showline=True, 
                linecolor=colors[2] if len(colors) > 2 else "#B6B6B6",
                zeroline=True, 
                zerolinecolor=colors[2] if len(colors) > 2 else "#B6B6B6"
            ),
            yaxis=dict(
                showgrid=False, 
                showline=True, 
                linecolor=colors[2] if len(colors) > 2 else "#B6B6B6",
                zeroline=True, 
                zerolinecolor=colors[2] if len(colors) > 2 else "#B6B6B6"
            )
        )
    )

def configure_altair_theme(colors, font_family):
    """Configure Altair theme"""
    def custom_theme():
        return {
            'config': {
                'view': {'strokeWidth': 0, 'fill': TRANS},
                'mark': {'color': colors[0] if colors else "#987F2F"},
                'axis': {
                    'labelFont': FONT_FAMILIES.get(font_family, FONT_FAMILIES["Red Hat Display"]),
                    'titleFont': FONT_FAMILIES.get(font_family, FONT_FAMILIES["Red Hat Display"]),
                    'labelColor': colors[2] if len(colors) > 2 else "#B6B6B6",
                    'titleColor': colors[2] if len(colors) > 2 else "#B6B6B6",
                    'gridColor': 'transparent',
                    'domainColor': colors[2] if len(colors) > 2 else "#B6B6B6",
                    'tickColor': colors[2] if len(colors) > 2 else "#B6B6B6"
                },
                'legend': {
                    'labelFont': FONT_FAMILIES.get(font_family, FONT_FAMILIES["Red Hat Display"]),
                    'titleFont': FONT_FAMILIES.get(font_family, FONT_FAMILIES["Red Hat Display"]),
                    'labelColor': colors[2] if len(colors) > 2 else "#B6B6B6",
                    'titleColor': colors[2] if len(colors) > 2 else "#B6B6B6"
                },
                'title': {
                    'font': FONT_FAMILIES.get(font_family, FONT_FAMILIES["Red Hat Display"]),
                    'color': colors[2] if len(colors) > 2 else "#B6B6B6"
                },
                'range': {
                    'category': colors if colors else DEFAULT_COLORS
                }
            }
        }
    
    alt.themes.register('custom', custom_theme)
    alt.themes.enable('custom')

# Apply themes
pio.templates["custom"] = create_plotly_template(st.session_state.colors, st.session_state.font_family)
pio.templates.default = "custom"
configure_altair_theme(st.session_state.colors, st.session_state.font_family)

# ── Utility Functions ─────────────────────────────────────────────────────
def parse_manual(txt: str) -> pd.DataFrame:
    """Parse manually entered data without automatic type conversion"""
    delim = next((d for d in ("\t",";","|",",") if d in txt), ",")
    # Read everything as string initially
    df = pd.read_csv(io.StringIO(txt), delimiter=delim, dtype=str)
    return df

def sample_df() -> pd.DataFrame:
    """Generate sample data"""
    df = pd.DataFrame({
        "Date": ['2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01'],
        "Revenue": ['42000', '48000', '45000', '52000'],
        "Profit": ['8400', '11200', '9000', '13500'],
        "Growth": ['5.2', '14.3', '-6.3', '15.6']
    })
    return df

def sample_map_df() -> pd.DataFrame:
    """Generate sample map data"""
    df = pd.DataFrame({
        "Country": ["United States", "Germany", "France", "Japan", "Brazil", "India", "Australia"],
        "Market Share": ['35.2', '18.7', '12.4', '15.8', '8.3', '6.1', '3.5']
    })
    return df
def is_valid_hex_color(color):
    """Validate hex color format"""
    if not color:
        return False
    return bool(re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color))

def safe_str(value):
    """Safely convert value to string, handling None values"""
    if value is None or value == "None":
        return ""
    return str(value)

def clean_dataframe_for_display(df):
    """Clean dataframe for display"""
    if df.empty:
        return df
    
    df_display = df.copy()
    df_display.columns = [safe_str(col) for col in df_display.columns]
    
    for col in df_display.columns:
        # Format datetime columns for better display
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            # Handle NaT (Not a Time) values
            df_display[col] = df_display[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
        elif df_display[col].dtype == 'object':
            df_display[col] = df_display[col].fillna('').apply(safe_str)
        else:
            df_display[col] = df_display[col].fillna(0)
    
    return df_display

def detect_column_type(series):
    """Detect the likely data type of a column"""
    # Remove NaN values for analysis
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return "text"
    
    # Check if it's already numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check if it looks like integers
        try:
            if (series_clean == series_clean.astype(int)).all():
                return "integer"
        except:
            pass
        return "decimal"
    
    # For string columns, try to detect pattern
    # Check if all values can be converted to numbers
    try:
        pd.to_numeric(series_clean, errors='raise')
        # If successful, it's numeric data stored as text
        # Check if decimals exist
        if any('.' in str(x) for x in series_clean):
            return "decimal"
        return "integer"
    except:
        pass
    
    # Check for date patterns
    sample_values = series_clean.head(10).astype(str)
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'^\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'^\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        r'^\d{1,2}\s+\w+\s+\d{4}',  # D Month YYYY
    ]
    
    for pattern in date_patterns:
        matches = sum(1 for val in sample_values if re.match(pattern, str(val)))
        if matches >= len(sample_values) * 0.5:  # At least 50% match
            return "date"
    
    return "text"

def convert_column_type(series, target_type, date_format=None):
    """Convert a column to the specified type"""
    try:
        if target_type == "integer":
            # First convert to numeric, then to int
            numeric_series = pd.to_numeric(series, errors='coerce')
            return numeric_series.astype('Int64')  # Nullable integer type
        
        elif target_type == "decimal":
            return pd.to_numeric(series, errors='coerce')
        
        elif target_type == "date":
            if date_format and date_format != "auto":
                return pd.to_datetime(series, format=date_format, errors='coerce')
            else:
                return pd.to_datetime(series, errors='coerce')
        
        elif target_type == "text":
            return series.astype(str).replace('nan', '')
        
        else:
            return series
            
    except Exception as e:
        st.error(f"Error converting column: {str(e)}")
        return series

def reshape_for_grouped_bar(df, x_col, y_cols):
    """Reshape dataframe for grouped bar charts"""
    try:
        # Comprehensive validation
        if df.empty:
            return pd.DataFrame()
            
        if not x_col or x_col == "None" or x_col == "":
            return pd.DataFrame()
            
        if not y_cols or not any(col and col != "None" and col != "" for col in y_cols):
            return pd.DataFrame()
        
        # Ensure we're working with a copy
        df_copy = df.copy()
        
        # Clean column names
        df_copy.columns = [safe_str(col) for col in df_copy.columns]
        x_col_clean = safe_str(x_col).strip()
        y_cols_clean = [safe_str(col).strip() for col in y_cols if col and col != "None" and col != ""]
        
        # Validate columns exist
        if x_col_clean not in df_copy.columns:
            st.warning(f"X column '{x_col_clean}' not found in data")
            return pd.DataFrame()
            
        available_y_cols = [col for col in y_cols_clean if col in df_copy.columns]
        if not available_y_cols:
            st.warning("No valid Y columns found in data")
            return pd.DataFrame()
        
        # Filter to relevant columns only
        relevant_cols = [x_col_clean] + available_y_cols
        df_subset = df_copy[relevant_cols].copy()
        
        # Remove rows where x_col is null
        df_subset = df_subset.dropna(subset=[x_col_clean])
        
        if df_subset.empty:
            return pd.DataFrame()
        
        # Melt the dataframe
        df_melted = pd.melt(
            df_subset, 
            id_vars=[x_col_clean], 
            value_vars=available_y_cols,
            var_name='Series', 
            value_name='Value'
        )
        
        # Clean up the melted data
        df_melted = df_melted.dropna(subset=['Value'])  # Remove null values
        df_melted['Series'] = df_melted['Series'].apply(lambda x: safe_str(x).strip())
        df_melted[x_col_clean] = df_melted[x_col_clean].apply(lambda x: safe_str(x).strip())
        
        # Remove empty strings
        df_melted = df_melted[
            (df_melted['Series'] != '') & 
            (df_melted[x_col_clean] != '')
        ]
        
        return df_melted
    
    except Exception as e:
        st.error(f"Error reshaping data for grouped bar: {str(e)}")
        return pd.DataFrame()

def create_grouped_bar_chart(df, x_col, y_cols, title="", horizontal=False):
    """Create a grouped bar chart using Plotly"""
    try:
        # Ensure column names are strings
        x_col = safe_str(x_col)
        y_cols = [safe_str(col) for col in y_cols]
        title = safe_str(title)
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each y column
        for i, y_col in enumerate(y_cols):
            if y_col in df.columns:
                color = st.session_state.colors[i % len(st.session_state.colors)]
                
                if horizontal:
                    fig.add_trace(go.Bar(
                        y=df[x_col],
                        x=df[y_col],
                        name=y_col,
                        marker_color=color,
                        orientation='h'
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=df[x_col],
                        y=df[y_col],
                        name=y_col,
                        marker_color=color
                    ))
        
        # Update layout
        fig.update_layout(
            title=title,
            barmode='group',
            template='custom'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating grouped bar chart: {str(e)}")
        return None

def create_altair_chart(chart_type, df, x_col, y_cols, title, W, H, config):
    """Create Altair charts with advanced options"""
    df_melted = reshape_for_grouped_bar(df, x_col, y_cols)
    
    if df_melted.empty:
        return None
    
    base = alt.Chart(df_melted).properties(
        width=W * 0.85,
        height=H * 0.85,
        title=title if title else None
    )
    
    # Configure axes
    x_axis = alt.X(f'{x_col}:N', 
                   title=config['x_title'] if config['show_x_title'] else None,
                   axis=alt.Axis(labels=config['show_x_labels']))
    
    y_axis = alt.Y('Value:Q', 
                   title=config['y_title'] if config['show_y_title'] else None,
                   axis=alt.Axis(labels=config['show_y_labels']),
                   scale=alt.Scale(zero=config['y_zero']))
    
    if chart_type == "Grouped Bar":
        chart = base.mark_bar().encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color('Series:N', 
                          scale=alt.Scale(range=st.session_state.colors[:len(y_cols)]),
                          legend=alt.Legend() if config['show_legend'] else None),
            xOffset='Series:N'
        )
    else:  # Stacked Bar
        chart = base.mark_bar().encode(
            x=x_axis,
            y=alt.Y('sum(Value):Q', 
                   title=config['y_title'] if config['show_y_title'] else None,
                   axis=alt.Axis(labels=config['show_y_labels']),
                   scale=alt.Scale(zero=config['y_zero'])),
            color=alt.Color('Series:N',
                          scale=alt.Scale(range=st.session_state.colors[:len(y_cols)]),
                          legend=alt.Legend() if config['show_legend'] else None)
        )
    
    if config['show_values']:
        text = chart.mark_text(
            dy=-5 if chart_type == "Grouped Bar" else 0,
            fontSize=config['font_size'] * 0.8,
            font=FONT_FAMILIES[st.session_state.font_family]
        ).encode(
            text=alt.Text('Value:Q', format='.1f')
        )
        chart = chart + text
    
    return chart.configure(
        font=FONT_FAMILIES[st.session_state.font_family],
        fontSize=config['font_size']
    ).configure_axis(
        grid=config['show_grid'],
        gridColor='#E0E0E0' if config['show_grid'] else 'transparent'
    )

def create_map_chart(df, location_col, value_col, title="", width=800, height=600):
    """Create a choropleth map"""
    try:
        if df.empty or not location_col or not value_col:
            return None
            
        # Try to create a map based on location type
        location_col = safe_str(location_col)
        value_col = safe_str(value_col)
        title = safe_str(title)
        
        # Check if locations are country names/codes
        sample_locations = df[location_col].dropna().head(5).astype(str)
        
        # Try to match with country codes
        fig = px.choropleth(
            df,
            locations=location_col,
            color=value_col,
            title=title,
            width=width,
            height=height
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

# ── Main Application ──────────────────────────────────────────────────────
st.title("📊 Chart Builder")
st.markdown("*Professional data visualization with custom styling*")

# ── Sidebar: Data Upload & Configuration ──────────────────────────────────
with st.sidebar:
    st.header("📁 Data Input")
    
    tabs = st.tabs(["Upload", "Paste", "Sample"])
    
    with tabs[0]:
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    # Read all columns as strings initially
                    df_temp = pd.read_csv(uploaded_file, dtype=str)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    # Read Excel - all as strings to avoid automatic conversion
                    df_temp = pd.read_excel(uploaded_file, dtype=str)
                elif uploaded_file.name.endswith('.json'):
                    df_temp = pd.read_json(uploaded_file)
                    # Convert to strings
                    for col in df_temp.columns:
                        df_temp[col] = df_temp[col].astype(str)
                
                # Ensure column names are strings
                df_temp.columns = [safe_str(col) for col in df_temp.columns]
                
                # Store in session state
                st.session_state.df = df_temp
                st.session_state.types_configured = False
                
                st.success(f"✅ Loaded {len(st.session_state.df)} rows × {len(st.session_state.df.columns)} columns")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with tabs[1]:
        st.markdown("**Paste your data**")
        st.info("💡 Paste tab-separated, comma-separated, or other delimited data")
        
        text_input = st.text_area(
            "Data", 
            height=200,
            placeholder="Paste tab or comma-separated data here\nExample:\nName,Age,Score\nJohn,25,85\nJane,30,92",
            label_visibility="collapsed"
        )
        
        if st.button("📊 Process Data", type="primary"):
            if text_input.strip():
                try:
                    preview_df = parse_manual(text_input)
                    st.session_state.preview = preview_df
                    st.session_state.types_configured = False
                    st.success("✅ Data parsed successfully!")
                except Exception as e:
                    st.error(f"❌ Parse error: {str(e)}")
                    st.info("💡 Make sure your data is properly formatted with consistent delimiters")
        
        # Show preview and column naming if data was parsed
        if not st.session_state.preview.empty:
            st.markdown("**Preview of parsed data:**")
            preview_display = clean_dataframe_for_display(st.session_state.preview)
            st.dataframe(preview_display, use_container_width=True)
            
            with st.form("column_names"):
                st.markdown("**Name your columns**")
                columns = []
                cols_per_row = min(len(st.session_state.preview.columns), 4)
                col_layout = st.columns(cols_per_row)
                
                for i, col in enumerate(st.session_state.preview.columns):
                    with col_layout[i % cols_per_row]:
                        new_name = st.text_input(
                            f"Column {i+1}", 
                            value=safe_str(col), 
                            key=f"col_{i}"
                        )
                        columns.append(new_name if new_name else f"Column_{i+1}")
                
                if st.form_submit_button("✅ Apply Column Names", type="primary"):
                    st.session_state.preview.columns = columns
                    st.session_state.df = st.session_state.preview.copy()
                    st.session_state.types_configured = False
                    st.session_state.preview = pd.DataFrame()  # Clear preview
                    st.success("✅ Data ready - configure column types below")
                    st.rerun()
    
    with tabs[2]:
        st.markdown("**Sample datasets for testing**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Load Chart Sample", use_container_width=True):
                sample_data = sample_df()
                st.session_state.df = sample_data
                st.session_state.types_configured = False
                st.success("✅ Chart sample data loaded")
                st.info("💡 Tip: Set Date as 'date' type and the rest as 'decimal'")
                st.rerun()
        
        with col2:
            if st.button("🗺️ Load Map Sample", use_container_width=True):
                map_data = sample_map_df()
                st.session_state.df = map_data
                st.session_state.types_configured = False
                st.success("✅ Map sample data loaded")
                st.info("💡 Tip: Set Country as 'text' and Market Share as 'decimal'")
                st.rerun()
    
    # Check if data is loaded
    if st.session_state.df.empty:
        st.info("📊 Choose a data input method above to begin creating charts")
        st.stop()

# Continue only if we have data
if not st.session_state.df.empty:
    df = st.session_state.df.copy()
    
    # ── Column Type Configuration ─────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Column Configuration")
        
        if not st.session_state.types_configured:
            st.info("Configure your column types for better charts")
            
            # Auto-detect types
            detected_types = {}
            for col in df.columns:
                detected_types[col] = detect_column_type(df[col])
            
            # Allow user to modify types
            st.session_state.column_types = {}
            
            for col in df.columns:
                col_str = safe_str(col)
                st.session_state.column_types[col_str] = st.selectbox(
                    f"**{col_str}**",
                    ["text", "integer", "decimal", "date"],
                    index=["text", "integer", "decimal", "date"].index(detected_types[col]),
                    key=f"type_{col_str}"
                )
            
            if st.button("✅ Apply Column Types"):
                # Apply conversions
                for col, col_type in st.session_state.column_types.items():
                    if col in df.columns:
                        if col_type == "date":
                            # For date conversion, try automatic format detection
                            df[col] = convert_column_type(df[col], col_type)
                        else:
                            df[col] = convert_column_type(df[col], col_type)
                
                st.session_state.df = df
                st.session_state.types_configured = True
                st.success("✅ Column types applied!")
                st.rerun()
        
        else:
            st.success("✅ Column types configured")
            if st.button("🔄 Reconfigure Types"):
                st.session_state.types_configured = False
                st.rerun()
    
    # ── Style Configuration ───────────────────────────────────────────────
    with st.sidebar:
        st.header("🎨 Style Configuration")
        
        # Font selection
        font_family = st.selectbox(
            "Font Family",
            options=list(FONT_FAMILIES.keys()),
            index=list(FONT_FAMILIES.keys()).index(st.session_state.font_family),
            key="font_selector"
        )
        
        if font_family != st.session_state.font_family:
            st.session_state.font_family = font_family
            # Update themes
            pio.templates["custom"] = create_plotly_template(st.session_state.colors, font_family)
            configure_altair_theme(st.session_state.colors, font_family)
        
        # Color configuration
        st.subheader("Colors")
        
        colors = st.session_state.colors.copy()
        for i in range(len(colors)):
            new_color = st.color_picker(
                f"Color {i+1}",
                value=colors[i],
                key=f"color_{i}"
            )
            if is_valid_hex_color(new_color):
                colors[i] = new_color
        
        # Add/remove colors
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Color") and len(colors) < 10:
                colors.append("#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
        
        with col2:
            if st.button("➖ Remove Color") and len(colors) > 1:
                colors.pop()
        
        # Update colors if changed
        if colors != st.session_state.colors:
            st.session_state.colors = colors
            # Update themes
            pio.templates["custom"] = create_plotly_template(colors, st.session_state.font_family)
            configure_altair_theme(colors, st.session_state.font_family)
        
        # Reset to defaults
        if st.button("🔄 Reset to Defaults"):
            st.session_state.colors = DEFAULT_COLORS.copy()
            st.session_state.font_family = "Red Hat Display"
            pio.templates["custom"] = create_plotly_template(st.session_state.colors, st.session_state.font_family)
            configure_altair_theme(st.session_state.colors, st.session_state.font_family)
            st.rerun()

    # ── Main Content Area ─────────────────────────────────────────────────
    # Data preview
    st.header("📋 Data Preview")
    
    # Clean dataframe for display
    df_display = clean_dataframe_for_display(df)
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_usage:.1f} MB")
    
    # Display data
    st.dataframe(df_display, use_container_width=True, height=300)
    
    # ── Chart Configuration ───────────────────────────────────────────────
    st.header("📊 Chart Configuration")
    
    # Get column names as strings
    columns = [safe_str(col) for col in df.columns if col is not None]
    numeric_columns = []
    
    for col in df.columns:
        if col is not None and pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(safe_str(col))
    
    if not columns:
        st.error("No valid columns available for charting")
        st.stop()
        
    if not numeric_columns:
        st.warning("No numeric columns found. Some chart types may not be available.")
        numeric_columns = columns  # Fallback to all columns
    
    # Chart type selection
    chart_types = [
        "Line", "Bar", "Grouped Bar", "Stacked Bar", "Stacked Bar (Horizontal)",
        "Scatter", "Area", "Histogram", "Box", "Pie", "Donut", 
        "Radar", "Heatmap", "Waterfall", "Drawdown", "Line-Bar Combo", "Map"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox("Chart Type", chart_types, key="chart_type")
    
    with col2:
        title = st.text_input("Chart Title", value="", placeholder="Enter chart title")
    
    # Column selection based on chart type
    col1, col2 = st.columns(2)
    
    with col1:
        if chart_type in ["Pie", "Donut"]:
            x_col = st.selectbox("Category Column", columns, key="x_col", index=0 if columns else None)
            y_cols = [st.selectbox("Value Column", numeric_columns, key="y_col_single", index=0 if numeric_columns else None)]
        elif chart_type in ["Box", "Histogram"]:
            x_col = st.selectbox("Group Column (optional)", ["None"] + columns, key="x_col", index=0)
            if x_col == "None":
                x_col = None
            y_cols = [st.selectbox("Value Column", numeric_columns, key="y_col_single", index=0 if numeric_columns else None)]
        elif chart_type == "Map":
            x_col = st.selectbox("Location Column", columns, key="x_col", index=0 if columns else None)
            y_cols = [st.selectbox("Value Column", numeric_columns, key="y_col_single", index=0 if numeric_columns else None)]
        elif chart_type in ["Radar", "Heatmap"] or "Combo" in chart_type:
            x_col = st.selectbox("X-axis Column", columns, key="x_col", index=0 if columns else None)
            y_cols = st.multiselect("Y-axis Columns", numeric_columns, key="y_cols_multi", default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns[:1])
        else:
            x_col = st.selectbox("X-axis Column", columns, key="x_col", index=0 if columns else None)
            y_cols = st.multiselect("Y-axis Columns", numeric_columns, default=numeric_columns[:1] if numeric_columns else [], key="y_cols_multi")
    
    with col2:
        # Chart dimensions
        width = st.number_input("Width (px)", min_value=300, max_value=2000, value=800, step=50)
        height = st.number_input("Height (px)", min_value=200, max_value=1200, value=500, step=50)
        
        # Font size
        font_size = st.slider("Font Size", min_value=8, max_value=24, value=14)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            show_legend = st.checkbox("Show Legend", value=True)
            show_grid = st.checkbox("Show Grid", value=False)
            show_values = st.checkbox("Show Values on Chart", value=False)
            y_zero = st.checkbox("Start Y-axis at Zero", value=True)
        
        with col2:
            show_x_title = st.checkbox("Show X-axis Title", value=True)
            show_y_title = st.checkbox("Show Y-axis Title", value=True)
            show_x_labels = st.checkbox("Show X-axis Labels", value=True)
            show_y_labels = st.checkbox("Show Y-axis Labels", value=True)
        
        # Custom axis titles
        if show_x_title:
            x_title = st.text_input("X-axis Title", value=safe_str(x_col) if x_col else "")
        else:
            x_title = ""
        
        if show_y_title:
            if len(y_cols) == 1:
                y_title = st.text_input("Y-axis Title", value=safe_str(y_cols[0]) if y_cols else "")
            else:
                y_title = st.text_input("Y-axis Title", value="Value")
        else:
            y_title = ""

    # ── Chart Generation ──────────────────────────────────────────────────
    st.header("📈 Generated Chart")
    
    # Comprehensive input validation
    if not y_cols or any(col is None or col == "None" or col == "" for col in y_cols):
        st.warning("⚠️ Please select valid Y-axis columns")
        st.stop()
    
    if (x_col is None or x_col == "None" or x_col == "") and chart_type not in ["Histogram"]:
        st.warning("⚠️ Please select a valid X-axis column")
        st.stop()
    
    # Filter out None/invalid columns
    y_cols = [col for col in y_cols if col and col != "None" and col != "" and col in df.columns]
    if not y_cols:
        st.warning("⚠️ No valid Y-axis columns selected")
        st.stop()
    
    # Ensure x_col is valid if required
    if x_col and x_col != "None" and x_col != "" and x_col not in df.columns:
        st.warning(f"⚠️ Column '{x_col}' not found in data")
        st.stop()
    
    # Convert x_col to None if it's "None" string
    if x_col == "None" or x_col == "":
        x_col = None
    
    # Generate chart
    try:
        # Use Altair for grouped and stacked bar charts
        if chart_type in ["Grouped Bar", "Stacked Bar"]:
            # Create configuration object
            config = {
                'show_x_title': show_x_title,
                'show_y_title': show_y_title,
                'x_title': x_title,
                'y_title': y_title,
                'show_x_labels': show_x_labels,
                'show_y_labels': show_y_labels,
                'show_legend': show_legend,
                'show_grid': show_grid,
                'y_zero': y_zero,
                'show_values': show_values,
                'font_size': font_size
            }
            
            # Create Altair chart
            chart = create_altair_chart(chart_type, df, x_col, y_cols, title, width, height, config)
            if chart:
                st.altair_chart(chart, use_container_width=False)
                
                # Export options
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chart_{timestamp}"
                
                try:
                    chart_json = chart.to_json()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download JSON",
                            chart_json,
                            f"{filename}.json",
                            "application/json"
                        )
                    
                    with col2:
                        html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
                            <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
                            <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
                            <style>
                                body {{ margin: 0; padding: 20px; }}
                                #vis {{ margin: 0 auto; }}
                            </style>
                        </head>
                        <body>
                            <div id="vis"></div>
                            <script>
                                const spec = {chart_json};
                                vegaEmbed('#vis', spec, {{
                                    renderer: 'svg',
                                    actions: {{
                                        export: true,
                                        source: false,
                                        compiled: false,
                                        editor: false
                                    }}
                                }});
                            </script>
                        </body>
                        </html>
                        """
                        st.download_button(
                            "Download HTML",
                            html.encode(),
                            f"{filename}.html",
                            "text/html"
                        )
                except Exception as export_error:
                    st.warning(f"Export functionality not available: {str(export_error)}")
            else:
                st.error("Failed to create chart. Please check your data selection.")
        
        elif chart_type == "Map":
            # Generate map
            fig = create_map_chart(df, x_col, y_cols[0], title, width, height)
            if fig:
                st.plotly_chart(fig, use_container_width=False)
        
        else:
            # Generate Plotly chart
            fig = None
            
            if chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_cols, title=title)
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_cols, title=title)
            elif chart_type == "Area":
                fig = px.area(df, x=x_col, y=y_cols, title=title)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_cols, title=title, 
                           text_auto=show_values)
            elif chart_type == "Stacked Bar (Horizontal)":
                df_melted = reshape_for_grouped_bar(df, x_col, y_cols)
                if not df_melted.empty:
                    fig = px.bar(df_melted, y=x_col, x='Value', color='Series',
                               orientation='h', title=title, text_auto=show_values)
            elif chart_type == "Histogram":
                if x_col:
                    fig = px.histogram(df, x=x_col, y=y_cols[0], title=title)
                else:
                    fig = px.histogram(df, x=y_cols[0], title=title)
            elif chart_type == "Box":
                fig = px.box(df, x=x_col, y=y_cols[0], title=title)
            elif chart_type == "Pie":
                fig = px.pie(df, names=x_col, values=y_cols[0], title=title)
                if show_values:
                    fig.update_traces(textinfo='percent+label')
            elif chart_type == "Donut":
                fig = px.pie(df, names=x_col, values=y_cols[0], hole=.4, title=title)
                if show_values:
                    fig.update_traces(textinfo='percent+label')
            elif chart_type == "Radar":
                fig = go.Figure()
                for i, col in enumerate(y_cols):
                    fig.add_trace(go.Scatterpolar(
                        r=df[col], theta=df[x_col], fill='toself', name=col,
                        line_color=st.session_state.colors[i % len(st.session_state.colors)]
                    ))
                fig.update_layout(title=title, 
                                polar=dict(radialaxis=dict(visible=True)))
            elif chart_type == "Heatmap":
                if len(y_cols) > 1:
                    corr = df[y_cols].corr()
                    fig = px.imshow(corr, title=title or "Correlation Matrix",
                                  text_auto=True)
                else:
                    fig = px.density_heatmap(df, x=x_col, y=y_cols[0], title=title)
            elif chart_type == "Waterfall":
                fig = go.Figure(go.Waterfall(
                    x=df[x_col], y=df[y_cols[0]],
                    textposition="outside" if show_values else "none"
                ))
                fig.update_layout(title=title)
            elif chart_type == "Drawdown":
                # Calculate drawdown
                values = df[y_cols[0]]
                cumulative = (1 + values/100).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                
                fig = px.area(x=df[x_col], y=drawdown, title=title)
                fig.update_yaxis(tickformat='.1%')
            elif chart_type == "Line-Bar Combo":
                if len(y_cols) >= 2:
                    fig = go.Figure()
                    fig.add_bar(x=df[x_col], y=df[y_cols[0]], name=y_cols[0],
                              marker_color=st.session_state.colors[0])
                    fig.add_scatter(x=df[x_col], y=df[y_cols[1]], name=y_cols[1],
                                  yaxis="y2", mode='lines+markers',
                                  line_color=st.session_state.colors[1])
                    fig.update_layout(
                        title=title,
                        yaxis2=dict(overlaying='y', side='right')
                    )
            
            if fig:
                # Apply layout settings
                fig.update_layout(
                    width=width,
                    height=height,
                    showlegend=show_legend,
                    font=dict(
                        family=FONT_FAMILIES[st.session_state.font_family],
                        size=font_size,
                        color=st.session_state.colors[2] if len(st.session_state.colors) > 2 else "#666"
                    )
                )
                
                # Update axes
                fig.update_xaxes(
                    title_text=x_title if show_x_title else None,
                    showticklabels=show_x_labels,
                    showgrid=show_grid
                )
                
                fig.update_yaxes(
                    title_text=y_title if show_y_title else None,
                    showticklabels=show_y_labels,
                    showgrid=show_grid,
                    rangemode="tozero" if y_zero else "normal"
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=False)
                
                # Export options
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chart_{timestamp}"
                
                try:
                    # Generate exports
                    png_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
                    svg_bytes = fig.to_image(format="svg", width=width, height=height)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download PNG",
                            png_bytes,
                            f"{filename}.png",
                            "image/png"
                        )
                    
                    with col2:
                        st.download_button(
                            "Download SVG",
                            svg_bytes,
                            f"{filename}.svg",
                            "image/svg+xml"
                        )
                except Exception as export_error:
                    st.warning(f"Export functionality not available: {str(export_error)}")
                        
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")
        st.info("Check your data selection and try again")

# Footer
st.markdown("---")
st.markdown(
    """<div style='text-align:center;color:#999;font-size:12px;'>
    Chart Builder v3.2.3 · Professional visualization tool · Complete with paste functionality and fixed stacked bars
    </div>""",
    unsafe_allow_html=True
)
