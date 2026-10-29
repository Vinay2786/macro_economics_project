"""
Streamlit Dashboard:
ML-Based Analysis and Prediction of Investment Response to GDP Changes Using the Accelerator Principle

Modifications:
1. Complete, evenly shaped rectangular Title box centered across content width.
2. Graph 1: Removed all blue circular data markers and their legend entry; kept only the green fitted regression line.
3. Graph 4: Repositioned the positive and negative error annotations to a dedicated area above the plot grid to avoid any data overlap.
4. Preserved clean 2 x 2 matrix of equal-sized graphs with high-contrast readable styling.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from model import load_and_validate_data, run_accelerator_model

# Page configuration
st.set_page_config(
    page_title="Macroeconomics Accelerator Principle Project",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom high-contrast CSS ensuring total legibility and balanced rectangular geometry
st.markdown("""
<style>
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .dashboard-title-container {
        width: 100%;
        box-sizing: border-box;
        background-color: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 8px;
        padding: 18px 24px;
        margin-top: 0.3rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .dashboard-title-text {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a !important;
        line-height: 1.35;
        margin: 0;
        padding: 0;
        text-align: center;
        width: 100%;
    }
    .info-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        height: 100%;
        color: #0f172a !important;
    }
    .card-heading {
        font-size: 0.88rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .eq-text {
        font-size: 1.05rem;
        font-family: "Courier New", Courier, monospace;
        font-weight: 700;
        color: #0f172a;
        margin: 3px 0;
    }
    .param-def {
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.4;
        margin-top: 4px;
    }
    .coeff-text {
        font-size: 0.9rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-top: 4px;
    }
    .error-text {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.45;
        margin-top: 4px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data & Model (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_data_and_fit():
    clean_df, _ = load_and_validate_data()
    results = run_accelerator_model(clean_df)
    return results

results = load_data_and_fit()

train_df = results['train_df']
test_df = results['test_df']
full_df = results['full_pred_df']
modeling_df = full_df[full_df['Year'] >= 1991].copy()

alpha = results['metrics']['alpha']
beta = results['metrics']['beta']
alpha_billion = alpha / 1e9

# ---------------------------------------------------------
# 1. Project Title (Evenly Shaped Rectangular Banner)
# ---------------------------------------------------------
st.markdown("""
<div class="dashboard-title-container">
    <div class="dashboard-title-text">
        ML-Based Analysis and Prediction of Investment Response to GDP Changes Using the Accelerator Principle
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Model Information & Error-Term Formula Cards
# ---------------------------------------------------------
col_top_left, col_top_right = st.columns([1.18, 0.82])

with col_top_left:
    st.markdown(f"""
    <div class="info-card">
        <div class="card-heading">Model Specification & Fitted Parameters</div>
        <div class="eq-text">ŷ = b₀ + b₁x &nbsp;&nbsp;|&nbsp;&nbsp; GFCF(t) = α + β × ΔGDP(t) + ε(t)</div>
        <div class="param-def">
            • <strong>GFCF(t):</strong> actual real gross fixed capital formation in year t<br>
            • <strong>ΔGDP(t) = GDP(t) − GDP(t−1):</strong> annual change in real GDP<br>
            • <strong>α, β:</strong> fitted intercept and regression slope &nbsp;|&nbsp; <strong>ε(t):</strong> error term
        </div>
        <div class="coeff-text">
            Fitted Intercept (α): {alpha:,.2f} USD (${alpha_billion:.2f}B) &nbsp;|&nbsp; Fitted Slope (β): {beta:.6f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_top_right:
    st.markdown("""
    <div class="info-card">
        <div class="card-heading">Error-Term Formula</div>
        <div class="eq-text">ε(t) = Actual GFCF(t) − Predicted GFCF(t)</div>
        <div class="error-text">
            The error term represents the portion of investment not explained by the regression model. Positive errors indicate underprediction, while negative errors indicate overprediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2b. Collapsible Explanation: Calculation of Alpha and Beta
# ---------------------------------------------------------
with st.expander("How Are Alpha and Beta Calculated?", expanded=False):
    col_exp_b, col_exp_a = st.columns(2)
    with col_exp_b:
        st.markdown("**Slope Formula (β):**")
        st.code("β = Σ[(xᵢ − x̄)(yᵢ − ȳ)] / Σ[(xᵢ − x̄)²]", language="text")
        st.latex(r"\beta = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}")
        st.markdown("""
        - **β:** Fitted regression slope.
        - **xᵢ:** Change in GDP for year i.
        - **yᵢ:** Actual GFCF for year i.
        - **x̄:** Mean of the GDP changes in the training data.
        - **ȳ:** Mean of GFCF in the training data.
        - **Σ:** Summation over all training observations.
        """)
    with col_exp_a:
        st.markdown("**Intercept Formula (α):**")
        st.code("α = ȳ − βx̄", language="text")
        st.latex(r"\alpha = \bar{y} - \beta\bar{x}")
        st.markdown("""
        - **α:** Fitted regression intercept.
        - **ȳ:** Mean GFCF in the training data.
        - **β:** Fitted regression slope.
        - **x̄:** Mean GDP change in the training data.
        """)
    st.markdown("""
    <div style="font-size: 0.88rem; color: #334155; margin-top: 6px; padding-top: 6px; border-top: 1px solid #e2e8f0;">
        Both coefficients are calculated using the training period, 1991–2017, only. If all annual observations are available, the training sample contains 27 observations. The fitted coefficients remain fixed when predicting the testing period, 2018–2025.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper: Common Readable Layout Configuration & 2D Zoom/Pan
# ---------------------------------------------------------
GRAPH_HEIGHT = 420

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': [
        'pan2d',
        'zoom2d',
        'zoomIn2d',
        'zoomOut2d',
        'autoScale2d',
        'resetScale2d'
    ]
}

def apply_readable_layout(fig, title_text, x_title, y_title):
    fig.update_layout(
        dragmode='pan',
        title=dict(
            text=f"<b>{title_text}</b>",
            font=dict(size=14, color='#0f172a', family='Arial, sans-serif')
        ),
        xaxis=dict(
            title=dict(text=f"<b>{x_title}</b>", font=dict(size=12, color='#0f172a')),
            tickfont=dict(size=11, color='#0f172a', weight='bold'),
            gridcolor='#e2e8f0',
            zeroline=True,
            zerolinecolor='#cbd5e1',
            zerolinewidth=1.5,
            fixedrange=False
        ),
        yaxis=dict(
            title=dict(text=f"<b>{y_title}</b>", font=dict(size=12, color='#0f172a')),
            tickfont=dict(size=11, color='#0f172a', weight='bold'),
            gridcolor='#e2e8f0',
            zeroline=True,
            zerolinecolor='#cbd5e1',
            zerolinewidth=1.5,
            fixedrange=False
        ),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor='#ffffff',
            bordercolor='#cbd5e1',
            borderwidth=1,
            font=dict(size=10, color='#0f172a', weight='bold')
        ),
        margin=dict(l=65, r=25, t=80, b=55),
        height=GRAPH_HEIGHT
    )
    return fig

# ---------------------------------------------------------
# 3. Exactly Four Graphs in a 2 x 2 Matrix
# ---------------------------------------------------------

# =========================================================
# Top-Left: Graph 1 (Actual GFCF vs Change in GDP)
# Blue dots removed; only green fitted regression line shown.
# =========================================================
fig1 = go.Figure()

min_delta = modeling_df['Delta_GDP'].min() / 1e9
max_delta = modeling_df['Delta_GDP'].max() / 1e9
x_line = np.linspace(min_delta - 10, max_delta + 10, 100)
y_line = (alpha / 1e9) + beta * x_line

# Only the green fitted regression line (no blue circular dots or scatter points)
fig1.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode='lines',
    name='Fitted Regression Line (1991–2017)',
    line=dict(color='#16a34a', width=2.5),
    hoverinfo='skip'
))

apply_readable_layout(
    fig1,
    title_text="Graph 1: Actual GFCF vs Change in GDP",
    x_title="Change in GDP: ΔGDP (Billion Constant 2015 USD)",
    y_title="Actual GFCF (Billion Constant 2015 USD)"
)

# =========================================================
# Top-Right: Graph 2 (Actual vs Predicted GFCF — Training 1991–2017)
# =========================================================
fig2 = go.Figure()

# Actual GFCF (Training)
fig2.add_trace(go.Scatter(
    x=train_df['Year'],
    y=train_df['GFCF'] / 1e9,
    mode='lines+markers',
    name='Actual GFCF',
    line=dict(color='#1d4ed8', width=2.4),
    marker=dict(size=6, symbol='circle', color='#1d4ed8'),
    customdata=np.stack((
        train_df['Year'],
        train_df['GFCF'] / 1e9,
        train_df['pred_GFCF'] / 1e9,
        train_df['residual'] / 1e9
    ), axis=-1),
    hovertemplate=(
        '<b>Year %{x:.0f} (Train)</b><br>'
        'Actual GFCF: $%{customdata[1]:,.2f}B<br>'
        'Predicted GFCF: $%{customdata[2]:,.2f}B<br>'
        'Error ε(t): $%{customdata[3]:,.2f}B<extra></extra>'
    )
))

# Predicted GFCF (Training)
fig2.add_trace(go.Scatter(
    x=train_df['Year'],
    y=train_df['pred_GFCF'] / 1e9,
    mode='lines+markers',
    name='Predicted GFCF',
    line=dict(color='#ea580c', width=2.2, dash='dash'),
    marker=dict(size=6, symbol='square', color='#ea580c'),
    customdata=np.stack((
        train_df['Year'],
        train_df['GFCF'] / 1e9,
        train_df['pred_GFCF'] / 1e9,
        train_df['residual'] / 1e9
    ), axis=-1),
    hovertemplate=(
        '<b>Year %{x:.0f} (Train)</b><br>'
        'Actual GFCF: $%{customdata[1]:,.2f}B<br>'
        'Predicted GFCF: $%{y:,.2f}B<br>'
        'Error ε(t): $%{customdata[3]:,.2f}B<extra></extra>'
    )
))

apply_readable_layout(
    fig2,
    title_text="Graph 2: Actual vs Predicted GFCF — Training Period (1991–2017)",
    x_title="Year",
    y_title="GFCF (Billion Constant 2015 USD)"
)
fig2.update_xaxes(tickmode='linear', tick0=1991, dtick=3)

# =========================================================
# Bottom-Left: Graph 3 (Actual vs Predicted GFCF — Testing 2018–2025)
# =========================================================
fig3 = go.Figure()

# Actual GFCF (Testing)
fig3.add_trace(go.Scatter(
    x=test_df['Year'],
    y=test_df['GFCF'] / 1e9,
    mode='lines+markers',
    name='Actual GFCF',
    line=dict(color='#1d4ed8', width=2.4),
    marker=dict(size=7, symbol='circle', color='#1d4ed8'),
    customdata=np.stack((
        test_df['Year'],
        test_df['GFCF'] / 1e9,
        test_df['pred_GFCF'] / 1e9,
        test_df['residual'] / 1e9
    ), axis=-1),
    hovertemplate=(
        '<b>Year %{x:.0f} (Test)</b><br>'
        'Actual GFCF: $%{customdata[1]:,.2f}B<br>'
        'Predicted GFCF: $%{customdata[2]:,.2f}B<br>'
        'Error ε(t): $%{customdata[3]:,.2f}B<extra></extra>'
    )
))

# Predicted GFCF (Testing)
fig3.add_trace(go.Scatter(
    x=test_df['Year'],
    y=test_df['pred_GFCF'] / 1e9,
    mode='lines+markers',
    name='Predicted GFCF',
    line=dict(color='#dc2626', width=2.2, dash='dash'),
    marker=dict(size=7, symbol='square', color='#dc2626'),
    customdata=np.stack((
        test_df['Year'],
        test_df['GFCF'] / 1e9,
        test_df['pred_GFCF'] / 1e9,
        test_df['residual'] / 1e9
    ), axis=-1),
    hovertemplate=(
        '<b>Year %{x:.0f} (Test)</b><br>'
        'Actual GFCF: $%{customdata[1]:,.2f}B<br>'
        'Predicted GFCF: $%{y:,.2f}B<br>'
        'Error ε(t): $%{customdata[3]:,.2f}B<extra></extra>'
    )
))

# Emphasized points for 2020 & 2021
covid_df = test_df[test_df['Year'].isin([2020, 2021])].copy()

# Emphasized actual observations
fig3.add_trace(go.Scatter(
    x=covid_df['Year'],
    y=covid_df['GFCF'] / 1e9,
    mode='markers',
    name='COVID-19 Actual',
    marker=dict(size=13, color='#1d4ed8', symbol='circle', line=dict(width=2.5, color='#d97706')),
    showlegend=False,
    hovertemplate='<b>Year %{x:.0f} (COVID-19 Period)</b><br>Actual GFCF: $%{y:,.2f}B<extra></extra>'
))

# Emphasized predicted observations
fig3.add_trace(go.Scatter(
    x=covid_df['Year'],
    y=covid_df['pred_GFCF'] / 1e9,
    mode='markers',
    name='COVID-19 Predicted',
    marker=dict(size=13, color='#dc2626', symbol='square', line=dict(width=2.5, color='#d97706')),
    showlegend=False,
    hovertemplate='<b>Year %{x:.0f} (COVID-19 Period)</b><br>Predicted GFCF: $%{y:,.2f}B<extra></extra>'
))

# Subtle COVID-19 period background band
fig3.add_vrect(
    x0=2019.5, x1=2021.5,
    fillcolor='rgba(217, 119, 6, 0.08)',
    layer='below',
    line_width=1,
    line_dash='dot',
    line_color='#d97706'
)

# Clearly visible, unobtrusive annotation identifying the COVID-19 period
fig3.add_annotation(
    x=2020.5,
    y=covid_df['GFCF'].max() / 1e9 + 60,
    text='<b>COVID-19 period</b>',
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1.5,
    arrowcolor='#b45309',
    ax=0,
    ay=-32,
    font=dict(size=11, color='#92400e'),
    bgcolor='rgba(254, 243, 199, 0.95)',
    bordercolor='#f59e0b',
    borderwidth=1,
    borderpad=4
)

apply_readable_layout(
    fig3,
    title_text="Graph 3: Actual vs Predicted GFCF — Testing Period (2018–2025)",
    x_title="Year",
    y_title="GFCF (Billion Constant 2015 USD)"
)
fig3.update_xaxes(
    tickmode='array',
    tickvals=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    # Bold-highlight the years 2020 and 2021 on the x-axis
    ticktext=['2018', '2019', '<b>2020</b>', '<b>2021</b>', '2022', '2023', '2024', '2025']
)

# =========================================================
# Bottom-Right: Graph 4 (Error Term vs Year 1991–2025)
# Explanatory annotations repositioned above the plot area.
# =========================================================
fig4 = go.Figure()

error_labels = [
    'Underprediction (Actual > Predicted)' if r > 0 else 'Overprediction (Actual < Predicted)'
    for r in modeling_df['residual']
]

# Annual error series line with markers (full 35-year period: 1991–2025)
fig4.add_trace(go.Scatter(
    x=modeling_df['Year'],
    y=modeling_df['residual'] / 1e9,
    mode='lines+markers',
    name='Error ε(t) = Actual − Predicted',
    line=dict(color='#7c3aed', width=2.2),
    marker=dict(size=6, symbol='circle', color='#7c3aed'),
    customdata=error_labels,
    hovertemplate=(
        '<b>Year %{x:.0f}</b><br>'
        'Error ε(t): $%{y:,.2f}B<br>'
        '%{customdata}<extra></extra>'
    )
))

# Horizontal zero-error reference line
fig4.add_hline(
    y=0,
    line_dash='dash',
    line_color='#475569',
    line_width=1.5
)

# Place both explanatory annotations in dedicated space above the plotting region
# Positioned at y=1.04 above the plot grid to guarantee zero data overlap
fig4.add_annotation(
    xref="paper", yref="paper",
    x=0.0, y=1.04,
    xanchor="left", yanchor="bottom",
    text="<b>+ Error: Underprediction (Actual > Predicted)</b>",
    showarrow=False,
    font=dict(size=10, color="#15803d", weight='bold'),
    bgcolor="rgba(240, 253, 244, 0.95)",
    bordercolor="#86efac",
    borderwidth=1,
    borderpad=3
)
fig4.add_annotation(
    xref="paper", yref="paper",
    x=0.48, y=1.04,
    xanchor="left", yanchor="bottom",
    text="<b>− Error: Overprediction (Actual < Predicted)</b>",
    showarrow=False,
    font=dict(size=10, color="#b91c1c", weight='bold'),
    bgcolor="rgba(254, 242, 242, 0.95)",
    bordercolor="#fca5a5",
    borderwidth=1,
    borderpad=3
)

apply_readable_layout(
    fig4,
    title_text="Graph 4: Regression Error Term vs Year (1991–2025)",
    x_title="Year",
    y_title="Error: Actual − Predicted GFCF ($B)"
)
fig4.update_xaxes(tickmode='linear', tick0=1991, dtick=4)

# ---------------------------------------------------------
# 4. Render 2 x 2 Equal-Sized Matrix Layout
# ---------------------------------------------------------
# Row 1: Top-Left (Graph 1) & Top-Right (Graph 2)
row1_left, row1_right = st.columns(2)
with row1_left:
    st.plotly_chart(fig1, config=PLOTLY_CONFIG, width='stretch')
with row1_right:
    st.plotly_chart(fig2, config=PLOTLY_CONFIG, width='stretch')

# Row 2: Bottom-Left (Graph 4) & Bottom-Right (Graph 3)
row2_left, row2_right = st.columns(2)
with row2_left:
    st.plotly_chart(fig4, config=PLOTLY_CONFIG, width='stretch')
with row2_right:
    st.plotly_chart(fig3, config=PLOTLY_CONFIG, width='stretch')
