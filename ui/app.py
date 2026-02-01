"""
Meridian Dashboard - Enterprise ML Analytics Platform

This is the main Streamlit application providing interactive visualization
for Uplift Modeling, Demand Forecasting, and Price Optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from typing import Optional
import json

# Page configuration
st.set_page_config(
    page_title="Meridian Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/vasdz/Meridian',
        'Report a bug': 'https://github.com/vasdz/Meridian/issues',
        'About': '# Meridian\nEnterprise ML Analytics Platform for Retail'
    }
)

# Custom CSS for enterprise look
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --background-color: #0e1117;
        --card-background: #1e2130;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e2130, #2a2f42);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-delta-positive {
        color: #00cc66;
        font-size: 0.9rem;
    }
    
    .metric-delta-negative {
        color: #ff4444;
        font-size: 0.9rem;
    }
    
    /* Status indicators */
    .status-healthy {
        background-color: #00cc66;
        border-radius: 50%;
        width: 12px;
        height: 12px;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-warning {
        background-color: #ffcc00;
        border-radius: 50%;
        width: 12px;
        height: 12px;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-error {
        background-color: #ff4444;
        border-radius: 50%;
        width: 12px;
        height: 12px;
        display: inline-block;
        margin-right: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e2130, #0e1117);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #1a5a8a);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #888;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(145deg, #1e3a5f, #0d1f33);
        border-left: 4px solid #1f77b4;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1f77b4, transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> dict:
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "details": response.json()}
        return {"status": "unhealthy", "details": {"error": f"Status {response.status_code}"}}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "details": {"error": str(e)}}

def generate_sample_data():
    """Generate sample data for visualization."""
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=365, freq='D')

    # Demand data
    trend = np.linspace(100, 150, 365)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.normal(0, 10, 365)
    demand = trend + seasonality + noise

    demand_df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'forecast': demand + np.random.normal(0, 5, 365),
        'lower_bound': demand - 15,
        'upper_bound': demand + 15
    })

    return demand_df

# Sidebar
with st.sidebar:
    st.markdown('<h1 class="main-header">🎯 Meridian</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML Analytics Platform</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # API Status
    health = check_api_health()
    if health["status"] == "healthy":
        st.markdown('<span class="status-healthy"></span> API Connected', unsafe_allow_html=True)
    elif health["status"] == "unhealthy":
        st.markdown('<span class="status-warning"></span> API Degraded', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-error"></span> API Offline', unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    page = st.selectbox(
        "📊 Navigation",
        ["🏠 Overview", "🎯 Uplift Modeling", "📈 Demand Forecasting", "💰 Price Optimization", "🧪 A/B Testing", "⚙️ Settings"]
    )

    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh Data"):
        st.rerun()
    if st.button("📥 Export Report"):
        st.info("Report generation started...")

    st.markdown("---")

    # User info
    st.markdown("### 👤 User")
    st.markdown("**Admin User**")
    st.markdown("admin@meridian.io")

    st.markdown("---")
    st.markdown("v1.0.0 • © 2026 Meridian")

# Main content based on selected page
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">Dashboard Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time analytics and KPIs</p>', unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Revenue",
            value="$2.4M",
            delta="+12.5%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Uplift Impact",
            value="$340K",
            delta="+8.2%",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Forecast Accuracy",
            value="94.2%",
            delta="+2.1%",
            delta_color="normal"
        )

    with col4:
        st.metric(
            label="Active Experiments",
            value="12",
            delta="+3",
            delta_color="normal"
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Revenue Trend")
        demand_df = generate_sample_data()
        fig = px.area(
            demand_df, x='date', y='demand',
            title='Daily Revenue (Last 365 Days)',
            template='plotly_dark'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            title_font_color='white',
            xaxis_title='',
            yaxis_title='Revenue ($K)'
        )
        fig.update_traces(fill='tozeroy', line_color='#1f77b4', fillcolor='rgba(31,119,180,0.3)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Uplift Distribution")
        uplift_data = pd.DataFrame({
            'segment': ['High Value', 'Medium Value', 'Low Value', 'New Customers'],
            'uplift': [0.15, 0.08, 0.03, 0.12],
            'customers': [5000, 15000, 25000, 8000]
        })
        fig = px.bar(
            uplift_data, x='segment', y='uplift',
            color='customers',
            title='CATE by Customer Segment',
            template='plotly_dark',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # System health
    st.subheader("🖥️ System Health")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">API Latency (P95)</div>
            <div class="metric-value">45ms</div>
            <div class="metric-delta-positive">▼ 12ms from yesterday</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Inference Time</div>
            <div class="metric-value">120ms</div>
            <div class="metric-delta-positive">▼ 8ms from yesterday</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Error Rate</div>
            <div class="metric-value">0.02%</div>
            <div class="metric-delta-positive">▼ 0.01% from yesterday</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "🎯 Uplift Modeling":
    st.markdown('<h1 class="main-header">Uplift Modeling</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict individual treatment effects (CATE)</p>', unsafe_allow_html=True)

    tabs = st.tabs(["📊 Predictions", "📈 Analysis", "⚙️ Model Config"])

    with tabs[0]:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Customer Features")

            customer_id = st.text_input("Customer ID", value="cust_12345")
            age = st.slider("Age", 18, 80, 35)
            segment = st.selectbox("Segment", ["premium", "regular", "new"])
            recency = st.number_input("Days Since Last Purchase", 0, 365, 14)
            frequency = st.number_input("Purchase Frequency (per month)", 0.0, 30.0, 2.5)
            monetary = st.number_input("Average Order Value ($)", 0.0, 1000.0, 75.0)

            treatment = st.selectbox("Treatment Type", [
                "discount_10pct",
                "discount_20pct",
                "free_shipping",
                "loyalty_points"
            ])

            if st.button("🎯 Predict Uplift", type="primary"):
                with st.spinner("Running causal inference..."):
                    # Simulated prediction
                    import time
                    time.sleep(1)

                    cate = np.random.uniform(0.05, 0.20)
                    ci_lower = cate - 0.03
                    ci_upper = cate + 0.03

                    st.success("Prediction complete!")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("CATE", f"{cate:.1%}")
                    with col_b:
                        recommendation = "TREAT" if cate > 0.1 else "CONTROL"
                        st.metric("Recommendation", recommendation)

                    st.info(f"95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

        with col2:
            st.subheader("CATE Distribution")

            # Generate sample CATE distribution
            np.random.seed(42)
            cate_values = np.random.beta(2, 5, 1000) * 0.3

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=cate_values,
                nbinsx=50,
                marker_color='#1f77b4',
                opacity=0.7,
                name='CATE Distribution'
            ))
            fig.add_vline(x=0.1, line_dash="dash", line_color="red",
                         annotation_text="Treatment Threshold")
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title='CATE Distribution Across Customers',
                xaxis_title='CATE Value',
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Qini Curve Analysis")

        # Generate Qini curve data
        population = np.linspace(0, 1, 100)
        random_model = population
        perfect_model = np.sqrt(population)
        our_model = 0.3 * perfect_model + 0.7 * random_model

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=population, y=random_model, name='Random',
                                 line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=population, y=perfect_model, name='Perfect',
                                 line=dict(dash='dot', color='green')))
        fig.add_trace(go.Scatter(x=population, y=our_model, name='Causal Forest',
                                 line=dict(color='#1f77b4', width=3)))
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title='Qini Curve',
            xaxis_title='Population Targeted (%)',
            yaxis_title='Cumulative Uplift'
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AUUC Score", "0.342")
        with col2:
            st.metric("Qini Coefficient", "0.287")
        with col3:
            st.metric("Optimal Targeting %", "32%")

    with tabs[2]:
        st.subheader("Model Configuration")

        model_type = st.selectbox("Model Type", [
            "Causal Forest (EconML)",
            "X-Learner",
            "T-Learner",
            "S-Learner"
        ])

        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input("Number of Estimators", 100, 1000, 500)
            max_depth = st.number_input("Max Depth", 3, 20, 10)
        with col2:
            min_samples_leaf = st.number_input("Min Samples Leaf", 1, 100, 10)
            cv_folds = st.number_input("CV Folds", 2, 10, 5)

        if st.button("💾 Save Configuration"):
            st.success("Configuration saved successfully!")

elif page == "📈 Demand Forecasting":
    st.markdown('<h1 class="main-header">Demand Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-horizon probabilistic predictions</p>', unsafe_allow_html=True)

    tabs = st.tabs(["📊 Forecast", "📉 Backtesting", "⚙️ Configuration"])

    with tabs[0]:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Parameters")
            product_id = st.text_input("Product/SKU ID", "SKU-12345")
            store_id = st.selectbox("Store", ["Store A", "Store B", "Store C"])
            horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
            confidence = st.slider("Confidence Level", 80, 99, 95)

            if st.button("🔮 Generate Forecast", type="primary"):
                st.success("Forecast generated!")

        with col2:
            st.subheader("Forecast Results")

            demand_df = generate_sample_data()

            # Split into history and forecast
            history = demand_df.iloc[:-30]
            forecast = demand_df.iloc[-30:]

            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=history['date'], y=history['demand'],
                name='Historical',
                line=dict(color='#888')
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['date'], y=forecast['forecast'],
                name='Forecast',
                line=dict(color='#1f77b4', width=3)
            ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                y=forecast['upper_bound'].tolist() + forecast['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(31,119,180,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'
            ))

            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title='Demand Forecast with Prediction Intervals',
                xaxis_title='Date',
                yaxis_title='Demand (units)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("MAPE", "4.2%")
            with col_b:
                st.metric("RMSE", "12.3")
            with col_c:
                st.metric("Bias", "+1.2%")
            with col_d:
                st.metric("Coverage", "94.5%")

    with tabs[1]:
        st.subheader("Backtesting Results")
        st.info("Compare model performance across different time periods.")

        # Sample backtesting results
        backtest_data = pd.DataFrame({
            'Period': ['2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4'],
            'MAPE': [4.5, 3.8, 4.2, 4.1],
            'RMSE': [13.2, 11.5, 12.3, 11.9],
            'Coverage': [93.2, 95.1, 94.5, 94.8]
        })

        st.dataframe(backtest_data, use_container_width=True)

    with tabs[2]:
        st.subheader("Model Configuration")

        model_type = st.selectbox("Forecasting Model", [
            "DeepAR (GluonTS)",
            "N-BEATS",
            "Temporal Fusion Transformer",
            "Prophet"
        ])

        col1, col2 = st.columns(2)
        with col1:
            context_length = st.number_input("Context Length", 7, 365, 60)
            prediction_length = st.number_input("Prediction Length", 1, 90, 30)
        with col2:
            num_layers = st.number_input("Number of Layers", 1, 8, 3)
            hidden_size = st.number_input("Hidden Size", 16, 256, 64)

elif page == "💰 Price Optimization":
    st.markdown('<h1 class="main-header">Price Optimization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Maximize profit with elasticity-based pricing</p>', unsafe_allow_html=True)

    tabs = st.tabs(["💵 Optimizer", "📊 Elasticity", "📈 Simulation"])

    with tabs[0]:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Optimization Parameters")

            product = st.selectbox("Product Category", [
                "Electronics",
                "Apparel",
                "Groceries",
                "Home & Garden"
            ])

            current_price = st.number_input("Current Price ($)", 0.0, 1000.0, 49.99)
            cost = st.number_input("Unit Cost ($)", 0.0, 500.0, 25.0)
            min_price = st.number_input("Min Price ($)", 0.0, 1000.0, 30.0)
            max_price = st.number_input("Max Price ($)", 0.0, 1000.0, 80.0)

            objective = st.radio("Objective", ["Maximize Profit", "Maximize Revenue", "Maximize Volume"])

            if st.button("🎯 Optimize Price", type="primary"):
                with st.spinner("Running optimization..."):
                    import time
                    time.sleep(1.5)

                    optimal_price = 54.99
                    profit_increase = 12.5

                    st.success("Optimization complete!")
                    st.metric("Optimal Price", f"${optimal_price:.2f}", f"+{profit_increase}% profit")

        with col2:
            st.subheader("Price-Profit Curve")

            prices = np.linspace(30, 80, 100)
            # Simulated profit curve
            profit = -0.5 * (prices - 55) ** 2 + 200 + np.random.normal(0, 10, 100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prices, y=profit,
                mode='lines',
                name='Profit',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.add_vline(x=49.99, line_dash="dash", line_color="gray",
                         annotation_text="Current Price")
            fig.add_vline(x=54.99, line_dash="solid", line_color="green",
                         annotation_text="Optimal Price")
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title='Price vs Profit Analysis',
                xaxis_title='Price ($)',
                yaxis_title='Expected Profit ($)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Price Elasticity Analysis")

        # Elasticity by category
        categories = ['Electronics', 'Apparel', 'Groceries', 'Home & Garden', 'Beauty']
        elasticities = [-1.8, -2.3, -0.9, -1.5, -2.1]

        fig = px.bar(
            x=categories, y=elasticities,
            title='Price Elasticity by Category',
            template='plotly_dark',
            color=elasticities,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Category',
            yaxis_title='Price Elasticity'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <strong>Interpretation:</strong><br>
            • Elasticity < -1: Elastic demand (price sensitive)<br>
            • Elasticity > -1: Inelastic demand (price insensitive)<br>
            • Groceries show lowest elasticity (essential goods)
        </div>
        """, unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("Revenue Simulation")

        col1, col2 = st.columns(2)

        with col1:
            price_change = st.slider("Price Change (%)", -30, 30, 0)

        with col2:
            elasticity = st.slider("Elasticity Assumption", -3.0, -0.5, -1.5)

        # Calculate impact
        volume_change = price_change * elasticity
        revenue_change = (1 + price_change/100) * (1 + volume_change/100) - 1

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Price Change", f"{price_change:+}%")
        with col_b:
            st.metric("Volume Change", f"{volume_change:+.1f}%")
        with col_c:
            delta_color = "normal" if revenue_change > 0 else "inverse"
            st.metric("Revenue Impact", f"{revenue_change*100:+.1f}%")

elif page == "🧪 A/B Testing":
    st.markdown('<h1 class="main-header">A/B Testing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Design and analyze experiments</p>', unsafe_allow_html=True)

    tabs = st.tabs(["📋 Active Experiments", "📊 Analysis", "➕ New Experiment"])

    with tabs[0]:
        # Sample experiments
        experiments = pd.DataFrame({
            'ID': ['EXP-001', 'EXP-002', 'EXP-003', 'EXP-004'],
            'Name': ['Homepage Banner', 'Checkout Flow', 'Email Subject', 'Price Display'],
            'Status': ['🟢 Running', '🟢 Running', '🟡 Analyzing', '🔴 Concluded'],
            'Variant A': ['42.3%', '18.2%', '12.5%', '3.2%'],
            'Variant B': ['44.1%', '19.8%', '13.1%', '3.5%'],
            'P-Value': ['0.08', '0.02', '0.21', '0.04'],
            'Traffic': ['100K', '50K', '200K', '150K']
        })

        st.dataframe(experiments, use_container_width=True, hide_index=True)

        # Experiment details
        st.subheader("Experiment Details: EXP-002 (Checkout Flow)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Control (A)", "18.2%", help="Conversion rate")
        with col2:
            st.metric("Treatment (B)", "19.8%", "+8.8%", help="Conversion rate")
        with col3:
            st.metric("Statistical Power", "92%")
        with col4:
            st.metric("Days Running", "14")

        # Conversion over time
        days = range(1, 15)
        control = [17.5, 17.8, 18.0, 17.9, 18.1, 18.0, 18.2, 18.1, 18.3, 18.2, 18.1, 18.2, 18.2, 18.2]
        treatment = [17.6, 18.2, 18.5, 18.8, 19.0, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.7, 19.8, 19.8]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(days), y=control, name='Control (A)',
                                 line=dict(color='#888')))
        fig.add_trace(go.Scatter(x=list(days), y=treatment, name='Treatment (B)',
                                 line=dict(color='#1f77b4', width=3)))
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title='Conversion Rate Over Time',
            xaxis_title='Day',
            yaxis_title='Conversion Rate (%)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Statistical Analysis")

        col1, col2 = st.columns(2)

        with col1:
            experiment_id = st.selectbox("Select Experiment", ['EXP-001', 'EXP-002', 'EXP-003', 'EXP-004'])
        with col2:
            metric = st.selectbox("Metric", ['Conversion Rate', 'Revenue per User', 'Average Order Value'])

        # Posterior distribution
        st.subheader("Bayesian Analysis")

        x = np.linspace(-0.05, 0.15, 200)
        control_dist = np.exp(-0.5 * ((x - 0.02) / 0.02) ** 2)
        treatment_dist = np.exp(-0.5 * ((x - 0.04) / 0.02) ** 2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=control_dist, name='Control Posterior',
                                 fill='tozeroy', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=x, y=treatment_dist, name='Treatment Posterior',
                                 fill='tozeroy', line=dict(color='#1f77b4')))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title='Posterior Distribution of Effect Size',
            xaxis_title='Effect Size',
            yaxis_title='Density'
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("P(B > A)", "94.2%")
        with col_b:
            st.metric("Expected Lift", "+8.8%")
        with col_c:
            st.metric("Credible Interval", "[3.2%, 14.5%]")

    with tabs[2]:
        st.subheader("Create New Experiment")

        with st.form("new_experiment"):
            col1, col2 = st.columns(2)

            with col1:
                exp_name = st.text_input("Experiment Name")
                exp_hypothesis = st.text_area("Hypothesis")
                primary_metric = st.selectbox("Primary Metric", [
                    "Conversion Rate",
                    "Revenue per User",
                    "Click-through Rate"
                ])

            with col2:
                mde = st.slider("Minimum Detectable Effect (%)", 1, 20, 5)
                significance = st.slider("Significance Level", 0.01, 0.10, 0.05)
                power = st.slider("Statistical Power", 0.70, 0.95, 0.80)
                traffic_split = st.slider("Traffic Split (A/B)", 0.1, 0.9, 0.5)

            # Sample size calculation
            baseline_rate = 0.10
            z_alpha = 1.96
            z_beta = 0.84
            effect = baseline_rate * (mde / 100)
            n = 2 * ((z_alpha + z_beta) ** 2) * baseline_rate * (1 - baseline_rate) / (effect ** 2)

            st.info(f"📊 Required sample size: **{int(n):,}** per variant")

            submitted = st.form_submit_button("🚀 Launch Experiment", type="primary")
            if submitted:
                st.success("Experiment created successfully!")

elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Platform configuration</p>', unsafe_allow_html=True)

    tabs = st.tabs(["🔌 API", "🎨 Appearance", "🔔 Notifications", "👥 Team"])

    with tabs[0]:
        st.subheader("API Configuration")

        api_url = st.text_input("API Base URL", value="http://localhost:8000")
        api_key = st.text_input("API Key", type="password", value="••••••••••••")
        timeout = st.number_input("Request Timeout (seconds)", 5, 120, 30)

        if st.button("🔄 Test Connection"):
            health = check_api_health()
            if health["status"] == "healthy":
                st.success("✅ Connection successful!")
            else:
                st.error(f"❌ Connection failed: {health['details']}")

    with tabs[1]:
        st.subheader("Appearance")

        theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
        accent_color = st.color_picker("Accent Color", "#1f77b4")
        chart_style = st.selectbox("Chart Style", ["Plotly Dark", "Plotly Light", "Seaborn"])

        st.markdown("---")

        show_tips = st.checkbox("Show helpful tips", value=True)
        compact_mode = st.checkbox("Compact mode", value=False)

    with tabs[2]:
        st.subheader("Notifications")

        st.markdown("#### Email Notifications")
        email_daily = st.checkbox("Daily summary", value=True)
        email_experiment = st.checkbox("Experiment results", value=True)
        email_anomaly = st.checkbox("Anomaly alerts", value=True)

        st.markdown("#### Slack Integration")
        slack_webhook = st.text_input("Slack Webhook URL", type="password")
        slack_channel = st.text_input("Channel", "#meridian-alerts")

    with tabs[3]:
        st.subheader("Team Management")

        team_members = pd.DataFrame({
            'Name': ['Admin User', 'Data Scientist', 'ML Engineer'],
            'Email': ['admin@meridian.io', 'ds@meridian.io', 'ml@meridian.io'],
            'Role': ['Admin', 'Analyst', 'Developer'],
            'Status': ['🟢 Active', '🟢 Active', '🟡 Pending']
        })

        st.dataframe(team_members, use_container_width=True, hide_index=True)

        if st.button("➕ Invite Team Member"):
            st.info("Invitation dialog would open here")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Meridian Analytics Platform v1.0.0 | © 2026 All Rights Reserved | 
    <a href="https://github.com/vasdz/Meridian" style="color: #1f77b4;">GitHub</a> | 
    <a href="mailto:support@meridian.io" style="color: #1f77b4;">Support</a>
</div>
""", unsafe_allow_html=True)

