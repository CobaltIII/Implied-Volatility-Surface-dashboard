import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


st.set_page_config(page_title="IV dashboard", layout="wide")

def black_scholes_price(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(S, K, T, r, q, market_price, option_type="call"):
    try:
        return brentq(
            lambda sigma: black_scholes_price(S, K, T, r, q, sigma, option_type) - market_price,
            1e-5, 3.0
        )
    except:
        return np.nan

st.sidebar.markdown("## Made by [Dhruv Jaiswal](https://cobaltiii.github.io/)")
st.sidebar.write("This site visualizes the Implied Volatility (IV) data for options using data yfinance. It aims to give both, a broad view and deep statistical insight into IVs")
st.sidebar.header("Model Parameters")
r = st.sidebar.number_input("Risk-Free Rate", value=0.015, step=0.001)
q = st.sidebar.number_input("Dividend Yield", value=0.013, step=0.001)

st.sidebar.header("Visualization Parameters")
y_axis = st.sidebar.selectbox("Select Y-axis:", ["Strike Price ($)", "Moneyness (%)"])

st.sidebar.header("Ticker Symbol")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SPY").upper()

st.sidebar.header("Strike Price Filter")
min_strike_pct = st.sidebar.number_input("Min Strike (% of Spot)", value=80.0)
max_strike_pct = st.sidebar.number_input("Max Strike (% of Spot)", value=120.0)

# --- Fetch Data ---
st.title("Implied Volatility Surface Dashboard")
custom_cmap = LinearSegmentedColormap.from_list("custom", ["red", "black", "green"])
try:
    data = yf.Ticker(ticker)
    spot_price = data.history(period="1d")["Close"].iloc[-1]

    expirations = data.options[:5]  # limit for speed
    all_data = []

    for exp in expirations:
        option_chain = data.option_chain(exp)
        calls = option_chain.calls.copy()
        expiration_date = datetime.strptime(exp, "%Y-%m-%d")
        T = (expiration_date - datetime.today()).days / 365.0
        if T <= 0: continue

        calls = calls[(calls['strike'] > spot_price * min_strike_pct / 100) &
                      (calls['strike'] < spot_price * max_strike_pct / 100)]

        for _, row in calls.iterrows():
            K = row['strike']
            market_price = row['lastPrice']
            iv = implied_vol(spot_price, K, T, r, q, market_price)
            if not np.isnan(iv):
                all_data.append({
                    "T": T,
                    "K": K,
                    "Moneyness": 100 * K / spot_price,
                    "IV": iv * 100  # percentage
                })

    df = pd.DataFrame(all_data)
    col1, col2 = st.columns(2)    
    if df.empty:
        st.warning("No valid option data available for selected filters.")
    else:
        df = df.dropna()
        grid_x, grid_y = np.meshgrid(
            np.linspace(df["T"].min(), df["T"].max(), 50),
            np.linspace(df["K"].min(), df["K"].max(), 50)
            )
        grid_z = griddata(
            points=(df["T"], df["K"]),
            values=df["IV"],
            xi=(grid_x, grid_y),
            method='cubic'
            )
        fig = go.Figure(data=[go.Surface(
            x=grid_x,  # Time to Expiry (Years)
            y=grid_y,  # Strike Price
            z=grid_z,  # Implied Volatility (%)
            colorscale='tropic',
            colorbar=dict(title="IV (%)"),
            showscale=True
            )])
        
        fig.update_layout(
            title=f"Implied Volatility Surface for {ticker} Options",
            scene=dict(
                xaxis_title="Time to Expiration (Years)",
                yaxis_title="Strike Price ($)" if y_axis == "Strike Price ($)" else "Moneyness (%)",
                zaxis_title="Implied Volatility (%)"
            ),
            width = 150,
            height = 700,
            margin=dict(l=0, r=0, b=0, t=40)
            )
        with col1:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Current Price of {ticker}:** ${spot_price:.2f}")
            st.markdown("## A few tickers you can use:")
            st.markdown("""
                - **AAPL** – Apple Inc.
                - **MSFT** – Microsoft Corporation
                - **GOOGL** – Alphabet Inc.
                - **AMZN** – Amazon.com Inc.
                - **TSLA** – Tesla Inc.
                - **NVDA** – NVIDIA Corporation
                - **META** – Meta Platforms Inc.
                - **BRK-B** – Berkshire Hathaway Inc.
                - **JPM** – JPMorgan Chase & Co.
                - **NFLX** – Netflix Inc.
                - **ORCL** – Oracle Corporation Common Stock
                - **JNJ** – Johnson & Johnson Common Stock	        
                """)
            
    
    
    with col2:
        
        st.markdown("### Implied Volatility Distribution")

        iv_values = df["IV"]
        mean_iv = iv_values.mean()
        counts, bins = np.histogram(iv_values, bins=30)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        norm = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
        #cmap = plt.get_cmap("cool")
        cmap = plt.get_cmap("vanimo")
        colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.2f})' for r, g, b, a in [cmap(val) for val in norm]]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                marker=dict(color=colors),
                width=(bins[1] - bins[0]),
                name="IV Histogram"
            )
        )

        x_vals = np.linspace(iv_values.min(), iv_values.max(), 200)
        kde = gaussian_kde(iv_values)
        fig2.add_trace(
            go.Scatter(
                x=x_vals,
                y=kde(x_vals) * len(iv_values) * (iv_values.max() - iv_values.min()) / 30,
                mode='lines',
                line=dict(color='orange', width=3),
                name='Smoothed Curve (KDE)'
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[mean_iv, mean_iv],
                y=[0, max(np.histogram(iv_values, bins=30)[0])],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'Mean: {mean_iv:.2f}%'
            )
        )
        fig2.update_layout(
            title="Implied Volatility Distribution",
            xaxis_title="Implied Volatility (%)",
            yaxis_title="Count",
            width = 150,
            height = 650,
            margin=dict(l=0, r=0, b=0, t=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig2.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
        fig2.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Filtered Options Data")
        st.dataframe(df[["T", "K", "Moneyness", "IV"]].sort_values(by="T"), use_container_width=True)
    
    with st.container():
        st.markdown("### Volatility Smile / Skew")

        selected_expiries = st.multiselect(
            "Select Expiry Times (T) to Plot Smile/Skew",
            options=sorted(df['T'].unique()),
            default=sorted(df['T'].unique())[:3]
        )

        if selected_expiries:
            fig3 = go.Figure()
            for t in selected_expiries:
                group = df[df["T"] == t]
                fig3.add_trace(go.Scatter(
                    x=group["K"],
                    y=group["IV"],
                    mode='lines+markers',
                    name=f"T = {round(t, 3)} yrs"
                ))

            fig3.update_layout(
                title="Volatility Smile / Skew by Expiry",
                xaxis_title="Strike Price ($)",
                yaxis_title="Implied Volatility (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=650,
                margin=dict(l=0, r=0, b=0, t=40),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            fig3.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            fig3.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Select at least one expiry to show the smile plot.")

except Exception as e:
    st.error(f"Error fetching or plotting data: {e}")
    
