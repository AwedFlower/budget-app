import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="AI Wealth Tracker", layout="wide")

st.title("💰 Smart Budget & Income Analyzer")
st.write("Track your spending habits and see your predicted savings using AI.")

# --- sidebar ---
st.sidebar.header("💵 Monthly Cash Flow")
user_income = st.sidebar.number_input("Monthly Income ($)", min_value=0, value=2500)

st.sidebar.divider()
st.sidebar.header("📅 Next Month's Expenses")
user_food = st.sidebar.number_input("Food & Dining ($)", min_value=0, value=400)
user_util = st.sidebar.number_input("Utilities ($)", min_value=0, value=170)
user_ent = st.sidebar.number_input("Entertainment ($)", min_value=0, value=200)
user_rent = st.sidebar.number_input("Rent ($)", min_value=0, value=800)

# --- Processing ---
historical_data = """Month,Food,Utilities,Entertainment,Rent
1,350,150,200,800
2,380,165,180,800
3,340,140,210,800
4,410,170,150,800
5,390,160,250,800
6,420,185,190,800"""

df = pd.read_csv(io.StringIO(historical_data))
df["Total_Spend"] = df[["Food", "Utilities", "Entertainment", "Rent"]].sum(axis=1)

# Current Month Calculation
user_total_spend = user_food + user_util + user_ent + user_rent
net_savings = user_income - user_total_spend

# --- AI prediction ---
X = df[["Month"]].values
y = df["Total_Spend"].values
model = LinearRegression().fit(X, y)
predicted_spend = model.predict(np.array([[7]]))[0]
predicted_savings = user_income - predicted_spend

# --- Dashboard  ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Monthly Income", f"${user_income:,}")
    
with col2:
    st.metric(
        "Projected Net Savings", 
        f"${net_savings:,}",
        delta=f"{(net_savings/user_income)*100:.1f}% Savings Rate",
        delta_color="normal"
    )

with col3:
    st.metric(
        "AI Predicted Savings (Month 7)", 
        f"${predicted_savings:,.2f}",
        help="Based on your spending trend vs current income"
    )

# --- Visuals ---
st.divider()
fig, ax = plt.subplots(figsize=(10, 4))

# Historical Spending
ax.plot(df["Month"], df["Total_Spend"], marker="o", label="Past Spending", color="#1f77b4")

# Income Baseline
ax.axhline(y=user_income, color="green", linestyle="--", label="Income Level")

# Prediction Point
ax.scatter([7], [predicted_spend], color="red", label="AI Spend Forecast")

ax.fill_between(df["Month"], df["Total_Spend"], user_income, color='green', alpha=0.1, label="Savings Area")

ax.set_title("Spending vs. Income Trend")
ax.set_xlabel("Month")
ax.set_ylabel("USD ($)")
ax.legend()
st.pyplot(fig)

# --- Data Science fr ---
if net_savings < 0:
    st.error(f"⚠️ Warning: Your projected spending exceeds your income by ${abs(net_savings):,}. Consider adjusting your entertainment or food budget.")
elif net_savings > (user_income * 0.2):
    st.success("✅ Great job! You are saving more than 20% of your income.")
