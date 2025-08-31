# dashboard.py
# Streamlit KPI dashboard for trash collection multi-agent system

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def show_dashboard(sim, costs, rewards_hist=None):
    st.title("🚛 Trash Collection — KPI Dashboard")

    # --- Costs Summary
    st.header("Cost Breakdown (€ per day)")
    df_costs = pd.DataFrame([costs])
    st.dataframe(df_costs.style.format("{:.2f}"))

    fig, ax = plt.subplots()
    df_costs.iloc[0].plot(kind="bar", ax=ax, color=["#3498db","#2ecc71","#e67e22","#e74c3c","#9b59b6"])
    ax.set_ylabel("€")
    ax.set_title("Cost Breakdown")
    st.pyplot(fig)

    # --- Service KPIs
    st.header("Service KPIs")
    overflows = len([e for e in sim.events if e["type"]=="overflow"])
    pickups   = len([e for e in sim.events if e["type"]=="pickup"])
    drops     = len([e for e in sim.events if e["type"]=="drop"])
    st.metric("Overflows", overflows)
    st.metric("Pickups", pickups)
    st.metric("Drops", drops)

    # --- Efficiency KPIs
    st.header("Efficiency KPIs")
    total_km = sum(t.km_total for t in sim.trucks)
    total_kwh = sum(t.kwh_total for t in sim.trucks)
    st.metric("Total km driven", f"{total_km:.1f} km")
    st.metric("Total energy used", f"{total_kwh:.1f} units")

    # --- Learning curve if available
    if rewards_hist is not None:
        st.header("Learning Progress (Avg Reward per Episode)")
        fig2, ax2 = plt.subplots()
        ax2.plot(rewards_hist, label="Avg reward")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Reward")
        ax2.legend()
        st.pyplot(fig2)

    # --- Scenario compare toggle
    st.sidebar.header("Scenario Settings")
    n_trucks = st.sidebar.slider("Number of Trucks", 1, 10, len(sim.trucks))
    wage     = st.sidebar.number_input("Wage €/h", value=sim.cfg["WAGE_PER_HOUR"])
    overflow = st.sidebar.number_input("Overflow Penalty €", value=sim.cfg["OVERFLOW_PENALTY_EUR"])
    st.sidebar.write("Change parameters and rerun simulation for comparison.")
