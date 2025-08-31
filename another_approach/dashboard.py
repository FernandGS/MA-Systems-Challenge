# dashboard.py
# Streamlit KPI dashboard for trash collection multi-agent system
# Robust gegen Events ohne 't' und gegen leere Eventlisten.

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def _events_to_df(sim):
    """Konvertiert sim.events -> DataFrame und stellt sicher, dass 't' existiert."""
    if not sim.events:
        return pd.DataFrame(columns=["t","type","bin","truck","amount"])
    df = pd.DataFrame(sim.events)

    # Stelle 't' bereit (kann bei pickup/drop/recharge fehlen)
    if "t" not in df.columns:
        df["t"] = pd.NA
    # nach Möglichkeit numerisch machen
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    return df

def _plot_costs_bar(costs: dict):
    df_costs = pd.DataFrame([costs])
    st.dataframe(df_costs.style.format("{:.2f}"))

    fig, ax = plt.subplots()
    # feste Farben für Lesbarkeit (subset falls weniger Spalten)
    df_costs.iloc[0].plot(
        kind="bar", ax=ax,
        color=["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"][:len(df_costs.columns)]
    )
    ax.set_ylabel("€")
    ax.set_title("Cost Breakdown")
    st.pyplot(fig)

def _plot_event_timeline(df_events: pd.DataFrame):
    st.subheader("Events über die Zeit")
    if df_events.empty or "t" not in df_events.columns:
        st.info("Keine Events vorhanden.")
        return

    # nur Events mit Zeitstempel nehmen
    df = df_events.dropna(subset=["t"]).copy()
    if df.empty:
        st.info("Keine zeitgestempelten Events vorhanden.")
        return

    df["t"] = df["t"].round(0)
    counts = df.pivot_table(index="t", columns="type", aggfunc="size", fill_value=0)

    fig, ax = plt.subplots()
    counts.plot(ax=ax)
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Events")
    ax.set_title("Event-Anzahl pro Sekunde")
    st.pyplot(fig)

def _plot_truck_utilization(sim):
    st.subheader("Truck-Utilization")
    util = []
    for t in sim.trucks:
        util.append({
            "truck": t.tid,
            "km_total": t.km_total,
            "energy_used": t.kwh_total,
        })
    df = pd.DataFrame(util)
    st.dataframe(df.style.format({"km_total":"{:.2f}", "energy_used":"{:.2f}"}))

    fig1, ax1 = plt.subplots()
    df.set_index("truck")["km_total"].plot(kind="bar", ax=ax1)
    ax1.set_title("Distanz pro Truck (km)")
    ax1.set_ylabel("km")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    df.set_index("truck")["energy_used"].plot(kind="bar", ax=ax2)
    ax2.set_title("Energie pro Truck (units)")
    ax2.set_ylabel("units")
    st.pyplot(fig2)

def _plot_rewards(rewards_hist):
    st.subheader("Learning Progress (Avg Reward per Episode)")
    if not rewards_hist:
        st.info("Keine Rewards vorhanden.")
        return
    fig, ax = plt.subplots()
    ax.plot(rewards_hist, label="Avg reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    st.pyplot(fig)

def show_dashboard(sim, costs, rewards_hist=None):
    """Weiterhin für main.py verfügbar – erwartet, dass in Streamlit-Context aufgerufen wird."""
    st.title("🚛 Trash Collection — KPI Dashboard")

    # --- Costs Summary
    st.header("Cost Breakdown (€ per day)")
    _plot_costs_bar(costs)

    # --- Service KPIs
    st.header("Service KPIs")
    overflows = len([e for e in sim.events if e.get("type")=="overflow"])
    pickups   = len([e for e in sim.events if e.get("type")=="pickup"])
    drops     = len([e for e in sim.events if e.get("type")=="drop"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Overflows", overflows)
    col2.metric("Pickups", pickups)
    col3.metric("Drops", drops)

    # --- Efficiency KPIs
    st.header("Efficiency KPIs")
    total_km = sum(t.km_total for t in sim.trucks)
    total_kwh = sum(t.kwh_total for t in sim.trucks)
    col4, col5 = st.columns(2)
    col4.metric("Total km driven", f"{total_km:.1f} km")
    col5.metric("Total energy used", f"{total_kwh:.1f} units")

    # --- Event timeline
    df_events = _events_to_df(sim)
    _plot_event_timeline(df_events)

    # --- Per-truck utilization
    _plot_truck_utilization(sim)

    # --- Learning curve if available
    if rewards_hist is not None:
        _plot_rewards(rewards_hist)

    # --- Sidebar Info
    st.sidebar.header("Scenario Settings (Info)")
    st.sidebar.write(f"Trucks: {len(sim.trucks)}")
    st.sidebar.write(f"Wage €/h: {sim.cfg['WAGE_PER_HOUR']}")
    st.sidebar.write(f"Overflow Penalty €: {sim.cfg['OVERFLOW_PENALTY_EUR']}")
