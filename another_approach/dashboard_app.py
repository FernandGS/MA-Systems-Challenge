# dashboard_app.py
# Streamlit app to run a baseline sim or a short DQN training and visualize KPIs.

import streamlit as st
from config import CONFIG
from city import City
from sim import Simulation
from dashboard import show_dashboard  # <- use the correct function
from visualize import preview  # optional for local debug (won't render inside Streamlit)

# RL bits (shared policy)
from dqn_train import train_multi
from dqn_env import MultiTruckEnv

st.set_page_config(page_title="Trash RL KPI", layout="wide")

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.selectbox("Mode", ["Baseline", "Train DQN (short)"])
steps = st.sidebar.slider("Steps per day (episode length)", 100, 2000, CONFIG["STEPS_PER_DAY"], 50)

# Scenario knobs (apply to a copy of CONFIG)
cfg = CONFIG.copy()
cfg["STEPS_PER_DAY"] = steps
cfg["N_TRUCKS"] = st.sidebar.slider("Trucks", 1, 10, cfg["N_TRUCKS"])
cfg["N_BINS"] = st.sidebar.slider("Bins", 4, 30, cfg["N_BINS"])
cfg["BIN_CAPACITY"] = st.sidebar.number_input("Bin Capacity", value=cfg["BIN_CAPACITY"], min_value=10)
cfg["BIN_FILL_PER_STEP"] = (
    st.sidebar.number_input("Bin fill min / step", value=cfg["BIN_FILL_PER_STEP"][0], min_value=0),
    st.sidebar.number_input("Bin fill max / step", value=cfg["BIN_FILL_PER_STEP"][1], min_value=0),
)
cfg["WAGE_PER_HOUR"] = st.sidebar.number_input("Wage €/h", value=cfg["WAGE_PER_HOUR"], min_value=0.0, step=1.0)
cfg["OVERFLOW_PENALTY_EUR"] = st.sidebar.number_input("Overflow penalty €", value=cfg["OVERFLOW_PENALTY_EUR"], min_value=0.0, step=10.0)
cfg["ENERGY_EUR_PER_UNIT"] = st.sidebar.number_input("Energy cost per unit €", value=cfg["ENERGY_EUR_PER_UNIT"], min_value=0.0, step=0.01, format="%.2f")

# Wire the road planner into RL config if used (Simulation/City will provide)
cfg["plan_route_fn"] = None

st.title("🚛 Trash Collection — KPI Dashboard")

def _rollout_greedy_with_shared_agent(_cfg, agent):
    """Run one greedy episode using the already-trained shared policy and return (sim, costs)."""
    env = MultiTruckEnv(_cfg)
    agent.eps = 0.0  # greedy
    obs_all = env.reset()
    done = [False] * env.n_agents
    last_info = {}

    while not all(done):
        actions = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, _rewards, done, info = env.step(actions)
        last_info = info

    sim = env.sim
    costs = last_info.get("costs", {})
    return sim, costs

if mode == "Baseline":
    if st.button("Run Simulation"):
        city = City(cfg)
        cfg["plan_route_fn"] = city.plan_route
        sim = Simulation(cfg, city)
        sim.run(cfg["STEPS_PER_DAY"])
        costs = sim.summary_costs()
        show_dashboard(sim, costs)
    else:
        st.info("Adjust parameters in the sidebar, then click **Run Simulation**.")

else:  # Train DQN (short) with shared policy, then evaluate that policy
    episodes = st.sidebar.slider("Episodes", 5, 200, 30, 5)
    st.caption("Training maximizes reward (≈ minimizes cost). After training, we evaluate the learned policy (greedy).")
    if st.button("Train & Evaluate"):
        with st.spinner("Training shared policy..."):
            agents, rewards_hist, paths = train_multi(cfg, episodes=episodes, verbose=False)
            # shared policy => agents is [agent]
            agent = agents[0]
        st.success(f"Training complete. {'Saved model at ' + paths[0] if paths else 'No checkpoint saved.'}")

        # Evaluate the trained policy (greedy) to generate sim/costs/events
        sim, costs = _rollout_greedy_with_shared_agent(cfg, agent)
        show_dashboard(sim, costs, rewards_hist=rewards_hist)
    else:
        st.info("Pick a small number of episodes (e.g., 30–50) to get a feel for learning curves.")

st.divider()
# Optional: Load a previously exported JSON (from sim.export_json) and visualize
st.subheader("📁 Load exported simulation JSON")
upload = st.file_uploader("Upload JSON exported by sim.export_json()", type=["json"])
if upload is not None:
    import json
    data = json.load(upload)
    st.write("Config snapshot:", {k: data["cfg"].get(k) for k in ["N_TRUCKS","N_BINS","STEPS_PER_DAY","WAGE_PER_HOUR","OVERFLOW_PENALTY_EUR"]})

    # Quick inline view of costs and events
    costs = data.get("costs", {})
    if costs:
        st.write("Costs")
        import pandas as pd, matplotlib.pyplot as plt
        df_costs = pd.DataFrame([costs])
        st.dataframe(df_costs.style.format("{:.2f}"))
        fig, ax = plt.subplots()
        df_costs.iloc[0].plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # Show raw event counts
    events = data.get("events", [])
    st.write(f"Events loaded: {len(events)}")
    if events:
        import pandas as pd
        df_events = pd.DataFrame(events)
        st.dataframe(df_events.head(200))
