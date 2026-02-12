import pandas as pd

from src.environment import SimulationEnvironment


def _run_short(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = SimulationEnvironment(
        num_agents=20,
        memory_type="dynamic",
        decision_mode="cognitive_lockin",
        enable_normative=True,
        observation_k=3,
        crystal_threshold=2.0,
        random_seed=seed,
    )
    result = env.run(max_ticks=80, early_stop=False)
    history = env.get_history_dataframe().copy()
    agents = pd.DataFrame(result.agent_final_states)[["id", "first_crystallisation_tick", "has_norm"]].copy()
    return history, agents


def test_tick_update_order_is_explicit_and_fixed():
    env = SimulationEnvironment(num_agents=10, random_seed=1)
    assert env.get_tick_update_order() == [
        "pair_and_action",
        "observe_and_memory_update",
        "confidence_update",
        "normative_ddm_or_anomaly",
        "enforcement_signal_broadcast",
        "metrics_and_convergence",
    ]


def test_same_seed_produces_same_history_and_crystallisation_distribution():
    h1, a1 = _run_short(seed=1234)
    h2, a2 = _run_short(seed=1234)

    cols = ["tick", "norm_adoption_rate", "coordination_rate", "majority_fraction"]
    pd.testing.assert_frame_equal(h1[cols].reset_index(drop=True), h2[cols].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        a1.sort_values("id").reset_index(drop=True),
        a2.sort_values("id").reset_index(drop=True),
    )


def test_shock_injection_activates_persistent_violators():
    env = SimulationEnvironment(
        num_agents=20,
        enable_normative=True,
        observation_k=2,
        random_seed=42,
        shock_tick=0,
        shock_violator_fraction=0.2,
    )
    env.step()
    assert len(env._persistent_violator_strategy_by_id) == 4

