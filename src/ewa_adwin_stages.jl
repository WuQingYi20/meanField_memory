# ══════════════════════════════════════════════════════════════
# EWA-ADWIN Stage 1: Pair and Act
# ══════════════════════════════════════════════════════════════

"""
    ewa_adwin_pair_and_act!(agents, ws, params, rng)

Form random pairs, select actions via logit probability matching, record coordination.
Identical logic to ewa_pair_and_act! but typed on EWAADWINAgent.
"""
function ewa_adwin_pair_and_act!(agents::Vector{EWAADWINAgent}, ws::EWAWorkspace,
                                  params::EWAADWINParams, rng::AbstractRNG)
    N = params.N
    n_pairs = N ÷ 2

    # Random permutation
    randperm!(rng, ws.perm)

    for k in 1:n_pairs
        i = ws.perm[2k - 1]
        j = ws.perm[2k]
        ws.pair_i[k] = i
        ws.pair_j[k] = j

        # Action selection: probability matching with logit probability
        ws.action[i] = rand(rng) < agents[i].prob_A ? STRATEGY_A : STRATEGY_B
        ws.action[j] = rand(rng) < agents[j].prob_A ? STRATEGY_A : STRATEGY_B

        # Record coordination
        ws.coordinated[k] = ws.action[i] == ws.action[j]
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════
# EWA-ADWIN Stage 2: Update Attractions (ADWIN replaces phi/rho)
# ══════════════════════════════════════════════════════════════

"""
    _ewa_adwin_update_prob!(agent::EWAADWINAgent, lambda::Float64)

Recompute logit probability from attractions using log-sum-exp stable softmax.
"""
@inline function _ewa_adwin_update_prob!(agent::EWAADWINAgent, lambda::Float64)
    a = lambda * agent.attract_A
    b = lambda * agent.attract_B
    m = max(a, b)
    ea = exp(a - m)
    eb = exp(b - m)
    agent.prob_A = ea / (ea + eb)
    return nothing
end

"""
    ewa_adwin_update_attractions!(agents, ws, params)

ADWIN-based attraction update. Each tick's reinforcement is fed to ADWIN
which manages the adaptive window. Attractions = windowed means.

For pure coordination game: pi(k, s_j) = I(k == s_j).
Reinforcement: (delta + (1-delta)*I(k==s_i)) * pi(k, s_j)
"""
function ewa_adwin_update_attractions!(agents::Vector{EWAADWINAgent}, ws::EWAWorkspace,
                                        params::EWAADWINParams)
    N = params.N
    n_pairs = N ÷ 2
    delta = params.delta

    # Build partner lookup from pairs
    for k in 1:n_pairs
        i = ws.pair_i[k]
        j = ws.pair_j[k]
        ws.partner_id[i] = j
        ws.partner_id[j] = i
        ws.partner_action[i] = ws.action[j]
        ws.partner_action[j] = ws.action[i]
    end

    # Update each agent's attractions via ADWIN
    for i in 1:N
        s_i = ws.action[i]
        s_j = ws.partner_action[i]

        # Pure coordination payoffs: pi(k, s_j) = I(k == s_j)
        pi_A = s_j == STRATEGY_A ? 1.0 : 0.0
        pi_B = s_j == STRATEGY_B ? 1.0 : 0.0

        # Reinforcement weights (same formula as EWA)
        reinf_A = (delta + (1.0 - delta) * (s_i == STRATEGY_A ? 1.0 : 0.0)) * pi_A
        reinf_B = (delta + (1.0 - delta) * (s_i == STRATEGY_B ? 1.0 : 0.0)) * pi_B

        # Feed to ADWIN (adaptive window manages history)
        adwin_add!(agents[i].adwin_A, reinf_A)
        adwin_add!(agents[i].adwin_B, reinf_B)

        # Attractions = windowed means
        agents[i].attract_A = adwin_mean(agents[i].adwin_A)
        agents[i].attract_B = adwin_mean(agents[i].adwin_B)

        # Recompute logit probability
        _ewa_adwin_update_prob!(agents[i], params.lambda)
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════
# EWA-ADWIN Stage 3: Metrics
# ══════════════════════════════════════════════════════════════

"""
    ewa_adwin_metrics(agents, ws, t, history, tick_count, params)

Compute per-tick metrics. Uses prob_A as belief analog. Normative fields are 0.
"""
function ewa_adwin_metrics(agents::Vector{EWAADWINAgent}, ws::EWAWorkspace,
                            t::Int, history::Vector{TickMetrics}, tick_count::Int,
                            params::EWAADWINParams)
    N = params.N
    n_pairs = N ÷ 2

    # Fraction A
    n_A = 0
    for i in 1:N
        if ws.action[i] == STRATEGY_A
            n_A += 1
        end
    end
    fraction_A = n_A / N

    # Coordination rate
    n_coord = 0
    for k in 1:n_pairs
        if ws.coordinated[k]
            n_coord += 1
        end
    end
    coord_rate = n_coord / n_pairs

    # Belief accuracy and consensus (using prob_A as belief analog)
    sum_error = 0.0
    sum_b = 0.0
    for i in 1:N
        sum_error += abs(agents[i].prob_A - fraction_A)
        sum_b += agents[i].prob_A
    end
    belief_error = sum_error / N
    mean_b = sum_b / N

    sum_var = 0.0
    for i in 1:N
        sum_var += (agents[i].prob_A - mean_b)^2
    end
    belief_var = sum_var / N

    # Convergence: behavioral + belief only (no crystallization check)
    majority = max(fraction_A, 1.0 - fraction_A)
    layers_met = majority >= params.thresh_majority &&
                 belief_error < params.thresh_belief_error &&
                 belief_var < params.thresh_belief_var

    if layers_met
        prev = tick_count > 0 ? history[tick_count].convergence_counter : 0
        conv_counter = prev + 1
    else
        conv_counter = 0
    end

    return TickMetrics(
        t, fraction_A, 0.0, coord_rate, 0,
        0.0, 0,
        belief_error, belief_var, 0.0, conv_counter,
    )
end
