# ══════════════════════════════════════════════════════════════
# Stage 1: Pair and Action
# ══════════════════════════════════════════════════════════════

"""
    compute_effective_belief!(agent::AgentState, params::SimulationParams)

Compute compliance and b_eff_A from current normative state and experience belief.
"""
@inline function compute_effective_belief!(agent::AgentState, params::SimulationParams)
    if agent.r != NO_NORM
        agent.compliance = agent.sigma ^ params.k
        c = agent.compliance
        b_norm_A = agent.r == STRATEGY_A ? 1.0 : 0.0
        agent.b_eff_A = c * b_norm_A + (1.0 - c) * agent.b_exp_A
    else
        agent.compliance = 0.0
        agent.b_eff_A = agent.b_exp_A
    end
    return nothing
end

"""
    map_predict(b_eff_A::Float64, rng::AbstractRNG)::Int8

MAP prediction: deterministic except for exact ties (DD-10).
"""
@inline function map_predict(b_eff_A::Float64, rng::AbstractRNG)::Int8
    if b_eff_A > 0.5
        return STRATEGY_A
    elseif b_eff_A < 0.5
        return STRATEGY_B
    else
        return rand(rng) < 0.5 ? STRATEGY_A : STRATEGY_B
    end
end

"""
    stage_1_pair_and_act!(agents, ws, params, rng)

Form random pairs, compute effective beliefs, select actions, make predictions.
"""
function stage_1_pair_and_act!(agents::Vector{AgentState}, ws::TickWorkspace,
                                params::SimulationParams, rng::AbstractRNG)
    N = params.N
    n_pairs = N ÷ 2

    # 1a. Random permutation (reuse buffer)
    randperm!(rng, ws.perm)

    for k in 1:n_pairs
        i = ws.perm[2k - 1]
        j = ws.perm[2k]
        ws.pair_i[k] = i
        ws.pair_j[k] = j

        # 1b. Compute effective belief
        compute_effective_belief!(agents[i], params)
        compute_effective_belief!(agents[j], params)

        # 1c. Action selection: probability matching
        ws.action[i] = rand(rng) < agents[i].b_eff_A ? STRATEGY_A : STRATEGY_B
        ws.action[j] = rand(rng) < agents[j].b_eff_A ? STRATEGY_A : STRATEGY_B

        # 1d. MAP prediction (deterministic, tie-break random)
        ws.prediction[i] = map_predict(agents[i].b_eff_A, rng)
        ws.prediction[j] = map_predict(agents[j].b_eff_A, rng)

        # 1e. Record coordination
        ws.coordinated[k] = ws.action[i] == ws.action[j]
    end

    return nothing
end

"""
    stage_1_pair_and_act_network!(agents, ws, params, rng, network)

Network-constrained pairing via greedy maximal matching.
Shuffle agents, iterate: for each unmatched agent, pick a random available neighbor.
Unmatched agents get partner_id[i] = 0.
"""
function stage_1_pair_and_act_network!(agents::Vector{AgentState}, ws::TickWorkspace,
                                       params::SimulationParams, rng::AbstractRNG,
                                       network::Vector{Vector{Int}})
    N = params.N

    # Reset availability and partner lookup
    fill!(ws.available, true)
    fill!(ws.partner_id, 0)
    fill!(ws.partner_action, Int8(0))

    # Random permutation for visit order
    randperm!(rng, ws.perm)

    n_pairs = 0

    for idx in 1:N
        i = ws.perm[idx]
        ws.available[i] || continue

        # Find available neighbors
        neighbors = network[i]
        # Shuffle neighbor selection by picking a random available one
        best_j = 0
        n_avail = 0
        for j in neighbors
            if ws.available[j]
                n_avail += 1
                # Reservoir sampling: pick uniformly at random
                if rand(rng, 1:n_avail) == 1
                    best_j = j
                end
            end
        end

        best_j == 0 && continue  # no available neighbor

        # Form pair
        ws.available[i] = false
        ws.available[best_j] = false
        n_pairs += 1
        ws.pair_i[n_pairs] = i
        ws.pair_j[n_pairs] = best_j

        # Compute effective beliefs
        compute_effective_belief!(agents[i], params)
        compute_effective_belief!(agents[best_j], params)

        # Action selection
        ws.action[i] = rand(rng) < agents[i].b_eff_A ? STRATEGY_A : STRATEGY_B
        ws.action[best_j] = rand(rng) < agents[best_j].b_eff_A ? STRATEGY_A : STRATEGY_B

        # MAP prediction
        ws.prediction[i] = map_predict(agents[i].b_eff_A, rng)
        ws.prediction[best_j] = map_predict(agents[best_j].b_eff_A, rng)

        # Partner lookup
        ws.partner_id[i] = best_j
        ws.partner_id[best_j] = i
        ws.partner_action[i] = ws.action[best_j]
        ws.partner_action[best_j] = ws.action[i]

        # Coordination
        ws.coordinated[n_pairs] = ws.action[i] == ws.action[best_j]
    end

    ws.n_pairs_formed = n_pairs

    return nothing
end

"""
    stage_1_pair_and_act_roundrobin!(agents, ws, params, rng, schedule)

Round-robin pairing: read the next round from the pre-computed schedule,
then advance the round counter (reshuffling at cycle boundaries).
All agents are matched every tick.
"""
function stage_1_pair_and_act_roundrobin!(agents::Vector{AgentState}, ws::TickWorkspace,
                                           params::SimulationParams, rng::AbstractRNG,
                                           schedule::RoundRobinSchedule)
    N = params.N
    n_pairs = N ÷ 2

    # Look up current round
    round_idx = schedule.round_order[schedule.current_idx]

    for k in 1:n_pairs
        i = schedule.schedule_i[k, round_idx]
        j = schedule.schedule_j[k, round_idx]
        ws.pair_i[k] = i
        ws.pair_j[k] = j

        # 1b. Compute effective belief
        compute_effective_belief!(agents[i], params)
        compute_effective_belief!(agents[j], params)

        # 1c. Action selection: probability matching
        ws.action[i] = rand(rng) < agents[i].b_eff_A ? STRATEGY_A : STRATEGY_B
        ws.action[j] = rand(rng) < agents[j].b_eff_A ? STRATEGY_A : STRATEGY_B

        # 1d. MAP prediction (deterministic, tie-break random)
        ws.prediction[i] = map_predict(agents[i].b_eff_A, rng)
        ws.prediction[j] = map_predict(agents[j].b_eff_A, rng)

        # 1e. Record coordination
        ws.coordinated[k] = ws.action[i] == ws.action[j]
    end

    # Advance round counter; reshuffle at cycle boundary
    schedule.current_idx += 1
    if schedule.current_idx > schedule.n_rounds
        schedule.current_idx = 1
        shuffle!(rng, schedule.round_order)
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════
# Stage 2: Observe and Update Experience Memory
# ══════════════════════════════════════════════════════════════

"""
    compute_b_exp_A(agent::AgentState)::Float64

Recompute experience belief (A component) from FIFO using last w entries.
"""
@inline function compute_b_exp_A(agent::AgentState)::Float64
    n_A, total = count_strategy_A(agent.fifo, agent.w)
    if total == 0
        return 0.5
    end
    return n_A / total
end

"""
    stage_2_observe_and_memory!(agents, ws, params, rng; use_network=false)

Add partner's strategy to FIFO, sample V additional observations, recompute b_exp.
When use_network=true, partner_id is already set by stage_1_pair_and_act_network!
and unmatched agents (partner_id==0) are skipped.
"""
function stage_2_observe_and_memory!(agents::Vector{AgentState}, ws::TickWorkspace,
                                      params::SimulationParams, rng::AbstractRNG;
                                      use_network::Bool=false)
    N = params.N
    n_pairs = use_network ? ws.n_pairs_formed : N ÷ 2

    # Build partner lookup from pairs (only for complete graph path)
    if !use_network
        for k in 1:n_pairs
            i = ws.pair_i[k]
            j = ws.pair_j[k]
            ws.partner_id[i] = j
            ws.partner_id[j] = i
            ws.partner_action[i] = ws.action[j]
            ws.partner_action[j] = ws.action[i]
        end
    end

    # Build observations for each agent
    obs_pos = 1  # current write position in obs_pool

    for i in 1:N
        ws.obs_offset[i] = obs_pos

        # Skip unmatched agents on network topology
        if use_network && ws.partner_id[i] == 0
            continue
        end

        # 2a. Always include partner's action as first observation
        ws.obs_pool[obs_pos] = ws.partner_action[i]
        obs_pos += 1

        # 2b. Add partner's strategy to FIFO (DD-1: partner only)
        push!(agents[i].fifo, ws.partner_action[i])

        # 2c. V additional observations from other interactions (global, not network-constrained)
        if params.V > 0
            # Build eligible list: pairs not involving agent i
            n_eligible = 0
            for k in 1:n_pairs
                if ws.pair_i[k] != i && ws.pair_j[k] != i
                    n_eligible += 1
                    ws.eligible_buf[n_eligible] = k
                end
            end

            n_sample = min(params.V, n_eligible)
            if n_sample > 0
                # Sample without replacement from eligible pairs
                if n_sample == n_eligible
                    # Take all eligible
                    for s in 1:n_eligible
                        k = ws.eligible_buf[s]
                        # Observe one random participant's strategy (DD-4)
                        if rand(rng) < 0.5
                            ws.obs_pool[obs_pos] = ws.action[ws.pair_i[k]]
                        else
                            ws.obs_pool[obs_pos] = ws.action[ws.pair_j[k]]
                        end
                        obs_pos += 1
                    end
                else
                    # Fisher-Yates partial shuffle for sampling
                    sampled = sample(rng, view(ws.eligible_buf, 1:n_eligible), n_sample; replace=false)
                    for k in sampled
                        if rand(rng) < 0.5
                            ws.obs_pool[obs_pos] = ws.action[ws.pair_i[k]]
                        else
                            ws.obs_pool[obs_pos] = ws.action[ws.pair_j[k]]
                        end
                        obs_pos += 1
                    end
                end
            end
        end

        # 2d. Recompute experience belief from FIFO
        agents[i].b_exp_A = compute_b_exp_A(agents[i])
    end

    ws.obs_offset[N + 1] = obs_pos
    return nothing
end

# ══════════════════════════════════════════════════════════════
# Stage 3: Confidence Update
# ══════════════════════════════════════════════════════════════

"""
    stage_3_confidence!(agents, ws, params; use_network=false)

Update predictive confidence and window size based on prediction accuracy.
When use_network=true, skip unmatched agents (no prediction was made).
"""
function stage_3_confidence!(agents::Vector{AgentState}, ws::TickWorkspace,
                              params::SimulationParams; use_network::Bool=false)
    for i in 1:params.N
        # Skip unmatched agents on network topology
        if use_network && ws.partner_id[i] == 0
            continue
        end

        partner_act = ws.partner_action[i]
        my_pred = ws.prediction[i]
        correct = (my_pred == partner_act)

        if correct
            agents[i].C = agents[i].C + params.alpha * (1.0 - agents[i].C)
        else
            agents[i].C = agents[i].C * (1.0 - params.beta)
        end

        # Clamp (defensive)
        agents[i].C = clamp(agents[i].C, 0.0, 1.0)

        # Recompute window size
        agents[i].w = params.w_base + floor(Int, agents[i].C * (params.w_max - params.w_base))
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════
# Stage 4: Normative Update
# ══════════════════════════════════════════════════════════════

"""
    ddm_update!(agent::AgentState, params::SimulationParams)

Pre-crystallisation DDM evidence accumulation (DD-11, DD-12).

Uses b_exp (experience belief) as input signal instead of raw single-tick
observations. No additive noise — sampling noise is captured by b_exp's
finite-window variance. This ensures crystallisation requires a sustained
directional signal, not a random walk.
"""
function ddm_update!(agent::AgentState, params::SimulationParams)
    # a) Signed consistency from experience belief (DD-11)
    f_diff = 2.0 * agent.b_exp_A - 1.0   # = b_exp_A - b_exp_B, ∈ [-1, 1]

    # b) Drift: confidence-gated
    drift = (1.0 - agent.C) * f_diff

    # c) Signal push from previous tick's enforcement (DD-5, DD-8)
    signal_push = 0.0
    if agent.pending_signal != NO_SIGNAL
        direction = agent.pending_signal == STRATEGY_A ? 1.0 : -1.0
        signal_push = params.Phi * (1.0 - agent.C) * params.gamma_signal * direction
        agent.pending_signal = NO_SIGNAL  # consumed
    end

    # d) Evidence accumulation (no noise — DD-12)
    agent.e += drift + signal_push

    # e) Crystallisation check
    if abs(agent.e) >= params.theta_crystal
        agent.r = agent.e > 0 ? STRATEGY_A : STRATEGY_B
        agent.sigma = params.sigma_0
        agent.a = 0
        # e is NOT reset; irrelevant post-crystallisation
    end

    return nothing
end

"""
    post_crystal_update!(agent::AgentState, agent_id::Int,
                         obs_pool::Vector{Int8}, obs_start::Int, obs_end::Int,
                         ws::TickWorkspace, params::SimulationParams)

Post-crystallisation: anomaly tracking, strengthening, crisis, dissolution.
"""
function post_crystal_update!(agent::AgentState, agent_id::Int,
                               obs_pool::Vector{Int8}, obs_start::Int, obs_end::Int,
                               ws::TickWorkspace, params::SimulationParams)
    norm = agent.r

    # Separate partner observation (first) from V observations
    partner_action = obs_pool[obs_start]

    # Count violations and conformities from V observations
    v_violations = 0
    v_conform = 0
    for idx in (obs_start + 1):obs_end
        if obs_pool[idx] != norm
            v_violations += 1
        else
            v_conform += 1
        end
    end

    # Handle partner observation
    partner_conform = 0
    partner_violation = 0
    enforcement_triggered = false

    if partner_action == norm
        partner_conform = 1
    else
        # Partner violated — enforce or accumulate? (DD-6, DD-7)
        can_enforce = (params.Phi > 0) && (agent.sigma > params.theta_enforce)
        if can_enforce
            enforcement_triggered = true
            # Partner violation NOT counted as anomaly
        else
            partner_violation = 1
        end
    end

    # Totals
    total_violations = v_violations + partner_violation
    total_conform = v_conform + partner_conform

    # Batch strengthening (DD-3)
    for _ in 1:total_conform
        agent.sigma = min(1.0, agent.sigma + params.alpha_sigma * (1.0 - agent.sigma))
    end

    # Batch anomaly accumulation (DD-3)
    agent.a += total_violations

    # Crisis check (once, after all updates) (DD-3)
    if agent.a >= params.theta_crisis
        agent.sigma *= params.lambda_crisis
        agent.a = 0

        # Dissolution check
        if agent.sigma < params.sigma_min
            agent.r = NO_NORM
            agent.e = 0.0
            agent.sigma = 0.0
            agent.a = 0
        end
    end

    # Record enforcement intent for Stage 5
    if enforcement_triggered
        ws.enforce_target[agent_id] = ws.partner_id[agent_id]
        ws.enforce_strategy[agent_id] = norm
    end

    # Consume wasted pending signal (DD-8)
    agent.pending_signal = NO_SIGNAL

    return nothing
end

"""
    stage_4_normative!(agents, ws, params, rng; use_network=false)

Update normative memory: DDM (pre-crystallisation) or anomaly/crisis (post-crystallisation).
When use_network=true, skip unmatched agents (no observations to process).
"""
function stage_4_normative!(agents::Vector{AgentState}, ws::TickWorkspace,
                             params::SimulationParams, rng::AbstractRNG;
                             use_network::Bool=false)
    N = params.N

    # Reset enforcement intents
    fill!(ws.enforce_target, 0)

    for i in 1:N
        # Skip unmatched agents on network topology
        if use_network && ws.partner_id[i] == 0
            continue
        end

        obs_start = ws.obs_offset[i]
        obs_end = ws.obs_offset[i + 1] - 1

        if agents[i].r == NO_NORM
            # Pre-crystallisation: DDM (uses b_exp, not raw observations — DD-11)
            ddm_update!(agents[i], params)
        else
            # Post-crystallisation: anomaly / strengthening / crisis
            post_crystal_update!(agents[i], i, ws.obs_pool, obs_start, obs_end, ws, params)
        end
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════
# Stage 5: Partner-Directed Enforcement
# ══════════════════════════════════════════════════════════════

"""
    stage_5_enforce!(agents, ws, params)

Write pending_signal to enforcement targets. One-tick delay (DD-5).
"""
function stage_5_enforce!(agents::Vector{AgentState}, ws::TickWorkspace,
                           params::SimulationParams)
    count = 0

    for i in 1:params.N
        if ws.enforce_target[i] != 0
            partner_id = ws.enforce_target[i]
            agents[partner_id].pending_signal = ws.enforce_strategy[i]
            count += 1
        end
    end

    ws.num_enforcements = count
    return nothing
end

# ══════════════════════════════════════════════════════════════
# Stage 6: Metrics and Norm Detection
# ══════════════════════════════════════════════════════════════

"""
    count_consecutive_ticks(history, tick_count, predicate)

Count consecutive ticks at the END of history satisfying predicate.
"""
function count_consecutive_ticks(history::Vector{TickMetrics}, tick_count::Int, predicate)
    n = 0
    for idx in tick_count:-1:1
        if predicate(history[idx])
            n += 1
        else
            break
        end
    end
    return n
end

"""
    all_layers_met(fraction_A, belief_error, belief_var,
                   frac_crystallised, frac_dominant_norm, params)::Bool

Check whether all 3 continuous layers exceed their convergence thresholds.
"""
function all_layers_met(fraction_A::Float64, belief_error::Float64, belief_var::Float64,
                        frac_crystallised::Float64, frac_dominant_norm::Float64,
                        params::SimulationParams)::Bool
    majority = max(fraction_A, 1.0 - fraction_A)
    return majority >= params.thresh_majority &&
           belief_error < params.thresh_belief_error &&
           belief_var < params.thresh_belief_var &&
           frac_crystallised >= params.thresh_crystallised &&
           frac_dominant_norm >= params.thresh_dominant_norm
end

"""
    stage_6_metrics(agents, ws, t, history, tick_count, params; use_network=false)

Compute all per-tick metrics. Read-only (no state modification).
When use_network=true, uses ws.n_pairs_formed instead of N÷2 for coordination rate.
"""
function stage_6_metrics(agents::Vector{AgentState}, ws::TickWorkspace,
                          t::Int, history::Vector{TickMetrics}, tick_count::Int,
                          params::SimulationParams; use_network::Bool=false)
    N = params.N
    n_pairs = use_network ? ws.n_pairs_formed : N ÷ 2

    # Fraction A
    n_A = 0
    for i in 1:N
        if ws.action[i] == STRATEGY_A
            n_A += 1
        end
    end
    fraction_A = n_A / N

    # Mean confidence
    sum_C = 0.0
    for i in 1:N
        sum_C += agents[i].C
    end
    mean_C = sum_C / N

    # Coordination rate
    n_coord = 0
    for k in 1:n_pairs
        if ws.coordinated[k]
            n_coord += 1
        end
    end
    coord_rate = n_pairs > 0 ? n_coord / n_pairs : 0.0

    # Crystallisation stats
    num_cryst = 0
    sum_sigma = 0.0
    for i in 1:N
        if agents[i].r != NO_NORM
            num_cryst += 1
            sum_sigma += agents[i].sigma
        end
    end
    mean_sigma = num_cryst > 0 ? sum_sigma / num_cryst : 0.0

    # Belief accuracy and consensus (using b_exp_A, not b_eff_A)
    sum_error = 0.0
    sum_b = 0.0
    for i in 1:N
        sum_error += abs(agents[i].b_exp_A - fraction_A)
        sum_b += agents[i].b_exp_A
    end
    belief_error = sum_error / N
    mean_b = sum_b / N

    sum_var = 0.0
    for i in 1:N
        sum_var += (agents[i].b_exp_A - mean_b)^2
    end
    belief_var = sum_var / N

    # Fraction of crystallised agents holding the dominant norm
    frac_dominant_norm = 0.0
    if num_cryst > 0
        n_norm_A = 0
        n_norm_B = 0
        for i in 1:N
            if agents[i].r == STRATEGY_A
                n_norm_A += 1
            elseif agents[i].r == STRATEGY_B
                n_norm_B += 1
            end
        end
        frac_dominant_norm = max(n_norm_A, n_norm_B) / N
    end

    # Convergence counter: all 3 layers met
    frac_crystallised = num_cryst / N
    layers_met = all_layers_met(fraction_A, belief_error, belief_var,
                                frac_crystallised, frac_dominant_norm, params)
    if layers_met
        prev = tick_count > 0 ? history[tick_count].convergence_counter : 0
        conv_counter = prev + 1
    else
        conv_counter = 0
    end

    return TickMetrics(
        t, fraction_A, mean_C, coord_rate, num_cryst,
        mean_sigma, ws.num_enforcements,
        belief_error, belief_var, frac_dominant_norm, conv_counter,
    )
end
