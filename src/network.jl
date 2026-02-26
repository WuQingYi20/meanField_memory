# ══════════════════════════════════════════════════════════════
# Network Topology Generation
# ══════════════════════════════════════════════════════════════

# ── Round-Robin Schedule ─────────────────────────────────────

"""
    RoundRobinSchedule

Pre-computed round-robin tournament schedule. Every pair of agents meets exactly
once per cycle (N-1 rounds). Between cycles, the round order is reshuffled.

All agents are matched every tick (like complete graph), but pair assignment
follows a deterministic schedule within each cycle.
"""
mutable struct RoundRobinSchedule
    N::Int
    n_rounds::Int               # N - 1
    n_pairs::Int                # N ÷ 2
    schedule_i::Matrix{Int}     # (n_pairs, n_rounds) — first agent in each pair
    schedule_j::Matrix{Int}     # (n_pairs, n_rounds) — second agent in each pair
    round_order::Vector{Int}    # shuffled permutation of 1:n_rounds
    current_idx::Int            # position in round_order (1-based, wraps)
end

"""
    generate_roundrobin_schedule(N::Int, rng::AbstractRNG)

Generate a round-robin tournament schedule for N agents (N must be even).

Uses the classic polygon algorithm: fix agent N, rotate agents 1..N-1 through
N-1 rounds. Each round produces N/2 pairs; every pair appears exactly once
per cycle.
"""
function generate_roundrobin_schedule(N::Int, rng::AbstractRNG)
    iseven(N) || throw(ArgumentError("N must be even for round-robin, got $N"))
    N >= 4 || throw(ArgumentError("N must be ≥ 4 for round-robin, got $N"))

    n_rounds = N - 1
    n_pairs = N ÷ 2
    nm1 = N - 1  # number of rotating players

    schedule_i = Matrix{Int}(undef, n_pairs, n_rounds)
    schedule_j = Matrix{Int}(undef, n_pairs, n_rounds)

    for r in 0:(n_rounds - 1)
        # Pair 1: rotating position 1 ↔ fixed player N
        schedule_i[1, r + 1] = (r % nm1) + 1
        schedule_j[1, r + 1] = N

        # Remaining pairs: position p ↔ position (N+1-p) among rotating players
        for p in 2:n_pairs
            player_a = ((p - 1 + r) % nm1) + 1
            player_b = ((N - p + r) % nm1) + 1
            schedule_i[p, r + 1] = player_a
            schedule_j[p, r + 1] = player_b
        end
    end

    round_order = collect(1:n_rounds)
    shuffle!(rng, round_order)

    return RoundRobinSchedule(N, n_rounds, n_pairs, schedule_i, schedule_j, round_order, 1)
end

# ── Type alias for network argument ─────────────────────────

const NetworkArg = Union{Nothing, Vector{Vector{Int}}, RoundRobinSchedule}

# ── Topology dispatcher ─────────────────────────────────────

"""
    generate_network(topology::Symbol, N::Int, k::Int; p::Float64=0.1, rng::AbstractRNG=Random.default_rng())

Generate a network for the given topology. Returns:
- `nothing` for :complete
- `RoundRobinSchedule` for :roundrobin
- `Vector{Vector{Int}}` adjacency list for :ring, :smallworld, :scalefree

- `:complete`    — returns nothing (use standard mean-field pairing)
- `:roundrobin`  — round-robin tournament schedule (k is ignored)
- `:ring`        — ring lattice with k/2 neighbors each side
- `:smallworld`  — Watts-Strogatz rewiring with probability p
- `:scalefree`   — Barabási-Albert preferential attachment with m = k÷2
"""
function generate_network(topology::Symbol, N::Int, k::Int;
                           p::Float64=0.1, rng::AbstractRNG=Random.default_rng())
    topology == :complete && return nothing
    topology == :roundrobin && return generate_roundrobin_schedule(N, rng)
    topology == :ring && return generate_ring_lattice(N, k)
    topology == :smallworld && return generate_smallworld(N, k, p, rng)
    topology == :scalefree && return generate_scalefree(N, k ÷ 2, rng)
    throw(ArgumentError("Unknown topology: $topology. Use :complete, :roundrobin, :ring, :smallworld, or :scalefree"))
end

"""
    generate_ring_lattice(N, k)

Ring lattice: each node connects to k/2 nearest neighbors on each side.
k must be even and ≥ 2.
"""
function generate_ring_lattice(N::Int, k::Int)
    k >= 2 || throw(ArgumentError("k must be ≥ 2, got $k"))
    iseven(k) || throw(ArgumentError("k must be even for ring lattice, got $k"))
    half_k = k ÷ 2

    adj = [Int[] for _ in 1:N]
    for i in 1:N
        for d in 1:half_k
            j = mod1(i + d, N)
            if j != i
                push!(adj[i], j)
                push!(adj[j], i)
            end
        end
    end

    # Deduplicate (each edge added from both endpoints)
    for i in 1:N
        sort!(adj[i])
        unique!(adj[i])
    end

    return adj
end

"""
    generate_smallworld(N, k, p, rng)

Watts-Strogatz small-world: start from ring lattice(N, k), then rewire each
edge with probability p to a random non-neighbor.
"""
function generate_smallworld(N::Int, k::Int, p::Float64, rng::AbstractRNG)
    adj = generate_ring_lattice(N, k)

    # Build edge set for O(1) lookup
    edge_set = Set{Tuple{Int,Int}}()
    for i in 1:N
        for j in adj[i]
            if i < j
                push!(edge_set, (i, j))
            end
        end
    end

    # Rewire: iterate over original ring edges
    half_k = k ÷ 2
    for i in 1:N
        for d in 1:half_k
            j = mod1(i + d, N)
            u, v = minmax(i, j)
            if rand(rng) < p && (u, v) in edge_set
                # Pick a new target for i
                new_j = rand(rng, 1:N)
                attempts = 0
                while new_j == i || minmax(i, new_j) in edge_set
                    new_j = rand(rng, 1:N)
                    attempts += 1
                    attempts > 100 && break
                end
                if attempts <= 100
                    # Remove old edge
                    delete!(edge_set, (u, v))
                    filter!(!=(j), adj[i])
                    filter!(!=(i), adj[j])

                    # Add new edge
                    push!(edge_set, minmax(i, new_j))
                    push!(adj[i], new_j)
                    push!(adj[new_j], i)
                end
            end
        end
    end

    return adj
end

"""
    generate_scalefree(N, m, rng)

Barabási-Albert preferential attachment. Start with a clique of m+1 nodes,
then attach each new node to m existing nodes with probability proportional
to degree.
"""
function generate_scalefree(N::Int, m::Int, rng::AbstractRNG)
    m >= 1 || throw(ArgumentError("m must be ≥ 1, got $m"))
    m + 1 <= N || throw(ArgumentError("N must be > m, got N=$N, m=$m"))

    adj = [Int[] for _ in 1:N]

    # Initial clique of m+1 nodes
    for i in 1:(m+1)
        for j in (i+1):(m+1)
            push!(adj[i], j)
            push!(adj[j], i)
        end
    end

    # Degree-weighted stub list for O(1) preferential attachment sampling
    stubs = Int[]
    for i in 1:(m+1)
        for _ in 1:length(adj[i])
            push!(stubs, i)
        end
    end

    # Add remaining nodes
    for new_node in (m+2):N
        targets = Set{Int}()
        while length(targets) < m
            candidate = stubs[rand(rng, 1:length(stubs))]
            if candidate != new_node
                push!(targets, candidate)
            end
        end

        for t in targets
            push!(adj[new_node], t)
            push!(adj[t], new_node)
            push!(stubs, new_node)
            push!(stubs, t)
        end
    end

    return adj
end

"""
    network_stats(network)

Return (mean_degree, min_degree, max_degree, std_degree) for logging.
"""
function network_stats(network::Vector{Vector{Int}})
    degrees = [length(adj) for adj in network]
    μ = mean(degrees)
    σ = std(degrees; corrected=false)
    return (mean_degree=μ, min_degree=minimum(degrees),
            max_degree=maximum(degrees), std_degree=σ)
end
