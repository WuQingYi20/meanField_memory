# ══════════════════════════════════════════════════════════════
# Network Topology Generation
# ══════════════════════════════════════════════════════════════

"""
    generate_network(topology::Symbol, N::Int, k::Int; p::Float64=0.1, rng::AbstractRNG=Random.default_rng())

Generate an adjacency list for the given topology. Returns `nothing` for :complete.

- `:complete` — returns nothing (use standard mean-field pairing)
- `:ring`     — ring lattice with k/2 neighbors each side
- `:smallworld` — Watts-Strogatz rewiring with probability p
- `:scalefree`  — Barabási-Albert preferential attachment with m = k÷2
"""
function generate_network(topology::Symbol, N::Int, k::Int;
                           p::Float64=0.1, rng::AbstractRNG=Random.default_rng())
    topology == :complete && return nothing
    topology == :ring && return generate_ring_lattice(N, k)
    topology == :smallworld && return generate_smallworld(N, k, p, rng)
    topology == :scalefree && return generate_scalefree(N, k ÷ 2, rng)
    throw(ArgumentError("Unknown topology: $topology. Use :complete, :ring, :smallworld, or :scalefree"))
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
