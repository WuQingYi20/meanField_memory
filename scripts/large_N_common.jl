#!/usr/bin/env julia
#
# Shared infrastructure for large-N scaling experiments.
# Included by: large_N_sweep.jl, large_N_ablation.jl,
#              large_N_cascade.jl, large_N_causal.jl
#

using CSV, DataFrames, Dates

# ══════════════════════════════════════════════════════════════
# 1. Incremental CSV write (crash-safe, append-aware)
# ══════════════════════════════════════════════════════════════

"""
    incremental_csv_write(filepath, rows; append=true)

Write `rows` (vector of NamedTuples) to CSV.
- If the file does not exist, create it with a header row.
- If the file exists and `append=true`, append without header.
- Flushes after write to guard against crashes.
"""
function incremental_csv_write(filepath::String, rows; append::Bool=true)
    if isempty(rows)
        return
    end
    mkpath(dirname(filepath))
    if isfile(filepath) && append
        open(filepath, "a") do io
            CSV.write(io, rows; append=true)
            flush(io)
        end
    else
        CSV.write(filepath, rows)
    end
end

# ══════════════════════════════════════════════════════════════
# 2. Load completed trials for checkpoint / resume
# ══════════════════════════════════════════════════════════════

"""
    load_completed_trials(filepath, key_cols) → Set{Tuple}

Read an existing CSV and return a set of completed key-column
combinations so that they can be skipped on restart.

Example:
    done = load_completed_trials("sweep.csv", [:N, :condition, :trial])
    if (1000, "D_full_model", 3) in done
        # skip
    end
"""
function load_completed_trials(filepath::String, key_cols::Vector{Symbol})
    completed = Set{Tuple}()
    if !isfile(filepath)
        return completed
    end
    df = CSV.read(filepath, DataFrame)
    for row in eachrow(df)
        key = Tuple(row[c] for c in key_cols)
        push!(completed, key)
    end
    return completed
end

# ══════════════════════════════════════════════════════════════
# 3. Adaptive T_MAX based on population size
# ══════════════════════════════════════════════════════════════

"""
    select_T_MAX(N) → Int

Large populations may need more ticks to converge.
"""
function select_T_MAX(N::Int)::Int
    N <= 500  && return 3000
    N <= 2000 && return 5000
    return 10000
end

# ══════════════════════════════════════════════════════════════
# 4. Progress logging with timestamp
# ══════════════════════════════════════════════════════════════

"""
    progress_log(msg)

Print a timestamped progress message to stdout.
"""
function progress_log(msg::String)
    ts = Dates.format(now(), "HH:MM:SS")
    println("[$ts] $msg")
    flush(stdout)
end
