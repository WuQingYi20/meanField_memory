#!/usr/bin/env julia
#
# Plot full-range speedup: N = 4..20,000
# Combines small_N_speedup.csv + large_N/ablation_3x2_N*.csv
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV, DataFrames, Statistics, Printf
using Plots; gr()

const RESULTS = joinpath(@__DIR__, "..", "results")
const FIGDIR  = joinpath(@__DIR__, "..", "figures")

# ── 1. Small N data (already summarised) ──
small = CSV.read(joinpath(RESULTS, "small_N_speedup.csv"), DataFrame)

# ── 2. Medium N data from 3x2 ablation (N=100, 500 from the main ablation) ──
# We need to also get N=100, 500 from the original ablation
ablation_path = joinpath(RESULTS, "ablation_3x2_report.md")

# ── 3. Large N data from ablation_3x2_N*.csv ──
large_Ns = [1000, 2000, 5000, 10000, 20000]
rows = []

for N in large_Ns
    path = joinpath(RESULTS, "large_N", "ablation_3x2_N$(N).csv")
    if !isfile(path)
        println("Missing: $path")
        continue
    end
    df = CSV.read(path, DataFrame)

    # Dyn_noNorm = experiential only, Dyn_Norm = full model
    off = filter(r -> r.condition == "Dyn_noNorm" && r.converged, df)
    on  = filter(r -> r.condition == "Dyn_Norm"   && r.converged, df)

    n_off_total = nrow(filter(r -> r.condition == "Dyn_noNorm", df))
    n_on_total  = nrow(filter(r -> r.condition == "Dyn_Norm",   df))

    if nrow(off) == 0 || nrow(on) == 0
        println("N=$N: insufficient converged trials (off=$(nrow(off)), on=$(nrow(on)))")
        continue
    end

    mean_off = mean(off.conv_tick)
    mean_on  = mean(on.conv_tick)
    se_off   = std(off.conv_tick) / sqrt(nrow(off))
    se_on    = std(on.conv_tick) / sqrt(nrow(on))
    speedup  = mean_off / mean_on

    push!(rows, (N=N, mean_tick_off=mean_off, mean_tick_on=mean_on,
                 se_off=se_off, se_on=se_on, speedup=speedup,
                 conv_rate_off=nrow(off)/n_off_total,
                 conv_rate_on=nrow(on)/n_on_total))

    @printf("N=%6d: off=%.1f (n=%d/%d)  on=%.1f (n=%d/%d)  speedup=%.1fx\n",
            N, mean_off, nrow(off), n_off_total, mean_on, nrow(on), n_on_total, speedup)
end

# ── 4. Also get N=100 and N=500 from the main sweep ──
sweep_path = joinpath(RESULTS, "large_N", "sweep_N_extended.csv")
if isfile(sweep_path)
    sweep_df = CSV.read(sweep_path, DataFrame)
    # The sweep uses condition labels like "D_full" and "B_lockin"
    # We need Dyn_noNorm (B_lockin has w_base=2,w_max=6,norm=false)
    # and Dyn_Norm (D_full has w_base=2,w_max=6,norm=true)
    # Actually let's check what conditions exist
    println("\nSweep conditions: ", unique(sweep_df.condition))
end

# Also check the original 3x2 ablation for N=100, 500
for N in [100, 500]
    # Check if there's ablation data in the results directory
    for pattern in ["ablation_3x2_results.csv", "convergence_cascade_summary.csv"]
        path = joinpath(RESULTS, pattern)
        if isfile(path)
            df = CSV.read(path, DataFrame)
            if hasproperty(df, :N) && N in df.N
                sub = filter(r -> r.N == N, df)
                println("Found N=$N in $pattern: $(nrow(sub)) rows")
                if hasproperty(sub, :condition)
                    println("  Conditions: ", unique(sub.condition))
                end
            end
        end
    end
end

# ── 5. Combine all data ──
# Small N
all_N       = Float64.(small.N)
all_speedup = Float64.(small.speedup)
all_se      = [s * sqrt((r.se_off/r.mean_tick_off)^2 + (r.se_on/r.mean_tick_on)^2)
               for (s, r) in zip(small.speedup, eachrow(small))]

# Large N
for r in rows
    push!(all_N, r.N)
    push!(all_speedup, r.speedup)
    se_sp = r.speedup * sqrt((r.se_off/r.mean_tick_off)^2 + (r.se_on/r.mean_tick_on)^2)
    push!(all_se, se_sp)
end

# Sort by N
order = sortperm(all_N)
all_N       = all_N[order]
all_speedup = all_speedup[order]
all_se      = all_se[order]

println("\n", "=" ^ 60)
println("Combined speedup data:")
println("=" ^ 60)
for i in eachindex(all_N)
    @printf("N=%6.0f  speedup=%6.1fx  SE=%.2f\n", all_N[i], all_speedup[i], all_se[i])
end

# ── 6. Plot ──
mkpath(FIGDIR)

p = plot(all_N, all_speedup,
    seriestype = :scatter,
    yerror = all_se,
    label = "Normative speedup",
    color = :royalblue,
    markerstrokecolor = :royalblue,
    markersize = 6,
    xlabel = "Population size N",
    ylabel = "Speedup ratio (Dyn-only / Dyn+Norm)",
    title = "Normative Speedup Across Population Sizes",
    legend = :topleft,
    size = (800, 500),
    grid = true, gridalpha = 0.3,
    fontfamily = "sans-serif", titlefontsize = 12,
    xscale = :log10,
    yscale = :log10,
    minorgrid = true, minorgridalpha = 0.15,
)

# Connecting line
plot!(p, all_N, all_speedup,
    label = "", lw = 2.5, color = :royalblue, alpha = 0.5)

# Horizontal dashed line at y=1
hline!(p, [1.0], color = :gray, ls = :dash, lw = 1.5,
    label = "No speedup (ratio = 1)")

# Annotate convergence failure region
if any(r -> r.conv_rate_off < 1.0, rows)
    fail_N = [r.N for r in rows if r.conv_rate_off < 1.0]
    fail_sp = [r.speedup for r in rows if r.conv_rate_off < 1.0]
    scatter!(p, fail_N, fail_sp,
        label = "Exp-only conv. < 100%",
        color = :red, markershape = :diamond, markersize = 8,
        markerstrokecolor = :red, alpha = 0.7)
end

figpath = joinpath(FIGDIR, "fig_N_speedup.png")
savefig(p, figpath)
println("\nFigure saved to $figpath")
