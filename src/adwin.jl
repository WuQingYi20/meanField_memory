# ══════════════════════════════════════════════════════════════
# ADWIN2 — Adaptive Windowing (Bifet & Gavalda, 2007)
#
# Exponential histogram approach: automatically grows window
# during stable periods and shrinks upon distribution shift.
# ══════════════════════════════════════════════════════════════

mutable struct ADWIN2
    bucket_sum::Vector{Float64}    # sum of values per bucket
    bucket_count::Vector{Int}      # number of elements per bucket (power of 2)
    n_buckets::Int                 # active bucket count
    M::Int                         # max buckets per level (default 5)
    delta::Float64                 # confidence parameter
    total::Float64                 # window sum
    count::Int                     # window element count
end

"""
    ADWIN2(; M=5, delta=0.01)

Create an empty ADWIN2 instance. Pre-allocates 128 bucket slots
(sufficient for M=5 with windows up to 2^21).
"""
function ADWIN2(; M::Int=5, delta::Float64=0.01)
    max_slots = 128
    return ADWIN2(
        zeros(Float64, max_slots),   # bucket_sum
        zeros(Int, max_slots),       # bucket_count
        0,                           # n_buckets
        M,                           # M
        delta,                       # delta
        0.0,                         # total
        0,                           # count
    )
end

"""
    adwin_mean(adw::ADWIN2)

Return the mean of the current window. Returns 0.0 if empty.
"""
@inline function adwin_mean(adw::ADWIN2)
    adw.count == 0 && return 0.0
    return adw.total / adw.count
end

"""
    adwin_count(adw::ADWIN2)

Return the number of elements in the current window.
"""
@inline adwin_count(adw::ADWIN2) = adw.count

"""
    adwin_reset!(adw::ADWIN2)

Clear all buckets and reset the window to empty.
"""
function adwin_reset!(adw::ADWIN2)
    fill!(adw.bucket_sum, 0.0)
    fill!(adw.bucket_count, 0)
    adw.n_buckets = 0
    adw.total = 0.0
    adw.count = 0
    return nothing
end

"""
    adwin_add!(adw::ADWIN2, value::Float64)

Add a new observation. Inserts at front (bucket 1), compresses
the exponential histogram, then runs Hoeffding-bound change detection.
"""
function adwin_add!(adw::ADWIN2, value::Float64)
    # Insert new bucket at position 1 (shift existing right)
    n = adw.n_buckets
    if n + 1 > length(adw.bucket_sum)
        error("ADWIN2: bucket overflow — increase pre-allocated slots")
    end

    # Shift existing buckets right by 1
    for i in (n + 1):-1:2
        adw.bucket_sum[i] = adw.bucket_sum[i - 1]
        adw.bucket_count[i] = adw.bucket_count[i - 1]
    end
    adw.bucket_sum[1] = value
    adw.bucket_count[1] = 1
    adw.n_buckets += 1

    adw.total += value
    adw.count += 1

    # Compress: merge when a level has M+2 buckets
    _adwin_compress!(adw)

    # Detect change and shrink if needed
    _adwin_detect_and_shrink!(adw)

    return nothing
end

"""
    _adwin_compress!(adw::ADWIN2)

Merge oldest pair of buckets at each level when level has M+2 buckets.
Buckets at level ℓ have count = 2^ℓ. Level 0 buckets have count=1, etc.
"""
function _adwin_compress!(adw::ADWIN2)
    # Scan from newest (bucket 1) to oldest, counting buckets per level
    i = 1
    while i <= adw.n_buckets
        level_size = adw.bucket_count[i]  # expected count for this level

        # Count consecutive buckets with this count
        j = i
        while j <= adw.n_buckets && adw.bucket_count[j] == level_size
            j += 1
        end
        n_at_level = j - i  # number of buckets at this level

        if n_at_level >= adw.M + 2
            # Merge the two oldest at this level (positions j-1 and j-2)
            merge_pos = j - 2
            adw.bucket_sum[merge_pos] += adw.bucket_sum[merge_pos + 1]
            adw.bucket_count[merge_pos] = level_size * 2

            # Shift everything after merge_pos+1 left by 1
            for k in (merge_pos + 1):(adw.n_buckets - 1)
                adw.bucket_sum[k] = adw.bucket_sum[k + 1]
                adw.bucket_count[k] = adw.bucket_count[k + 1]
            end
            adw.bucket_sum[adw.n_buckets] = 0.0
            adw.bucket_count[adw.n_buckets] = 0
            adw.n_buckets -= 1

            # Recheck from the merged position (now at next level)
            # Don't advance i — the merged bucket might trigger another merge
        else
            i = j  # skip to next level
        end
    end
    return nothing
end

"""
    _adwin_detect_and_shrink!(adw::ADWIN2)

Scan O(log W) cut points (bucket boundaries) from oldest to newest.
For each cut: W0 = older portion, W1 = newer portion.
If |mean(W1) - mean(W0)| >= epsilon (Hoeffding bound), drop W0.
"""
function _adwin_detect_and_shrink!(adw::ADWIN2)
    adw.n_buckets <= 1 && return nothing

    while adw.n_buckets > 1
        found_change = false

        # Accumulate from the oldest bucket inward
        n0 = 0       # elements in W0 (old tail)
        sum0 = 0.0   # sum of W0

        # Scan from oldest to newest-1
        for i in adw.n_buckets:-1:2
            n0 += adw.bucket_count[i]
            sum0 += adw.bucket_sum[i]

            n1 = adw.count - n0      # elements in W1 (newer head)
            n1 <= 0 && continue

            sum1 = adw.total - sum0
            mean0 = sum0 / n0
            mean1 = sum1 / n1

            # Hoeffding bound with harmonic mean
            m = 1.0 / (1.0 / n0 + 1.0 / n1)  # harmonic mean
            n_total = Float64(adw.count)
            epsilon = sqrt(log(4.0 * n_total / adw.delta) / (2.0 * m))

            if abs(mean1 - mean0) >= epsilon
                # Drop the oldest bucket
                _adwin_drop_oldest!(adw)
                found_change = true
                break
            end
        end

        !found_change && break
    end

    return nothing
end

"""
    _adwin_drop_oldest!(adw::ADWIN2)

Remove the oldest (rightmost) bucket from the window.
"""
function _adwin_drop_oldest!(adw::ADWIN2)
    adw.n_buckets <= 0 && return nothing

    idx = adw.n_buckets
    adw.total -= adw.bucket_sum[idx]
    adw.count -= adw.bucket_count[idx]
    adw.bucket_sum[idx] = 0.0
    adw.bucket_count[idx] = 0
    adw.n_buckets -= 1

    return nothing
end
