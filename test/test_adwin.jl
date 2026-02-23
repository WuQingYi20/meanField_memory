using Test
using DualMemoryABM

@testset "ADWIN2 Tests" begin

    # ── A1: Empty ADWIN has count=0, mean=0 ──
    @testset "A1: Empty state" begin
        adw = ADWIN2(M=5, delta=0.01)
        @test adwin_count(adw) == 0
        @test adwin_mean(adw) == 0.0
        @test adw.n_buckets == 0
    end

    # ── A2: Single add → count=1, mean=value ──
    @testset "A2: Single observation" begin
        adw = ADWIN2(M=5, delta=0.01)
        adwin_add!(adw, 0.7)
        @test adwin_count(adw) == 1
        @test adwin_mean(adw) ≈ 0.7
    end

    # ── A3: Stationary stream → window grows ──
    @testset "A3: Stationary stream grows" begin
        adw = ADWIN2(M=5, delta=0.01)
        # Feed 200 observations from a stationary stream (all 0.5)
        for _ in 1:200
            adwin_add!(adw, 0.5)
        end
        # Window should retain most observations (no change detected)
        @test adwin_count(adw) >= 100  # should keep most
        @test adwin_mean(adw) ≈ 0.5 atol=0.01
    end

    # ── A4: Sudden change detection ──
    @testset "A4: Sudden change detection" begin
        adw = ADWIN2(M=5, delta=0.01)

        # Phase 1: 200 observations of 0.8
        for _ in 1:200
            adwin_add!(adw, 0.8)
        end
        count_before = adwin_count(adw)

        # Phase 2: 200 observations of 0.2 (sudden shift)
        for _ in 1:200
            adwin_add!(adw, 0.2)
        end

        # After change, ADWIN should have shrunk — mean should be closer to 0.2
        @test adwin_mean(adw) < 0.5
        # The window should not contain the full 400 observations
        @test adwin_count(adw) < 400
    end

    # ── A5: Gradual change detection ──
    @testset "A5: Gradual change" begin
        adw = ADWIN2(M=5, delta=0.01)

        # Feed gradually increasing values
        for t in 1:300
            val = 0.3 + 0.4 * (t / 300.0)  # 0.3 → 0.7
            adwin_add!(adw, val)
        end

        # Mean should reflect recent values more than old ones
        # (ADWIN may or may not detect gradual change depending on rate)
        @test adwin_mean(adw) > 0.4  # at least above the initial mean
        @test adwin_count(adw) >= 1
    end

    # ── A6: Compression invariant ──
    @testset "A6: Compression invariant (at most M+1 buckets per level)" begin
        M = 5
        adw = ADWIN2(M=M, delta=0.01)

        # Feed many observations to trigger compression
        for _ in 1:500
            adwin_add!(adw, 0.5)
        end

        # Count buckets per level (level = log2(bucket_count))
        level_counts = Dict{Int,Int}()
        for i in 1:adw.n_buckets
            bc = adw.bucket_count[i]
            level = 0
            tmp = bc
            while tmp > 1
                level += 1
                tmp ÷= 2
            end
            level_counts[level] = get(level_counts, level, 0) + 1
        end

        # Each level should have at most M+1 buckets
        for (level, cnt) in level_counts
            @test cnt <= M + 1
        end
    end

    # ── A7: Reset clears everything ──
    @testset "A7: Reset" begin
        adw = ADWIN2(M=5, delta=0.01)
        for _ in 1:50
            adwin_add!(adw, 0.6)
        end
        @test adwin_count(adw) > 0

        adwin_reset!(adw)
        @test adwin_count(adw) == 0
        @test adwin_mean(adw) == 0.0
        @test adw.n_buckets == 0
        @test adw.total == 0.0
    end

    # ── A8: Values in [0,1] → mean in [0,1] ──
    @testset "A8: Mean bounded in [0,1]" begin
        adw = ADWIN2(M=5, delta=0.01)
        for _ in 1:100
            adwin_add!(adw, rand())
        end
        @test 0.0 <= adwin_mean(adw) <= 1.0
    end

end  # @testset "ADWIN2 Tests"
