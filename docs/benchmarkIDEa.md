按生命周期阶段组织的指标
涌现阶段
对称打破时间 (symmetry_break_tick)：fraction_A 首次离开 [0.4, 0.6] 区间的 tick。测量的是从均匀状态到出现方向性偏移需要多久。DM_base 靠采样噪声，EWA 靠 λ 放大，DM_full 额外有 DDM——三者应该有不同的分布形状。
涌现路径的异质性 (agent_divergence_at_break)：对称打破时刻各 agent 信念的标准差。EWA 中所有 agent 的 attraction 应该比较同步，DM 中因为 confidence 分化应该有更大的个体差异。
谁先动 (first_mover_confidence)：最早稳定选择多数策略的那批 agent 的平均 confidence。这直接检验你的 H2——低 confidence agents 应该先动。对 EWA 来说这个指标没有对应物，这本身就是现象差异。
扩散阶段
60% → 95% majority 的时间 (diffusion_duration)：从初步多数到压倒性多数需要多久。你已有的数据显示 DM_base 在这个阶段特别慢（长尾锁定），DM_full 靠 enforcement 加速，EWA 靠 λ 放大。
扩散路径形状 (majority_trajectory)：记录每 tick 的 fraction_A 序列。EWA 应该是 sigmoid 形（平滑 S 曲线），DM_base 应该有一个长平台然后突然加速（自催化），DM_full 应该有阶梯状特征（每一波 crystallisation 推动一次跳跃）。不需要单一数字指标，直接画轨迹曲线对比。
Crystallisation cascade (crystal_wave_ticks)：仅 DM_full 有。记录每 tick 新增 crystallised agents 的数量。如果存在 cascade（enforcement 触发链式 crystallisation），这个序列会有明显的 burst 而非均匀分布。EWA 和 DM_base 没有这个指标，这本身就是现象差异。
稳定阶段
均衡的内部结构：在系统行为收敛后（比如 fraction_A > 0.95 持续 50 ticks 后），快照各模型的内部状态。DM_full 应该有高 norm strength、高 confidence、一致的 crystallised rule。DM_base 有高 confidence 但没有 norm。EWA 有收敛的 attraction 但没有任何规范表征。这个快照本身就是定性证据。
Enforcement 活跃度 (enforcement_rate_at_steady)：稳定期每 tick 的 enforcement 事件数。如果规范已完全内化，enforcement 应该趋近于零——不是因为没有能力 enforce，而是因为没人违反。这是一个只有你的模型能产生的"沉默的狗"式证据。
受扰阶段（perturbation experiment）
这是最关键的阶段。设计：在稳定后的某个 tick（比如 tick 300），随机选 20% agents 强制重置。
扰动设计的具体操作：
juliafunction perturb!(agents, params, rng; frac=0.20)
    n_perturb = round(Int, frac * params.N)
    targets = sample(rng, 1:params.N, n_perturb; replace=false)
    for i in targets
        # 翻转到少数策略 — 通过重置信念
        agents[i].b_exp = [0.5, 0.5]
        agents[i].C = params.C0
        agents[i].w = params.w_base + floor(Int, params.C0 * (params.w_max - params.w_base))
        # 清空 FIFO
        empty!(agents[i].fifo)
        # 对 DM_full：dissolution
        if agents[i].r !== nothing
            agents[i].r = nothing
            agents[i].sigma = 0.0
            agents[i].a = 0
            agents[i].e = 0.0
        end
    end
end
恢复时间 (recovery_ticks)：从扰动到重新达到 95% majority 的 tick 数。
恢复深度 (perturbation_depth)：扰动后 fraction_A 的最低点。DM_full 的 crystallised agents 有 compliance 撑着，最低点应该更浅。
恢复路径 (recovery_trajectory)：扰动后的 fraction_A 序列。DM_full 预测是 V 形快速恢复（enforcement 主动纠正）。DM_base 预测是缓慢漂移恢复（靠经验冲刷）。EWA 预测取决于 λ。
Norm survival rate：扰动后，未被 perturb 的 crystallised agents 中有多少触发了 crisis？有多少 dissolve 了？这量化了规范的韧性。只有 DM_full 有这个指标。
Re-crystallisation rate：被 perturb 的 agents 有多少重新 crystallise 了？crystallise 到了哪个 norm？如果 enforcement 有效，他们应该快速 re-crystallise 到多数 norm。
崩溃与重生（extreme perturbation）
用更大的扰动（比如 50%）测试规范是否能存活。预测：DM_full 在 20% 扰动下恢复，在 50% 扰动下可能触发大规模 crisis → dissolution → 重新涌现（可能翻转到另一个 norm）。这种"规范死亡与重生"的动态是只有你的模型能产生的现象。
测量架构
juliastruct LifecycleMetrics
    # 涌现
    symmetry_break_tick::Int          # frac_A 首次离开 [0.4, 0.6]
    first_mover_mean_C::Float64       # 前 10% 收敛 agents 的平均 C

    # 扩散
    diffusion_duration::Int           # 60% → 95% majority 的 tick 数
    crystal_cascade_bursts::Int       # crystallisation burst 的次数（仅 DM_full）

    # 稳定
    steady_state_tick::Int            # 95% majority 持续 50 ticks 的起始 tick
    enforcement_at_steady::Float64    # 稳定后平均 enforcement/tick

    # 扰动恢复
    recovery_ticks::Int               # 扰动后恢复到 95% 的时间
    perturbation_depth::Float64       # fraction_A 最低点
    norm_survival_rate::Float64       # 未 perturb agents 保持 norm 的比例
    recrystallisation_rate::Float64   # perturb agents 重新 crystallise 的比例

    # 崩溃/重生（极端扰动）
    norm_flip::Bool                   # 重生后 norm 是否翻转
end
这些指标中，有些所有模型都能计算（symmetry_break_tick、diffusion_duration、recovery_ticks），有些只有 DM_full 能产生（crystal_cascade、norm_survival、recrystallisation）。后者本身就是论证——不是你的模型在相同指标上更好，而是你的模型能产生其他模型结构性无法表达的量。