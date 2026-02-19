# 给导师的邮件草稿

---

**Subject**: Dual-Memory Norm Emergence Model — Simulation Progress Update

---

Dear Professor [Name],

Thank you for the suggestions in our last meeting regarding norm emergence timing and the role of communication. I've made significant progress since then and wanted to share the current results.

## What's been done

Based on your feedback, I refactored the convergence detection system from a single norm-level metric to a **3-layer convergence definition**:
- **Behavioral layer**: ≥95% agents choosing the same strategy
- **Belief layer**: individual beliefs aligned with actual population behavior
- **Crystallisation layer**: ≥80% agents holding a norm, ≥90% agreeing on *which* norm

Convergence requires all three layers sustained for 50 consecutive ticks. This gives us a much richer picture of *how* norms emerge, not just *whether* they do.

I then ran systematic parameter sweeps across 5 dimensions:

### Key findings

**1. Mechanism hierarchy (2×2 factorial)**

The normative memory mechanism (DDM-based crystallisation + enforcement) is ~23× faster than the pure experience-memory baseline and achieves 100% convergence. The dynamic window (lock-in) mechanism provides an additional ~30% speedup on top. Without any active mechanism, convergence fails at large populations (N≥200).

**2. Tipping point structure**

All conditions show a clear 3-phase pattern: noise → takeoff → avalanche. The normative mechanism converts random early fluctuations into crystallised norms within ~13 ticks, creating a self-reinforcing cascade.

**3. Crystallisation threshold (θ) — most sensitive parameter**

There is a sharp phase transition at θ≈7. Below this, behavioral convergence is the bottleneck; above, crystallisation becomes the bottleneck due to a **confidence-DDM paradox**: once agents can predict well (high confidence), the DDM drift channel closes (drift = (1-C)×f_diff → 0), preventing late crystallisation. The sweet spot is θ≈2-3.

**4. Visibility × Enforcement (V×Φ) — connects to your suggestion about communication**

This is the most interesting new finding. I swept visibility (V = additional observations per tick, 0-10) crossed with enforcement strength (Φ = 0-2):
- **Enforcement without visibility is catastrophic**: at V=0, enabling enforcement (Φ>0) drops convergence from 100% to only 10%. Enforcement signals push agents toward conflicting norms.
- **Even minimal visibility (V=1) fully rescues the system**: 100% convergence restored at all Φ.
- **Interpretation**: Information must precede enforcement for norms to emerge. "Blind enforcement" fragments rather than coordinates.

> **Note**: The V×Φ results are preliminary — I haven't fully theorised the mechanism yet. The pattern is robust across 30 trials per cell, but the exact causal pathway from V to DDM evidence to norm alignment needs further analysis.

### What this means for the paper

The model demonstrates that:
1. Individual-level memory mechanisms (experience + normative) are sufficient for norm emergence without centralized coordination
2. The interplay between confidence and norm crystallisation creates non-trivial dynamics (paradox, churn, phase transitions)
3. Social observation (V) and normative enforcement (Φ) have a strong interaction — this connects directly to your point about the role of communication in norm emergence

### Next steps I'm considering

1. Deep-dive into the V×Φ mechanism (per-agent trace analysis of the "blind enforcement" failure)
2. Possibly explore how V maps to network structure (currently V draws from random pairs; could use social network topology)
3. Begin drafting for EUMAS/AAMAS — the 3-layer convergence framework + θ phase transition + V×Φ interaction seems like enough for a contribution

## Attachments / things to review

I've prepared the following documents:

1. **`results/sweep_analysis_report.md`** — Full analysis report with all sweep results, tables, and interpretations (recommended first read, ~10 pages)
2. **`docs/implementation_spec_v5.md`** — Complete technical specification of the model (data structures, algorithms, edge cases). This is the authoritative reference for the implementation.
3. **`scripts/`** — All sweep scripts are reproducible (Julia, single command each)

I would be happy to walk through any of these in our next meeting. In particular, I'd like your thoughts on:
- Whether the V×Φ finding is worth pursuing as a central contribution
- How to frame the confidence-DDM paradox (design flaw vs. interesting emergent constraint?)
- Target venue and scope

Best regards,
[Your name]

---

# 中文备注：给你的建议

## 邮件策略

这封邮件的重点是：
1. **回应他上次的建议**（norm emergence timing → 3-layer convergence; communication → V×Φ sweep）
2. **展示有实质性进展**（5维扫描，定量结果）
3. **提出一个有趣的新发现**（blind enforcement trap），让他有东西可以讨论
4. **标注V×Φ结果是初步的**，避免被追问机制细节

## 应该给他看什么

### 必看
- **`results/sweep_analysis_report.md`**（分析报告）：这是最重要的，包含所有结论和数据表。建议他先看 §1-2（实验设计+基础结果），然后跳到 §8（V×Φ）和 §9（总结）。

### 选择性地看
- **`docs/implementation_spec_v5.md`**（技术规范）：如果他想了解模型细节。但这个文件很长且技术性强，建议只在他有具体问题时再给。
- **Factorial CSV / trajectories**：如果他想看原始数据。

### 不建议直接给
- 代码文件（除非他主动要求）
- 个别sweep的raw CSV（太碎片化）

## 会议准备

建议你准备以下几个"故事"来讲述：

1. **开场**："上次你建议我关注norm什么时候产生，以及沟通的角色。我做了两件事——重新定义了convergence为3层结构，然后做了visibility × enforcement的扫描。"

2. **亮点**："最有趣的发现是blind enforcement trap——如果agents能enforce但看不到其他人在做什么，norm emergence反而会崩溃。这和现实世界的直觉是一致的：没有信息基础的执法只会制造混乱。"

3. **讨论点**："θ的phase transition是个有趣的理论问题——confidence-DDM paradox说明高自信反而阻碍了个人层面的norm crystallisation。这可以解释为什么在高度可预测的环境中，formal norm仍然可能不emerge。"

4. **下一步的提问**："您觉得V×Φ这个finding够不够作为一个contribution？或者我需要再加网络结构的维度？"
