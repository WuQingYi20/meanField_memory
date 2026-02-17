# 复杂模型的机制梳理方法

> 一个把"N个互相纠缠的机制"变成"可读、可实现、可debug的结构化文档"的pipeline。
> 从V5.1 dual-memory模型的实际经验中提炼。

---

## Pipeline总览

```
Step 1: 找骨架        → 层、桥接变量、信息流方向
Step 2: 层内排序      → 每层的因果链 / 生命周期
Step 3: 逐机制填卡片  → 固定模板，不遗漏
Step 4: 标注失败模式  → 每个机制错了会怎样
Step 5: 追踪涌现性质  → 哪些行为不属于任何单个机制
Step 6: 反向审计      → 从预期结果倒推，检查是否有断链
```

---

## Step 1：找骨架

**目标：** 把N个机制压缩成"几个子系统 + 几根管道"。

**方法：** 列出所有机制之间的依赖关系，然后问一个问题——

> 如果我要把这个模型拆给三个人分别实现，我在接口处需要传递哪些变量？

这些变量就是**桥接变量**。子系统之间只通过桥接变量通信，子系统内部的复杂性被封装。

**操作：**
1. 列出所有状态变量
2. 对每个变量，标注"谁写、谁读"
3. 跨子系统的读写 → 桥接变量
4. 子系统内部的读写 → 内部细节

**检验标准：** 桥接变量应该很少（3-5个）。如果超过7个，说明你的分层方式不对，子系统之间耦合太紧。

**V5.1的结果：**
```
经验层 ──C_i──→ 规范层 ──σ_i──→ 决策层
                  ↑ p_i ←── 执行机制
```
三个桥接变量。任何读者看到这张图就知道整体结构。

**常见陷阱：**
- 把所有东西放在一层里（"这些机制都互相影响所以没法分层"）→ 几乎总是可以分的，关键是找到信息流的主方向
- 把utility function当桥接变量 → 太宽泛，没有压缩信息

---

## Step 2：层内排序

**目标：** 每层内部的机制按因果链排列，让读者线性阅读一遍就理解整个flow。

**方法：** 问两个问题——

> 这一层里，什么东西是被*先*计算出来、然后*被*其他东西使用的？

> 如果我要给一个完全不懂这个模型的人讲这一层，我会从哪里开始讲？

**三种常见排序原则（选一个，保持一致）：**

| 原则 | 适用场景 | 例子 |
|---|---|---|
| **时间顺序** | 有明确的tick/step执行顺序 | 经验层：data → belief → action → feedback |
| **生命周期** | 对象有birth-life-death | 规范层：formation → maintenance → crisis → dissolution |
| **依赖拓扑** | 纯计算图，无时间概念 | 编译器pass：parse → type-check → optimize → codegen |

**V5.1的选择：**
- 经验层：时间顺序（memory → belief → action → confidence → window）
- 规范层：生命周期（DDM → crystallisation → strengthening → anomaly → crisis）
- 执行机制：因果链（trigger → collect → broadcast → receive）

**检验标准：** 每个机制最多依赖排在它前面的机制。如果出现"机制7依赖机制9"，说明排序不对或者有循环依赖需要标注。

**常见陷阱：**
- 按"重要性"排（最重要的放最前）→ 读者无法跟随因果链
- 按"复杂度"排（简单的先讲）→ 可能把因果链打断

---

## Step 3：逐机制填卡片

**目标：** 每个机制用固定模板描述，保证不遗漏关键信息。

**模板（6个字段）：**

```markdown
### 机制 X：[名称]

**是什么（What）**
一句话 + 核心方程。读者看完这段应该能写出伪代码。

**为什么这样设计（Why this way）**
和至少一个替代方案对比。不是"为什么需要这个机制"
（那是architecture层面的问题），而是"为什么用这个
functional form而不是另一个"。

**数值与依据（Parameters）**
表格：参数名 | 值 | 来源（文献/校准目标/直觉）
区分三个tier：
  (i)   有named study → 给citation
  (ii)  从behavioral target推导 → 给推导过程
  (iii) 探索性 → 标注需要sensitivity analysis

**失败模式（Failure Modes）**
如果这个机制实现错了或参数选错了，simulation会
表现出什么symptom？这是debug时的查找表。

**连接（Connections）**
- 输入 ← 哪些机制/变量
- 输出 → 哪些机制/变量
只列跨机制的连接，不列内部细节。

**可选性（Optionality）**
这个机制能否被关掉（设参数=0）？关掉后模型会
失去什么能力？这帮助判断机制的necessity。
```

**为什么每个字段都不能省：**

| 字段 | 省了会怎样 |
|---|---|
| What | 实现者猜你的意图，猜错了 |
| Why this way | 后续改进时不知道当初为什么选这个form，改了又改回来 |
| Parameters | 参数来源不透明，reviewer问"这个0.3哪来的"你答不上 |
| Failure modes | Debug时盲目搜索，浪费几天 |
| Connections | 改一个机制时不知道会影响什么，引入unintended side effects |
| Optionality | 不知道哪些是核心哪些是可选，complexity无法降低 |

**填卡片的顺序：** 先填所有机制的What（快速过一遍），再回来逐个填Why和Parameters。最后填Failure Modes和Connections。原因：填What的过程中你会发现有些机制的边界不对（太大或太小），此时调整成本最低。

---

## Step 4：标注失败模式

**目标：** 每个机制一行"如果X错了，你会看到Y"。

**方法：** 对每个机制，问三个问题——

> 1. 如果这个机制被完全关掉（参数=0或跳过），什么宏观行为会消失？
> 2. 如果这个机制的符号写反了（+变-，> 变 <），什么行为会反转？
> 3. 如果这个机制的参数偏大/偏小一个数量级，什么行为会过度/不足？

**V5.1的例子：**

| 机制 | 关掉 | 符号反 | 参数偏大 | 参数偏小 |
|---|---|---|---|---|
| 5. Dynamic window | Baseline和lock-in条件无差异 | C高→window小：高confidence agent反而volatile | W_max=60：belief过度稳定，无法适应变化 | W_max=3：和fixed几乎无区别 |
| 7. Confidence gate | 所有agent同速crystallise，不论confidence | 高C agent更容易crystallise（反"copy when uncertain"） | — | — |
| 9. Strengthening | Crisis后norm永久削弱，无recovery | — | α_σ=0.05：norm几乎不可动摇 | α_σ=0.0005：recovery要1000 ticks |
| 11. Crisis | Norm不可动摇，shock实验无效 | — | θ_crisis=3：norm极脆弱 | θ_crisis=100：norm dogmatic |
| 14. Signal push | 无enforcement cascade，H4失败 | Push反向：enforcement抑制crystallisation | γ=20：单signal→instant crystallisation | γ=0.2：signal忽略不计 |

**这张表的价值：** 当你跑出意外结果时，不用猜——直接查表。"Norm永远不dissolve" → 查机制10/11 → 是不是高σ agent没accumulate anomaly？

---

## Step 5：追踪涌现性质

**目标：** 标注哪些行为是由机制组合涌现的，不属于任何单个机制。

**方法：** 列出模型的所有预期宏观行为（hypotheses），对每个行为问——

> 这个行为能归因到单个机制吗？还是需要多个机制协作？

如果需要多个机制，画出最小的机制子集和它们的interaction path。

**V5.1的例子：**

| 涌现行为 | 最小机制子集 | Interaction path |
|---|---|---|
| Loop 1：认知锁定 | 4 + 5 + 1 + 3 | prediction → C → window → belief → action → prediction |
| Loop 2：社会放大 | 2 + 3 + observation | consistent action → observed → belief shift → coordination |
| Loop 3：规范cascade | 6 + 8 + 12 + 13 + 14 | consensus → DDM → crystallise → enforce → accelerate DDM |
| Crisis recovery | 9 + 11 | crisis降σ → violations停止 → strengthening恢复σ |
| Tipping point | 8 + 12 + 13 | 少数crystallised agents enforce → 加速剩余agents → cascade |

**为什么这步重要：**
- 如果一个涌现行为没出现，你知道该检查哪些机制的组合
- 如果一个涌现行为意外出现（论文没预测的），你可以trace它来自哪些机制的interaction
- 这也是论文discussion section的素材——"我们发现X行为涌现于机制A和B的交互，这是模型的novel prediction"

---

## Step 6：反向审计

**目标：** 从预期结果倒推，检查pipeline中是否有断链。

**方法：** 拿出你的hypothesis list，对每个hypothesis做因果链分析——

> H_n 预测 [结果Y]。Y需要[行为X]。X需要[机制A的输出]进入[机制B的输入]。
> 这个连接在spec里有明确定义吗？在代码里有实现吗？

**V5.1的例子：**

```
H2: 低confidence agents先crystallise
    需要: DDM drift ∝ (1-C)
    检查: 机制7的公式里是(1-C_i)还是C_i？ ✓
    检查: Phase 6a用的是Phase 5更新后的C_i？ ✓
    检查: Phase 5在Phase 6a之前执行？ ✓ (Phase 5 → 6a)

H4: Enforcement加速cascade
    需要: enforcement signal → p_i → DDM drift增加 → 更快crystallise
    检查: p_i在6b被设置？ ✓
    检查: p_i在下一tick的6a被加到drift里？ ✓
    检查: p_i加完后被清零（不重复计算）？ ✓
    检查: 多个signals累加（不是覆盖）？ ✓

H6: Sustained anomalies → norm collapse
    需要: violations → anomaly count → crisis → σ下降 → dissolution
    检查: 高σ agents也accumulate anomaly？ ✓（修复后）
    检查: crisis后a_i归零（不重复trigger）？ ✓
    检查: dissolution后e_i归零（可以re-crystallise）？ ✓
```

**这步的价值：** 在写代码之前发现断链。我们在review spec的过程中发现的几个bug（高σ agent不accumulate anomaly、observations不影响action selection、single enforcer wins）都可以通过这步提前发现。

**最后一个check：** 对每个桥接变量，确认它的writer和reader在执行顺序中的先后关系。如果reader在writer之前执行，你要么看到的是stale value（可能是intentional的，比如Phase 3读上一tick的状态），要么是bug。

---

## 方法本身的局限

这个pipeline优化的是**可读性和可debug性**，不是**简洁性**或**数学优雅**。

适用于：
- 多机制的simulation模型（ABM、multi-agent RL、复杂系统）
- 需要多人协作实现的模型
- 需要逐步验证的模型（不能一次性验证整体）

不适用于：
- 单方程模型（直接写方程就够了）
- 已有成熟数学框架的模型（比如mean field game——用标准数学notation比这个pipeline更有效）
- 纯探索性的prototype（还没定型的模型，用这个pipeline太重了）