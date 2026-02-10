# Research Roadmap: Cognitive Lock-in and Norm Emergence

## 目标会议
- **EUMAS** (European Conference on Multi-Agent Systems): 欧洲多智能体系统顶会
- **AAMAS** (International Conference on Autonomous Agents and Multi-Agent Systems): 多智能体领域顶级会议

---

## Part 1: 现有实现总结

### 1.1 核心机制：认知锁定 (Cognitive Lock-in)

**理论基础**:
- Slovic (1993): 信任不对称性 - 建立慢，破坏快
- Miller (1956): 认知容量限制 (7±2)
- Bicchieri (2006): 规范需要经验预期和规范预期

**核心公式**:
```
信任更新 (成功): T_new = T + alpha * (1 - T)
信任更新 (失败): T_new = T * (1 - beta)
稳态信任: T* = (p * alpha) / (p * alpha + (1-p) * beta)
动态记忆窗口: window = base + round(trust * (max - base))
```

**反馈回路**:
```
协调成功 → 预测正确 → 信任↑ → 记忆窗口↑ → 信念稳定 → 规范固化
     ↑                                                    ↓
     ←←←←←←←←←←←← 更容易协调成功 ←←←←←←←←←←←←←←←←←←←←←←←←
```

### 1.2 已实现模块

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|------|
| 记忆系统 | `src/memory/` | 固定/衰减/动态记忆 | 完成 |
| 决策机制 | `src/decision/` | 认知锁定/双反馈/epsilon贪婪 | 完成 |
| 通信机制 | `src/communication/` | 规范信号/预游戏信号/阈值传染 | 完成 |
| 基础环境 | `src/environment.py` | 随机匹配+观察 | 完成 |
| 扩展环境 | `src/environment_extended.py` | 整合通信机制 | 完成 |
| 智能体 | `src/agent.py` | 整合记忆+决策+通信 | 完成 |
| 实验运行 | `experiments/runner.py` | 批量实验 | 完成 |
| 可视化 | `visualization/` | 实时+静态图表 | 基本完成 |

### 1.3 已验证发现

1. **动态记忆优于固定记忆**: 收敛时间快17%，共识度高4%
2. **信任不对称性关键**: beta/alpha ≈ 3:1 最优
3. **规模效应亚线性**: 收敛时间 ≈ O(N^0.6)
4. **路径依赖**: 两种策略胜出概率相近 (修复偏差后)
5. **理论-实证一致**: 稳态信任误差 < 10%

---

## Part 2: 短期扩展方向 (EUMAS 2025)

### 2.1 通信机制深入研究

**目标**: 系统研究三种通信机制对规范形成的影响

**实验设计**:
```python
conditions = [
    {"name": "baseline", "normative": False, "preplay": False, "threshold": False},
    {"name": "normative_only", "normative": True, "preplay": False, "threshold": False},
    {"name": "preplay_only", "normative": False, "preplay": True, "threshold": False},
    {"name": "threshold_only", "normative": False, "preplay": False, "threshold": True},
    {"name": "normative_preplay", "normative": True, "preplay": True, "threshold": False},
    {"name": "all_mechanisms", "normative": True, "preplay": True, "threshold": True},
]
```

**预期发现**:
- 预游戏信号: 加速收敛但可能锁定次优
- 规范信号: 增强共识强度
- 阈值传染: 减慢收敛但增加稳定性

**所需工作**:
- [ ] 设计对照实验矩阵
- [ ] 运行批量实验 (每条件100次)
- [ ] 分析交互效应
- [ ] 可视化比较图表

### 2.2 异质性智能体

**目标**: 研究智能体异质性对规范形成的影响

**异质性维度**:
1. **信任参数异质性**: alpha, beta 服从分布
2. **记忆类型异质性**: 混合记忆类型智能体
3. **初始信任异质性**: 高信任者 vs 低信任者

**实验设计**:
```python
# 信任参数服从Beta分布
alpha_distribution = Beta(2, 8)  # mean=0.2, concentrated
beta_distribution = Beta(4, 4)   # mean=0.5, dispersed

# 混合记忆类型
memory_mix = {"fixed": 0.3, "decay": 0.3, "dynamic": 0.4}
```

**预期发现**:
- 高信任者成为"规范锚点"
- 动态记忆智能体主导规范形成
- 参数异质性可能导致多规范共存

### 2.3 外部冲击与规范韧性

**目标**: 测试已建立规范对外部冲击的韧性

**冲击类型**:
1. **随机策略重置**: 随机选择k%智能体重置策略
2. **信任冲击**: 全体信任下降delta
3. **记忆冲击**: 清除所有记忆

**实验设计**:
```python
# 规范稳定后施加冲击
shock_timing = convergence_time + 50

shock_scenarios = [
    {"type": "strategy_reset", "fraction": 0.1},
    {"type": "strategy_reset", "fraction": 0.3},
    {"type": "trust_shock", "delta": -0.3},
    {"type": "memory_wipe", "fraction": 0.5},
]
```

**预期发现**:
- 认知锁定使规范具有韧性
- 小冲击: 快速恢复原规范
- 大冲击: 可能触发规范转换
- 信任冲击效果最显著

---

## Part 3: 中期扩展方向 (AAMAS 2026)

### 3.1 网络结构

**目标**: 从随机匹配扩展到结构化网络

**网络类型**:
1. **小世界网络** (Watts-Strogatz): 高聚类 + 短路径
2. **无标度网络** (Barabasi-Albert): 幂律度分布
3. **社区结构**: 多个密连子群

**关键问题**:
- 网络hub是否成为规范锚点?
- 社区结构是否导致局部规范?
- 网络重连如何影响规范传播?

**实现计划**:
```python
class NetworkedEnvironment(SimulationEnvironment):
    def __init__(self, network_type, **kwargs):
        self._network = create_network(network_type, num_agents)

    def get_interaction_pairs(self):
        # 基于网络边选择交互对
        edges = list(self._network.edges())
        return random.sample(edges, k=num_interactions)

    def get_observation_sources(self, agent_id):
        # 只能观察邻居
        return list(self._network.neighbors(agent_id))
```

### 3.2 多策略扩展

**目标**: 从2策略扩展到多策略协调

**挑战**:
- 信念从2维扩展到n维
- 收敛变得更困难
- 可能出现策略周期

**实现思路**:
```python
class MultiStrategyGame:
    def __init__(self, num_strategies):
        # 纯协调: 对角线为1，其他为0
        self._payoff = np.eye(num_strategies)

class MultiStrategyMemory:
    def get_strategy_distribution(self):
        # 返回n维概率向量
        counts = np.zeros(self._num_strategies)
        for interaction in self._history:
            counts[interaction.partner_strategy] += weight
        return counts / counts.sum()
```

### 3.3 学习与适应

**目标**: 智能体学习最优alpha-beta参数

**方法**:
1. **进化算法**: 参数作为基因，适应度=协调成功率
2. **强化学习**: 智能体学习调整参数
3. **贝叶斯更新**: 在线学习最优参数

**预期发现**:
- 进化稳定策略可能不唯一
- 环境变化频率影响最优参数
- 可能出现参数专业化

### 3.4 规范层级

**基于Bicchieri (2006)的规范层级**:
1. **行为规范**: 大多数人这样做
2. **认知规范**: 大多数人知道应该这样做
3. **共享规范**: 大多数人知道大多数人知道
4. **制度规范**: 有明确的惩罚/奖励

**实现思路**:
```python
class NormLevel(Enum):
    BEHAVIORAL = 1    # 行为收敛
    COGNITIVE = 2     # 信念收敛
    SHARED = 3        # 高阶信念收敛
    INSTITUTIONAL = 4 # 外部强制

def measure_norm_level(environment):
    behavioral = measure_strategy_convergence()
    cognitive = measure_belief_convergence()
    shared = measure_higher_order_belief()
    return classify_norm_level(behavioral, cognitive, shared)
```

---

## Part 4: 长期研究议程

### 4.1 理论贡献

1. **收敛时间界**: 推导解析表达式
2. **相变分析**: 识别临界参数值
3. **稳定性定理**: 证明哪些规范稳定

### 4.2 实证验证

1. **实验室实验**: 人类被试验证认知锁定
2. **田野研究**: 观察现实规范变化中的记忆效应
3. **历史分析**: 社会规范与外部冲击

### 4.3 应用领域

1. **在线社区规范**: 如何设计平台促进健康规范
2. **组织变革**: 如何打破不良规范锁定
3. **政策设计**: 信任冲击的规范影响

---

## Part 5: 投稿策略

### 5.1 EUMAS 2025 (短期目标)

**截止日期**: 通常5-6月

**投稿重点**:
- 基础模型 + 通信机制研究
- 强调理论基础 (Bicchieri, Skyrms, Centola)
- 8-12页短论文

**论文结构**:
1. Introduction: 规范形成的认知视角
2. Model: 认知锁定机制
3. Communication: 三种机制
4. Experiments: 比较分析
5. Discussion: 理论意义

### 5.2 AAMAS 2026 (中期目标)

**截止日期**: 通常10月

**投稿重点**:
- 完整模型 + 网络扩展 + 异质性
- 强调创新性和系统性
- 8页短论文 或 15页长论文

**论文结构**:
1. Introduction: 认知锁定的理论基础
2. Model: 完整数学描述
3. Analysis: 理论结果
4. Experiments: 大规模实验
5. Extensions: 网络、异质性
6. Conclusion: 广泛影响

### 5.3 备选会议

- **AAAI**: 人工智能顶会，强调创新
- **IJCAI**: 人工智能顶会，理论性强
- **ECAI**: 欧洲AI会议
- **COIN Workshop @ AAMAS**: 专注规范和制度

---

## Part 6: 时间线

### 2025 Q1 (1-3月)
- [ ] 完成通信机制对比实验
- [ ] 撰写EUMAS论文初稿
- [ ] 内部评审和修改

### 2025 Q2 (4-6月)
- [ ] 提交EUMAS (如果截止日期在此)
- [ ] 开始网络结构扩展
- [ ] 实现异质性智能体

### 2025 Q3 (7-9月)
- [ ] EUMAS结果/修改
- [ ] 完成网络实验
- [ ] 撰写AAMAS论文大纲

### 2025 Q4 (10-12月)
- [ ] 提交AAMAS
- [ ] 开始多策略扩展
- [ ] 规划实验室实验

---

## Part 7: 技术债务与改进

### 7.1 代码质量
- [ ] 添加完整单元测试
- [ ] 类型注解检查
- [ ] 文档字符串完善
- [ ] 性能优化 (大规模实验)

### 7.2 可复现性
- [ ] Docker容器
- [ ] 随机种子管理
- [ ] 配置版本控制
- [ ] 结果存档系统

### 7.3 可视化增强
- [ ] 交互式仪表板
- [ ] 论文级图表模板
- [ ] 动画导出 (GIF/MP4)

---

## 附录: 关键参考文献

### 规范理论
- Bicchieri, C. (2006). The Grammar of Society
- Bicchieri, C. (2017). Norms in the Wild
- Brennan et al. (2013). Explaining Norms

### 信号与协调
- Skyrms, B. (2010). Signals
- Lewis, D. (1969). Convention
- Crawford & Sobel (1982). Strategic Information Transmission

### 社会传染
- Centola, D. (2018). How Behavior Spreads
- Centola & Macy (2007). Complex Contagion

### 信任动态
- Slovic, P. (1993). Perceived Risk, Trust, and Democracy
- Kramer, R. (1999). Trust and Distrust in Organizations

### 记忆与决策
- Miller, G.A. (1956). The Magical Number Seven
- Hertwig et al. (2004). Decisions from Experience
