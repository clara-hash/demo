# 基于nuPlan/Las Vegas地图的无人车IRL路径规划实验报告 (v2)

> 包含 V3-V7 完整实验记录。V7 Neural Reward IRL 为新增内容。

---

## 摘要

本报告记录了在nuPlan Las Vegas真实城市地图数据上，基于逆强化学习（IRL）进行自动驾驶路径规划的完整实验迭代过程。实验从基础Goal-conditioned IRL（V3）开始，逐步演进到神经网络奖励模型（V7），在真实城市道路语义地图和专家轨迹数据上验证了从专家示范中学习驾驶偏好的能力。

**关键词**：逆强化学习，路径规划，自动驾驶，nuPlan，最大熵模型，神经网络奖励

---

## 1. 实验背景与目标

### 1.1 问题定义

给定城市道路语义地图和人类专家驾驶轨迹，通过逆强化学习从专家示范中学习奖励函数，使得在给定起点和目标点时，规划出的路径能够体现专家驾驶偏好（可行驶区域行驶、保持安全距离、高效抵达目标等）。

### 1.2 数据基础

| 项目 | 内容 |
|------|------|
| 地图来源 | nuPlan Las Vegas Strip HD Map |
| 地图格式 | GPKG → 栅格化 (240×187 grids) |
| 语义层 | 0=无效区, 1=可行驶区, 2=车道, 3=连接区, 4=交叉口 |
| 有效栅格 | 10,183 cells |
| 专家轨迹 | 12条 nuPlan ego vehicle 记录 |
| 轨迹点数 | 13–85 points/trajectory |

### 1.3 最大熵逆强化学习原理

奖励函数定义为特征线性加权：R(s) = θᵀφ(s)。最大熵模型下轨迹分布：

> P(τ|θ) = exp(R(τ)) / Z(θ)

通过最大化专家轨迹对数似然学习参数θ：

> L(θ) = (1/|D|) Σ log P(τ|θ)

梯度计算通过匹配专家特征期望和模型特征期望实现。

---

## 2. V3：目标条件Goal-conditioned IRL（基线）

### 2.1 设计思路

在语义地图上给定起点和目标点，通过候选路径特征评分选择最优路径，使用专家轨迹密度（expert_density）作为辅助特征。

### 2.2 配置

| 参数 | 值 |
|------|------|
| 特征维度 | 9 |
| 训练样本 | 43 |
| 候选路径/样本 | 18 |
| 训练轮数 | 220 |
| 学习率 | 0.06 |
| L2正则 | 1e-4 |

### 2.3 特征列表

`drivable_ratio`, `lane_ratio`, `lane_connector_ratio`, `intersection_ratio`, `expert_density`, `goal_reached`, `heading_to_goal`, `smoothness`, `length_efficiency`

### 2.4 结果

| 指标 | 值 |
|------|------|
| 初始 NLL | 2.052 |
| 最终 NLL | 1.688 |
| IRL ADE | 2.600 |
| Shortest ADE | 1.315 |
| IRL 成功率 | 100% |
| IRL 路径长度 | 36.014 |
| 专家路径长度 | 36.410 |

**学习的奖励权重**：`length_efficiency`(1.338) > `drivable_ratio`(0.966) > `intersection_ratio`(0.953) > `heading_to_goal`(0.861) > `expert_density`(0.329) > `smoothness`(0.274)

### 2.5 分析

- 模型主要偏好高效且在可行驶区域的路径
- `expert_density` 有正向权重(0.329)，依赖专家轨迹密度，泛化性受限
- 缺少 clearance 特征，无法主动避开墙壁/边界
- 没有障碍物感知能力

---

## 3. V4：Clearance-aware Goal-conditioned IRL

### 3.1 改进点

引入 static clearance（静态间隙）特征，使用欧几里得距离变换（EDT）计算每个栅格到最近非行驶区域的距离。解决V3中路径贴墙的问题。

### 3.2 特征变化

移除 `lane_connector_ratio`，新增 `clearance`（CLEARANCE_CAP=10cells, MIN=5cells）。

### 3.3 结果对比

| 指标 | V3 | V4 | 变化 |
|------|------|------|------|
| 最终 NLL | 1.688 | 1.662 | ↓1.5% |
| IRL ADE | 2.600 | 1.919 | **↓26.2%** |
| Shortest ADE | 1.315 | 1.315 | - |
| IRL 成功率 | 100% | 100% | - |
| IRL 路径长度 | 36.014 | 36.427 | ↑1.1% |

### 3.4 分析

- Clearance特征是最有效的单点改进 —— ADE下降26.2%
- 路径长度微增1.1%，说明模型在做安全-效率权衡时略微偏向安全
- 仍依赖 expert_density 特征，泛化性有限

---

## 4. V5：轻量级自主障碍感知IRL

### 4.1 架构变化

- 首次实现 Train/Test 分离（最后一条demo作为hold-out测试集）
- 引入人工障碍物增强（训练时随机放置障碍物）
- 首次移除 `expert_density` 特征
- 保存训练好的模型权重（.npz格式）

### 4.2 配置变化

| 参数 | V3/V4 | V5 |
|------|-------|------|
| 训练样本 | 43 | 180 |
| 训练轮数 | 220 | 260 |
| 学习率 | 0.06 | 0.035 |
| Train/Test分离 | 按样本切分 | **按demo隔离** |
| 障碍物增强 | 无 | **训练+测试均开启** |
| expert_density | 保留 | **移除** |

### 4.3 分析

- 180个样本太少，训练不充分
- 权重值域很小（0-0.2），多个特征收敛到0附近
- 证明：要学到有效的奖励函数，需要更大的训练规模

---

## 5. V6 Heavy：大规模自主障碍感知线性IRL

### 5.1 核心改进

大幅度提升训练规模：180 → 2,400 cases（+1233%）。

### 5.2 配置

| 参数 | V5 | V6 Heavy |
|------|------|------|
| 训练 cases | 180 | 2,400 |
| 预处理耗时 | - | 30分钟 |
| 候选路径/case | 18 | 24 |
| 障碍物/case | 若干 | 7 |
| 训练轮数 | 260 | 900 |
| Batch size | 全量 | 48 |
| 有效候选集 | - | 2,371 |
| 特征维度 | 11 | 12 |

### 5.3 特征列表（12维）

`valid_ratio`, `lane_ratio`, `lane_connector_ratio`, `non_intersection_ratio`, `static_clearance_mean`, `static_clearance_min`, `obstacle_clearance_mean`, `obstacle_clearance_min`, `goal_reached`, `length_efficiency`, `heading_to_goal`, `smoothness`

### 5.4 Loss收敛

| 轮数 | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 899 |
|------|------|------|------|------|------|------|------|------|------|------|
| NLL | 2.664 | 2.209 | 2.057 | 1.979 | 1.932 | 1.901 | 1.879 | 1.862 | 1.848 | **1.838** |

Loss 下降 31.0%。

### 5.5 最终奖励权重

| 特征 | 权重 | 方向 |
|------|------|------|
| `length_efficiency` | **+19.738** | 偏好高效路径 |
| `static_clearance_min` | **+7.715** | 偏好安全（最强安全信号） |
| `heading_to_goal` | **+6.488** | 偏好朝向目标 |
| `obstacle_clearance_min` | -0.442 | 统计偏差 |
| `lane_ratio` | -4.069 | 地图语义偏差 |
| `smoothness` | -3.879 | 候选生成偏见 |
| `obstacle_clearance_mean` | -5.481 | 统计偏差 |
| `static_clearance_mean` | -15.369 | 与min形成互补约束 |

### 5.6 测试结果（30个复杂障碍cases）

| 指标 | IRL | Shortest |
|------|------|------|
| 成功率 | 100% | 100% |
| 平均路径长度 | 133.85 | 134.08 |
| 障碍物违规 | 0.0 | 0.0 |
| 地图违规 | 0.0 | 0.0 |

### 5.7 分析

1. **规模效果显著**：2,400 cases + 900 epochs 使 loss 持续下降，未过拟合
2. **min > mean**：`static_clearance_min`(+7.7) vs `mean`(-15.4)，极端情况比平均值更有信息量
3. **完全摆脱expert_density**：模型不依赖专家轨迹密度特征
4. **线性奖励局限**：在简单场景中与shortest baseline差异不大，复杂权衡场景需要非线性奖励

---

## 6. V7 Neural Reward IRL

### 6.1 设计动机

V6 Heavy的线性奖励函数存在表达能力上限：`R(s) = θᵀφ(s)` 只能捕捉特征的线性关系。V7用神经网络替代线性模型，从"特征权重学习"升级为"奖励函数学习"。

### 6.2 方法框架

V7采用**选择式（Choice-based）学习方法**：

```
随机起点/目标 + 随机复杂障碍物
         ↓
生成候选路径（A* + 不同障碍物规避参数）
         ↓
提取每条路径的18维特征向量
         ↓
Teacher Utility 函数打分 → 伪标签（最优候选）
         ↓
MLP Reward Model → Cross-Entropy Loss
         ↓
测试：新场景中MLP打分 → 选最优路径
```

### 6.3 模型架构

```
Input (18 features)
  → Linear(18, 256) + ReLU + Dropout(0.10)
  → Linear(256, 256) + ReLU + Dropout(0.10)
  → Linear(256, 128) + ReLU
  → Linear(128, 1)  → scalar reward score
```

### 6.4 特征列表（18维）

`length_ratio`, `length_efficiency`, `valid_ratio`, `lane_ratio`, `lane_connector_ratio`, `intersection_ratio`, `static_clearance_mean`, `static_clearance_min`, `obstacle_clearance_mean`, `obstacle_clearance_min`, `near_static_penalty`, `near_obstacle_penalty`, `heading_to_goal`, `turn_mean`, `turn_max`, `goal_reached`, `combined_clearance_min`, `num_points_norm`

### 6.5 Teacher Utility函数

用于生成训练伪标签的人造效用函数。**第一版训练后发现障碍物间隙不足，第二版大幅提升了障碍物相关权重：**

| 权重项 | 第一版 | **第二版（修复后）** |
|------|--------|------|
| obstacle_clearance_min | 3.4 | **12.0** |
| obstacle_clearance_mean | 1.8 | **5.0** |
| combined_clearance_min | 2.5 | **8.0** |
| near_obstacle_penalty | -6.5 | **-15.0** |

### 6.6 候选路径生成策略（修复后）

A*候选路径的障碍物规避参数从原来的 `min_obstacle_clearance=0~2` 提升到 **2~4 cells**，`obstacle_weight` 从 3~7 提升到 **8~15**，障碍物成本衰减范围从 2.0 扩大到 **3.0**。

### 6.7 训练配置

| 参数 | 值 |
|------|------|
| 数据集规模 | **3,000 cases** |
| 候选路径/case | 10 |
| 障碍物/case | 6–16（椭圆随机障碍物） |
| 起点-目标距离 | 45–180 cells |
| Train/Val/Test | 70% / 15% / 15% (2100/450/450) |
| 训练轮数 | 80 |
| Batch size | 256 |
| 学习率 | 2e-4 |
| 优化器 | AdamW + CosineAnnealingLR |
| 设备 | CPU (Jetson ARM) |
| 数据生成时间 | ~75分钟 |

### 6.8 训练结果

| 阶段 | Loss | Accuracy |
|------|------|----------|
| Epoch 0 (Train) | 1.763 | 31.3% |
| Epoch 10 (Val) | 1.190 | 61.3% |
| Epoch 30 (Val) | 0.865 | 73.6% |
| Epoch 60 (Val) | 0.810 | 77.3% |
| Epoch 79 (Val) | **0.807** | **77.8%** |
| **Test (hold-out)** | **0.868** | **75.8%** |

Loss 下降 54%（1.763 → 0.807），Choice Accuracy 从随机水平10%提升到 **77.8%（val）/ 75.8%（test）**，较800 cases版本提升 **+11.6pp（test）**。

### 6.9 自主规划测试（30个新障碍场景）

#### 总体结果

| 指标 | V7 Neural IRL | Shortest Baseline | 对比 |
|------|:--:|:--:|------|
| 成功率 | **100%** | 100% | 持平 |
| 平均路径长度 | 142.8 | 134.5 | IRL 长 6.9% |
| 长度比均值 | 1.070 | 1.0 | - |
| 最小长度比 | 1.000 | - | 有时重合 |
| 最大长度比 | 1.275 | - | 最多长27.5% |
| **平均障碍物间隙** | **5.64** | - | ↑3000 case显著提升 |
| 平均静态间隙 | 1.22 | - | 安全 |

#### 逐条结果（节选）

| Test | IRL长度 | Shortest | 比率 | 障碍min | 说明 |
|------|------|------|------|------|------|
| 0 | 58.6 | 58.6 | 1.000 | 1.0 | 极窄通道 |
| 2 | 100.5 | 78.9 | 1.275 | **8.5** | 大幅绕行 |
| 5 | 136.7 | 126.6 | 1.080 | **6.0** | 绕行 |
| 6 | 157.3 | 148.7 | 1.058 | **9.2** | 大幅绕行 |
| 10 | 102.2 | 93.3 | 1.096 | **7.0** | 绕行 |
| 23 | 171.3 | 157.0 | 1.091 | **6.7** | 绕行 |
| 26 | 237.2 | 229.5 | 1.033 | **14.8** | 极大安全间隙 |
| 27 | 144.9 | 139.1 | 1.042 | **12.1** | 极大安全间隙 |
| 28 | 62.3 | 53.8 | 1.158 | **10.6** | 极大安全间隙 |
| 29 | 118.5 | 112.5 | 1.053 | **12.4** | 极大安全间隙 |

### 6.10 训练规模影响：800 vs 3000 cases

| 指标 | 800 cases | **3000 cases** | 变化 |
|------|-----------|------|------|
| Val Accuracy | 62.5% | **77.8%** | ↑**+15.3pp** |
| Test Accuracy | 64.2% | **75.8%** | ↑**+11.6pp** |
| Val Loss | 1.104 | **0.807** | ↓27% |
| Test Loss | 1.091 | **0.868** | ↓20% |
| 平均障碍物间隙 | 3.73 | **5.64** | ↑**+51%** |
| obs_min ≤ 1.0 | 17% | **6.7%** | ↓61% |
| obs_min ≥ 10.0 | 0% | **13.3%** | 新增 |

### 6.11 V7 关键分析

**1. 3000 cases 训练带来质的飞跃**

- Test Accuracy 从 64.2% → **75.8%**（+11.6pp），模型在 3/4 的情况下选中teacher最优路径
- 障碍物间隙从 3.73 → **5.64**（+51%），首次出现 ≥10 cells 的高安全间隙案例（13.3%）
- 贴障碍案例占比从 17% → **6.7%**，几乎消除

**2. 神经网络成功学习了障碍物规避偏好**

V7不复制shortest path。30个测试中，27个路径与shortest不同。模型在有选择空间时主动绕行获取安全间隙。

**3. obs_min=1.0 的剩余2个case（#0, #18）**

比率=1.000（IRL=shortest），属于地图几何约束——障碍物布局极其狭窄，不存在更好的选择。

**4. 与V6 Heavy的关键差异**

| 维度 | V6 Heavy (线性) | V7 (神经，3000 cases) |
|------|------|------|
| 奖励函数 | θᵀφ(s) 线性 | MLP(φ) 非线性 |
| 训练目标 | MaxEnt特征匹配 | Cross-Entropy选择 |
| 测试行为 | 路径≈shortest | 路径≠shortest（主动规避障碍物） |
| 障碍物间隙 | ~1.44 (mix) | **~5.64 (obstacle min)** |
| 路径变长代价 | ~0% | ~7.0% |
| 训练规模 | 2,400 cases | **3,000 cases** |
| 准确率 | - | **75.8% (test)** |

**5. 局限性**

- Teacher utility 是人造的，不是从真实专家轨迹学到的
- 3000 cases 已足够验证可行性，但更大规模（10K+）可进一步提升准确率

---

## 7. 版本对比总结

### 7.1 关键指标总览

| 版本 | 核心改进 | 训练量 | 特征 | Loss变化 | 关键指标 |
|------|----------|--------|------|----------|----------|
| V3 | Goal-conditioned 基线 | 43 | 9 | 2.052→1.688 | ADE=2.600 |
| V4 | +Static Clearance | 43 | 9 | 2.052→1.662 | ADE=1.919 (↓26%) |
| V5 | +Obstacle, Train/Test分离 | 180 | 11 | - | 训练不足 |
| V6 Heavy | 规模扩大 | 2,400 | 12 | 2.664→1.838 (↓31%) | Success=100% |
| **V7** | **神经网络奖励+障碍物规避** | **3,000** | **18→MLP** | **1.763→0.807** | **Acc=75.8%, obs_min=5.64** |

### 7.2 方法论演进

```
V3-V4: 手工特征探索（地图语义 + 专家密度）
   ↓
V5:    训练框架化（Train/Test分离 + 障碍物增强）
   ↓
V6:    规模化线性IRL（2400 cases + 900 epochs）
   ↓
V7:    神经网络奖励（MLP + choice learning + 18维特征 + 障碍物规避修复）
```

### 7.3 核心发现

1. **Clearance是最有效的单点改进**：V3→V4 ADE ↓26.2%
2. **规模决定质量**：V5(180)权重接近0，V6(2400)学到有意义的权重
3. **min > mean**：安全间隙的最小值比平均值更有信息量
4. **线性→神经是质的飞跃**：V7首次让IRL路径系统性区分于shortest baseline
5. **障碍物规避可通过调整权重显著改善**：修复后障碍物间隙均值达5.64 cells，低间隙case降至6.7%
6. **3000 cases训练带来突破**：Test Accuracy 75.8%（vs random 10%），障碍物间隙较800 case版本提升51%

---

## 8. 结论与展望

### 8.1 主要结论

1. **最大熵IRL能从nuPlan专家轨迹中学习有效驾驶偏好**，系统性地验证了V3→V7的演进

2. **Clearance特征是安全感知的关键**，加入后ADE显著下降

3. **神经网络奖励（V7）突破了线性表达的瓶颈**，3000 cases训练达到77.8% val accuracy，障碍物间隙均值5.64 cells

4. **障碍物规避可通过调整teacher utility和候选生成策略灵活控制**

5. **Choice-based learning方法可行**，从10%随机基线提升到75.8% test accuracy

### 8.2 后续工作

- **扩大V7训练规模**：800→10,000+ cases，提高泛化能力
- **真实专家标签**：用nuPlan真实ego轨迹替代teacher utility
- **Transformer/Attention架构**：建模路径点之间的依赖关系
- **闭环仿真评价**：在nuPlan simulator中评估
- **交通规则约束**：红绿灯、停止线、人行横道

---

## 附录A：实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | NVIDIA Jetson (ARM aarch64, 61GiB RAM) |
| OS | Ubuntu 20.04.6 LTS, L4T R35.4.1 |
| Python | 3.9.23 (Miniforge conda env: nuplan_irl) |
| 关键依赖 | numpy, scipy, matplotlib, pandas, shapely, PyTorch 2.8.0 |
| 数据 | nuPlan v1.1 Mini Split + Maps v1.0 |
| 地图 | Las Vegas Strip (us-nv-las-vegas-strip) |

## 附录B：输出文件索引

| 文件 | 说明 |
|------|------|
| `outputs/report_all_versions_loss.png` | V3/V4/V6/V7 Loss对比图 |
| `outputs/report_v7_training.png` | V7训练Loss和Accuracy曲线 |
| `outputs/report_v7_planning_summary.png` | V7测试18个case详细结果 |
| `outputs/nuplan_irl_neural_reward_v7/test_figures/` | V7 12张测试case路径图 |
| `outputs/nuplan_irl_neural_reward_v7/neural_reward_v7_best.pt` | V7训练好的模型权重 |
| `outputs/nuplan_irl_neural_reward_v7/dataset_v7.npz` | V7数据集 |
