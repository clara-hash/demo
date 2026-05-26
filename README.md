# nuPlan IRL 路径规划实验项目

基于 nuPlan Las Vegas 地图的逆强化学习（IRL）自动驾驶路径规划实验，从基础 Goal-conditioned IRL（V3）演进到神经网络奖励模型（V7）。

## 目录结构

```
nuplan_irl_project/
├── data/                                  # 原始数据
│   ├── maps/                              # nuPlan 地图 (GPKG格式)
│   │   ├── us-nv-las-vegas-strip/         #   主要实验区域
│   │   ├── us-ma-boston/
│   │   ├── us-pa-pittsburgh-hazelwood/
│   │   └── sg-one-north/
│   └── data/cache/mini/                   # nuPlan .db 数据库文件
│
├── scripts/                               # 实验脚本
│   ├── README.md                          #   脚本说明
│   ├── train_irl_neural_reward_v7.py      # ★ V7 神经奖励IRL（当前主脚本）
│   ├── train_multi_demo_big_map_irl.py    #   多示范大地图IRL训练
│   ├── build_las_vegas_big_map.py         #   GPKG → 栅格地图构建
│   ├── extract_las_vegas_multi_demos.py   #   多专家轨迹提取
│   ├── extract_ego_trajectory.py          #   单条ego轨迹提取
│   ├── real_nuplan_irl_from_csv.py        #   真实轨迹IRL实验
│   ├── quick_nuplan_irl.py                #   简化版IRL演示
│   ├── inspect_nuplan_db.py               #   数据库结构查看
│   ├── inspect_gpkg_layers.py             #   GPKG图层查看
│   ├── list_db_locations.py               #   按城市统计DB分布
│   ├── generate_experiment_report.py      #   V1报告生成脚本 (V3-V6)
│   ├── generate_v2_report.py              #   V2报告图表生成 (V3-V7)
│   ├── generate_v2_word_report.py         #   V2 Word报告生成
│   └── archive/                           #   历史版本 (V1-V6, 已废弃)
│       ├── train_irl_goal_path_planning_v3.py   # V3
│       ├── train_irl_goal_path_planning_v4.py   # V4
│       ├── train_irl_autonomous_model_v5.py     # V5
│       ├── train_irl_autonomous_model_v6_heavy.py# V6 Heavy
│       ├── irl_neural_reward_v7_common.py       # V7 公共函数(原始)
│       ├── train_irl_neural_reward_v7.py        # V7 训练(原始)
│       └── test_irl_*.py                        # 各版本测试脚本
│
├── outputs/                               # 实验输出
│   ├── IRL_实验报告_V3-V7_v2.docx          # ★ Word 正式报告 (最新)
│   ├── experiment_report_v2.md            # ★ Markdown 报告 (最新)
│   ├── IRL_实验报告_V3-V6.docx             #   Word 报告 (V1旧版)
│   ├── experiment_report.md               #   Markdown 报告 (V1旧版)
│   │
│   ├── nuplan_irl_neural_reward_v7/       # ★ V7 最终输出
│   │   ├── neural_reward_v7_best.pt       #   训练好的模型权重 (410KB)
│   │   ├── neural_reward_v7_last.pt       #   最后一个epoch模型
│   │   ├── dataset_v7.npz                 #   训练数据集 (800 cases)
│   │   ├── feature_stats_v7.npz           #   特征标准化参数
│   │   ├── model_config_v7.json           #   模型配置
│   │   ├── training_loss_v7.csv           #   训练loss记录
│   │   ├── neural_reward_training_v7.png  #   训练曲线图
│   │   ├── planning_summary_v7.png        #   测试综合结果图
│   │   └── test_figures/                  #   12张测试case路径图
│   │
│   ├── las_vegas_big_map/                 #   大地图构建输出
│   │   ├── las_vegas_big_map.npz          #   栅格地图 (240×187)
│   │   └── multi_expert_trajectories_grid.csv  # 专家轨迹栅格坐标
│   │
│   ├── las_vegas_big_map_irl/             #   多示范IRL训练结果
│   ├── las_vegas_multi_demo/              #   多专家轨迹提取结果
│   ├── nuplan_real/                       #   真实ego轨迹数据
│   │   ├── ego_trajectory.csv             #   ego位姿轨迹
│   │   └── ego_trajectory.png             #   轨迹可视化
│   │
│   ├── report_*.png                       #   报告图表集
│   │   ├── report_all_versions_loss.png   #   V3/V4/V6/V7 Loss对比
│   │   ├── report_v7_training.png         #   V7 训练曲线
│   │   ├── report_v7_planning_summary.png #   V7 测试详细对比
│   │   ├── report_v7_before_after.png     #   V7 障碍物间隙修复前后
│   │   ├── report_v7_differentiation.png  #   V6/V7 路径区分度
│   │   ├── report_loss_comparison.png     #   V3/V4/V6 Loss对比 (旧)
│   │   ├── report_ade_comparison.png      #   V3/V4 ADE对比
│   │   └── report_v6_weights.png          #   V6 奖励权重图
│   │
│   ├── db_structure.txt                   #   数据库表结构
│   ├── db_locations.csv                   #   各城市DB统计
│   ├── gpkg_layers.txt                    #   GPKG图层列表
│   └── archive/                           #   历史版本输出 (V1-V6)
│
└── .claude/                               # Claude 配置
```

## 实验版本演进

| 版本 | 脚本 | 核心特点 | 结果 |
|------|------|----------|------|
| V3 | `train_irl_goal_path_planning_v3.py` | Goal-conditioned IRL, 9特征, 43样本 | ADE=2.600 |
| V4 | `train_irl_goal_path_planning_v4.py` | +Static Clearance | ADE=1.919 (↓26%) |
| V5 | `train_irl_autonomous_model_v5.py` | +Obstacle, Train/Test分离, 180样本 | 训练不足 |
| V6 Heavy | `train_irl_autonomous_model_v6_heavy.py` | 规模扩大, 2400 cases, 900 epochs | Success=100% |
| **V7** | `train_irl_neural_reward_v7.py` | **神经网络奖励, 800 cases, 障碍物规避** | Acc=64.2%, obs_min=3.73 |

## 快速开始

### 环境

```bash
conda activate nuplan_irl
# Python 3.9, PyTorch 2.8, numpy, scipy, matplotlib, pandas, shapely
```

### 运行V7训练

```bash
python scripts/train_irl_neural_reward_v7.py \
  --total-cases 800 \
  --epochs 60 \
  --num-tests 24
```

### 使用已训练模型测试

```bash
python scripts/train_irl_neural_reward_v7.py \
  --skip-train \
  --num-tests 20
```

### 生成报告

```bash
python scripts/generate_v2_report.py           # 图表
python scripts/generate_v2_word_report.py      # Word 文档
```

## 数据来源

- **nuPlan v1.1 Mini Split**: 约 50 个 `.db` 文件，Las Vegas 区域占 48 个
- **nuPlan Maps v1.0**: Las Vegas Strip HD Map → 栅格化 (240×187 cells)
- **地图语义层**: 0=无效区, 1=可行驶区, 2=车道, 3=连接区, 4=交叉口

## 环境信息

| 项目 | 配置 |
|------|------|
| 硬件 | NVIDIA Jetson (ARM aarch64, 61GiB RAM) |
| OS | Ubuntu 20.04.6 LTS, L4T R35.4.1 |
| Python | 3.9.23 (Miniforge conda env: `nuplan_irl`) |
| 关键依赖 | numpy, scipy, matplotlib, pandas, shapely, PyTorch 2.8.0 |
