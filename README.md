# nuPlan IRL 路径规划实验

用逆强化学习（IRL）做nuPlan自动驾驶路径规划的实验代码，基于Las Vegas Strip地图。从最开始的线性奖励模型一路做到V7的神经网络奖励模型，主要是想看看能不能让规划出来的路径更贴近人类驾驶行为。

## 项目结构

```
.
├── scripts/
│   ├── train_irl_neural_reward_v7.py   # V7主脚本，一条龙：生成数据集→训练→测试→出图
│   ├── build_las_vegas_big_map.py      # 把nuPlan的GPKG地图转成栅格地图
│   └── extract_las_vegas_multi_demos.py # 从数据库里提取专家轨迹
├── outputs/
│   ├── las_vegas_big_map/              # 建好的栅格地图+专家轨迹可视化
│   └── nuplan_irl_neural_reward_v7/    # V7训练输出（模型权重、loss、测试图等）
└── data/                               # nuPlan原始数据，太大了不上传（https://www.nuscenes.org/nuplan#download）
```

## 思路演进

迭代过程：

- **V3**：最开始用 9 个手工特征做 Goal-conditioned IRL，算是把整个流程跑通了
- **V4**：加了 Static Clearance 特征，ADE 降了不少，说明离障碍物远一点确实有用
- **V5**：加了障碍物感知，也把训练集和测试集分开了，但训练得不太充分
- **V6**：暴力加大数据量（2400 cases）和训练轮数（900 epochs），效果好了但总感觉特征工程到头了
- **V7**：干脆把线性奖励换成一个小的 MLP 网络，用交叉熵损失代替最大熵 IRL 那套。核心思路是用一个 teacher utility function 给候选路径打分当伪标签，然后让网络去学怎么给路径排序，这样就不用死磕手工特征了

## 怎么跑？

环境用 conda：

```bash
conda activate nuplan_irl
```

训练 V7：

```bash
python scripts/train_irl_neural_reward_v7.py \
  --total-cases 800 \
  --epochs 60 \
  --num-tests 24
```

拿训练好的模型做测试：

```bash
python scripts/train_irl_neural_reward_v7.py \
  --skip-train \
  --num-tests 20
```

地图构建（nuPlan数据要去官网下载https://www.nuscenes.org/nuplan#download）：

```bash
python scripts/build_las_vegas_big_map.py
python scripts/extract_las_vegas_multi_demos.py
```

## 数据

- **nuPlan v1.1 Mini Split**：大概 50 个 `.db` 文件，大部分是 Las Vegas 的
- **nuPlan Maps v1.0**：Las Vegas Strip 高精地图，栅格化成 240×187
- **地图语义**：0=不可走, 1=可行驶, 2=车道, 3=连接区, 4=交叉口

## 环境

跑在Jetson上，纯CPU没有GPU，所以所有实验都是单进程跑的，batch size也比较小。

| 项目 | 配置 |
|------|------|
| 硬件 | NVIDIA Jetson (ARM aarch64, 61GiB RAM) |
| 系统 | Ubuntu 20.04.6 LTS |
| Python | 3.9 + PyTorch 2.8 + numpy/scipy/matplotlib/shapely |
