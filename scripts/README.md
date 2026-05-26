# Scripts 说明

## 实验流水线（按顺序执行）

```
1. 数据探索
   inspect_nuplan_db.py        → 查看 nuPlan .db 结构
   list_db_locations.py        → 统计各城市 DB 数量
   inspect_gpkg_layers.py      → 查看 GPKG 地图图层

2. 专家轨迹提取
   extract_ego_trajectory.py          → 单条 ego 轨迹提取
   extract_las_vegas_multi_demos.py   → Las Vegas 多条专家轨迹提取

3. 地图构建
   build_las_vegas_big_map.py   → 从 GPKG 构建大范围栅格地图

4. MaxEnt IRL 训练与路径生成
   quick_nuplan_irl.py                → 简化版（合成数据，验证算法流程）
   real_nuplan_irl_from_csv.py        → 单条真实轨迹 IRL
   train_multi_demo_big_map_irl.py    → 多示范大地图 IRL（最终版）
```

## 脚本职责

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `inspect_nuplan_db.py` | `.db` 文件 | `outputs/db_structure.txt` | 查看数据库表结构和样本 |
| `list_db_locations.py` | 所有 `.db` | `outputs/db_locations.csv` | 按 location 统计 DB 分布 |
| `inspect_gpkg_layers.py` | `map.gpkg` | `outputs/gpkg_layers.txt` | 列出 GPKG 所有图层 |
| `extract_ego_trajectory.py` | `.db` | `outputs/nuplan_real/ego_trajectory.csv/png` | 提取 ego 车辆位姿轨迹 |
| `extract_las_vegas_multi_demos.py` | 所有 `las_vegas` DB | `outputs/las_vegas_multi_demo/` | 多条专家轨迹筛选和保存 |
| `build_las_vegas_big_map.py` | `map.gpkg` + 轨迹 CSV | `outputs/las_vegas_big_map/` | 构建大范围语义栅格地图 |
| `quick_nuplan_irl.py` | 无（合成数据） | `outputs/quick_irl_*` | 简化版 MaxEnt IRL 演示 |
| `real_nuplan_irl_from_csv.py` | `ego_trajectory.csv` | `outputs/nuplan_real/real_irl_*` | 单条真实轨迹 IRL 训练 |
| `train_multi_demo_big_map_irl.py` | 大地图 + 多轨迹 | `outputs/las_vegas_big_map_irl/` | 多示范大地图 IRL（完整实验） |

## archive/

已废弃的历史版本脚本（v2~v7 迭代过程），仅供参考。
