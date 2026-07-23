# Adaptive Acceleration Factors

基于轨迹距离分析的**每帧自适应加速因子**方案。共 3 个脚本 + 1 个工具模块。

## 原理

通过距离度量（DTW 或 MSE）比较 **reference chunk（chunk_size）与 extended chunk（L=chunk_size×factor）** 的相似度，找到每帧可承受的最大加速因子。训练时按因子对 action chunk 做自适应重采样。

```
原始: frame t 训练 target = actions[t : t+chunk_size]
加速: frame t 训练 target = resample(actions[t : t+chunk_size×factor[t]], chunk_size)
```

## 项目结构

```
adaptive_acc/
├── adaptive_utils.py        # 共享工具（extract_action_chunk, resample_linear, 距离计算, HDF5 读写）
├── compute_factors.py       # 脚本1: 距离分析 → factors_{chunk}_{mode}_{thmode}_{value} (HDF5)
├── train.py                 # 脚本2: 训练 + --eval 评估
├── visualize.py             # 脚本3: 渲染视频（camera + 3D TCP + 因子曲线 overlay）
└── README.md                # 本文件
```

## 环境依赖

```bash
conda activate aloha
pip install dtaidistance
```

其余依赖 (`numpy`, `scipy`, `h5py`, `modern_robotics`, `torch`) 已包含在 aloha 环境中。

## HDF5 字段命名

因子写入 HDF5 的字段名遵循规则: `factors_{chunk_size}_{mode}_{threshold_mode}_{value}`

| 参数 | 字段名示例 |
|------|-----------|
| chunk=50, dtw, abs=0.03 | `factors_50_dtw_abs_0_03` |
| chunk=30, mse, rel=0.4 | `factors_30_mse_rel_0_4` |
| chunk=50, dtw, rel=0.2 | `factors_50_dtw_rel_0_2` |

同一 HDF5 可共存多组不同参数的因子，加载时精确匹配字段名。

## 脚本1: compute_factors.py — 计算因子

```bash
# DTW + 绝对阈值
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human \
    --chunk_size 50 \
    --mode dtw --threshold_mode abs --threshold 0.03

# MSE + 相对阈值
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human \
    --chunk_size 30 \
    --mode mse --threshold_mode rel --ratio 0.4

# 使用 6D TCP (FK from qpos)
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human \
    --chunk_size 50 \
    --mode dtw --threshold_mode abs --threshold 0.03 \
    --tcp_only
```

**参数**:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--task_name` | 必填 | 任务名，对应 `data/` 下数据集目录 |
| `--chunk_size` | 必填 | action chunk 长度，与训练一致 |
| `--mode` | `dtw` | 距离度量: `dtw` 或 `mse` |
| `--threshold_mode` | 必填 | 阈值模式: `abs`（绝对）或 `rel`（相对比例） |
| `--threshold` | — | abs 模式必填，绝对阈值 |
| `--ratio` | — | rel 模式必填，比例值 |
| `--tcp_only` | False | FK 转 6D TCP，否则用 14D action |
| `--max_factor` | 5.0 | 最大加速因子 |
| `--scale_step` | 0.1 | 因子搜索步长 |
| `--frame_stride` | 1 | 帧采样间隔 |
| `--dtw_window` | 5 | DTW Sakoe-Chiba 窗口（0=无约束） |
| `--filter_window` | 9 | 中值滤波窗口 |
| `--episodes 0 1 2` | 全部 | 指定 episode |
| `--num_workers` | 0 | 并行数（0=自动） |

**阈值模式详解**:

- **abs**: 每帧最优因子 = 距离 < `threshold` 的最大 scale_factor
- **rel**: 每帧 `dynamic_threshold = dist[1.0] + ratio × (dist[max_factor] - dist[1.0])`，
  最优因子 = 距离 < dynamic_threshold 的最大 scale_factor

**末尾帧**: 剩余帧 ≤ chunk_size → 距离返回 inf，factor 退回 1.0。

**输出**: 写入 `data/{task_name}/episode_*.hdf5`，终端打印统计摘要。

## 脚本2: train.py — 训练 / 评估

```bash
# 训练 (--mode --threshold_mode --threshold 须与 compute_factors 一致)
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --speedup --temporal_agg \
    --mode dtw --threshold_mode abs --threshold 0.03 \
    --num_epochs 16000 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0

# 评估
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --speedup --temporal_agg \
    --num_epochs 0 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0 --eval

# DP + rel 模式
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/DP/sim_insertion_human/ \
    --policy_class DP --chunk_size 48 --speedup --temporal_agg \
    --mode dtw --threshold_mode rel --ratio 0.2 \
    --num_epochs 16000 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0
```

**参数**:

| 参数 | 说明 |
|------|------|
| `--eval` | 评估模式（不训练） |
| `--speedup` | 启用自适应因子压缩 |
| `--mode` | speedup 时必填，距离度量 |
| `--threshold_mode` | speedup 时必填，阈值模式 |
| `--threshold` | abs 模式时必填 |
| `--ratio` | rel 模式时必填 |
| `--temporal_agg` | 时序聚合 |
| 其余 | 与原 `imitate_episodes.py` 一致 |

## 脚本3: visualize.py — 渲染视频

视频左上角叠加 3D TCP 轨迹，右上角叠加因子曲线指示器。

额外依赖: `opencv-python` (cv2)

```bash
python adaptive_acc/visualize.py \
    --hdf5_path data/sim_insertion_human/episode_0.hdf5 \
    --output_dir ./videos \
    --camera top --chunk_size 50 --fps 50 \
    --mode dtw --threshold_mode abs --threshold 0.03
```

**参数**:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--hdf5_path` | 必填 | HDF5 episode 文件路径 |
| `--output_dir` | `./videos` | 输出目录 |
| `--camera` | `top` | 相机名 |
| `--chunk_size` | 50 | 需与 compute_factors 一致 |
| `--mode` | 必填 | 需与 compute_factors 一致 |
| `--threshold_mode` | 必填 | 需与 compute_factors 一致 |
| `--threshold` | — | abs 模式必填 |
| `--ratio` | — | rel 模式必填 |
| `--fps` | 50 | 视频帧率 |

## 典型命令速查

```bash
# === sim_insertion_human, DTW + abs ===

# 1. 计算因子
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human --chunk_size 50 \
    --mode dtw --threshold_mode abs --threshold 0.03

# 2. 训练
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --speedup --temporal_agg \
    --mode dtw --threshold_mode abs --threshold 0.03 \
    --num_epochs 16000 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0

# 3. 评估
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --speedup --temporal_agg \
    --num_epochs 0 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0 --eval

# 4. 可视化
python adaptive_acc/visualize.py \
    --hdf5_path data/sim_insertion_human/episode_0.hdf5 \
    --camera top --chunk_size 50 \
    --mode dtw --threshold_mode abs --threshold 0.03


# === sim_insertion_human, MSE + rel ===

# 1. 计算因子
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human --chunk_size 30 \
    --mode mse --threshold_mode rel --ratio 0.4

# 2. 训练
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human_mse_rel/ \
    --policy_class ACT --chunk_size 30 --speedup --temporal_agg \
    --mode mse --threshold_mode rel --ratio 0.4 \
    --num_epochs 16000 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0
```
