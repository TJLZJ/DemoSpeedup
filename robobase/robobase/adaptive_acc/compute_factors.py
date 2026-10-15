import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from .adaptive_utils import (
    extract_action_chunk,
    resample_linear,
    compute_mse_distance,
    compute_dtw_distance,
)


def _compute_frame_factor(
    traj,
    t,
    chunk_size,
    scale_factors,
    threshold,
    threshold_mode,
    ratio,
    mode,
    thresh_at_1,
    thresh_at_max,
):
    ref_chunk = extract_action_chunk(traj, t, chunk_size)
    candidate_sf = 1.0

    for sf in scale_factors:
        L = max(1, int(chunk_size * sf))
        ext_chunk = extract_action_chunk(traj, t, L)
        if mode == "mse":
            dist = compute_mse_distance(ref_chunk, ext_chunk, chunk_size)
        elif mode == "dtw":
            dist = compute_dtw_distance(ref_chunk, ext_chunk)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if threshold_mode == "abs":
            if dist <= threshold:
                candidate_sf = sf
            else:
                break
        elif threshold_mode == "rel":
            d_thresh = thresh_at_1 + ratio * (thresh_at_max - thresh_at_1)
            if dist <= d_thresh:
                candidate_sf = sf
            else:
                break

    return candidate_sf


def _compute_initial_thresholds(
    traj, t, chunk_size, max_factor, scale_step, mode
):
    scale_factors = np.arange(1.0, max_factor + scale_step / 2, scale_step)
    ref_chunk = extract_action_chunk(traj, t, chunk_size)
    dist_at_1 = None
    dist_at_max = None

    for sf in scale_factors:
        L = max(1, int(chunk_size * sf))
        ext_chunk = extract_action_chunk(traj, t, L)
        if mode == "mse":
            dist = compute_mse_distance(ref_chunk, ext_chunk, chunk_size)
        elif mode == "dtw":
            dist = compute_dtw_distance(ref_chunk, ext_chunk)
        else:
            dist = 0.0

        if abs(sf - 1.0) < 1e-6:
            dist_at_1 = dist
        if abs(sf - max_factor) < 1e-6:
            dist_at_max = dist

    return dist_at_1, dist_at_max


def compute_factors_single_demo(
    traj,
    chunk_size,
    max_factor=3.0,
    scale_step=0.2,
    threshold=0.05,
    threshold_mode="abs",
    ratio=0.4,
    mode="mse",
    filter_window=9,
):
    scale_factors = np.arange(1.0, max_factor + scale_step / 2, scale_step)
    N = len(traj)
    factors = np.ones(N, dtype=np.float64)

    if N < chunk_size:
        return factors

    max_t = N - chunk_size

    for t in range(0, max_t):
        if threshold_mode == "rel":
            d1, dmax = _compute_initial_thresholds(
                traj, t, chunk_size, max_factor, scale_step, mode
            )
        else:
            d1, dmax = None, None

        factors[t] = _compute_frame_factor(
            traj,
            t,
            chunk_size,
            scale_factors,
            threshold,
            threshold_mode,
            ratio,
            mode,
            d1,
            dmax,
        )

    for t in range(max_t, N):
        factors[t] = 1.0

    from scipy.signal import medfilt

    w = int(filter_window)
    if w % 2 == 0:
        w += 1
    if w > 1 and len(factors) > w:
        factors = medfilt(factors, w)

    return factors


def _worker_compute(args_tuple):
    (
        ep_idx,
        traj,
        chunk_size,
        max_factor,
        scale_step,
        threshold,
        threshold_mode,
        ratio,
        mode,
        filter_window,
    ) = args_tuple

    factors = compute_factors_single_demo(
        traj=traj,
        chunk_size=chunk_size,
        max_factor=max_factor,
        scale_step=scale_step,
        threshold=threshold,
        threshold_mode=threshold_mode,
        ratio=ratio,
        mode=mode,
        filter_window=filter_window,
    )
    stats = {
        "mean_factor": float(np.mean(factors)),
        "max_factor": float(np.max(factors)),
        "min_factor": float(np.min(factors)),
        "spedup_ratio": float(np.mean(factors > 1.0)),
    }
    return ep_idx, factors, stats


def compute_factors_for_demos(
    demos,
    adaptive_cfg,
    env_factory=None,
    num_workers=0,
):
    trajectories = []
    for demo in demos:
        traj = _extract_trajectory(demo, adaptive_cfg, env_factory)
        trajectories.append(traj)

    chunk_size = adaptive_cfg.get("chunk_size", 24)
    max_factor = adaptive_cfg.get("max_factor", 3.0)
    scale_step = adaptive_cfg.get("scale_step", 0.2)
    threshold = adaptive_cfg.get("threshold", 0.05)
    threshold_mode = adaptive_cfg.get("threshold_mode", "abs")
    ratio = adaptive_cfg.get("ratio", 0.4)
    mode = adaptive_cfg.get("mode", "mse")
    filter_window = adaptive_cfg.get("filter_window", 9)

    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() // 2)

    args_list = [
        (
            ep_idx,
            traj,
            chunk_size,
            max_factor,
            scale_step,
            threshold,
            threshold_mode,
            ratio,
            mode,
            filter_window,
        )
        for ep_idx, traj in enumerate(trajectories)
    ]

    if num_workers > 1 and len(args_list) > 1:
        with mp.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_worker_compute, args_list),
                    total=len(args_list),
                    desc="Computing Factors",
                    unit="episode",
                )
            )
    else:
        results = []
        for args in tqdm(args_list, desc="Computing Factors", unit="episode"):
            results.append(_worker_compute(args))

    results.sort(key=lambda x: x[0])
    factors_list = [r[1] for r in results]

    print("\n===== Factor Statistics =====")
    print(f"{'Episode':>8} {'Mean':>8} {'Max':>8} {'Min':>8} {'Speedup%':>10}")
    for ep_idx, _, stats in results:
        print(
            f"{ep_idx:>8} {stats['mean_factor']:>8.3f} {stats['max_factor']:>8.3f} "
            f"{stats['min_factor']:>8.3f} {stats['spedup_ratio']:>9.1%}"
        )

    return factors_list


def _extract_trajectory(demo, adaptive_cfg, env_factory=None):
    traj_type = adaptive_cfg.get("traj", "raw_action")

    if traj_type == "raw_action":
        actions = []
        for step in demo:
            if len(step) >= 2:
                info = step[-1]
            elif len(step) == 1:
                info = step[0]
            else:
                info = {}
            demo_action = info.get("demo_action", None)
            if demo_action is not None:
                actions.append(demo_action)
        return np.array(actions)

    elif traj_type == "tcp":
        qpos_list = []
        for step in demo:
            obs = None
            for s in step:
                if isinstance(s, dict) and "qpos" in s:
                    obs = s
                    break
                elif isinstance(s, dict) and "low_dim_state" in s:
                    obs = s
            if obs is not None:
                if "qpos" in obs:
                    qpos_list.append(obs["qpos"])
                else:
                    qpos_list.append(np.zeros(14))
        qpos_array = np.array(qpos_list)

        left_qpos = qpos_array[:, :7]
        right_qpos = qpos_array[:, 7:14]
        left_tcp = []
        right_tcp = []
        for i in range(len(qpos_array)):
            lq = left_qpos[i]
            rq = right_qpos[i]
            try:
                import modern_robotics.core as mr

                M = np.array([
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1],
                ])
                Slist = np.array([
                    [0, 0, 1, 0, 0, 0],
                    [0, 1, 0, -0.1, 0, 0],
                    [0, 1, 0, -0.2, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, -1, 0, -0.2, 0, 0],
                    [0, 1, 0, -0.1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ]).T
                T_left = mr.FKinSpace(M, Slist, lq.tolist())
                left_tcp.append(T_left[:3, 3])
                T_right = mr.FKinSpace(M, Slist, rq.tolist())
                right_tcp.append(T_right[:3, 3])
            except (ImportError, Exception):
                return np.array(actions) if 'actions' in dir() else np.zeros((len(demo), 6))

        return np.hstack([np.array(left_tcp), np.array(right_tcp)])

    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
