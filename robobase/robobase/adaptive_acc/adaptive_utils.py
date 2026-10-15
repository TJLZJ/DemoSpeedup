import numpy as np
import torch
import os


def extract_action_chunk(actions, start_idx, length, pad_last_frame=True):
    if start_idx + length > len(actions):
        if pad_last_frame:
            chunk = np.zeros((length, actions.shape[1]), dtype=actions.dtype)
            available = len(actions) - start_idx
            chunk[:available] = actions[start_idx:]
            if available > 0 and available < length:
                chunk[available:] = actions[-1]
            return chunk
        else:
            return actions[start_idx:]
    return actions[start_idx : start_idx + length]


def resample_linear(actions, original_length, target_length):
    if original_length == target_length:
        return actions.copy()
    orig_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    result = np.zeros((target_length, actions.shape[1]), dtype=actions.dtype)
    for d in range(actions.shape[1]):
        result[:, d] = np.interp(
            target_indices, orig_indices, actions[:, d]
        )
    return result


def compute_mse_distance(reference_chunk, extended_chunk, chunk_size):
    resampled = resample_linear(extended_chunk, len(extended_chunk), chunk_size)
    return np.mean((reference_chunk - resampled) ** 2)


def compute_dtw_distance(reference_chunk, extended_chunk):
    try:
        from dtaidistance import dtw_ndim
    except ImportError:
        raise ImportError(
            "dtaidistance is required for DTW mode. "
            "Install it via: pip install dtaidistance"
        )
    return dtw_ndim.distance(
        reference_chunk.astype(np.float64), extended_chunk.astype(np.float64)
    )


def process_action_adaptive(action_data, factor, chunk_size, action_len):
    L = max(1, int(chunk_size * factor))
    L = min(L, action_len)
    extracted = extract_action_chunk(action_data, 0, L)
    resampled = resample_linear(extracted, L, chunk_size)
    if len(resampled) < chunk_size:
        padded = np.zeros((chunk_size, resampled.shape[1]), dtype=resampled.dtype)
        padded[: len(resampled)] = resampled
        resampled = padded
    result = np.zeros((chunk_size, action_data.shape[1]), dtype=action_data.dtype)
    result[:chunk_size] = resampled
    return result


def _build_factor_subdir(adaptive_cfg):
    mode = adaptive_cfg.get("mode", "mse")
    th_mode = adaptive_cfg.get("threshold_mode", "abs")
    chunk = adaptive_cfg.get("chunk_size", 24)
    if th_mode == "abs":
        val = str(adaptive_cfg.get("threshold", 0.05)).replace(".", "_")
    else:
        val = str(adaptive_cfg.get("ratio", 0.4)).replace(".", "_")
    return f"c{chunk}_{mode}_{th_mode}_{val}"


def load_factors_npy(base_dir, episode, adaptive_cfg=None):
    if adaptive_cfg is not None:
        subdir = _build_factor_subdir(adaptive_cfg)
        dir_path = os.path.join(base_dir, subdir)
    else:
        dir_path = base_dir
    file_path = os.path.join(dir_path, f"factors_{episode}.npy")
    if not os.path.exists(file_path):
        return None
    return np.load(file_path)


def save_factors_npy(data, base_dir, episode, adaptive_cfg=None):
    if adaptive_cfg is not None:
        subdir = _build_factor_subdir(adaptive_cfg)
        dir_path = os.path.join(base_dir, subdir)
    else:
        dir_path = base_dir
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"factors_{episode}.npy")
    np.save(file_path, data)
    return file_path
