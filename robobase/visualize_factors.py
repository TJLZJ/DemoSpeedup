from pathlib import Path
import os
import hydra
from omegaconf import OmegaConf
import numpy as np

from robobase.adaptive_acc import (
    load_factors_npy,
    visualize_factors_for_demos,
)
from robobase.envs.bigym import BiGymEnvFactory


def _adapt_dir_suffix(adaptive):
    if adaptive is None or adaptive is False or not hasattr(adaptive, "get"):
        return "default"
    mode = adaptive.get("mode", "default")
    th_mode = adaptive.get("threshold_mode", "default")
    chunk = adaptive.get("chunk_size", 24)
    if th_mode == "abs":
        val = str(adaptive.get("threshold", 0.05)).replace(".", "_")
    elif th_mode == "rel":
        val = str(adaptive.get("ratio", 0.4)).replace(".", "_")
    else:
        val = th_mode
    return f"c{chunk}_{mode}_{th_mode}_{val}"


OmegaConf.register_new_resolver("adapt_dir", _adapt_dir_suffix, replace=True)


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    cfg.pixels = True

    factory = BiGymEnvFactory()
    factory.collect_or_fetch_demos(cfg, cfg.demos)
    factory.post_collect_or_fetch_demos(cfg, Path.cwd())

    demos_raw = []
    for d_idx, demo in enumerate(factory._raw_demos):
        success_val = factory._demos[d_idx][0][-1].get("demo", 0)
        if success_val == 1:
            frames = [ts.observation for ts in demo.timesteps]
            demos_raw.append(frames)

    print(f"Loaded {len(demos_raw)} demos with RGB frames out of {len(factory._demos)} total.")
    if demos_raw:
        print(f"Steps per demo: {len(demos_raw[0])}, Obs keys: {list(demos_raw[0][0].keys())}")

    save_dir = Path(hydra.utils.get_original_cwd()) / "exp_local/factors" / cfg.env.task_name
    factors = []
    episode = 0
    while True:
        f = load_factors_npy(str(save_dir), episode, adaptive_cfg=cfg.adaptive)
        if f is None:
            break
        factors.append(f)
        episode += 1

    if not factors:
        print("No factors found! Run compute_factors.py first.")
        return

    demos_wrapped = [
        [(obs,) for obs in d[: len(factors[i])]]
        for i, d in enumerate(demos_raw)
    ]
    for i in range(len(factors)):
        print(f"  ep{i}: factors={len(factors[i])}, frames={len(demos_wrapped[i])}")

    chunk_size = cfg.adaptive.get("chunk_size", 24)
    output_dir = Path.cwd()
    visualize_factors_for_demos(demos_wrapped, factors, str(output_dir), chunk_size=chunk_size)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
