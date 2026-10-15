from pathlib import Path
import os
import hydra
from omegaconf import OmegaConf
import numpy as np

from robobase.adaptive_acc import compute_factors_for_demos, save_factors_npy
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
    factory = BiGymEnvFactory()
    factory.collect_or_fetch_demos(cfg, cfg.demos)

    eval_env = factory.make_eval_env(cfg, Path.cwd())
    factory.post_collect_or_fetch_demos(cfg, Path.cwd())
    eval_env.close()

    demos = [d for d in factory._demos if d[0][-1]["demo"] == 1]
    print(f"Filtered {len(demos)} successful demos out of {len(factory._demos)} total.")

    factors = compute_factors_for_demos(demos, cfg.adaptive, factory)

    save_dir = Path(hydra.utils.get_original_cwd()) / "exp_local/factors" / cfg.env.task_name
    for ep_idx, factor in enumerate(factors):
        path = save_factors_npy(factor, str(save_dir), ep_idx, adaptive_cfg=cfg.adaptive)
        print(f"Saved: {path}")

    print("\nAll factors computed and saved successfully.")


if __name__ == "__main__":
    main()
