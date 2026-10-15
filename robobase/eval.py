from pathlib import Path
import os
import hydra
from omegaconf import OmegaConf


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
    from robobase.workspace import Workspace

    root_dir = Path.cwd()

    workspace = Workspace(cfg)
    # put cups: 'exp_local/pixel_act/bigym_put_cups_20241217133245/snapshots/17500_snapshot.pt'
    # sandwich remove: 'exp_local/pixel_act/bigym_sandwich_remove_20241217133401/snapshots/20000_snapshot.pt'
    snapshot = workspace.work_dir/'snapshots/best_snapshot.pt'
    assert snapshot.exists()
    print(f"resuming: {snapshot}")
    workspace.load_snapshot(snapshot)
    workspace.eval()


if __name__ == "__main__":
    main()
