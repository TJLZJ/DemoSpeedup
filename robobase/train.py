from pathlib import Path
import os
import hydra
from omegaconf import OmegaConf

os.environ['WANDB_BASE_URL']='https://api.wandb.ai'


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

    snapshot = workspace.work_dir/'snapshots/snapshot.pt'
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
