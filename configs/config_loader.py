import json
import os
from types import SimpleNamespace
from copy import deepcopy


# ============================================================================
# Recursive namespace so config values can be accessed as cfg.training.lr
# instead of cfg["training"]["lr"], while still behaving like a dict when
# needed (cfg.to_dict()).
# ============================================================================
class ConfigNamespace(SimpleNamespace):
    """
    Thin wrapper around SimpleNamespace that:
      - recursively converts nested dicts into ConfigNamespace
      - keeps the original dict around (self._raw) so it can be dumped back
        to JSON exactly as it was loaded (e.g. when saving a config copy
        next to a checkpoint)
    """

    def __init__(self, data: dict):
        self._raw = deepcopy(data)
        converted = {}
        for key, value in data.items():
            converted[key] = self._convert(value)
        super().__init__(**converted)

    @staticmethod
    def _convert(value):
        if isinstance(value, dict):
            return ConfigNamespace(value)
        if isinstance(value, list):
            return [ConfigNamespace._convert(v) for v in value]
        return value

    def to_dict(self) -> dict:
        """Return the original (unconverted) dict, exactly as loaded."""
        return deepcopy(self._raw)

    def __repr__(self):
        return f"ConfigNamespace({self._raw})"


# ============================================================================
# Required top-level sections / keys. If any of these are missing the config
# is rejected -- better to fail at startup than half-way through training.
# ============================================================================
_REQUIRED_TOP_LEVEL_KEYS = [
    "experiment",
    "model",
    "data",
    "training",
    "loss",
    "evaluation",
    "checkpointing",
    "logging",
]

_REQUIRED_MODEL_KEYS = [
    "dino_ground_checkpoint",
    "dino_satellite_checkpoint",
    "text_encoder_checkpoint",
    "freeze_dino",
    "freeze_text_encoder",
    "embed_dim",
]

_REQUIRED_TRAINING_KEYS = [
    "batch_size",
    "stage_schedule",
    "optimizer",
]


def _validate_raw_config(raw: dict):
    missing_top = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in raw]
    if missing_top:
        raise ValueError(f"Config is missing required top-level sections: {missing_top}")

    missing_model = [k for k in _REQUIRED_MODEL_KEYS if k not in raw["model"]]
    if missing_model:
        raise ValueError(f"Config['model'] is missing required keys: {missing_model}")

    missing_train = [k for k in _REQUIRED_TRAINING_KEYS if k not in raw["training"]]
    if missing_train:
        raise ValueError(f"Config['training'] is missing required keys: {missing_train}")

    if "stage1_epochs" not in raw["training"]["stage_schedule"]:
        raise ValueError("Config['training']['stage_schedule'] must define 'stage1_epochs'")
    if "total_epochs" not in raw["training"]["stage_schedule"]:
        raise ValueError("Config['training']['stage_schedule'] must define 'total_epochs'")

    # ------------------------------------------------------------------
    # Hard policy checks. These flags are core to this pipeline's design
    # and should not silently be turned off by a bad config edit.
    # ------------------------------------------------------------------
    if raw["model"].get("freeze_dino") is not True:
        raise ValueError("model.freeze_dino must be true -- DINOv3 must remain frozen")
    if raw["model"].get("freeze_text_encoder") is not True:
        raise ValueError("model.freeze_text_encoder must be true -- the text encoder must remain frozen")
    if raw["checkpointing"].get("save_only_trainable_weights") is not True:
        raise ValueError(
            "checkpointing.save_only_trainable_weights must be true -- "
            "this pipeline never saves frozen encoder weights"
        )
    if raw["evaluation"].get("disable_mid_training_eval") is not True:
        raise ValueError(
            "evaluation.disable_mid_training_eval must be true -- "
            "this pipeline does not run evaluation during training"
        )


def load_config(json_path: str) -> ConfigNamespace:
    """
    Load and validate the JSON config, returning a ConfigNamespace.

    Also creates the checkpoint/log/info directories declared in the config
    if they don't already exist, so the training script doesn't need to.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    _validate_raw_config(raw)

    cfg = ConfigNamespace(raw)

    os.makedirs(cfg.checkpointing.save_dir, exist_ok=True)
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    if hasattr(cfg.logging, "info_dir"):
        os.makedirs(cfg.logging.info_dir, exist_ok=True)

    return cfg


def save_config_copy(cfg: ConfigNamespace, dest_path: str):
    """
    Write the original (un-mutated) config dict to dest_path.
    Used when checkpointing, so every checkpoint carries the exact config
    it was trained under.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=4)


def get_active_stage(cfg: ConfigNamespace, epoch: int) -> int:
    """
    Fixed, epoch-count based stage switch (no validation-based switching).

    epoch is 0-indexed.
    Returns 1 for Stage 1 (global warmup), 2 for Stage 2 (patch-MIL detective).
    """
    stage1_epochs = cfg.training.stage_schedule.stage1_epochs
    return 1 if epoch < stage1_epochs else 2


if __name__ == "__main__":
    # Quick smoke test: python utils/config_loader.py configs/geo_dino_siglip_qformer.json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/geo_dino_siglip_qformer.json"
    cfg = load_config(path)
    print(f"Loaded config for experiment: {cfg.experiment.exp_id}")
    print(f"Stage 1 epochs: {cfg.training.stage_schedule.stage1_epochs}")
    print(f"Total epochs:   {cfg.training.stage_schedule.total_epochs}")
    for epoch in range(cfg.training.stage_schedule.total_epochs):
        stage = get_active_stage(cfg, epoch)
        print(f"  epoch {epoch+1}/{cfg.training.stage_schedule.total_epochs} -> Stage {stage}")