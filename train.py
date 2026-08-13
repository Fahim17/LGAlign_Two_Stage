import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from configs.config_loader import load_config, save_config_copy
from models.custom_model import GeoDinoSiglipQFormer
from losses import Stage1ContrastiveLoss, RegionVotingLoss

# ----------------------------------------------------------------------------
# Re-use the OLD dataloader structure on purpose. These dataset classes
# already return (anchor, positive, negative, txt, idx) -- we keep that
# 5-tuple unpacking unchanged below; `negative` is accepted (per spec, the
# dataloader must keep returning it) but not used in any loss -- Stage 1
# relies on in-batch negatives, and Stage 2 relies on hard candidates mined
# from Stage 1 global retrieval instead (see Stage2CandidateDataset below).
# ----------------------------------------------------------------------------
from datasets.CVUSA_dataset import CVUSA_dataset_cropped
from datasets.CVACT_dataset import CVACT_dataset_cropped
from datasets.VIGOR_dataset import VIGOR_dataset_cropped
from datasets.GAMa_dataset import GAMa_dataset_cropped


def time_stamp(msg=""):
    print(f"\n[{datetime.now()}] {msg}\n")


# ============================================================================
# Dataset / dataloader setup (old-style, dataset class picked by cfg.data.dataset_name)
# ----------------------------------------------------------------------------
# The old dataset classes only accept a single `transform` argument, so they
# can't size ground vs satellite crops differently themselves. We resize to
# the LARGER of the two working resolutions (satellite, per
# cfg.data.image_size_satellite) and convert to a tensor only here --
# normalization is deliberately NOT done in this transform. Ground images
# are downsampled to their own (smaller) resolution and both branches are
# normalized with their own DINOv3-checkpoint-matched stats inside the
# training loop, via prepare_ground_satellite_tensors() below.
# ============================================================================
def build_dataloader(cfg):
    d = cfg.data

    base_size = d.image_size_satellite
    base_transform = transforms.Compose([
        transforms.Resize((base_size, base_size)),
        transforms.ToTensor(),
    ])

    if d.dataset_name == "CVUSA":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}", header=None)
        dataset = CVUSA_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    elif d.dataset_name == "CVACT":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}")
        dataset = CVACT_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    elif d.dataset_name == "VIGOR":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}")
        dataset = VIGOR_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    elif d.dataset_name == "GAMa":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}")
        dataset = GAMa_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    else:
        raise ValueError(f"Unknown dataset_name: {d.dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=d.num_workers,
    )
    return loader


# ============================================================================
# Per-branch resolution + normalization
# ----------------------------------------------------------------------------
# anchor/positive/negative arrive pre-resized to cfg.data.image_size_satellite
# (un-normalized) from the dataloader above. Ground (anchor) is downsampled
# to its own working resolution; satellite (positive/negative) stays at its
# native loaded size. Each branch is then normalized with the stats matching
# its own DINOv3 checkpoint (web-pretrained vs SAT-493M-pretrained).
# ============================================================================
def build_norm_tensors(cfg, device):
    d = cfg.data
    g_mean = torch.tensor(d.ground_norm_mean, device=device).view(1, 3, 1, 1)
    g_std = torch.tensor(d.ground_norm_std, device=device).view(1, 3, 1, 1)
    s_mean = torch.tensor(d.satellite_norm_mean, device=device).view(1, 3, 1, 1)
    s_std = torch.tensor(d.satellite_norm_std, device=device).view(1, 3, 1, 1)
    return g_mean, g_std, s_mean, s_std


def prepare_ground_satellite_tensors(cfg, anchor, positive, negative, device, norm_tensors):
    """
    `negative` is accepted only to match the old dataloader's 5-tuple return
    -- this pipeline never reads it (Stage 1 relies on in-batch negatives;
    Stage 2 relies on hard candidates mined separately). It is deliberately
    left on the CPU, unresized, and unnormalized, so it costs no GPU
    transfer time or memory.
    """
    g_mean, g_std, s_mean, s_std = norm_tensors
    d = cfg.data

    anchor = anchor.to(device, non_blocking=True)
    positive = positive.to(device, non_blocking=True)

    if anchor.shape[-1] != d.image_size_ground or anchor.shape[-2] != d.image_size_ground:
        anchor = F.interpolate(
            anchor, size=(d.image_size_ground, d.image_size_ground), mode="bilinear", align_corners=False
        )

    anchor = (anchor - g_mean) / g_std
    positive = (positive - s_mean) / s_std

    return anchor, positive, negative


def apply_text_dropout(texts, drop_prob):
    """
    Randomly drops captions during training so the model cannot rely only on text.
    `texts` is usually a tuple/list of strings from the DataLoader.
    """
    if drop_prob <= 0:
        return texts

    out = []
    for text in list(texts):
        if torch.rand(()) < drop_prob:
            out.append("")
        else:
            out.append(text)
    return out


# ============================================================================
# Stage 2 hard-candidate mining
# ----------------------------------------------------------------------------
# Runs Stage 1 global retrieval over the full training set once, in eval
# mode, with shuffle=False so row order is a stable index. For each query i,
# the top hard_candidate_pool_k satellites are cached as that query's
# negative pool (with i itself removed). Stage 2 batches then sample
# train_candidates_per_query - 1 negatives from that per-query pool.
#
# Returns the result in memory -- on-disk caching (so a Stage 2 crash
# doesn't require re-mining from scratch) is handled by the caller,
# run_stage2_epochs(), not here. See its mined-candidates cache-dir logic.
#
# Uses its own mining_batch_size (stage2_region_voting.mining_batch_size),
# separate from Stage 1's cfg.training.batch_size -- this pass runs entirely
# under @torch.no_grad() with no backward graph or optimizer state, so it
# can usually tolerate a much larger batch size than Stage 1 training can.
# Falls back to cfg.training.batch_size if mining_batch_size isn't set, for
# backward compatibility with older configs.
#
# Q_global/S_global accumulate on CPU (cheap per-batch transfer), but the
# N x N score matrix itself is computed on GPU (device), since that matmul
# + topk is the dominant cost for large N and is far slower on CPU. Only the
# reduced [N, k] result is moved back to CPU.
#
# NOTE: mining must stay on Stage 1 global embeddings -- encode_query_stage2()
# / encode_satellite_stage2() return the frozen Stage 1 global embedding as
# their FIRST output specifically so this function (and anything else doing
# global retrieval) can keep ignoring the second (Stage 2) output below.
# Mining must never use Stage 2 region-voting scores.
# ============================================================================
@torch.no_grad()
def build_hard_candidate_cache(cfg, model, device, pool_k):
    d = cfg.data
    base_size = d.image_size_satellite
    base_transform = transforms.Compose([
        transforms.Resize((base_size, base_size)),
        transforms.ToTensor(),
    ])

    if d.dataset_name == "CVUSA":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}", header=None)
        dataset = CVUSA_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    elif d.dataset_name == "CVACT":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}")
        dataset = CVACT_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    elif d.dataset_name == "VIGOR":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}")
        dataset = VIGOR_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    elif d.dataset_name == "GAMa":
        train_data = pd.read_csv(f"{d.data_root}/{d.train_csv}")
        dataset = GAMa_dataset_cropped(df=train_data, path=d.data_root, transform=base_transform, train=True, lang=d.lang)
    else:
        raise ValueError(f"Unknown dataset_name: {d.dataset_name}")

    mining_batch_size = getattr(cfg.training.stage2_region_voting, "mining_batch_size", cfg.training.batch_size)
    mining_loader = DataLoader(dataset, batch_size=mining_batch_size, shuffle=False, num_workers=d.num_workers)
    norm_tensors = build_norm_tensors(cfg, device)
    g_mean, g_std, s_mean, s_std = norm_tensors

    model.trainable_modules.eval()

    Q_global, S_global = [], []
    for anchor, positive, negative, txt, idx in tqdm(mining_loader, desc="Mining: encoding train set"):
        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)

        if anchor.shape[-1] != d.image_size_ground or anchor.shape[-2] != d.image_size_ground:
            anchor = F.interpolate(anchor, size=(d.image_size_ground, d.image_size_ground),
                                    mode="bilinear", align_corners=False)
        anchor = (anchor - g_mean) / g_std
        positive = (positive - s_mean) / s_std

        # First output of each is the frozen Stage 1 global embedding --
        # mining is Stage 1-only by design, so the Stage 2 second output
        # (query/patch tokens) is intentionally discarded here.
        q_global, _ = model.encode_query_stage2(anchor, txt)
        s_global, _ = model.encode_satellite_stage2(positive)

        Q_global.append(q_global.cpu())
        S_global.append(s_global.cpu())

    Q_global = torch.cat(Q_global, dim=0).to(device)   # [N, D]
    S_global = torch.cat(S_global, dim=0).to(device)   # [N, D]
    N = Q_global.shape[0]

    global_scores = Q_global @ S_global.t()   # [N, N], computed on GPU
    # exclude the true positive from its own negative pool
    global_scores.fill_diagonal_(float("-inf"))

    k = min(pool_k, N - 1)
    _, topk_idx = torch.topk(global_scores, k=k, dim=1)   # [N, k]

    return topk_idx.cpu()   # row i = i's hard-negative candidate pool


# ============================================================================
# Stage 2 dataset wrapper
# ----------------------------------------------------------------------------
# Wraps an existing *_dataset_cropped instance (already constructed with the
# usual transform/path/lang args) plus a hard-negative index cache. Reuses
# the underlying dataset's own __getitem__ to load each candidate satellite
# by index, so no per-dataset-class duplication is needed here.
# `negative` from the underlying dataset's 5-tuple is never used for Stage 2.
# ============================================================================
class Stage2CandidateDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, hard_negative_idx, num_candidates):
        self.base_dataset = base_dataset
        self.hard_negative_idx = hard_negative_idx     # [N, pool_k]
        self.num_candidates = num_candidates             # C, including the positive

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        anchor, positive, _negative, txt, _idx = self.base_dataset[idx]

        pool = self.hard_negative_idx[idx]
        num_negatives = self.num_candidates - 1
        sampled = pool[torch.randperm(pool.shape[0])[:num_negatives]]

        candidate_satellites = [positive]
        for neg_idx in sampled.tolist():
            _, neg_sat, _, _, _ = self.base_dataset[neg_idx]
            candidate_satellites.append(neg_sat)
        candidate_satellites = torch.stack(candidate_satellites, dim=0)   # [C, 3, H, W]

        return anchor, txt, candidate_satellites, 0, idx


# ============================================================================
# Stage 2 dataloader -- uses its own batch size
# (cfg.training.stage2_region_voting.stage2_batch_size), independent from
# Stage 1's cfg.training.batch_size. Stage 2's effective per-step memory is
# stage2_batch_size * train_candidates_per_query satellite encodes through
# frozen DINOv3, which is why this is a separate, usually much smaller, knob.
# Falls back to cfg.training.batch_size if stage2_batch_size isn't set, for
# backward compatibility with older configs.
# ============================================================================
def build_stage2_dataloader(cfg, base_dataset, hard_negative_idx):
    rv_cfg = cfg.training.stage2_region_voting
    stage2_dataset = Stage2CandidateDataset(
        base_dataset, hard_negative_idx, num_candidates=rv_cfg.train_candidates_per_query
    )
    stage2_batch_size = getattr(rv_cfg, "stage2_batch_size", 4)
    print(f"[DEBUG] stage2_batch_size resolved to: {stage2_batch_size}")
    return DataLoader(
        stage2_dataset,
        batch_size=stage2_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )


# ============================================================================
# Stage 2 satellite shift augmentation
# ----------------------------------------------------------------------------
# Random translation applied to every candidate satellite (positive AND hard
# negatives) so Stage 2 can't shortcut to "always vote center" -- it must
# actually find the query-compatible region wherever it ends up. Never
# applied to the ground image or text. Reflection padding keeps the shifted
# tensor full-sized with no black borders, and the shift magnitude is capped
# well below image size so the true region is never pushed fully out of frame.
#
# Fully vectorized -- no per-sample Python loop, no .item() calls. The
# original implementation looped over B*C samples one at a time, calling
# F.pad + a CPU .item() sync per sample; that scales linearly with B*C and
# never benefits from a larger batch (each iteration is still one sample's
# work), which is what was making stage2_batch_size increases not help.
# This version pads the whole batch once, then gathers each sample's H x W
# crop via batched advanced indexing -- same reflect-then-crop semantics and
# same input/output shapes, but as a handful of GPU ops instead of B*C.
# ============================================================================
def apply_satellite_shift(candidate_satellites, shift_pixels):
    """
    candidate_satellites: [B, C, 3, H, W]
    Returns the same shape, each [3, H, W] slice independently shifted.
    """
    if shift_pixels <= 0:
        return candidate_satellites

    B, C, ch, H, W = candidate_satellites.shape
    device = candidate_satellites.device
    N = B * C
    flat = candidate_satellites.view(N, ch, H, W)

    # Pad the entire batch once (reflection, same as before).
    padded = F.pad(flat, (shift_pixels, shift_pixels, shift_pixels, shift_pixels), mode="reflect")
    # padded: [N, ch, H + 2*shift_pixels, W + 2*shift_pixels]

    # Per-sample random shift, same range as before: randint(-shift_pixels, shift_pixels + 1)
    dx = torch.randint(-shift_pixels, shift_pixels + 1, (N,), device=device)
    dy = torch.randint(-shift_pixels, shift_pixels + 1, (N,), device=device)

    # Per-sample crop top-left corner, same formula as the original loop:
    # x0 = shift_pixels - dx, y0 = shift_pixels - dy
    x0 = shift_pixels - dx   # [N]
    y0 = shift_pixels - dy   # [N]

    # Build per-sample row/col index grids and gather via advanced indexing,
    # instead of slicing one sample at a time.
    row_idx = y0.view(N, 1, 1) + torch.arange(H, device=device).view(1, H, 1)   # [N, H, 1]
    col_idx = x0.view(N, 1, 1) + torch.arange(W, device=device).view(1, 1, W)   # [N, 1, W]
    row_idx = row_idx.expand(N, H, W)   # [N, H, W]
    col_idx = col_idx.expand(N, H, W)   # [N, H, W]

    batch_idx = torch.arange(N, device=device).view(N, 1, 1).expand(N, H, W)   # [N, H, W]

    # Index into padded: [N, ch, Hp, Wp] -> gather [N, H, W] per channel.
    shifted = padded[batch_idx.unsqueeze(1).expand(N, ch, H, W),
                      torch.arange(ch, device=device).view(1, ch, 1, 1).expand(N, ch, H, W),
                      row_idx.unsqueeze(1).expand(N, ch, H, W),
                      col_idx.unsqueeze(1).expand(N, ch, H, W)]

    return shifted.view(B, C, ch, H, W)


# ============================================================================
# Optimizer / scheduler -- trainable parameters ONLY (Stage 1)
# ============================================================================
def build_optimizer(cfg, model):
    opt_cfg = cfg.training.optimizer
    params = [p for p in model.trainable_modules.parameters() if p.requires_grad]

    if opt_cfg.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay, betas=tuple(opt_cfg.betas)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

    stage1_epochs = cfg.training.stage_schedule.stage1_epochs
    warmup_epochs = getattr(cfg.training.scheduler, "warmup_epochs", 0)

    # Stage 1-only cosine schedule (with warmup), sized off stage1_epochs.
    # Stage 2 always gets its own freshly-built optimizer/scheduler (see
    # build_stage2_optimizer below) -- this schedule never runs past
    # stage1_epochs in the current training loop.
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)

        progress = (epoch - warmup_epochs) / max(1, stage1_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ============================================================================
# Stage 1 -> Stage 2 warm start
# ----------------------------------------------------------------------------
# Thin wrapper around model.initialize_stage2_from_stage1(), gated on
# cfg.training.stage2_init_from_stage1 (defaults to True). Called from both
# transition paths in run_training()/train() -- the natural Stage 1 -> Stage 2
# handoff within one run, and the --resume-from-stage1 path -- so the
# warm-start behavior (and its on/off config knob) stays in one place
# instead of being duplicated at each call site.
# ============================================================================
def maybe_warm_start_stage2(cfg, model):
    if getattr(cfg.training, "stage2_init_from_stage1", True):
        model.initialize_stage2_from_stage1()
        log_line(cfg, "Warm-started Stage 2 query pathway (stage2_proj_ground/stage2_proj_text/"
                      "stage2_query_qformer) from Stage 1 weights.")
    else:
        log_line(cfg, "Skipping Stage 2 warm start (stage2_init_from_stage1=False) -- "
                      "Stage 2 query pathway keeps its random initialization.")


# ============================================================================
# Stage 2 optimizer rebuild
# ----------------------------------------------------------------------------
# Called once: either right after Stage 1 finishes naturally, or by
# run_training() before resuming via --resume-from-stage1. Builds a fresh
# optimizer over ONLY the parameters still trainable after
# freeze_stage1_for_stage2() -- stage2_proj_ground, stage2_proj_text,
# stage2_query_qformer, stage2_proj_satellite, stage2_region_voting_qformer.
# Stage 1's optimizer state is intentionally discarded, not migrated, since
# none of the frozen Stage 1 params are being optimized anymore.
#
# IMPORTANT: call model.freeze_stage1_for_stage2() BEFORE this function --
# it filters trainable_params on requires_grad, so the freeze must already
# be in effect or this will build an optimizer over the wrong (still
# unfrozen) parameter set.
#
# The cosine schedule is sized off stage2_epoch_count = total_epochs -
# stage1_epochs (the FIXED Stage 2 length per the config), not off whatever
# global epoch Stage 1 actually ended on -- so Stage 2 always gets a full
# cosine decay over its own fixed duration, regardless of resume point.
# ============================================================================
def build_stage2_optimizer(cfg, model):
    opt_cfg = cfg.training.optimizer
    trainable_params = [p for p in model.trainable_modules.parameters() if p.requires_grad]

    if opt_cfg.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.training.stage2_lr, weight_decay=opt_cfg.weight_decay, betas=tuple(opt_cfg.betas)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

    stage1_epochs = cfg.training.stage_schedule.stage1_epochs
    total_epochs = cfg.training.stage_schedule.total_epochs
    stage2_epoch_count = max(1, total_epochs - stage1_epochs)

    def lr_lambda(stage2_local_epoch):
        progress = stage2_local_epoch / stage2_epoch_count
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ============================================================================
# Logging helpers
# ============================================================================
def log_line(cfg, msg):
    print(msg)
    log_path = os.path.join(cfg.logging.log_dir, f"{cfg.experiment.exp_id}.log")
    with open(log_path, "a") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------------------------
# Trainable-parameter summary -- factored out of print_startup_summary so it
# can also be called again right after freeze_stage1_for_stage2(), where the
# trainable set shrinks. Filters on requires_grad (see the matching fix in
# GeoDinoSiglipQFormer.get_trainable_parameter_names()), so calling this
# again post-freeze prints only the names/count still actually trainable
# -- after the Stage 2 refactor, that's stage2_proj_ground.*, stage2_proj_text.*,
# stage2_query_qformer.*, stage2_proj_satellite.*, stage2_region_voting_qformer.*
# -- instead of repeating the original Stage 1 list unchanged.
# ----------------------------------------------------------------------------
def print_trainable_summary(cfg, model, header=None):
    if header:
        log_line(cfg, f"\n{header}")
    log_line(cfg, f"Trainable parameter count: {model.count_trainable_parameters():,}")
    log_line(cfg, "Trainable parameter names:")
    for name in model.get_trainable_parameter_names():
        log_line(cfg, f"  {name}")


def print_startup_summary(model, cfg):
    log_line(cfg, "=" * 70)
    log_line(cfg, f"Experiment: {cfg.experiment.exp_id}")
    log_line(cfg, f"Mid-training evaluation disabled: {cfg.evaluation.disable_mid_training_eval}")
    log_line(cfg, f"Save only trainable weights: {cfg.checkpointing.save_only_trainable_weights}")
    log_line(cfg, "-" * 70)
    log_line(cfg, "Frozen encoders (requires_grad=False, never checkpointed):")
    log_line(cfg, f"  DINOv3 ground encoder      : {sum(p.numel() for p in model.dino_ground.parameters()):,} params")
    log_line(cfg, f"  DINOv3 satellite encoder   : {sum(p.numel() for p in model.dino_satellite.parameters()):,} params")
    log_line(cfg, f"  Text encoder ({cfg.model.text_encoder_type})    : {sum(p.numel() for p in model.text_model.parameters()):,} params")
    log_line(cfg, "-" * 70)
    print_trainable_summary(cfg, model)
    log_line(cfg, "=" * 70)


# ============================================================================
# Checkpointing -- trainable Q-Former weights only
# ----------------------------------------------------------------------------
# Filename includes a _stage2 suffix once checkpointing during Stage 2, so
# Stage 1 and Stage 2 checkpoints never collide and are easy to tell apart
# in a directory listing. epoch is always the GLOBAL epoch number (continued
# count across both stages), not a per-stage local counter.
# ============================================================================
def save_checkpoint(cfg, model, optimizer, scheduler, epoch, stage, loss_history):
    ckpt_dir = cfg.checkpointing.save_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    suffix = "_stage2" if stage == 2 else ""
    ckpt_path = os.path.join(ckpt_dir, f"{cfg.experiment.exp_id}_epoch{epoch+1}{suffix}.pt")
    torch.save({
        "epoch": epoch,
        "stage": stage,
        "model_trainable_state_dict": model.get_trainable_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_history": loss_history,
        "config": cfg.to_dict(),
    }, ckpt_path)

    if cfg.checkpointing.save_config_copy:
        save_config_copy(cfg, os.path.join(ckpt_dir, f"{cfg.experiment.exp_id}_config.json"))

    # # keep only the last N checkpoints
    # keep_last_n = cfg.checkpointing.keep_last_n
    # existing = sorted(
    #     [f for f in os.listdir(ckpt_dir) if f.startswith(cfg.experiment.exp_id) and f.endswith(".pt")],
    #     key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
    # )
    # for stale in existing[:-keep_last_n]:
    #     os.remove(os.path.join(ckpt_dir, stale))

    return ckpt_path


def load_checkpoint_for_resume(cfg, model, optimizer, scheduler, ckpt_path):
    """
    Reloads a checkpoint by:
      1. (DINOv3 / SigLIP2 are already loaded fresh from cfg.model.* checkpoint
         names inside GeoDinoSiglipQFormer.__init__ -- nothing to do here for them)
      2. restoring only the trainable Q-Former-related weights
      3. restoring optimizer / scheduler state

    NOTE: this generic resume path is for resuming WITHIN the same stage
    (e.g. a Stage 1 run interrupted and restarted before reaching Stage 2).
    For resuming a Stage 1 checkpoint directly INTO Stage 2, use
    run_training(..., resume_from_stage1=...) / --resume-from-stage1
    instead -- that path builds a fresh Stage 2 optimizer/scheduler and
    does not call this function.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_trainable_state_dict(ckpt["model_trainable_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["stage"], ckpt["loss_history"]


# ============================================================================
# Training loop
# ----------------------------------------------------------------------------
# Stage 1 runs as a normal global-epoch loop, started at start_epoch (0 for
# a fresh run).
#
# Stage 2 ALWAYS runs for exactly (total_epochs - stage1_epochs) epochs,
# counted on its own local counter -- regardless of what global epoch
# number Stage 1 actually stopped on (e.g. if Stage 1 was resumed and ran
# past its configured stage1_epochs). Global epoch numbers used for
# logging/checkpoint filenames during Stage 2 are derived as
# stage2_start_global_epoch + stage2_local_epoch, where
# stage2_start_global_epoch is whatever global epoch Stage 2 actually
# begins at in THIS run (the next epoch after Stage 1 ends, or
# checkpoint_epoch + 1 when resuming via --resume-from-stage1).
#
# start_epoch: global epoch to begin at (0 for a fresh run; checkpoint_epoch+1
#              when resuming via --resume-from-stage1).
# loss_history: pre-existing loss history to append to (empty list for a
#              fresh run; the resumed checkpoint's history otherwise).
# resume_into_stage2: True when the caller (run_training, via
#              --resume-from-stage1) has already warm-started + frozen
#              Stage 1, built the Stage 2 optimizer/scheduler, and is
#              handing off directly into Stage 2 -- in that case this loop
#              skips Stage 1 entirely and runs only the fixed-length
#              Stage 2 block below. `optimizer`/`scheduler` passed in are
#              then the STAGE 2 ones already.
# resume_from_stage1_path: the --resume-from-stage1 checkpoint path, when
#              resume_into_stage2 is True (None otherwise). Used only to
#              name/locate the mined hard-negative-candidates cache for
#              THIS checkpoint's weights -- see run_stage2_epochs below.
# ============================================================================
def train(cfg, model, train_loader, optimizer, scheduler, device,
          start_epoch=0, loss_history=None, resume_into_stage2=False, resume_from_stage1_path=None):
    stage1_loss_fn = Stage1ContrastiveLoss(temperature=cfg.loss.stage1_temperature)

    rv_cfg = cfg.training.stage2_region_voting
    region_voting_loss_fn = RegionVotingLoss(
        temperature=cfg.loss.stage2_temperature,
        positive_entropy_weight=rv_cfg.positive_entropy_weight,
        negative_entropy_weight=rv_cfg.negative_entropy_weight,
        negative_peak_weight=rv_cfg.negative_peak_weight,
    )

    norm_tensors = build_norm_tensors(cfg, device)

    stage1_epochs = cfg.training.stage_schedule.stage1_epochs
    total_epochs = cfg.training.stage_schedule.total_epochs
    stage2_epoch_count = total_epochs - stage1_epochs   # ALWAYS fixed, per spec

    use_amp = cfg.training.use_mixed_precision and device.type == "cuda"
    # bfloat16, not float16: T5 (the default text encoder) is well known to
    # produce inf/NaN under float16 autocast -- its activations (particularly
    # around the relative position bias added before softmax, and inside
    # T5LayerNorm's variance computation) routinely exceed fp16's ~65504 max.
    # bf16 has the same exponent range as fp32 (no overflow, just reduced
    # mantissa precision) and Ampere+ GPUs (e.g. the A6000) run it at the
    # same speed as fp16. GradScaler is an fp16-specific workaround for its
    # narrow range -- not needed here, so it's left disabled.
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    loss_history = loss_history if loss_history is not None else []
    time_stamp("Training start" if start_epoch == 0 else f"Training resumed at global epoch {start_epoch}")

    def run_stage1_epochs(from_epoch, to_epoch_exclusive):
        """Runs Stage 1 global epochs [from_epoch, to_epoch_exclusive)."""
        for epoch in range(from_epoch, to_epoch_exclusive):
            model.dino_ground.eval()
            model.dino_satellite.eval()
            model.text_model.eval()
            model.trainable_modules.train()

            log_line(cfg, f"\nEpoch {epoch+1} (Stage 1, {epoch+1}/{stage1_epochs}) -- global epoch {epoch+1}")

            running_loss = []
            for anchor, positive, negative, txt, idx in tqdm(train_loader):
                anchor, positive, negative = prepare_ground_satellite_tensors(
                    cfg, anchor, positive, negative, device, norm_tensors
                )

                optimizer.zero_grad()

                text_dropout_prob = getattr(cfg.training, "text_dropout_prob", 0.0)
                txt = apply_text_dropout(txt, text_dropout_prob)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    query_embedding, satellite_embedding, aux = model(
                        q=anchor, r=positive, t=txt, stage=1, isTrain=True, isQuery=True, return_aux=True
                    )
                    loss = stage1_loss_fn(query_embedding, satellite_embedding)

                scaler.scale(loss).backward()
                if cfg.training.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.trainable_modules.parameters(), cfg.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                running_loss.append(loss.item())

            scheduler.step()
            mean_loss = float(np.mean(running_loss))
            loss_history.append(mean_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            log_line(cfg, f"Epoch {epoch+1}/{total_epochs} Stage 1 Loss: {mean_loss:.4f} LR: {current_lr:.2e}")

            if (epoch + 1) % cfg.checkpointing.save_every_n_epochs == 0 or (epoch + 1) == stage1_epochs:
                ckpt_path = save_checkpoint(cfg, model, optimizer, scheduler, epoch, stage=1, loss_history=loss_history)
                log_line(cfg, f"Saved trainable-weights checkpoint -> {ckpt_path}")

            # ------------------------------------------------------------
            # No mid-training evaluation by design (cfg.evaluation.disable_mid_training_eval).
            # Run a separate evaluation script after training finishes instead.
            # ------------------------------------------------------------

    def run_stage2_epochs(stage2_optimizer, stage2_scheduler, global_epoch_offset):
        """
        Runs Stage 2 for EXACTLY stage2_epoch_count epochs (local counter
        0..stage2_epoch_count-1), regardless of what global epoch Stage 1
        actually ended on. global_epoch_offset is the global epoch number
        of the FIRST Stage 2 epoch in this run (e.g. 20 if Stage 1 finished
        at epoch 19, or checkpoint_epoch+1 when resuming).

        Mined hard-negative candidates are cached to disk in a folder named
        after whichever Stage 1 checkpoint produced the current model
        weights, e.g. "<checkpoint_path>_mined_candidates/hard_negative_idx.pt"
        -- derived here, not passed in, from resume_into_stage2 /
        resume_from_stage1_path (both in train()'s enclosing scope):
          - resuming via --resume-from-stage1: that checkpoint's own path.
          - natural Stage 1 -> Stage 2 transition in this same run: the
            last Stage 1 checkpoint's path, i.e. the file save_checkpoint()
            wrote at epoch stage1_epochs.
        If that folder already exists, mining is skipped entirely and the
        cached indices are loaded instead -- this is what lets a crashed
        Stage 2 run resume without repeating the (slow) mining pass, as
        long as the underlying Stage 1 weights haven't changed.
        """
        if resume_into_stage2 and resume_from_stage1_path is not None:
            source_checkpoint = resume_from_stage1_path
        else:
            source_checkpoint = os.path.join(
                cfg.checkpointing.save_dir, f"{cfg.experiment.exp_id}_epoch{stage1_epochs}.pt"
            )

        cache_dir = source_checkpoint + "_mined_candidates"
        cache_path = os.path.join(cache_dir, "hard_negative_idx.pt")

        base_dataset = train_loader.dataset

        if os.path.exists(cache_path):
            log_line(cfg, f"\nFound existing mined candidates -> {cache_path} -- skipping mining.")
            hard_negative_idx = torch.load(cache_path, map_location="cpu")
        else:
            log_line(cfg, f"\nMining hard candidates (pool_k={rv_cfg.hard_candidate_pool_k})...")
            hard_negative_idx = build_hard_candidate_cache(cfg, model, device, rv_cfg.hard_candidate_pool_k)
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(hard_negative_idx, cache_path)
            log_line(cfg, f"Saved mined candidates -> {cache_path}")

        stage2_loader = build_stage2_dataloader(cfg, base_dataset, hard_negative_idx)

        g_mean, g_std, s_mean, s_std = norm_tensors
        d = cfg.data

        for stage2_local_epoch in range(stage2_epoch_count):
            global_epoch = global_epoch_offset + stage2_local_epoch

            model.dino_ground.eval()
            model.dino_satellite.eval()
            model.text_model.eval()
            # Stage 2: do NOT call model.trainable_modules.train() -- that
            # recurses into every submodule, including the frozen Stage 1
            # ones, and would silently re-enable their dropout even though
            # freeze_stage1_for_stage2() already set them to .eval(). Put
            # only the still-trainable Stage 2 submodules into .train();
            # explicitly re-assert .eval() on the frozen ones so a stray
            # earlier .train() call elsewhere can't leave them in the
            # wrong mode going into this epoch.
            for name in ("stage2_proj_ground", "stage2_proj_text", "stage2_query_qformer",
                         "stage2_proj_satellite", "stage2_region_voting_qformer"):
                model.trainable_modules[name].train()
            for name in ("proj_ground", "proj_text", "query_qformer",
                         "satellite_global_head", "retrieval_proj_query", "retrieval_proj_satellite"):
                model.trainable_modules[name].eval()

            log_line(
                cfg,
                f"\nEpoch {global_epoch+1} (Stage 2, {stage2_local_epoch+1}/{stage2_epoch_count}) "
                f"-- global epoch {global_epoch+1}"
            )

            running_loss = []
            running_ce = []
            running_pos_entropy = []
            running_neg_entropy = []
            running_neg_peak = []

            # Stage 2 diagnostics (pos/neg score gap), accumulated per epoch
            # alongside the existing loss-component running lists above.
            running_pos_score_mean = []
            running_neg_score_mean = []
            running_max_neg_score_mean = []
            running_margin = []
            running_train_acc = []

            for anchor, txt, candidate_satellites, target, idx in tqdm(stage2_loader):
                anchor = anchor.to(device, non_blocking=True)
                candidate_satellites = candidate_satellites.to(device, non_blocking=True)   # [B, C, 3, H, W]
                target = torch.zeros(anchor.shape[0], dtype=torch.long, device=device)

                # ground: resize + normalize (same as Stage 1)
                if anchor.shape[-1] != d.image_size_ground or anchor.shape[-2] != d.image_size_ground:
                    anchor = F.interpolate(anchor, size=(d.image_size_ground, d.image_size_ground),
                                            mode="bilinear", align_corners=False)
                anchor = (anchor - g_mean) / g_std

                # candidate satellites: shift augmentation, then normalize
                candidate_satellites = apply_satellite_shift(
                    candidate_satellites, rv_cfg.satellite_shift_pixels
                )
                B, C, ch, H, W = candidate_satellites.shape
                candidate_satellites = (candidate_satellites - s_mean) / s_std

                text_dropout_prob = getattr(cfg.training, "text_dropout_prob", 0.0)
                txt = apply_text_dropout(txt, text_dropout_prob)

                stage2_optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    # query_global (1st output) is the frozen Stage 1 embedding --
                    # unused here, Stage 2 training only needs the trainable
                    # Stage 2 query tokens (2nd output).
                    _, query_tokens = model.encode_query_stage2(anchor, txt)

                    candidate_flat = candidate_satellites.view(B * C, ch, H, W)
                    _, sat_patches_flat = model.encode_satellite_stage2(candidate_flat)
                    sat_patches = sat_patches_flat.view(B, C, *sat_patches_flat.shape[1:])

                    candidate_scores, vote_maps, region_maps = model.score_stage2_candidates(
                        query_tokens=query_tokens,
                        candidate_sat_patches=sat_patches,
                        return_maps=True,
                    )

                    loss_dict = region_voting_loss_fn(
                        candidate_scores=candidate_scores,
                        region_maps=region_maps,
                        target=target,
                    )
                    loss = loss_dict["loss"]

                scaler.scale(loss).backward()
                if cfg.training.grad_clip_norm:
                    scaler.unscale_(stage2_optimizer)
                    trainable_now = [p for p in model.trainable_modules.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_now, cfg.training.grad_clip_norm)
                scaler.step(stage2_optimizer)
                scaler.update()

                running_loss.append(loss.item())
                running_ce.append(loss_dict["ce"].item())
                running_pos_entropy.append(loss_dict["pos_entropy"].item())
                running_neg_entropy.append(loss_dict["neg_entropy"].item())
                running_neg_peak.append(loss_dict["neg_peak"].item())

                # ----------------------------------------------------------------
                # Stage 2 diagnostics: candidate index 0 is always the positive
                # (see Stage2CandidateDataset.__getitem__ -- positive is always
                # prepended before sampled hard negatives), so candidate_scores[:, 0]
                # is the positive score and candidate_scores[:, 1:] are negatives.
                # ----------------------------------------------------------------
                with torch.no_grad():
                    pos_score = candidate_scores[:, 0]
                    neg_scores = candidate_scores[:, 1:]
                    max_neg_score, _ = neg_scores.max(dim=1)
                    margin = pos_score - max_neg_score
                    train_acc = (pos_score > max_neg_score).float()

                running_pos_score_mean.append(pos_score.mean().item())
                running_neg_score_mean.append(neg_scores.mean().item())
                running_max_neg_score_mean.append(max_neg_score.mean().item())
                running_margin.append(margin.mean().item())
                running_train_acc.append(train_acc.mean().item())

            stage2_scheduler.step()
            mean_loss = float(np.mean(running_loss))
            loss_history.append(mean_loss)

            current_lr = stage2_optimizer.param_groups[0]["lr"]
            log_line(
                cfg,
                f"Epoch {global_epoch+1} Stage 2 ({stage2_local_epoch+1}/{stage2_epoch_count}) "
                f"Loss: {mean_loss:.4f} "
                f"CE: {np.mean(running_ce):.4f} "
                f"PosEntropy: {np.mean(running_pos_entropy):.4f} "
                f"NegEntropy: {np.mean(running_neg_entropy):.4f} "
                f"NegPeak: {np.mean(running_neg_peak):.4f} "
                f"LR: {current_lr:.2e}"
            )
            log_line(
                cfg,
                f"Epoch {global_epoch+1} Stage 2 diagnostics: "
                f"pos_score_mean: {np.mean(running_pos_score_mean):.4f} "
                f"neg_score_mean: {np.mean(running_neg_score_mean):.4f} "
                f"max_neg_score_mean: {np.mean(running_max_neg_score_mean):.4f} "
                f"margin: {np.mean(running_margin):.4f} "
                f"train_acc: {np.mean(running_train_acc):.4f}"
            )

            if (stage2_local_epoch + 1) % cfg.checkpointing.save_every_n_epochs == 0 or \
               (stage2_local_epoch + 1) == stage2_epoch_count:
                ckpt_path = save_checkpoint(
                    cfg, model, stage2_optimizer, stage2_scheduler, global_epoch, stage=2, loss_history=loss_history
                )
                log_line(cfg, f"Saved trainable-weights checkpoint -> {ckpt_path}")

            # ------------------------------------------------------------
            # No mid-training evaluation by design (cfg.evaluation.disable_mid_training_eval).
            # Run a separate evaluation script after training finishes instead.
            # ------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dispatch: either resume straight into Stage 2 (skip Stage 1 entirely
    # in this process), or run Stage 1 (from start_epoch) then Stage 2.
    # ------------------------------------------------------------------
    if resume_into_stage2:
        run_stage2_epochs(optimizer, scheduler, global_epoch_offset=start_epoch)
    else:
        run_stage1_epochs(start_epoch, stage1_epochs)
        maybe_warm_start_stage2(cfg, model)
        model.freeze_stage1_for_stage2()
        model.unfreeze_stage2_for_stage2()
        print_trainable_summary(cfg, model, header="After freeze_stage1_for_stage2():")
        stage2_optimizer, stage2_scheduler = build_stage2_optimizer(cfg, model)
        run_stage2_epochs(stage2_optimizer, stage2_scheduler, global_epoch_offset=stage1_epochs)

    time_stamp("Training complete")
    return loss_history


# ============================================================================
# Entry point
# ----------------------------------------------------------------------------
# run_training() is the reusable entry point -- main.py imports and calls
# this directly when invoked in "train" mode. This file is still runnable
# standalone too: python train.py --config ...
#
# resume_from_stage1: optional path to a Stage 1 checkpoint. When given,
# Stage 1 is skipped entirely -- the model's trainable weights are loaded
# from that checkpoint (strict=False, since an old Stage 1 checkpoint has
# no stage2_* keys yet -- the missing keys are expected and logged below),
# Stage 2's query pathway is warm-started from the loaded Stage 1 weights,
# Stage 1 is frozen, a fresh Stage 2 optimizer/scheduler is built (the
# Stage 1 optimizer state in the checkpoint is intentionally NOT loaded,
# since it covers a different, now-frozen parameter set), and training
# continues from checkpoint_epoch + 1 using the GLOBAL epoch counter (so a
# checkpoint saved at epoch 19 resumes at epoch 20, not epoch 0). Stage 2
# then runs for exactly (total_epochs - stage1_epochs) epochs regardless of
# what epoch the checkpoint came from.
# ============================================================================
def run_training(
    config_path: str = "configs/geo_dino_t5_qformer.json",
    resume_from_stage1: str = None,
    gpu_id: int = None,
):
    cfg = load_config(config_path)
    assert cfg.evaluation.disable_mid_training_eval, "this training script does not support mid-training eval"

    if gpu_id is not None:
        if gpu_id < 0:
            raise ValueError("gpu_id must be a non-negative integer")
        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU {gpu_id} was requested, but CUDA is not available")
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu_id} was requested, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(cfg.experiment.seed)

    model = GeoDinoSiglipQFormer(cfg).to(device)
    # Stage 2 modules default to requires_grad=True at construction (same as
    # everything else in trainable_modules), but Stage 1 never forwards
    # through them. Freeze them up front so Stage 1's optimizer param group,
    # trainable-parameter count, and logged trainable-parameter names only
    # ever reflect the Stage 1 modules actually being trained. Reversed by
    # model.unfreeze_stage2_for_stage2() at the Stage 1 -> Stage 2 transition
    # below (both the natural in-run transition and --resume-from-stage1).
    model.freeze_stage2_until_stage2()
    print_startup_summary(model, cfg)

    train_loader = build_dataloader(cfg)

    if resume_from_stage1 is not None:
        log_line(cfg, f"\nResuming from Stage 1 checkpoint: {resume_from_stage1}")
        ckpt = torch.load(resume_from_stage1, map_location="cpu")

        if ckpt.get("stage") != 1:
            raise ValueError(
                f"--resume-from-stage1 expects a Stage 1 checkpoint, but {resume_from_stage1} "
                f"was saved with stage={ckpt.get('stage')!r}. Resuming a Stage 2 checkpoint into "
                f"Stage 2 is not supported by this flag."
            )

        # strict=False: an old Stage 1 checkpoint predates the stage2_* modules,
        # so it will be missing those keys -- that's expected, not an error.
        # Anything unexpected (a key in the checkpoint with no matching module
        # on this model) is logged too, in case the checkpoint is stale/mismatched.
        load_result = model.load_trainable_state_dict(ckpt["model_trainable_state_dict"], strict=False)
        log_line(cfg, f"Loaded trainable weights from checkpoint epoch {ckpt['epoch']} (stage {ckpt['stage']}).")
        log_line(cfg, f"  Missing keys (expected: new Stage 2 query/satellite modules): {load_result.missing_keys}")
        log_line(cfg, f"  Unexpected keys: {load_result.unexpected_keys}")

        maybe_warm_start_stage2(cfg, model)

        model.freeze_stage1_for_stage2()
        model.unfreeze_stage2_for_stage2()
        print_trainable_summary(cfg, model, header="After freeze_stage1_for_stage2():")
        stage2_optimizer, stage2_scheduler = build_stage2_optimizer(cfg, model)

        start_epoch = ckpt["epoch"] + 1   # global epoch -- first Stage 2 epoch number for THIS run
        loss_history = list(ckpt.get("loss_history", []))

        loss_history = train(
            cfg, model, train_loader, stage2_optimizer, stage2_scheduler, device,
            start_epoch=start_epoch,
            loss_history=loss_history,
            resume_into_stage2=True,
            resume_from_stage1_path=resume_from_stage1,
        )
    else:
        optimizer, scheduler = build_optimizer(cfg, model)
        loss_history = train(cfg, model, train_loader, optimizer, scheduler, device)

    return model, loss_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/geo_dino_t5_qformer.json")
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=None,
                        help="CUDA device index, e.g. 0 or 1")
    parser.add_argument(
        "--resume-from-stage1", type=str, default=None,
        help="Path to a Stage 1 checkpoint. Skips Stage 1 entirely, warm-starts + freezes it, and "
             "runs Stage 2 for exactly (total_epochs - stage1_epochs) epochs, with global epoch "
             "numbering continuing from checkpoint_epoch + 1."
    )
    args = parser.parse_args()
    run_training(args.config, resume_from_stage1=args.resume_from_stage1, gpu_id=args.gpu_id)


if __name__ == "__main__":
    main()
