import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from configs.config_loader import load_config
from models.custom_model import GeoDinoSiglipQFormer
from train import build_norm_tensors

from datasets.CVUSA_dataset import CVUSA_dataset_cropped
from datasets.CVACT_dataset import CVACT_dataset_cropped
from datasets.VIGOR_dataset import VIGOR_dataset_cropped
from datasets.GAMa_dataset import GAMa_dataset_cropped


# ============================================================================
# Model loading -- reload frozen DINOv3 / SigLIP2 fresh from the config's
# checkpoint names (inside GeoDinoSiglipQFormer.__init__), then restore ONLY
# the trainable Q-Former-related weights from the checkpoint file. The
# checkpoint never contains frozen-encoder weights, so this is the only
# valid way to reconstruct a working model from one.
# ============================================================================
def load_eval_model(cfg, checkpoint_path, device):
    model = GeoDinoSiglipQFormer(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_trainable_state_dict(ckpt["model_trainable_state_dict"])

    model.eval()
    print(
        f"Loaded trainable weights from {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', '?')}, stage {ckpt.get('stage', '?')})"
    )
    return model


# ============================================================================
# Eval dataloader -- same shared-transform approach as training (resize to
# satellite resolution, ToTensor only; ground is downsampled and both
# branches normalized with their own stats below), val split, no shuffling
# so row order stays a stable index into Q_*/S_* (row i's query is assumed
# to match row i's satellite).
# ============================================================================
def build_eval_dataloader(cfg, eval_batch_size):
    d = cfg.data

    base_size = d.image_size_satellite
    base_transform = transforms.Compose([
        transforms.Resize((base_size, base_size)),
        transforms.ToTensor(),
    ])

    if d.dataset_name == "CVUSA":
        val_data = pd.read_csv(f"{d.data_root}/{d.val_csv}", header=None)
        dataset = CVUSA_dataset_cropped(df=val_data, path=d.data_root, transform=base_transform, train=False, lang=d.lang)
    elif d.dataset_name == "CVACT":
        val_data = pd.read_csv(f"{d.data_root}/{d.val_csv}")
        dataset = CVACT_dataset_cropped(df=val_data, path=d.data_root, transform=base_transform, train=False, lang=d.lang)
    elif d.dataset_name == "VIGOR":
        val_data = pd.read_csv(f"{d.data_root}/{d.val_csv}")
        dataset = VIGOR_dataset_cropped(df=val_data, path=d.data_root, transform=base_transform, train=False, lang=d.lang)
    elif d.dataset_name == "GAMa":
        val_data = pd.read_csv(f"{d.data_root}/{d.val_csv}")
        dataset = GAMa_dataset_cropped(df=val_data, path=d.data_root, transform=base_transform, train=False, lang=d.lang)
    else:
        raise ValueError(f"Unknown dataset_name: {d.dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,  # order must stay stable -- it's the index used for ground-truth matching below
        num_workers=d.num_workers,
    )
    return loader


def _resize_normalize_ground(anchor, device, d, g_mean, g_std):
    anchor = anchor.to(device, non_blocking=True)
    if anchor.shape[-1] != d.image_size_ground or anchor.shape[-2] != d.image_size_ground:
        anchor = torch.nn.functional.interpolate(
            anchor, size=(d.image_size_ground, d.image_size_ground), mode="bilinear", align_corners=False
        )
    return (anchor - g_mean) / g_std


def _normalize_satellite(positive, device, s_mean, s_std):
    positive = positive.to(device, non_blocking=True)
    return (positive - s_mean) / s_std


# ============================================================================
# Step 1: encode every query and every satellite ONCE
# ----------------------------------------------------------------------------
# Q_global / S_global: Stage 1-style pooled embeddings, used for the cheap
#   global retrieval pass (Step 2).
# Q_tokens / S_patches: raw normalized query tokens and satellite patch tokens,
#   used by the Satellite Region Voting Former during reranking (Step 3).
#
# Uses encode_query_stage2() / encode_satellite_stage2() so this matches
# exactly what training calls -- the same frozen Stage 1 query tokens and
# the same proj_satellite-projected patch tokens feed the Stage 2 scorer
# in both places.
# ============================================================================
@torch.no_grad()
def encode_all(cfg, model, dataloader, device, norm_tensors):
    d = cfg.data
    g_mean, g_std, s_mean, s_std = norm_tensors

    Q_global, Q_tokens = [], []
    S_global, S_patches = [], []

    for anchor, positive, negative, txt, idx in tqdm(dataloader, desc="Encoding queries + satellites"):
        anchor = _resize_normalize_ground(anchor, device, d, g_mean, g_std)
        positive = _normalize_satellite(positive, device, s_mean, s_std)

        q_global, q_tokens = model.encode_query_stage2(anchor, txt)
        s_global, s_patches = model.encode_satellite_stage2(positive)

        Q_global.append(q_global.cpu())
        Q_tokens.append(q_tokens.cpu())
        S_global.append(s_global.cpu())
        S_patches.append(s_patches.cpu())

    Q_global = torch.cat(Q_global, dim=0)      # [N, D]
    Q_tokens = torch.cat(Q_tokens, dim=0)       # [N, Nq, D], frozen Stage 1 query tokens
    S_global = torch.cat(S_global, dim=0)       # [N, D]
    S_patches = torch.cat(S_patches, dim=0)     # [N, Np, D], proj_satellite-projected patch tokens
    return Q_global, Q_tokens, S_global, S_patches


# ============================================================================
# Step 2: cheap global retrieval -- top-K candidates per query
# ----------------------------------------------------------------------------
# Also reports Global R@1/5/10 directly (diagonal = ground truth, row i's
# query matches row i's satellite) and Global top-K recall, since reranking
# in Step 3 can only ever help within whatever Step 2 already retrieved.
# ============================================================================
def global_retrieve(Q_global, S_global, k, device):
    N = Q_global.shape[0]
    global_scores = (Q_global.to(device) @ S_global.to(device).t()).cpu()   # [N, N]

    topk_scores, topk_idx = torch.topk(global_scores, k=min(k, N), dim=1)    # [N, k]

    true_idx = torch.arange(N)
    rank_of_true = (global_scores > global_scores[true_idx, true_idx].unsqueeze(1)).sum(dim=1)  # [N]

    global_recall = {
        "global_R@1": (rank_of_true < 1).float().mean().item() * 100.0,
        "global_R@5": (rank_of_true < 5).float().mean().item() * 100.0,
        "global_R@10": (rank_of_true < 10).float().mean().item() * 100.0,
        f"global_top{k}_recall": (rank_of_true < k).float().mean().item() * 100.0,
    }
    return topk_idx, topk_scores, global_recall


# ============================================================================
# Step 3: Satellite Region Voting rerank -- only within each query's top-K
# ----------------------------------------------------------------------------
# Calls model.score_stage2_candidates() -- the same Stage 2 scorer used
# during training -- on chunks of each query's top-K candidates, so eval
# uses the identical region-voting mechanism end to end. Candidates are
# processed in chunks to avoid putting [batch, topK, Np, D] all on GPU at once.
# ============================================================================
def row_zscore(x, eps=1e-6):
    """
    Normalize scores independently per query row.

    This is important because global dot-product scores and region-voting
    scores live on different numeric scales. Per-row z-scoring lets
    fusion_lambda control how much the region score perturbs the global
    ranking for each query.
    """
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)


@torch.no_grad()
def region_voting_rerank(
    model,
    Q_tokens,
    S_patches,
    topk_idx,
    topk_scores,
    device,
    gather_batch_size=16,
    candidate_chunk_size=100,
    fusion_lambda=0.3,
):
    """
    Rerank Stage-1 top-K candidates using a fused score:

        combined_score = zscore(global_topk_score)
                         + fusion_lambda * zscore(region_voting_score)

    fusion_lambda=0.0 reproduces the original global top-K ordering.
    Larger values give more authority to the Stage-2 region-voting score.
    """
    N, K = topk_idx.shape
    reranked_idx = torch.empty_like(topk_idx)

    use_amp = device.type == "cuda"

    for start in tqdm(range(0, N, gather_batch_size), desc="Region voting reranking"):
        end = min(start + gather_batch_size, N)
        b = end - start

        batch_q_tokens = Q_tokens[start:end].to(device)     # [b, Nq, D]
        batch_topk_idx = topk_idx[start:end]                # [b, K]
        batch_topk_scores = topk_scores[start:end].float()  # [b, K], CPU

        region_score_chunks = []

        for c_start in range(0, K, candidate_chunk_size):
            c_end = min(c_start + candidate_chunk_size, K)
            k_chunk = c_end - c_start

            idx_chunk = batch_topk_idx[:, c_start:c_end]    # [b, k_chunk]

            candidate_patches = S_patches[idx_chunk.reshape(-1)].to(device)
            candidate_patches = candidate_patches.view(b, k_chunk, *S_patches.shape[1:])

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                candidate_scores, _, _ = model.score_stage2_candidates(
                    query_tokens=batch_q_tokens,
                    candidate_sat_patches=candidate_patches,
                    return_maps=False,
                )

            region_score_chunks.append(candidate_scores.float().cpu())

        region_scores = torch.cat(region_score_chunks, dim=1)  # [b, K]

        global_z = row_zscore(batch_topk_scores)
        region_z = row_zscore(region_scores)
        combined_scores = global_z + fusion_lambda * region_z

        order = torch.argsort(combined_scores, dim=1, descending=True)
        reranked_idx[start:end] = torch.gather(batch_topk_idx, 1, order)

    return reranked_idx


def rerank_recall(reranked_idx):
    N = reranked_idx.shape[0]
    true_idx = torch.arange(N).unsqueeze(1)
    match_position = (reranked_idx == true_idx).float().argmax(dim=1)
    found = (reranked_idx == true_idx).any(dim=1)

    # if the true match wasn't in the top-K candidates at all, it can never
    # be "found" by reranking -- treat its rank as infinite (never within R@k)
    match_position = torch.where(found, match_position, torch.full_like(match_position, N))

    return {
        "region_voting_R@1": (match_position < 1).float().mean().item() * 100.0,
        "region_voting_R@5": (match_position < 5).float().mean().item() * 100.0,
        "region_voting_R@10": (match_position < 10).float().mean().item() * 100.0,
    }


# ============================================================================
# Entry point
# ----------------------------------------------------------------------------
# run_eval() is the reusable entry point -- main.py imports and calls this
# directly when invoked in "eval" mode. `stage` must be passed explicitly:
#   stage=1 -> Global retrieval only (cheap, query-independent satellite
#              embeddings -- but see staleness caveat for long Stage-2 runs).
#   stage=2 -> Full retrieve-then-rerank pipeline described above, reporting
#              both Global and Region-Voting-rerank metrics.
# This file is still runnable standalone: python eval.py --config ... --checkpoint ... --stage 2
# ============================================================================
def run_eval(
    config_path: str = "configs/geo_dino_t5_qformer.json",
    checkpoint_path: str = None,
    stage: int = None,
    rerank_topk: int = 100,
    eval_batch_size: int = 64,
    gather_batch_size: int = 16,
    candidate_chunk_size: int = 100,
    fusion_lambda: float = 0.3,
):
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for evaluation")
    if stage not in (1, 2):
        raise ValueError("stage must be explicitly passed as 1 or 2")

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_eval_model(cfg, checkpoint_path, device)
    eval_loader = build_eval_dataloader(cfg, eval_batch_size)
    norm_tensors = build_norm_tensors(cfg, device)

    print("\nStep 1: encoding all queries and satellites once...")
    Q_global, Q_tokens, S_global, S_patches = encode_all(cfg, model, eval_loader, device, norm_tensors)

    print(f"\nStep 2: global retrieval (top-{rerank_topk} candidates per query)...")
    topk_idx, topk_scores, global_recall = global_retrieve(Q_global, S_global, rerank_topk, device)

    results = dict(global_recall)
    print("\nGlobal retrieval results:")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%")

    if stage == 2:
        print(
            f"\nStep 3: Fused global + region-voting reranking within top-{rerank_topk} "
            f"(fusion_lambda={fusion_lambda})..."
        )
        reranked_idx = region_voting_rerank(
            model,
            Q_tokens,
            S_patches,
            topk_idx,
            topk_scores,
            device,
            gather_batch_size=gather_batch_size,
            candidate_chunk_size=candidate_chunk_size,
            fusion_lambda=fusion_lambda,
        )
        region_voting_results = rerank_recall(reranked_idx)
        results.update(region_voting_results)

        print("\nFused global + region-voting rerank results:")
        for k, v in region_voting_results.items():
            print(f"  {k}: {v:.2f}%")

        print(
            f"\nNote: region voting reranking can only recover matches already inside the "
            f"global top-{rerank_topk} -- global_top{rerank_topk}_recall above is its ceiling."
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/geo_dino_t5_qformer.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--rerank-topk", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gather-batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=100)
    parser.add_argument(
        "--fusion-lambda", type=float, default=0.3,
        help="Weight on per-query z-scored region-voting scores in fused reranking. "
             "0.0 reproduces pure global ranking; 0.3 is a conservative default."
    )
    args = parser.parse_args()
    run_eval(
        args.config,
        args.checkpoint,
        args.stage,
        rerank_topk=args.rerank_topk,
        eval_batch_size=args.eval_batch_size,
        gather_batch_size=args.gather_batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
        fusion_lambda=args.fusion_lambda,
    )


if __name__ == "__main__":
    main()