"""Create a CVUSA qualitative retrieval figure for selected query IDs.

The output has one row per query: ground image followed by its top-5
retrieved satellite images. Correct retrievals have green borders and
incorrect retrievals have red borders.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from configs.config_loader import load_config
from eval import load_eval_model, row_zscore
from train import build_norm_tensors


DEFAULT_QUERY_IDS = [
    "0014696",
    "0042062",
    "0041907",
    "0032946",
    "0036212",
    "0007872",
    "0028715",
]
DEFAULT_CONFIG = "configs/geo_dino_t5_qformer_image_only.json"
DEFAULT_CHECKPOINT = (
    "weights/geo_dino_t5_qformer_image_only/"
    "geo_dino_t5_qformer_image_only_epoch40_stage2.pt"
)


def image_id(path):
    """Return a zero-padded filename stem such as 0014696."""
    return Path(str(path)).stem


def select_device(gpu_id):
    if gpu_id is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_id < 0:
        raise ValueError("--gpu-id must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU {gpu_id} was requested, but CUDA is unavailable")
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {gpu_id} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible"
        )
    return torch.device(f"cuda:{gpu_id}")


class SatelliteDataset(Dataset):
    def __init__(self, rows, data_root, transform, indices=None):
        self.rows = rows
        self.data_root = Path(data_root)
        self.transform = transform
        self.indices = list(range(len(rows))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row_index = self.indices[item]
        path = self.data_root / str(self.rows.iloc[row_index, 0])
        image = self.transform(Image.open(path).convert("RGB"))
        return image, row_index


class SelectedQueryDataset(Dataset):
    def __init__(self, rows, texts, row_indices, data_root, transform):
        self.rows = rows
        self.texts = texts
        self.row_indices = list(row_indices)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, item):
        row_index = self.row_indices[item]
        path = self.data_root / str(self.rows.iloc[row_index, 1])
        image = self.transform(Image.open(path).convert("RGB"))
        # The image-only model discards text tokens, but its current API still
        # accepts a text argument, so pass the row's real caption.
        return image, str(self.texts.iloc[row_index]), row_index


def find_query_rows(rows, query_ids):
    sat_id_to_row = {image_id(path): i for i, path in enumerate(rows.iloc[:, 0])}
    ground_id_to_row = {image_id(path): i for i, path in enumerate(rows.iloc[:, 1])}
    selected = []
    for query_id in query_ids:
        query_id = str(query_id).zfill(7)
        if query_id in sat_id_to_row:
            selected.append(sat_id_to_row[query_id])
        elif query_id in ground_id_to_row:
            selected.append(ground_id_to_row[query_id])
        else:
            raise ValueError(f"Query ID {query_id} was not found in the validation split")
    return selected


@torch.no_grad()
def encode_gallery(model, loader, device, sat_mean, sat_std):
    embeddings = []
    indices = []
    for images, row_indices in tqdm(loader, desc="Encoding satellite gallery"):
        images = (images.to(device, non_blocking=True) - sat_mean) / sat_std
        global_embeddings, _ = model.encode_satellite_stage2(images)
        embeddings.append(global_embeddings.cpu())
        indices.append(row_indices)
    embeddings = torch.cat(embeddings)
    indices = torch.cat(indices)
    order = torch.argsort(indices)
    return embeddings[order]


@torch.no_grad()
def encode_queries(model, loader, device, size, ground_mean, ground_std):
    global_embeddings = []
    query_tokens = []
    row_indices = []
    for images, texts, indices in tqdm(loader, desc="Encoding selected queries"):
        images = images.to(device, non_blocking=True)
        if images.shape[-2:] != (size, size):
            images = torch.nn.functional.interpolate(
                images, size=(size, size), mode="bilinear", align_corners=False
            )
        images = (images - ground_mean) / ground_std
        global_query, tokens = model.encode_query_stage2(images, list(texts))
        global_embeddings.append(global_query.cpu())
        query_tokens.append(tokens.cpu())
        row_indices.append(indices)
    return (
        torch.cat(global_embeddings),
        torch.cat(query_tokens),
        torch.cat(row_indices),
    )


@torch.no_grad()
def encode_candidate_patches(
    model, rows, candidate_indices, data_root, transform, batch_size,
    num_workers, device, sat_mean, sat_std,
):
    unique_indices = sorted(set(candidate_indices.reshape(-1).tolist()))
    dataset = SatelliteDataset(rows, data_root, transform, unique_indices)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    patch_by_index = {}
    for images, indices in tqdm(loader, desc="Encoding rerank candidates"):
        images = (images.to(device, non_blocking=True) - sat_mean) / sat_std
        _, patches = model.encode_satellite_stage2(images)
        for row_index, patch_tokens in zip(indices.tolist(), patches.cpu()):
            patch_by_index[row_index] = patch_tokens
    return patch_by_index


@torch.no_grad()
def rerank_selected(
    model, query_tokens, candidate_indices, candidate_global_scores,
    patch_by_index, device, candidate_chunk_size, fusion_lambda,
):
    reranked = []
    for query_number in tqdm(range(len(query_tokens)), desc="Region-voting rerank"):
        indices = candidate_indices[query_number]
        score_parts = []
        query = query_tokens[query_number : query_number + 1].to(device)
        for start in range(0, len(indices), candidate_chunk_size):
            chunk_indices = indices[start : start + candidate_chunk_size].tolist()
            patches = torch.stack([patch_by_index[i] for i in chunk_indices]).to(device)
            patches = patches.unsqueeze(0)
            scores, _, _ = model.score_stage2_candidates(
                query_tokens=query,
                candidate_sat_patches=patches,
                return_maps=False,
            )
            score_parts.append(scores.squeeze(0).float().cpu())
        region_scores = torch.cat(score_parts).unsqueeze(0)
        global_scores = candidate_global_scores[query_number].float().unsqueeze(0)
        combined = row_zscore(global_scores) + fusion_lambda * row_zscore(region_scores)
        order = torch.argsort(combined.squeeze(0), descending=True)
        reranked.append(indices[order])
    return torch.stack(reranked)


def prepare_display_image(path, size=512):
    """Center-crop and resize an image to a common square display size."""
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def draw_figure(rows, query_rows, retrieved_rows, data_root, output):
    data_root = Path(data_root)
    n_rows = len(query_rows)
    fig, axes = plt.subplots(n_rows, 6, figsize=(15, 2.65 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row_number, (query_row, retrievals) in enumerate(zip(query_rows, retrieved_rows)):
        ground_path = data_root / str(rows.iloc[query_row, 1])
        axes[row_number, 0].imshow(prepare_display_image(ground_path))

        for rank, retrieved_row in enumerate(retrievals.tolist(), start=1):
            retrieved_row = int(retrieved_row)
            axis = axes[row_number, rank]
            sat_path = data_root / str(rows.iloc[retrieved_row, 0])
            axis.imshow(prepare_display_image(sat_path))
            correct = retrieved_row == query_row
            color = "#14a44d" if correct else "#dc3545"
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(4.5)

        for axis in axes[row_number]:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.05, hspace=0.05)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved qualitative figure to {output}")


def main():
    parser = argparse.ArgumentParser(description="Create a CVUSA top-5 qualitative retrieval figure")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--query-ids", nargs="+", default=DEFAULT_QUERY_IDS)
    parser.add_argument("--output", default="qualitative_results/cvusa_top5.png")
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--rerank-topk", type=int, default=100)
    parser.add_argument("--candidate-chunk-size", type=int, default=25)
    parser.add_argument(
        "--fusion-lambda", type=float, default=0.3,
        help="Weight of the z-scored Stage 2 region score relative to the global score",
    )
    args = parser.parse_args()

    if args.rerank_topk < 5:
        parser.error("--rerank-topk must be at least 5")

    cfg = load_config(args.config)
    if cfg.data.dataset_name != "CVUSA":
        raise ValueError(f"Expected a CVUSA config, got {cfg.data.dataset_name!r}")
    device = select_device(args.gpu_id)
    print(f"Using device: {device}")
    model = load_eval_model(cfg, args.checkpoint, device, stage=2)

    data_root = Path(cfg.data.data_root)
    split_path = data_root / cfg.data.val_csv
    rows = pd.read_csv(split_path, header=None).reset_index(drop=True)
    text_path = data_root / "lang" / f"{cfg.data.lang}_val-19zl.csv"
    texts = pd.read_csv(text_path)["Text"].reset_index(drop=True)
    if len(texts) != len(rows):
        raise ValueError(f"Split has {len(rows)} rows but caption file has {len(texts)} rows")

    normalized_query_ids = [str(query_id).zfill(7) for query_id in args.query_ids]
    query_rows = find_query_rows(rows, normalized_query_ids)
    num_workers = cfg.data.num_workers if args.num_workers is None else args.num_workers

    base_transform = transforms.Compose([
        transforms.Resize((cfg.data.image_size_satellite, cfg.data.image_size_satellite)),
        transforms.ToTensor(),
    ])
    norm_tensors = build_norm_tensors(cfg, device)
    ground_mean, ground_std, sat_mean, sat_std = norm_tensors

    gallery_loader = DataLoader(
        SatelliteDataset(rows, data_root, base_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    query_loader = DataLoader(
        SelectedQueryDataset(rows, texts, query_rows, data_root, base_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    gallery_global = encode_gallery(model, gallery_loader, device, sat_mean, sat_std)
    query_global, query_tokens, encoded_query_rows = encode_queries(
        model, query_loader, device, cfg.data.image_size_ground, ground_mean, ground_std
    )
    if encoded_query_rows.tolist() != query_rows:
        raise RuntimeError("Selected-query order changed unexpectedly")

    global_scores = query_global.to(device) @ gallery_global.to(device).T
    candidate_scores, candidate_indices = torch.topk(
        global_scores, k=min(args.rerank_topk, len(rows)), dim=1
    )
    candidate_scores = candidate_scores.cpu()
    candidate_indices = candidate_indices.cpu()
    del global_scores

    patch_by_index = encode_candidate_patches(
        model, rows, candidate_indices, data_root, base_transform,
        args.batch_size, num_workers, device, sat_mean, sat_std,
    )
    reranked = rerank_selected(
        model, query_tokens, candidate_indices, candidate_scores,
        patch_by_index, device, args.candidate_chunk_size, args.fusion_lambda,
    )
    draw_figure(
        rows, query_rows, reranked[:, :5], data_root, args.output,
    )


if __name__ == "__main__":
    main()
