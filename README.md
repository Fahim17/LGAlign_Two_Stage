# LG-Align: Language-Guided Global Retrieval to Local Region Voting

<div style="text-align: center;">
  <p align="center">
    <b>Fahimul Aleem · Raiyaan Abdullah · Fahimul Aleem · Shruti Vyas</b>
  </p>
</div>

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/paper_1.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-245b4c.svg)](https://fahim17.github.io/LGAlign_Two_Stage/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **LG-Align**, a two-stage language-guided framework for cross-view geolocalization with randomly oriented, limited-field-of-view ground images.

Given a random 90° crop from a ground panorama and a natural-language description of the visible scene, LG-Align retrieves the corresponding satellite image. It first performs efficient global retrieval, then re-ranks the strongest candidates by locating query-consistent regions within each satellite image.

> **Paper:** *LG-Align: Language-Guided Global Retrieval to Local Region Voting*  
> GAIA @ ECCV 2026 submission · [Read the paper](paper/paper_1.pdf)

## Highlights

- Does not assume a panorama, north alignment, or known camera orientation at inference time.
- Fuses limited-FoV visual evidence and scene-level language using a Global Q-Former.
- Re-ranks hard satellite candidates with patch-level region voting.
- Keeps the large image and text encoders frozen and checkpoints only the lightweight trainable modules.
- Achieves **23.62 R@1**, **49.61 R@5**, **60.41 R@10**, and **82.15 R@1%** on the paper's limited-FoV CVUSA evaluation.

## Method

LG-Align separates full-gallery retrieval from more expensive local matching.

### Stage 1: language-guided global retrieval

1. A frozen DINOv3 encoder extracts patch tokens from the limited-FoV ground image.
2. A frozen FLAN-T5 encoder extracts tokens from its scene description.
3. Learned projections map both modalities to a shared space.
4. A Global Q-Former with 32 learnable queries fuses the ground and text tokens.
5. Mean-pooled query features and a global satellite embedding are trained with symmetric InfoNCE.

This stage searches the full satellite gallery efficiently and produces the candidate pool used by Stage 2.

### Stage 2: local region-voting re-ranking

1. The frozen Stage 1 retriever mines hard satellite candidates.
2. A Local Q-Former, initialized once from the Stage 1 query pathway, learns tokens specialized for local matching.
3. A Voting Q-Former cross-attends those query tokens to each candidate's satellite patch tokens.
4. Token-gated attention forms a spatial vote map.
5. Local 3 × 3 max pooling converts the strongest query-consistent region into a candidate score.

At inference time, the Stage 2 score is fused with the Stage 1 global score to re-rank only the top candidates.

## Results

Results reported in the paper for the random limited-FoV CVUSA protocol:

| Method | R@1 | R@5 | R@10 | R@1% |
|---|---:|---:|---:|---:|
| Sample4Geo | 16.31 | 33.93 | 42.94 | 72.99 |
| L2LTR | 16.58 | 36.49 | 42.55 | 76.12 |
| GeoDTR | 13.20 | 30.33 | 39.58 | 73.58 |
| TransGeo | 20.72 | 42.31 | 51.52 | 81.84 |
| **LG-Align** | **23.62** | **49.61** | **60.41** | **82.15** |

### Ablations

| Query modalities | R@1 | R@5 | R@10 | R@1% |
|---|---:|---:|---:|---:|
| Text only | 1.94 | 6.95 | 11.59 | 40.45 |
| Ground image only | 15.20 | 39.51 | 52.68 | 80.53 |
| **Ground image + text** | **23.62** | **49.61** | **60.41** | **82.15** |

| Retrieval stage | R@1 | R@5 | R@10 | R@1% |
|---|---:|---:|---:|---:|
| Stage 1 global retrieval | 11.41 | 29.28 | 39.76 | 73.94 |
| **Stage 1 + Stage 2 re-ranking** | **23.62** | **49.61** | **60.41** | **82.15** |

Language supplies complementary semantic context, while region voting more than doubles Stage 1 Recall@1.

## Repository layout

```text
LGAlign_Two_Stage/
├── main.py                 # unified training/evaluation CLI
├── train.py                # Stage 1 and Stage 2 training
├── eval.py                 # global retrieval and region-voting evaluation
├── models/
│   └── custom_model.py     # DINOv3 + T5/SigLIP2 + Q-Former model
├── datasets/               # CVUSA, CVACT, VIGOR, and GAMa loaders
├── configs/                # experiment configurations
├── crcv_scripts/           # example Slurm jobs
├── qualitative_cvusa.py    # qualitative retrieval visualization
└── paper/paper_1.pdf       # manuscript
```

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/Fahim17/LGAlign_Two_Stage.git
cd LGAlign_Two_Stage

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install transformers
```

Choose the PyTorch command appropriate for your CUDA installation from the [official PyTorch installer](https://pytorch.org/get-started/locally/) if CUDA 12.1 is not suitable.

The default configuration uses gated DINOv3 models from Hugging Face. Request access to both model repositories and authenticate locally:

- [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [`facebook/dinov3-vitl16-pretrain-sat493m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m)

```bash
huggingface-cli login
```

Never place a Hugging Face token directly in source code or commit it to Git.

## Data preparation

Download CVUSA and arrange the processed limited-FoV data as follows:

```text
CVUSA/
├── splits/
│   ├── train-19zl.csv
│   └── val-19zl.csv
├── lang/
│   ├── T1_train-19zl.csv
│   └── T1_val-19zl.csv
├── streetview/             # limited-FoV ground images
└── bingmap/                # satellite images
```

The split CSV files have no header and use:

```text
<satellite-relative-path>,<ground-relative-path>
```

Each language CSV must contain a `Text` column aligned row-for-row with its corresponding split. The paper creates each query by taking a random horizontal crop representing a 90° FoV and generates its description with the prompt `Describe the context of the image.`

Before running the code, copy a configuration and edit at least these fields:

```json
{
  "data": {
    "data_root": "/absolute/path/to/CVUSA",
    "train_csv": "splits/train-19zl.csv",
    "val_csv": "splits/val-19zl.csv",
    "lang": "T1"
  }
}
```

For example:

```bash
cp configs/geo_dino_t5_qformer_5e-05.json configs/my_cvusa_experiment.json
```

Also update `checkpointing.save_dir`, `logging.log_dir`, and `logging.info_dir` if you want different output locations.

## Training

Run the complete two-stage schedule (20 Stage 1 epochs followed by 20 Stage 2 epochs in the paper configuration):

```bash
python main.py \
  --mode train \
  --config configs/my_cvusa_experiment.json \
  --gpu-id 0
```

Stage 1 checkpoints contain only the trainable projections and Q-Former-related weights; the frozen DINOv3 and text encoders are reloaded from Hugging Face. Before Stage 2, the code mines hard negatives from Stage 1 retrievals and caches them beside the source checkpoint.

To skip Stage 1 and start Stage 2 from an existing Stage 1 checkpoint:

```bash
python main.py \
  --mode train \
  --config configs/my_cvusa_experiment.json \
  --resume-from-stage1 weights/<experiment>/<stage1-checkpoint>.pt \
  --gpu-id 0
```

## Evaluation

Evaluate Stage 1 global retrieval only:

```bash
python main.py \
  --mode eval \
  --config configs/my_cvusa_experiment.json \
  --checkpoint weights/<experiment>/<stage1-checkpoint>.pt \
  --stage 1 \
  --gpu-id 0
```

Evaluate the full global-retrieval and region-voting pipeline:

```bash
python main.py \
  --mode eval \
  --config configs/my_cvusa_experiment.json \
  --checkpoint weights/<experiment>/<stage2-checkpoint>.pt \
  --stage 2 \
  --rerank-topk 200 \
  --fusion-lambda 0.3 \
  --gpu-id 0
```

Useful memory controls for Stage 2 evaluation are `--eval-batch-size`, `--gather-batch-size`, and `--candidate-chunk-size`. The re-ranker can only recover a correct match already present in the Stage 1 top-K candidate set.

## Paper configuration

| Setting | Value |
|---|---|
| Ground encoder | DINOv3 ViT-L/16, LVD-1689M (frozen) |
| Satellite encoder | DINOv3 ViT-L/16, SAT-493M (frozen) |
| Text encoder | FLAN-T5-Large (frozen) |
| Ground resolution | 224 × 224 |
| Satellite resolution | 320 × 320 (20 × 20 patches) |
| Shared dimension | 768 |
| Global/Local Q-Former | 32 queries, 4 layers, 8 heads |
| Voting Q-Former | 2 layers, 8 heads, token gating |
| Stage 1 / Stage 2 epochs | 20 / 20 |
| Stage 1 / Stage 2 learning rate | 1e-4 / 5e-5 |
| Stage 1 / Stage 2 batch size | 150 / 24 |
| Stage 2 training candidates | 16 (1 positive + 15 hard negatives) |
| Hard-negative pool | Top 200 Stage 1 candidates |
| Region window | 3 × 3, max pooling |
| Temperature | 0.07 for both stages |

## Citation

If you find this work useful, please cite the paper. Replace the anonymous author field when the final publication metadata is available.

```bibtex
@inproceedings{fahimul2026lgalign,
  title     = {LG-Align: Language-Guided Global Retrieval to Local Region Voting},
  author    = {Aleem, Fahimul and Abdullah, Raiyaan and Vyas, Shruti},
  booktitle = {GAIA at ECCV},
  year      = {2026}
}
```

## Acknowledgements

This project builds on CVUSA, DINOv3, FLAN-T5, and the Q-Former architecture, as well as prior work in cross-view geolocalization and multimodal retrieval.

## License

No license has been specified yet. Until a license is added, the repository's contents remain under the copyright holder's default rights.
