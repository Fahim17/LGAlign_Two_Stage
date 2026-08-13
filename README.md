# LG-Align: Language-Guided Global Retrieval to Local Region Voting

<!-- > **Anonymous GAIA @ ECCV 2026 Submission** -->

LG-Align is a two-stage framework for **cross-view image geolocalization under random limited field-of-view (FoV)**. Instead of assuming a north-aligned or panoramic ground query, LG-Align works with a randomly cropped **90° ground-view image** and augments it with a natural-language scene description.

The key idea is simple:

1. **Retrieve globally** using fused ground-image and language features.
2. **Re-rank locally** by finding the satellite region that best matches the limited-FoV query.

---

## Overview

Cross-view geolocalization matches a ground-level image to its corresponding satellite image. This becomes significantly harder when the ground query:

- covers only a small portion of the scene,
- can face an arbitrary azimuth direction,
- has no reliable orientation prior, and
- shares limited direct visual appearance with the overhead view.

LG-Align addresses this ambiguity by combining **visual evidence**, **language semantics**, and **local region voting**.

### Main Contributions

- A challenging **random 90° limited-FoV CVUSA setting** without fixed orientation assumptions.
- Language descriptions as an auxiliary query modality for cross-view matching.
- A **two-stage retrieval framework**:
  - Stage 1: global multimodal retrieval.
  - Stage 2: local region-aware re-ranking.
- A region-voting mechanism that explicitly searches for query-consistent local regions inside candidate satellite images.
- Strong performance gains over representative cross-view geolocalization baselines.

---

## Method

### Stage 1 — Global Retrieval

The limited-FoV ground image and its text description are encoded separately and projected into a shared embedding space.

A **Global Q-Former** uses learnable query tokens to fuse the projected ground-image and text features. The resulting query representation is compared with a global satellite embedding using a symmetric InfoNCE objective.

This stage efficiently retrieves an initial set of satellite candidates from the full gallery.

### Stage 2 — Local Region Voting

Stage 2 re-ranks the strongest Stage-1 candidates.

A **Local Q-Former**, initialized from the Stage-1 Global Q-Former, produces query tokens specialized for local matching. A **Voting Q-Former** then cross-attends these query tokens to satellite patch tokens.

The resulting attention maps are interpreted as spatial votes. After local pooling, the strongest regional response becomes the candidate score.

This allows LG-Align to identify a small satellite region that is consistent with the partial ground observation rather than relying only on global pooled similarity.

---

## Dataset

Experiments are conducted on **CVUSA**.

- **44,416** ground-satellite pairs
- **80%** training split
- **20%** evaluation split
- Ground panorama → random **90° FoV crop**
- Each crop is paired with a generated natural-language scene description

<!-- We refer to this setting as **CVUSA**, representing **Limited FoV + Language descriptions**. -->

---

## Results

Performance on the random limited-FoV CVUSA evaluation setting:

| Method | R@1 | R@5 | R@10 | R@1% |
|---|---:|---:|---:|---:|
| TransGeo | 20.72 | 42.31 | 51.52 | 81.84 |
| Sample4Geo | 16.31 | 33.93 | 42.94 | 72.99 |
| L2LTR | 16.58 | 36.49 | 42.55 | 76.12 |
| GeoDTR | 13.20 | 30.33 | 39.58 | 73.58 |
| GeoDTR+ | 8.47 | 322.60* | 31.58 | 69.48 |
| **LG-Align** | **23.62** | **49.61** | **60.41** | **82.15** |

LG-Align achieves the best reported result across all four retrieval metrics in this evaluation.

<!-- > **Note:** The current manuscript reports `322.60` for GeoDTR+ R@5, which appears inconsistent with the surrounding recall values. It is reproduced here from the submitted manuscript and should be verified before the public README is finalized. -->

---

<!-- ## Ablation Studies -->

### Query Modality

| Variant | R@1 | R@5 | R@10 | R@1% |
|---|---:|---:|---:|---:|
| w/o text | 15.20 | 39.51 | 52.68 | 80.53 |
| w/o ground image | 1.94 | 6.95 | 11.59 | 40.45 |
| **Ground image + text** | **23.62** | **49.61** | **60.41** | **82.15** |

Language is most effective as **complementary semantic guidance** rather than a replacement for the visual ground query.

### Effect of Stage 2 Re-ranking

| Setting | R@1 | R@5 | R@10 | R@1% |
|---|---:|---:|---:|---:|
| Stage 1 global retrieval | 11.41 | 29.28 | 39.76 | 73.94 |
| **Stage 1 + Stage 2** | **23.62** | **49.61** | **60.41** | **82.15** |

Stage 2 more than doubles Recall@1, showing the importance of local satellite-region matching under random limited FoV.

---

## Model Configuration

Key implementation details reported in the paper:

- Ground image resolution: `224 × 224`
- Satellite image resolution: `320 × 320`
- Shared embedding dimension: `768`
- Global Q-Former:
  - 32 query tokens
  - 4 layers
  - 8 attention heads
- Local Q-Former:
  - same architecture as Global Q-Former
  - initialized from Stage 1
- Voting Q-Former:
  - 2 layers
  - 8 attention heads
  - token gating
- Stage 1 training: 20 epochs
- Stage 2 training: 20 epochs
- Optimizer: AdamW
- Stage 1 learning rate: `1e-4`
- Stage 2 learning rate: `5e-5`
- Stage 2 candidates per query: 16
  - 1 positive
  - 15 hard negatives
- Hard negatives mined from top-200 Stage-1 retrievals
- Region pooling window: `3 × 3`

---

<!-- ## Repository Structure

```text
LG-Align/
├── README.md
├── index.html
├── style.css
├── assets/
│   ├── figures/
│   └── ...
├── paper/
│   └── paper.pdf
└── .nojekyll
```

The included static project page can be hosted directly with **GitHub Pages**.

--- -->

<!-- ## GitHub Pages

To publish the project website:

1. Push this repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, select:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/ (root)`
4. Save the configuration.

GitHub will then publish the project page from `index.html`. -->

---

## Code

<!-- Training and evaluation code will be released after the anonymous review period. -->

---

## Paper

**LG-Align: Language-Guided Global Retrieval to Local Region Voting**

<!-- Anonymous GAIA @ ECCV 2026 Submission. -->

---

## Citation

<!-- BibTeX will be updated after publication. For the anonymous submission period, you may use:

```bibtex
@inproceedings{lgalign2026,
  title     = {LG-Align: Language-Guided Global Retrieval to Local Region Voting},
  author    = {Anonymous},
  booktitle = {GAIA @ ECCV},
  year      = {2026}
}
``` -->

---

## Acknowledgements
<!-- 
This project builds on the CVUSA benchmark and prior work in cross-view image geolocalization, multimodal retrieval, and vision-language representation learning. -->

---

## License

<!-- A license will be added when the code and dataset resources are publicly released. -->