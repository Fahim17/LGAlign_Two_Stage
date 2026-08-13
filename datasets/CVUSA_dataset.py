import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ============================================================================
# Clean dataset classes for the DINOv3 + SigLIP2 + Q-Former pipeline.
# ----------------------------------------------------------------------------
# The old *_dataset_cropped classes (CVUSA_dataset.py, CVACT_dataset.py, ...)
# accept a `transform` argument but never actually apply it -- the relevant
# lines are commented out, and a CLIP AutoProcessor (built from attributes.py
# / hypm at both import time and __init__ time) runs unconditionally instead.
# That means every image gets resized to CLIP's fixed size and normalized
# with CLIP's stats, regardless of what `transform` we pass in -- which
# silently breaks this pipeline's per-branch (ground vs satellite) sizing
# and DINOv3-specific normalization (our own normalization in
# prepare_ground_satellite_tensors() would then double-normalize an
# already-CLIP-normalized tensor).
#
# These classes keep the same I/O contract as the old ones (constructor
# signature, CSV/path conventions, 5-tuple return) so train.py / eval.py's
# dataloader-building code barely changes, but with `transform` genuinely
# applied and zero dependency on attributes.py.
# ============================================================================
class CVUSA_dataset_cropped(Dataset):
    """
    Replacement for the old CVUSA_dataset_cropped in datasets/CVUSA_dataset.py
    -- same class name on purpose (drop-in replacement), but living in this
    file (datasets/geo_dataset.py) instead, since the old one is kept around
    unmodified as reference material.

    Column convention (matches the original CVUSA splits CSV, header=None):
      column 0 -> satellite image relative path
      column 1 -> ground/street-view image relative path

    Returns (matching the old dataloader's 5-tuple, kept for compatibility
    with train.py / eval.py's unpacking):
      ground_img, sat_img, negative_img, text, idx

    `negative_img` is just `sat_img` duplicated -- the old class never used a
    real distinct negative either (its hard-negative lookup was commented
    out); it's kept in the return only so the 5-tuple shape matches, and is
    never read downstream -- Stage 1 and Stage 2 both rely on in-batch
    negatives instead.
    """

    def __init__(self, df, path, train=True, transform=None, lang="T1"):
        # reset_index so positional access (iloc/values) and label access
        # (.loc, used for the text lookup below) always agree, even if the
        # caller's df came in with a non-contiguous index (e.g. after a
        # pandas filter) -- the old class assumed they already matched.
        self.data_csv = df.reset_index(drop=True)
        self.is_train = train
        self.transform = transform
        self.path = path
        self.lang = lang

        self.sat_images = self.data_csv.iloc[:, 0].values
        self.ground_images = self.data_csv.iloc[:, 1].values

        # numeric id parsed from the satellite filename, e.g. ".../0000001.jpg" -> 1
        self.data_csv["idx"] = self.data_csv[0].map(lambda x: int(x.split("/")[-1].split(".")[0]))

        lang_csv = f"{lang}_train-19zl.csv" if self.is_train else f"{lang}_val-19zl.csv"
        self.text_df = pd.read_csv(f"{self.path}/lang/{lang_csv}")

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, item):
        ground_path = f"{self.path}/{self.ground_images[item]}"
        sat_path = f"{self.path}/{self.sat_images[item]}"

        ground_img = Image.open(ground_path).convert("RGB")
        sat_img = Image.open(sat_path).convert("RGB")
        text = self.text_df["Text"].loc[item]

        if self.transform is not None:
            ground_img = self.transform(ground_img)
            sat_img = self.transform(sat_img)

        negative_img = sat_img  # unused downstream -- kept only for dataloader-shape compatibility

        return ground_img, sat_img, negative_img, text, self.data_csv["idx"][item]