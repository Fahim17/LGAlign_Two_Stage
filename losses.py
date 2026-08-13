import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Stage 1: Global warmup loss
# Standard symmetric InfoNCE / contrastive retrieval loss between a pooled
# query embedding (ground + text) and a pooled satellite embedding.
# Other satellite images in the batch act as in-batch negatives.
# ============================================================================
class Stage1ContrastiveLoss(nn.Module):
    """
    query_emb, sat_emb: [B, D], both expected to already be L2-normalized.
    Symmetric cross entropy over the [B, B] similarity matrix, diagonal = positives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb: torch.Tensor, sat_emb: torch.Tensor):
        B = query_emb.shape[0]
        device = query_emb.device

        # [B, B] cosine-similarity logits (already normalized embeddings)
        logits = (query_emb @ sat_emb.t()) / self.temperature

        labels = torch.arange(B, device=device)

        loss_q2s = F.cross_entropy(logits, labels)        # query retrieves correct satellite
        loss_s2q = F.cross_entropy(logits.t(), labels)     # satellite retrieves correct query

        loss = 0.5 * (loss_q2s + loss_s2q)
        return loss


# ============================================================================
# Stage 2: Region Voting loss
# ----------------------------------------------------------------------------
# Each query is scored against C candidate satellites (candidate 0 = positive,
# 1..C-1 = hard negatives mined from Stage 1 global retrieval). The model
# produces one scalar candidate_score per candidate (from local max-pooled
# region voting, not a pooled-embedding contrastive score) and one
# [20, 20] region_map per candidate (the spatial vote distribution).
#
# Three terms:
#   1. Cross entropy over candidate_scores, target = 0 (positive is index 0).
#   2. Positive concentration: the positive's region_map should be a sharp,
#      low-entropy spike (the model commits to one location).
#   3. Negative spread: negative region_maps should be high-entropy /
#      low-peak (the model should NOT find a confident match in a wrong
#      satellite -- a confident peak there is the failure mode this term
#      penalizes directly, on top of whatever the CE term alone catches).
#
# NOTE on entropy normalization: region_maps are already nonnegative
# vote/attention mass (not logits), so entropy is computed by normalizing
# to a probability distribution via simple sum-normalization (p = map /
# map.sum()), NOT softmax. Softmax on already-nonnegative values that are
# close in magnitude flattens the distribution almost to uniform regardless
# of how concentrated the actual votes are, which would make this term
# measure something close to noise rather than true concentration/spread.
# Sum-normalization preserves the actual relative vote mass, so entropy
# faithfully reflects whether votes are concentrated or spread out.
# ============================================================================
class RegionVotingLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        positive_entropy_weight: float = 0.02,
        negative_entropy_weight: float = 0.10,
        negative_peak_weight: float = 0.10,
    ):
        super().__init__()
        self.temperature = temperature
        self.positive_entropy_weight = positive_entropy_weight
        self.negative_entropy_weight = negative_entropy_weight
        self.negative_peak_weight = negative_peak_weight

    @staticmethod
    def _entropy(prob, eps=1e-8):
        """prob: [..., N], already a valid probability distribution over the last dim."""
        return -(prob * torch.log(prob + eps)).sum(dim=-1)

    @staticmethod
    def _normalize_mass(flat_map, eps=1e-8):
        """
        flat_map: [..., N] nonnegative vote/attention mass.
        Returns a true probability distribution via sum-normalization
        (NOT softmax) -- see class docstring for why.
        """
        flat_map = flat_map.clamp_min(eps)
        return flat_map / flat_map.sum(dim=-1, keepdim=True)

    def forward(self, candidate_scores: torch.Tensor, region_maps: torch.Tensor, target: torch.Tensor):
        """
        candidate_scores: [B, C]
        region_maps:      [B, C, H, W]  (H=W=20 with current config)
        target:           [B]           -- all zeros (candidate 0 is positive)
        """
        B, C = candidate_scores.shape

        # ------------------------------------------------------------------
        # 1. Candidate-selection cross entropy.
        # ------------------------------------------------------------------
        logits = candidate_scores / self.temperature
        ce = F.cross_entropy(logits, target)

        # ------------------------------------------------------------------
        # 2. Positive concentration -- minimize entropy of the positive's
        # normalized spatial vote distribution.
        # ------------------------------------------------------------------
        pos_map = region_maps[:, 0]                              # [B, H, W]
        pos_prob = self._normalize_mass(pos_map.flatten(1))       # [B, H*W]
        positive_entropy_loss = self._entropy(pos_prob).mean()

        # ------------------------------------------------------------------
        # 3. Negative spread -- maximize entropy of negative normalized vote
        # distributions (implemented as minimizing -entropy), and directly
        # penalize any sharp negative peak.
        # ------------------------------------------------------------------
        if C > 1:
            neg_map = region_maps[:, 1:]                              # [B, C-1, H, W]
            neg_prob = self._normalize_mass(neg_map.flatten(2))       # [B, C-1, H*W]
            neg_entropy = self._entropy(neg_prob)                     # [B, C-1], true entropy (not negated)
            negative_entropy_loss = -neg_entropy.mean()               # loss term: minimize -entropy = maximize entropy
            negative_peak_loss = neg_map.flatten(2).max(dim=-1).values.mean()
        else:
            neg_entropy = torch.zeros((), device=candidate_scores.device)
            negative_entropy_loss = torch.zeros((), device=candidate_scores.device)
            negative_peak_loss = torch.zeros((), device=candidate_scores.device)

        loss = (
            ce
            + self.positive_entropy_weight * positive_entropy_loss
            + self.negative_entropy_weight * negative_entropy_loss
            + self.negative_peak_weight * negative_peak_loss
        )

        return {
            "loss": loss,
            "ce": ce,
            "pos_entropy": positive_entropy_loss,
            "neg_entropy_loss": negative_entropy_loss,   # what's actually minimized (= -entropy)
            "neg_entropy": neg_entropy.mean() if C > 1 else negative_entropy_loss,  # true entropy, for clean logging
            "neg_peak": negative_peak_loss,
        }