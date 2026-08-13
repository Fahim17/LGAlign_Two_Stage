import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import json
from pathlib import Path


# ============================================================================
# DINOv3 Model Registration for transformers
# ============================================================================
def _register_dinov3_if_needed():
    """
    Registers DINOv3 configuration and model with transformers if not already registered.
    This is a workaround for older transformers versions that don't have DINOv3 support.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if "dinov3_vit" in CONFIG_MAPPING:
            return  # Already registered
    except:
        pass
    
    # Try to import vision transformer which is the base
    try:
        from transformers.models.vision_transformer import ViTConfig, ViTModel
        
        # Create a simple DINOv3 config class by inheriting from ViT
        class DINOv3Config(ViTConfig):
            model_type = "dinov3_vit"
        
        # Register it
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers import config
        CONFIG_MAPPING.register("dinov3_vit", DINOv3Config)
        
        # Also register the model
        from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
        MODEL_MAPPING.register("dinov3_vit", ViTModel)
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.register("dinov3_vit", ViTModel)
    except Exception as e:
        pass  # If registration fails, we'll handle it in the loader


_register_dinov3_if_needed()


# ============================================================================
# Frozen encoder loaders
# ----------------------------------------------------------------------------
# DINOv3 and the text encoder are loaded from the checkpoint *names* declared
# in the JSON config. Their weights are NEVER part of this model's saved
# checkpoint -- on reload, they are simply re-instantiated from these same
# names.
# ============================================================================
def _load_frozen_dino(checkpoint_name: str):
    """
    Loads a frozen DINOv3 vision backbone via the HF AutoModel API.
    NOTE: facebook/dinov3-* checkpoints are gated on Hugging Face -- request
    access on the model page and run `huggingface-cli login` locally first.
    """
    try:
        model = AutoModel.from_pretrained(checkpoint_name)
    except ValueError as e:
        if "dinov3_vit" in str(e):
            # Fallback: Try loading as Vision Transformer
            from transformers import ViTModel
            print(f"  [Warning] DINOv3 not natively supported by this transformers version.")
            print(f"  [Info] Loading {checkpoint_name} as ViT instead...")
            model = ViTModel.from_pretrained(checkpoint_name)
        else:
            raise
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_frozen_text_encoder(encoder_type: str, checkpoint_name: str):
    """
    Loads a frozen text encoder, dispatching on encoder_type:
      - "t5":      T5EncoderModel (encoder-only stack). Relative position
                   bias -- no hard short-context cap, suited to long
                   multi-sentence captions. This is the default.
      - "siglip2": the SigLIP2 TEXT tower only (full model's .text_model).
                   Hard 64-token architectural cap -- enforced in
                   _extract_text_tokens() below, not left as a config knob.

    Returns (text_model, tokenizer).
    """
    if encoder_type == "t5":
        text_model = T5EncoderModel.from_pretrained(checkpoint_name)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    elif encoder_type == "siglip2":
        full_model = AutoModel.from_pretrained(checkpoint_name)
        text_model = full_model.text_model
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    else:
        raise ValueError(f"Unsupported text_encoder_type: {encoder_type!r} (expected 't5' or 'siglip2')")

    text_model.eval()
    for p in text_model.parameters():
        p.requires_grad = False
    return text_model, tokenizer


# ============================================================================
# Query Q-Former (Stage A)
# ----------------------------------------------------------------------------
# A small set of learnable query tokens cross-attends over a joined context
# of [ground DINOv3 patch tokens, text tokens], interleaved with
# self-attention among the query tokens themselves. This is the "clue
# extraction" stage: query tokens end up representing things like road
# layout, houses, parked cars, etc., pulled from the ground image + caption.
# The frozen text encoder does not need to be pre-aligned with images --
# this module is what learns the cross-modal fusion, from scratch.
#
# This same class is instantiated twice in GeoDinoSiglipQFormer: once as the
# frozen Stage 1 "query_qformer", and once as the trainable Stage 2
# "stage2_query_qformer" -- an exact architectural copy, warm-started from
# Stage 1's weights and then trained independently (see
# initialize_stage2_from_stage1() below). The two never share parameters.
# ============================================================================
class _QFormerLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, q, context, context_key_padding_mask=None):
        # self-attention among query tokens
        h = self.norm1(q)
        sa_out, _ = self.self_attn(h, h, h)
        q = q + sa_out

        # cross-attention: query tokens look at the ground+text context
        h = self.norm2(q)
        ca_out, _ = self.cross_attn(h, context, context, key_padding_mask=context_key_padding_mask)
        q = q + ca_out

        # feed-forward
        h = self.norm3(q)
        q = q + self.ffn(h)
        return q


class QueryQFormer(nn.Module):
    def __init__(self, dim, num_query_tokens, num_layers, num_heads, ffn_dim, dropout):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, dim) * 0.02)
        self.layers = nn.ModuleList(
            [_QFormerLayer(dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, ground_tokens_proj, text_tokens_proj, text_attention_mask=None):
        """
        ground_tokens_proj: [B, Ng, D]  -- projected ground DINOv3 patch tokens
        text_tokens_proj:    [B, Nt, D]  -- projected text encoder tokens
        text_attention_mask: [B, Nt]     -- 1 for real tokens, 0 for padding

        Returns:
            query_tokens: [B, Nq, D]  -- per-token output, used by Stage 2
            pooled_query: [B, D]      -- mean-pooled, used by Stage 1
        """
        B = ground_tokens_proj.shape[0]
        q = self.query_tokens.expand(B, -1, -1).contiguous()

        context = torch.cat([ground_tokens_proj, text_tokens_proj], dim=1)  # [B, Ng+Nt, D]

        key_padding_mask = None
        if text_attention_mask is not None:
            ground_mask = torch.ones(
                B, ground_tokens_proj.shape[1], device=ground_tokens_proj.device, dtype=text_attention_mask.dtype
            )
            full_mask = torch.cat([ground_mask, text_attention_mask], dim=1)  # [B, Ng+Nt]
            key_padding_mask = full_mask == 0  # True = ignore this position

        for layer in self.layers:
            q = layer(q, context, key_padding_mask)

        q = self.final_norm(q)
        pooled_query = q.mean(dim=1)
        return q, pooled_query


# ============================================================================
# Satellite Q-Former layer (Stage 2)
# ----------------------------------------------------------------------------
# Stage 2 query tokens (trainable -- warm-started from Stage 1's query
# pathway, then trained independently; see stage2_query_qformer /
# initialize_stage2_from_stage1()) cross-attend to candidate satellite patch
# tokens. One lightweight refinement layer: cross-attention then a small
# FFN, returning the attention weights so the caller can use them as
# region votes.
# ============================================================================
class SatelliteQFormerLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, query_tokens, sat_patch_tokens):
        q = self.norm_q(query_tokens)
        s = self.norm_s(sat_patch_tokens)

        attn_out, attn_weights = self.cross_attn(
            q, s, s,
            need_weights=True,
            average_attn_weights=True,
        )

        query_tokens = query_tokens + attn_out
        query_tokens = query_tokens + self.ffn(self.norm_ffn(query_tokens))

        return query_tokens, attn_weights


# ============================================================================
# Satellite Region Voting Former (Stage 2)
# ----------------------------------------------------------------------------
# Replaces the old SatelliteDetectiveQFormer / PatchMILLoss pooled-embedding
# contrastive design. This module learns a brand-new local matching function:
# it takes the trainable Stage 2 query tokens (warm-started from Stage 1,
# then trained independently of it -- see stage2_query_qformer and
# initialize_stage2_from_stage1() on GeoDinoSiglipQFormer) and projected
# satellite patch tokens, and locates, via attention, the local region
# inside a candidate satellite that the query is consistent with. Output is
# a per-candidate scalar score plus a spatial vote map, not a pooled
# embedding -- there is no embedding-level contrastive loss here; scoring
# happens entirely through RegionVotingLoss on candidate_scores / region_maps
# (see losses/region_voting.py).
# ============================================================================
class SatelliteRegionVotingFormer(nn.Module):
    def __init__(self, dim, num_layers, num_heads, ffn_dim, dropout=0.1, use_token_gating=True):
        super().__init__()
        self.dim = dim
        self.use_token_gating = use_token_gating

        self.q_input_norm = nn.LayerNorm(dim)
        self.s_input_norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([
            SatelliteQFormerLayer(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Token gate: learns a per-query-token weight when pooling the
        # per-token attention maps into a single vote map (Section 6 of
        # agent_instructions.md). Only built if use_token_gating is True;
        # otherwise the caller falls back to a plain mean over tokens.
        if self.use_token_gating:
            self.token_gate_mlp = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )
        else:
            self.token_gate_mlp = None

    def forward(self, query_tokens, sat_patch_tokens):
        """
        query_tokens:     [N, Nq, D]  -- N = B*C flattened query/candidate pairs
        sat_patch_tokens: [N, Np, D]

        Returns:
            updated_query_tokens: [N, Nq, D]
            attention_map:        [N, Nq, Np]  -- last layer's attention, averaged over heads
        """
        q = self.q_input_norm(query_tokens)
        s = self.s_input_norm(sat_patch_tokens)

        attn = None
        for layer in self.layers:
            q, attn = layer(q, s)

        return q, attn

    def vote_map_from_attention(self, updated_query_tokens, attention_map, grid_size):
        """
        updated_query_tokens: [N, Nq, D]
        attention_map:        [N, Nq, Np]
        grid_size:             int, sqrt(Np) -- e.g. 20 for 400 satellite patches

        Returns:
            vote_map_2d: [N, grid_size, grid_size]
        """
        if self.use_token_gating:
            gate_logits = self.token_gate_mlp(updated_query_tokens)         # [N, Nq, 1]
            gate = F.softmax(gate_logits, dim=1)                            # softmax over Nq
            vote_map = (gate * attention_map).sum(dim=1)                    # [N, Np]
        else:
            vote_map = attention_map.mean(dim=1)                           # [N, Np]

        N = vote_map.shape[0]
        return vote_map.view(N, grid_size, grid_size)


# ============================================================================
# Main model
# ============================================================================
class GeoDinoSiglipQFormer(nn.Module):
    """
    Ground(+text)-to-satellite geolocalization model.

    Frozen:    DINOv3 ground encoder (web-pretrained), DINOv3 satellite
               encoder (SAT-493M-pretrained), text encoder (T5 by default;
               SigLIP2 also supported -- see cfg.model.text_encoder_type).

    Stage 1 trainable: Query Q-Former (proj_ground, proj_text, query_qformer),
               satellite global pooling/projection head, final retrieval
               projection layers. Used for global contrastive retrieval and
               for mining Stage 2's hard negatives.

    Stage 2 trainable: an independent local patch reranker --
               stage2_proj_ground, stage2_proj_text, stage2_query_qformer
               (an exact architectural copy of the Stage 1 query pathway,
               warm-started from Stage 1 weights via
               initialize_stage2_from_stage1() and then trained
               independently -- NOT weight-tied to Stage 1), plus
               stage2_proj_satellite and stage2_region_voting_qformer (the
               local region-voting matcher). Once Stage 2 begins,
               freeze_stage1_for_stage2() freezes everything under "Stage 1
               trainable" above.

    NOTE: all Stage 2 modules above default to requires_grad=True at
    construction (same as everything else in trainable_modules). Callers
    that run a separate Stage 1 phase before Stage 2 should call
    freeze_stage2_until_stage2() right after construction so Stage 1's
    optimizer/param-group/logging only ever sees the Stage 1 modules, then
    call unfreeze_stage2_for_stage2() (alongside freeze_stage1_for_stage2())
    at the Stage 1 -> Stage 2 transition. See run_training() in train.py.

    All trainable modules (both Stage 1 and Stage 2) are grouped under
    self.trainable_modules so they can be saved/loaded independently of the
    frozen encoders.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m = cfg.model

        self.embed_dim = m.embed_dim
        self.remove_cls_token = m.remove_cls_token
        self.remove_register_tokens = m.remove_register_tokens
        self.num_register_tokens = m.num_register_tokens
        self.text_encoder_type = m.text_encoder_type
        self.query_modality = getattr(cfg.model, "query_modality", "both")
        if self.query_modality not in ("both", "image", "text"):
            raise ValueError(
                f"Unsupported query_modality: {self.query_modality!r} "
                "(expected 'both', 'image', or 'text')"
            )
        print(f"Query modality: {self.query_modality}")

        assert m.freeze_dino, "this implementation always freezes DINOv3"
        assert m.freeze_text_encoder, "this implementation always freezes the text encoder"

        # --------------------------------------------------------------
        # Frozen encoders (intentionally NOT part of self.trainable_modules).
        # Ground and satellite each get their own DINOv3 checkpoint --
        # ground is web-pretrained (LVD-1689M), satellite is pretrained
        # directly on satellite imagery (SAT-493M).
        # --------------------------------------------------------------
        self.dino_ground = _load_frozen_dino(m.dino_ground_checkpoint)
        self.dino_satellite = _load_frozen_dino(m.dino_satellite_checkpoint)
        self.text_model, self.text_tokenizer = _load_frozen_text_encoder(
            m.text_encoder_type, m.text_encoder_checkpoint
        )

        dino_ground_dim = m.dino_ground_hidden_dim
        dino_satellite_dim = m.dino_satellite_hidden_dim
        text_dim = m.text_encoder_hidden_dim
        embed_dim = m.embed_dim

        # --------------------------------------------------------------
        # Trainable modules -- grouped in one ModuleDict for easy
        # state-dict save/load without touching the frozen encoders above.
        # --------------------------------------------------------------
        qq_cfg = m.query_qformer

        # Stage 2 Query Q-Former config: use cfg.model.stage2_query_qformer
        # if present, else fall back to the Stage 1 query_qformer config so
        # the two pathways are an exact architectural copy by default.
        stage2_qq_cfg = getattr(m, "stage2_query_qformer", None) or qq_cfg

        # Stage 2 region-voting matcher config. Renamed from
        # "stage2_satellite_qformer" -> "stage2_region_voting_qformer" for
        # clarity; old configs that still use the pre-rename key keep working.
        sd_cfg = getattr(m, "stage2_region_voting_qformer", None)
        if sd_cfg is None:
            sd_cfg = m.stage2_satellite_qformer  # backward-compat with pre-rename configs

        self.trainable_modules = nn.ModuleDict({
            # ----------------------------------------------------------
            # Stage 1: global query/satellite pathway. Frozen once
            # freeze_stage1_for_stage2() is called.
            # ----------------------------------------------------------
            "proj_ground": nn.Linear(dino_ground_dim, embed_dim),
            "proj_text": nn.Linear(text_dim, embed_dim),

            "query_qformer": QueryQFormer(
                dim=embed_dim,
                num_query_tokens=qq_cfg.num_query_tokens,
                num_layers=qq_cfg.num_layers,
                num_heads=qq_cfg.num_heads,
                ffn_dim=qq_cfg.ffn_dim,
                dropout=qq_cfg.dropout,
            ),

            # Stage 1 global satellite pooling/projection head
            "satellite_global_head": nn.Sequential(
                nn.Linear(dino_satellite_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            ),

            # Final retrieval projection layers (applied right before L2 norm)
            "retrieval_proj_query": nn.Linear(embed_dim, embed_dim),
            "retrieval_proj_satellite": nn.Linear(embed_dim, embed_dim),

            # ----------------------------------------------------------
            # Stage 2: independent local patch reranker. This is an exact
            # architectural copy of the Stage 1 query pathway above
            # (stage2_proj_ground/stage2_proj_text/stage2_query_qformer
            # mirror proj_ground/proj_text/query_qformer), warm-started
            # from Stage 1's weights via initialize_stage2_from_stage1()
            # and then trained independently -- this is a one-time weight
            # *copy*, not weight tying: the two pathways hold separate
            # parameter tensors from construction onward.
            # ----------------------------------------------------------
            "stage2_proj_ground": nn.Linear(dino_ground_dim, embed_dim),
            "stage2_proj_text": nn.Linear(text_dim, embed_dim),
            "stage2_query_qformer": QueryQFormer(
                dim=embed_dim,
                num_query_tokens=stage2_qq_cfg.num_query_tokens,
                num_layers=stage2_qq_cfg.num_layers,
                num_heads=stage2_qq_cfg.num_heads,
                ffn_dim=stage2_qq_cfg.ffn_dim,
                dropout=stage2_qq_cfg.dropout,
            ),

            # Stage 2 satellite patch projection -- projects raw DINOv3
            # satellite patch tokens to embed_dim for the region voting
            # matcher below. This is unambiguously a Stage 2 module (Stage 1
            # never uses patch-level satellite tokens), so it is NOT
            # warm-started from satellite_global_head -- that head builds a
            # global pooled embedding, a different role from per-patch local
            # tokens. It keeps its normal (random) initialization.
            "stage2_proj_satellite": nn.Linear(dino_satellite_dim, embed_dim),

            # Stage 2 local region-voting matcher (formerly named
            # "stage2_satellite_qformer"). Takes stage2_query_qformer's
            # output tokens and stage2_proj_satellite's patch tokens and
            # produces the spatial vote/region maps + candidate score.
            "stage2_region_voting_qformer": SatelliteRegionVotingFormer(
                dim=embed_dim,
                num_layers=sd_cfg.num_layers,
                num_heads=sd_cfg.num_heads,
                ffn_dim=sd_cfg.ffn_dim,
                dropout=sd_cfg.dropout,
                use_token_gating=getattr(sd_cfg, "use_token_gating", True),
            ),
        })

        # Stage 2 region-pooling settings (used by score_stage2_candidates).
        rv_cfg = getattr(cfg.training, "stage2_region_voting", None)
        self.region_window_size = getattr(rv_cfg, "region_window_size", 3) if rv_cfg else 3
        self.region_pooling = getattr(rv_cfg, "region_pooling", "max") if rv_cfg else "max"

    # ====================================================================
    # Frozen-encoder feature extraction
    # ====================================================================
    def _extract_dino_patch_tokens(self, dino_model, images):
        """
        Returns DINOv3 patch tokens with CLS / register tokens stripped.
        images: [B, C, H, W] -- already resized/normalized by the caller.
        """
        # Hugging Face ViT checkpoints enforce their pretraining resolution
        # (224x224 for these DINO checkpoints) unless positional-embedding
        # interpolation is explicitly enabled.  The satellite branch is
        # intentionally configured at 320x320, so allow ViT to interpolate
        # its learned 14x14 positional grid to the input patch grid.
        native_size = getattr(dino_model.config, "image_size", None)
        if isinstance(native_size, int):
            native_size = (native_size, native_size)
        elif native_size is not None:
            native_size = tuple(native_size)
        interpolate_pos_encoding = native_size is not None and images.shape[-2:] != native_size

        with torch.no_grad():
            outputs = dino_model(
                pixel_values=images,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
            hidden = outputs.last_hidden_state  # [B, 1 + num_register + Np, D]

        start = 0
        if self.remove_cls_token:
            start += 1

        if self.remove_register_tokens and self.num_register_tokens > 0:
            total_tokens = hidden.shape[1]
            expected_patches = (images.shape[-1] // 16) * (images.shape[-2] // 16)
            # If removing the configured register tokens leaves a perfect square grid,
            # do it; otherwise, assume the model does not include register tokens.
            candidate_tokens = total_tokens - start - self.num_register_tokens
            if candidate_tokens >= 0 and int(round(candidate_tokens ** 0.5)) ** 2 == candidate_tokens:
                start += self.num_register_tokens
            else:
                direct_tokens = total_tokens - start
                if direct_tokens >= 0 and int(round(direct_tokens ** 0.5)) ** 2 == direct_tokens:
                    # No register tokens were present.
                    pass
                elif direct_tokens == expected_patches:
                    pass
                elif candidate_tokens == expected_patches:
                    start += self.num_register_tokens
                else:
                    # Fall back to removing tokens only if that yields the expected patch count.
                    if candidate_tokens == expected_patches:
                        start += self.num_register_tokens
                    elif direct_tokens == expected_patches:
                        pass
                    else:
                        # If neither matches expected integer grid, preserve default behavior.
                        start += self.num_register_tokens

        patch_tokens = hidden[:, start:, :]
        return patch_tokens

    def _extract_text_tokens(self, texts):
        """
        texts: list[str]
        Returns token-level text features plus the attention mask so the
        Query Q-Former can ignore padding.

        SigLIP2 was trained with a hard fixed-length padding of 64 tokens
        (its position-embedding table has exactly 64 slots) -- max_length is
        FORCED to 64 here whenever text_encoder_type == "siglip2", regardless
        of cfg.data.text_max_length, since anything else would silently
        degrade quality or simply not run. T5 uses relative position bias
        with no such hard cap, so for "t5" cfg.data.text_max_length is just a
        truncation safety margin -- raise it in the config if your captions
        are longer.
        """
        device = next(self.text_model.parameters()).device

        if self.text_encoder_type == "siglip2":
            max_len = 64
            padding = "max_length"
        else:
            max_len = getattr(self.cfg.data, "text_max_length", 128)
            padding = "longest"

        tokenized = self.text_tokenizer(
            texts, padding=padding, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.text_model(**tokenized)
            text_tokens = outputs.last_hidden_state  # [B, T, D]

        return text_tokens, tokenized.get("attention_mask", None)

    def _apply_query_modality(self, ground_tokens_proj, text_tokens_proj, text_attention_mask):
        """
        Removes projected query-side context tokens according to cfg.model.query_modality
        while preserving the same QueryQFormer architecture and parameters.
        """
        if self.query_modality == "both":
            return ground_tokens_proj, text_tokens_proj, text_attention_mask

        if self.query_modality == "image":
            text_tokens_proj = text_tokens_proj[:, :0, :]
            if text_attention_mask is not None:
                text_attention_mask = text_attention_mask[:, :0]
        elif self.query_modality == "text":
            ground_tokens_proj = ground_tokens_proj[:, :0, :]

        return ground_tokens_proj, text_tokens_proj, text_attention_mask

    # ====================================================================
    # Forward -- Stage 1 only.
    # ----------------------------------------------------------------------------
    # Stage 2 no longer trains through a single joint forward() call -- it
    # uses encode_query_stage2() / encode_satellite_stage2() /
    # score_stage2_candidates() instead (see train.py / eval.py), since
    # Stage 2 needs one query scored against C candidate satellites rather
    # than a single 1:1 paired (q, r) forward pass.
    # ====================================================================
    def forward(self, q, r, t, stage=1, isTrain=True, isQuery=True, return_aux=True):
        """
        q: [B, C, H, W]  ground/query image tensor
        r: [B, C, H, W]  satellite/reference image tensor
        t: list[str]     text descriptions, one per ground image
        stage: kept for interface compatibility -- this method only
               implements Stage 1; anything else raises.
        isTrain, isQuery: kept for interface compatibility with the old model; unused internally
        return_aux: whether to populate / return the auxiliary dict

        Returns:
            query_embedding:     [B, D]  L2-normalized
            satellite_embedding: [B, D]  L2-normalized (global satellite embedding)
            aux: dict
        """
        if stage != 1:
            raise ValueError(
                "GeoDinoSiglipQFormer.forward() only implements Stage 1. "
                "Use encode_query_stage2/encode_satellite_stage2/score_stage2_candidates for Stage 2."
            )

        tm = self.trainable_modules
        aux = {"stage": stage, "shapes": {}}

        # ----------------------------------------------------------------
        # Query side: ground DINOv3 patches + text tokens -> Query Q-Former
        # ----------------------------------------------------------------
        ground_patches = self._extract_dino_patch_tokens(self.dino_ground, q)      # [B, Ng, dino_ground_dim]
        ground_patches_proj = tm["proj_ground"](ground_patches)                    # [B, Ng, D]

        text_tokens, text_mask = self._extract_text_tokens(t)                      # [B, Nt, text_dim]
        text_tokens_proj = tm["proj_text"](text_tokens)                            # [B, Nt, D]

        ground_patches_proj, text_tokens_proj, text_mask = self._apply_query_modality(
            ground_patches_proj, text_tokens_proj, text_mask
        )
        query_tokens, pooled_query = tm["query_qformer"](ground_patches_proj, text_tokens_proj, text_mask)
        query_embedding = F.normalize(tm["retrieval_proj_query"](pooled_query), dim=-1)  # [B, D]

        aux["shapes"]["ground_patches"] = tuple(ground_patches.shape)
        aux["shapes"]["text_tokens"] = tuple(text_tokens.shape)
        aux["shapes"]["query_tokens"] = tuple(query_tokens.shape)

        # ----------------------------------------------------------------
        # Satellite side: DINOv3 patches, global pooled embedding only
        # (Stage 1 has no patch-level objective).
        # ----------------------------------------------------------------
        sat_patches = self._extract_dino_patch_tokens(self.dino_satellite, r)      # [B, Np, dino_satellite_dim]
        aux["shapes"]["sat_patches"] = tuple(sat_patches.shape)

        sat_global = sat_patches.mean(dim=1)
        sat_global_emb = tm["satellite_global_head"](sat_global)
        satellite_embedding = F.normalize(
            tm["retrieval_proj_satellite"](sat_global_emb), dim=-1
        )

        if return_aux:
            aux.update({
                "query_tokens": F.normalize(query_tokens, dim=-1),
            })

        return query_embedding, satellite_embedding, aux

    # ====================================================================
    # Stage 2 encode-only methods
    # ----------------------------------------------------------------------------
    # Each method below computes the frozen Stage 1 encoder feature once
    # (ground DINOv3 patches + text tokens for the query side, satellite
    # DINOv3 patches for the satellite side) and feeds it through BOTH the
    # Stage 1 pathway (frozen, once freeze_stage1_for_stage2() has been
    # called) and the new trainable Stage 2 pathway, so the expensive
    # frozen-encoder forward pass is never duplicated.
    #
    # The Stage 1 output (first return value of each) remains the
    # frozen, Stage-1-style pooled/global embedding used for global
    # retrieval and hard-negative mining. The Stage 2 output (second
    # return value) is produced by the new trainable Stage 2 query /
    # satellite-patch pathway and is what feeds
    # score_stage2_candidates()'s region-voting matcher. See eval.py /
    # train.py for how these two are combined.
    # ====================================================================
    def encode_query_stage2(self, q, t):
        """
        Encodes ground+text only -- no satellite needed.

        Returns:
            query_global:        [B, D]      L2-normalized Stage 1 pooled query
                                              embedding (frozen once
                                              freeze_stage1_for_stage2() is
                                              called).
            stage2_query_tokens: [B, Nq, D]  L2-normalized Stage 2 Query
                                              Q-Former output tokens
                                              (trainable; warm-started from
                                              Stage 1, then trained
                                              independently).
        """
        tm = self.trainable_modules

        ground_patches = self._extract_dino_patch_tokens(self.dino_ground, q)
        text_tokens, text_mask = self._extract_text_tokens(t)

        # --- Stage 1 query path (frozen once Stage 1 -> Stage 2 freeze runs) ---
        ground_patches_proj = tm["proj_ground"](ground_patches)
        text_tokens_proj = tm["proj_text"](text_tokens)
        ground_patches_proj, text_tokens_proj, stage1_text_mask = self._apply_query_modality(
            ground_patches_proj, text_tokens_proj, text_mask
        )
        _, pooled_query = tm["query_qformer"](ground_patches_proj, text_tokens_proj, stage1_text_mask)
        query_global = F.normalize(tm["retrieval_proj_query"](pooled_query), dim=-1)

        # --- Stage 2 query path (trainable, warm-started from Stage 1) ---
        stage2_ground_patches_proj = tm["stage2_proj_ground"](ground_patches)
        stage2_text_tokens_proj = tm["stage2_proj_text"](text_tokens)
        stage2_ground_patches_proj, stage2_text_tokens_proj, stage2_text_mask = self._apply_query_modality(
            stage2_ground_patches_proj, stage2_text_tokens_proj, text_mask
        )
        stage2_query_tokens, _ = tm["stage2_query_qformer"](
            stage2_ground_patches_proj, stage2_text_tokens_proj, stage2_text_mask
        )
        stage2_query_tokens = F.normalize(stage2_query_tokens, dim=-1)

        return query_global, stage2_query_tokens

    def encode_satellite_stage2(self, r):
        """
        Encodes a satellite image only -- no query needed.

        Returns:
            sat_global:          [B, D]      L2-normalized Stage 1 global
                                              satellite embedding (frozen
                                              once freeze_stage1_for_stage2()
                                              is called).
            stage2_sat_patches:  [B, Np, D]  L2-normalized, stage2_proj_satellite
                                              -projected patch tokens (input to
                                              the Stage 2 region-voting matcher).
        """
        tm = self.trainable_modules

        sat_patches_raw = self._extract_dino_patch_tokens(self.dino_satellite, r)

        # --- Stage 1 global satellite path (frozen once Stage 2 begins) ---
        sat_global_raw = tm["satellite_global_head"](sat_patches_raw.mean(dim=1))
        sat_global = F.normalize(tm["retrieval_proj_satellite"](sat_global_raw), dim=-1)

        # --- Stage 2 satellite patch path (trainable) ---
        stage2_sat_patches_proj = tm["stage2_proj_satellite"](sat_patches_raw)
        stage2_sat_patches = F.normalize(stage2_sat_patches_proj, dim=-1)

        return sat_global, stage2_sat_patches

    # ====================================================================
    # Stage 2 candidate scoring -- region voting
    # ----------------------------------------------------------------------------
    # query_tokens is one query, broadcast across its C candidates.
    # candidate_sat_patches holds the positive (index 0) plus hard
    # negatives. Returns one scalar score and one spatial vote/region map
    # per candidate -- there is no pooled-embedding contrastive score here.
    # ====================================================================
    def score_stage2_candidates(self, query_tokens, candidate_sat_patches, return_maps=True):
        """
        query_tokens:          [B, Nq, D]
        candidate_sat_patches: [B, C, Np, D]

        Returns:
            candidate_scores: [B, C]
            vote_maps:        [B, C, grid, grid]  (None if return_maps=False)
            region_maps:      [B, C, grid, grid]  (None if return_maps=False)
        """
        tm = self.trainable_modules
        B, C, Np, D = candidate_sat_patches.shape
        Nq = query_tokens.shape[1]
        grid_size = int(round(Np ** 0.5))
        assert grid_size * grid_size == Np, f"Np={Np} is not a perfect square; can't reshape to a grid"

        flat_q = query_tokens[:, None, :, :].expand(B, C, Nq, D).reshape(B * C, Nq, D)
        flat_s = candidate_sat_patches.reshape(B * C, Np, D)

        region_voting_qformer = tm["stage2_region_voting_qformer"]
        updated_query_tokens, attention_map = region_voting_qformer(flat_q, flat_s)   # [B*C, Nq, D], [B*C, Nq, Np]

        vote_map_2d = region_voting_qformer.vote_map_from_attention(
            updated_query_tokens, attention_map, grid_size
        )                                                                    # [B*C, grid, grid]
        vote_map_2d = vote_map_2d.view(B, C, grid_size, grid_size)

        # Local region pooling -- sum votes inside a region_window_size x
        # region_window_size window via a stride-1 ones-kernel conv (keeps
        # the full grid resolution), so a vote merely adjacent to the true
        # patch still contributes -- a single isolated 1-patch peak isn't
        # required to win.
        window = self.region_window_size
        kernel = torch.ones(1, 1, window, window, device=vote_map_2d.device, dtype=vote_map_2d.dtype)
        padded = vote_map_2d.view(B * C, 1, grid_size, grid_size)
        region_map = F.conv2d(padded, kernel, padding=window // 2)
        region_map = region_map.view(B, C, grid_size, grid_size)

        if self.region_pooling == "max":
            candidate_scores = region_map.flatten(2).max(dim=-1).values
        elif self.region_pooling == "mean":
            candidate_scores = region_map.flatten(2).mean(dim=-1)
        else:
            raise ValueError(f"Unsupported region_pooling: {self.region_pooling!r}")

        if return_maps:
            return candidate_scores, vote_map_2d, region_map
        return candidate_scores, None, None

    # ====================================================================
    # Stage 1 -> Stage 2 warm start
    # ----------------------------------------------------------------------------
    # One-time copy of Stage 1's query-pathway weights into the new Stage 2
    # query pathway. This is a snapshot COPY via load_state_dict, not weight
    # tying -- stage2_proj_ground/stage2_proj_text/stage2_query_qformer were
    # already constructed as separate parameter tensors in __init__, so after
    # this call the two pathways hold independent values that can (and will,
    # once training proceeds) diverge from each other.
    #
    # Call this once, right before freeze_stage1_for_stage2(): either after
    # Stage 1 finishes naturally in this run, or immediately after loading an
    # old Stage 1 checkpoint via --resume-from-stage1.
    # ====================================================================
    def initialize_stage2_from_stage1(self):
        """
        Copies:
            proj_ground   -> stage2_proj_ground
            proj_text     -> stage2_proj_text
            query_qformer -> stage2_query_qformer

        Raises a RuntimeError with a clear message if the source and target
        shapes/keys don't match -- this would most likely mean
        cfg.model.stage2_query_qformer was configured with a different
        architecture than cfg.model.query_qformer, which breaks the "exact
        architectural copy" warm-start this method is meant to perform.
        """
        tm = self.trainable_modules
        warm_start_pairs = (
            ("proj_ground", "stage2_proj_ground"),
            ("proj_text", "stage2_proj_text"),
            ("query_qformer", "stage2_query_qformer"),
        )
        for src_name, dst_name in warm_start_pairs:
            src_state_dict = tm[src_name].state_dict()
            try:
                tm[dst_name].load_state_dict(src_state_dict, strict=True)
            except RuntimeError as e:
                raise RuntimeError(
                    f"initialize_stage2_from_stage1(): could not warm-start '{dst_name}' from "
                    f"'{src_name}' -- shape/key mismatch. Is cfg.model.stage2_query_qformer "
                    f"architecturally identical to cfg.model.query_qformer? Original error: {e}"
                ) from e

    # ====================================================================
    # Stage 1 -> Stage 2 freezing
    # ====================================================================
    def freeze_stage1_for_stage2(self):
        """
        Freezes the Stage 1 query/satellite global pathway so Stage 2 trains
        only its own independent modules: stage2_proj_ground,
        stage2_proj_text, stage2_query_qformer, stage2_proj_satellite, and
        stage2_region_voting_qformer. Idempotent -- safe to call more than
        once.

        NOTE: resume-into-Stage-2 (loading a checkpoint that already has
        these frozen, then continuing Stage 2 training) is out of scope --
        load_checkpoint_for_resume() does not currently call this method or
        rebuild the optimizer before loading state.
        """
        stage1_module_names = (
            "proj_ground", "proj_text", "query_qformer",
            "satellite_global_head", "retrieval_proj_query", "retrieval_proj_satellite",
        )
        for name in stage1_module_names:
            for p in self.trainable_modules[name].parameters():
                p.requires_grad = False
            self.trainable_modules[name].eval()

    # ====================================================================
    # Stage 2 modules: kept frozen until Stage 2 actually begins
    # ----------------------------------------------------------------------------
    # The Stage 2 modules (stage2_proj_ground, stage2_proj_text,
    # stage2_query_qformer, stage2_proj_satellite, stage2_region_voting_qformer)
    # live in the same trainable_modules ModuleDict as the Stage 1 ones from
    # construction onward, so without this they would default to
    # requires_grad=True and get swept into the Stage 1 optimizer's param
    # group even though Stage 1 never produces a forward/backward pass
    # through them. They likely wouldn't receive gradients in practice (no
    # Stage 1 forward path touches them), but they would still inflate
    # count_trainable_parameters()/get_trainable_parameter_names() and make
    # the Stage 1 optimizer's param groups misleading. Call
    # freeze_stage2_until_stage2() right after construction (see
    # run_training()) and unfreeze_stage2_for_stage2() at the Stage 1 ->
    # Stage 2 transition, alongside freeze_stage1_for_stage2().
    # ====================================================================
    def freeze_stage2_until_stage2(self):
        for name in (
            "stage2_proj_ground",
            "stage2_proj_text",
            "stage2_query_qformer",
            "stage2_proj_satellite",
            "stage2_region_voting_qformer",
        ):
            for p in self.trainable_modules[name].parameters():
                p.requires_grad = False
            self.trainable_modules[name].eval()

    def unfreeze_stage2_for_stage2(self):
        for name in (
            "stage2_proj_ground",
            "stage2_proj_text",
            "stage2_query_qformer",
            "stage2_proj_satellite",
            "stage2_region_voting_qformer",
        ):
            for p in self.trainable_modules[name].parameters():
                p.requires_grad = True

    # ====================================================================
    # Checkpointing helpers -- trainable weights only
    # ----------------------------------------------------------------------------
    # get_trainable_state_dict() always returns ALL trainable_modules
    # submodules (it calls .state_dict() on the whole ModuleDict, which has
    # nothing to do with requires_grad) -- so Stage 2 checkpoints still
    # contain the frozen Stage 1 weights too, not just the actively-training
    # Stage 2 modules. That's intentional: a Stage 2 checkpoint needs the
    # Stage 1 query-side weights to be usable standalone for eval/resume,
    # not just whatever happens to be unfrozen at the time.
    #
    # get_trainable_parameter_names() is different on purpose -- it's a
    # log-only display helper (see print_trainable_summary() in train.py),
    # so it DOES filter on requires_grad, to show only what's currently
    # being optimized at this point in training (shrinks after
    # freeze_stage1_for_stage2() is called).
    # ====================================================================
    def get_trainable_state_dict(self):
        return self.trainable_modules.state_dict()

    def load_trainable_state_dict(self, state_dict, strict=True):
        """
        strict=True (default): used for new Stage 2 checkpoints, which
        contain every trainable_modules key.
        strict=False: used for loading an old Stage 1 checkpoint (e.g. via
        --resume-from-stage1) that predates the stage2_* modules -- missing
        keys for those new modules are expected in that case.

        Returns whatever nn.Module.load_state_dict() returns (an
        _IncompatibleKeys namedtuple with .missing_keys / .unexpected_keys),
        so callers using strict=False can inspect/log what was missing.
        """
        return self.trainable_modules.load_state_dict(state_dict, strict=strict)

    def get_trainable_parameter_names(self):
        return [
            f"trainable_modules.{n}"
            for n, p in self.trainable_modules.named_parameters()
            if p.requires_grad
        ]

    def get_frozen_parameter_names(self):
        names = []
        names += [f"dino_ground.{n}" for n, _ in self.dino_ground.named_parameters()]
        names += [f"dino_satellite.{n}" for n, _ in self.dino_satellite.named_parameters()]
        names += [f"text_model.{n}" for n, _ in self.text_model.named_parameters()]
        return names

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.trainable_modules.parameters() if p.requires_grad)

    def count_frozen_parameters(self):
        frozen = (
            list(self.dino_ground.parameters())
            + list(self.dino_satellite.parameters())
            + list(self.text_model.parameters())
        )
        return sum(p.numel() for p in frozen)
