import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel


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
    model = AutoModel.from_pretrained(checkpoint_name)
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
# Frozen Stage 1 query tokens cross-attend to candidate satellite patch
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
# contrastive design. Stage 1 query tokens are frozen by the time this module
# is trained (see GeoDinoSiglipQFormer.freeze_stage1_for_stage2) -- this
# module only learns to project satellite patches and locate, via attention,
# the local region inside a candidate satellite that the query is consistent
# with. Output is a per-candidate scalar score plus a spatial vote map, not
# a pooled embedding -- there is no embedding-level contrastive loss here;
# scoring happens entirely through RegionVotingLoss on candidate_scores /
# region_maps (see losses/region_voting.py).
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
    Stage 1 trainable: Query Q-Former, projection layers, satellite global
               pooling/projection head, final retrieval projection layers.
    Stage 2 trainable: proj_satellite, Satellite Region Voting Former --
               everything else (including the Stage 1 modules above) is
               frozen via freeze_stage1_for_stage2() once Stage 2 begins.
    All trainable modules are grouped under self.trainable_modules so they
    can be saved/loaded independently of the frozen encoders.
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
        sd_cfg = m.stage2_satellite_qformer

        self.trainable_modules = nn.ModuleDict({
            "proj_ground": nn.Linear(dino_ground_dim, embed_dim),
            "proj_text": nn.Linear(text_dim, embed_dim),
            "proj_satellite": nn.Linear(dino_satellite_dim, embed_dim),

            "query_qformer": QueryQFormer(
                dim=embed_dim,
                num_query_tokens=qq_cfg.num_query_tokens,
                num_layers=qq_cfg.num_layers,
                num_heads=qq_cfg.num_heads,
                ffn_dim=qq_cfg.ffn_dim,
                dropout=qq_cfg.dropout,
            ),

            "stage2_satellite_qformer": SatelliteRegionVotingFormer(
                dim=embed_dim,
                num_layers=sd_cfg.num_layers,
                num_heads=sd_cfg.num_heads,
                ffn_dim=sd_cfg.ffn_dim,
                dropout=sd_cfg.dropout,
                use_token_gating=getattr(sd_cfg, "use_token_gating", True),
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
        with torch.no_grad():
            outputs = dino_model(pixel_values=images)
            hidden = outputs.last_hidden_state  # [B, 1 + num_register + Np, D]

        start = 0
        if self.remove_cls_token:
            start += 1
        if self.remove_register_tokens:
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
    # Query tokens here come from the Stage 1 Query Q-Former, frozen once
    # freeze_stage1_for_stage2() has been called. Satellite patches are
    # projected (proj_satellite) but otherwise raw -- the Satellite Region
    # Voting Former itself (not this method) does the cross-attention work.
    # See eval.py / train.py for how these two are combined with
    # score_stage2_candidates().
    # ====================================================================
    def encode_query_stage2(self, q, t):
        """
        Encodes ground+text only -- no satellite needed.

        Returns:
            query_global:  [B, D]      L2-normalized Stage 1-style pooled query embedding
            query_tokens:  [B, Nq, D]  L2-normalized Stage 1 Query Q-Former output tokens
        """
        tm = self.trainable_modules

        ground_patches = self._extract_dino_patch_tokens(self.dino_ground, q)
        ground_patches_proj = tm["proj_ground"](ground_patches)

        text_tokens, text_mask = self._extract_text_tokens(t)
        text_tokens_proj = tm["proj_text"](text_tokens)

        query_tokens, pooled_query = tm["query_qformer"](ground_patches_proj, text_tokens_proj, text_mask)
        query_global = F.normalize(tm["retrieval_proj_query"](pooled_query), dim=-1)
        query_tokens = F.normalize(query_tokens, dim=-1)

        return query_global, query_tokens

    def encode_satellite_stage2(self, r):
        """
        Encodes a satellite image only -- no query needed.

        Returns:
            sat_global:  [B, D]      L2-normalized Stage 1-style global satellite embedding
            sat_patches: [B, Np, D]  L2-normalized, proj_satellite-projected patch tokens
                                     (input to the Satellite Region Voting Former)
        """
        tm = self.trainable_modules

        sat_patches_raw = self._extract_dino_patch_tokens(self.dino_satellite, r)
        sat_global_raw = tm["satellite_global_head"](sat_patches_raw.mean(dim=1))
        sat_global = F.normalize(tm["retrieval_proj_satellite"](sat_global_raw), dim=-1)

        sat_patches_proj = tm["proj_satellite"](sat_patches_raw)
        sat_patches = F.normalize(sat_patches_proj, dim=-1)

        return sat_global, sat_patches

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

        sat_qformer = tm["stage2_satellite_qformer"]
        updated_query_tokens, attention_map = sat_qformer(flat_q, flat_s)   # [B*C, Nq, D], [B*C, Nq, Np]

        vote_map_2d = sat_qformer.vote_map_from_attention(
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
    # Stage 1 -> Stage 2 freezing
    # ====================================================================
    def freeze_stage1_for_stage2(self):
        """
        Freezes the Stage 1 query branch so Stage 2 trains only
        proj_satellite + stage2_satellite_qformer. Idempotent -- safe to
        call more than once.

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
    # Checkpointing helpers -- trainable weights only
    # ----------------------------------------------------------------------------
    # get_trainable_state_dict() always returns ALL eight trainable_modules
    # submodules (it calls .state_dict() on the whole ModuleDict, which has
    # nothing to do with requires_grad) -- so Stage 2 checkpoints still
    # contain the frozen Stage 1 weights too, not just proj_satellite /
    # stage2_satellite_qformer. That's intentional: a Stage 2 checkpoint
    # needs the Stage 1 query-side weights to be usable standalone for
    # eval/resume, not just the two modules actively training at the time.
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
        self.trainable_modules.load_state_dict(state_dict, strict=strict)

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