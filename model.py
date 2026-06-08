import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import config
from mamba_ssm import Mamba2


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding.

    Encodes absolute position so attention pooling knows
    where each token is relative to the sequence center.
    """

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class ChromaRegressor(nn.Module):
    """Cell-line-specific histone mark predictor.

    Architecture: CNN → Cell-Conditioned BiMamba → Cross-Attention → Per-Mark Heads

    Key design: RNA (cell-line fingerprint) is injected at THREE levels:
      1. Per-layer FiLM before each Mamba layer  (cell-type-aware sequence modeling)
      2. Cross-attention after Mamba              (position-specific cell conditioning)
      3. Concatenation at regression heads        (direct cell-type signal to output)
    """

    def __init__(self):
        super().__init__()
        num_rna_tokens = getattr(config, "NUM_RNA_TOKENS", 8)
        cross_attn_heads = getattr(config, "CROSS_ATTN_HEADS", 4)

        # ── 1. DNA Encoder (Multi-Resolution CNN Stem) ──────────────────────
        # Branch A: motif scale
        self.cnn_a = nn.Sequential(
            nn.Conv1d(4, 128, 15, stride=1, padding=7), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 9, stride=4, padding=4), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, 9, stride=4, padding=4), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, 5, stride=4, padding=2), nn.BatchNorm1d(256), nn.GELU(),
        )
        # Branch B: nucleosome scale
        self.cnn_b = nn.Sequential(
            nn.Conv1d(4, 128, 15, stride=1, padding=7), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 15, stride=8, padding=7), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, 9, stride=8, padding=4), nn.BatchNorm1d(256), nn.GELU(),
        )
        # Branch C: domain scale
        self.cnn_c = nn.Sequential(
            nn.Conv1d(4, 64, 31, stride=4, padding=15), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 15, stride=4, padding=7), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 9, stride=4, padding=4), nn.BatchNorm1d(256), nn.GELU(),
        )
        # Merge projection
        self.cnn_merge = nn.Sequential(
            nn.Conv1d(768, config.D_MODEL, 1), nn.BatchNorm1d(config.D_MODEL), nn.GELU()
        )

        # ── 2. RNA Encoder (deeper MLP) ─────────────────────────────────
        # 4000 gene expression values → rich cell-type embedding.
        # Three layers instead of two: the extra capacity helps compress
        # the high-dimensional RNA space into a truly informative embedding.
        self.rna_mlp = nn.Sequential(
            nn.Linear(config.RNA_INPUT_DIM, 1024),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(1024, config.RNA_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.RNA_HIDDEN_DIM, config.RNA_HIDDEN_DIM),
        )

        # ── 3. Per-Layer FiLM Conditioning ──────────────────────────────
        # Each Mamba layer gets its own γ, β so that cell-type modulates
        # DNA features BEFORE sequence modeling - not just once at the end.
        # Layer 1 learns coarse cell-type patterns; layer N learns fine-
        # grained position-specific effects.
        self.film_gammas = nn.ModuleList([
            nn.Linear(config.RNA_HIDDEN_DIM, config.D_MODEL)
            for _ in range(config.N_LAYERS)
        ])
        self.film_betas = nn.ModuleList([
            nn.Linear(config.RNA_HIDDEN_DIM, config.D_MODEL)
            for _ in range(config.N_LAYERS)
        ])

        # ── 4. Bidirectional Mamba-2 SSM ────────────────────────────────
        # DNA is symmetric: a peak at position X influences features both
        # upstream and downstream. Forward + backward scans give full
        # bidirectional context at O(L) cost.
        self.mamba_fwd = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.D_MODEL),
                Mamba2(
                    d_model=config.D_MODEL,
                    d_state=getattr(config, "D_STATE", 32),
                    d_conv=getattr(config, "D_CONV", 4),
                    expand=getattr(config, "MAMBA_EXPAND", 2),
                ),
            )
            for _ in range(config.N_LAYERS)
        ])
        self.mamba_bwd = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.D_MODEL),
                Mamba2(
                    d_model=config.D_MODEL,
                    d_state=getattr(config, "D_STATE", 32),
                    d_conv=getattr(config, "D_CONV", 4),
                    expand=getattr(config, "MAMBA_EXPAND", 2),
                ),
            )
            for _ in range(config.N_LAYERS)
        ])
        self.mamba_norm = nn.LayerNorm(config.D_MODEL)

        # ── 5. Cross-Attention: Position-Specific Cell Conditioning ─────
        # FiLM applies the SAME γ/β to every position. Cross-attention lets
        # each DNA position query the RNA independently:
        #   - A CpG-island position attends to "developmental genes"
        #   - A repeat-element position attends to "heterochromatin genes"
        # This is genuine position × cell-type interaction.
        self.num_rna_tokens = num_rna_tokens
        self.rna_to_tokens = nn.Linear(
            config.RNA_HIDDEN_DIM, num_rna_tokens * config.D_MODEL
        )
        self.cross_attn = nn.MultiheadAttention(
            config.D_MODEL,
            num_heads=cross_attn_heads,
            batch_first=True,
            dropout=config.DROPOUT,
        )
        self.cross_attn_norm = nn.LayerNorm(config.D_MODEL)

        # ── 5b. Sinusoidal Positional Encoding ─────────────────────────────
        # Mamba has implicit positional info from sequential scanning, but the
        # downstream attention pooling is permutation-invariant. Adding explicit
        # PE after Mamba+cross-attention lets the pooling head know where
        # each token is relative to the center of the window.
        self.pos_enc = SinusoidalPE(config.D_MODEL, max_len=4096)

        # ── 6. Per-Mark Attention Pooling ───────────────────────────────
        # Each mark has its own attention module + query because:
        #   - H3K4me3 needs narrow 4K context (sharp promoter peaks)
        #   - H3K27me3 needs full 32K context (broad repressive domains)
        self.pool_attns = nn.ModuleList([
            nn.MultiheadAttention(
                config.D_MODEL, num_heads=1, batch_first=True
            )
            for _ in range(config.NUM_MARKS)
        ])
        self.mark_queries = nn.Parameter(
            torch.randn(config.NUM_MARKS, 1, config.D_MODEL)
        )

        # ── 7. RNA Head Projection ───────────────────────────────────────
        # Project RNA embedding to D_MODEL space so it matches the scale
        # and distribution of the DNA pooled features before concatenation.
        self.rna_head_proj = nn.Sequential(
            nn.Linear(config.RNA_HIDDEN_DIM, config.D_MODEL),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
        )

        # ── 8. Independent Per-Mark Regression Heads ────────────────────
        # Each mark gets its own full MLP (no shared layer) so that
        # activating marks (H3K27ac, H3K4me1) don't interfere with
        # repressive marks (H3K27me3, H3K9me3) during training.
        self.mark_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.D_MODEL * 2, config.HEAD_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.HEAD_HIDDEN_DIM, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(config.NUM_MARKS)
        ])

        # ── 9. Track Head (Auxiliary Peak Localisation) ────────────────────
        # RNA FiLM gate: RNA selects which DNA features matter for this cell type
        self.track_rna_film = nn.Linear(config.D_MODEL, config.D_MODEL * 2)
        
        # A single 1×1 convolution over the post-Mamba token sequence.
        # Input:  [B, D_MODEL, NUM_BINS]   (CNN token features, already at 64bp res)
        # Output: [B, NUM_MARKS, NUM_BINS] (logit per bin per mark)
        # No pooling - preserves full spatial resolution.
        # Sigmoid is applied in the loss, not here, so we can use BCEWithLogitsLoss.
        self.track_head = nn.Conv1d(
            config.D_MODEL, config.NUM_MARKS, kernel_size=1
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _context_token_bounds(self, mark_name, seq_len):
        context_bp = getattr(config, "MARK_CONTEXT_BP", {}).get(
            mark_name, config.WINDOW_SIZE
        )
        context_bp = max(1, min(int(context_bp), int(config.WINDOW_SIZE)))
        context_tokens = max(
            1, int(round(seq_len * context_bp / config.WINDOW_SIZE))
        )

        center = seq_len // 2
        half = context_tokens // 2
        start = max(0, center - half)
        end = min(seq_len, start + context_tokens)
        start = max(0, end - context_tokens)
        return start, end

    def _pool_mark_features(self, dna_feat):
        batch_size, seq_len, _ = dna_feat.shape
        pooled = []

        for m_idx, mark_name in enumerate(config.TARGET_MARKS):
            start, end = self._context_token_bounds(mark_name, seq_len)
            mark_tokens = dna_feat[:, start:end, :]
            q = self.mark_queries[m_idx].unsqueeze(0).expand(batch_size, -1, -1)
            attn_out, _ = self.pool_attns[m_idx](
                query=q, key=mark_tokens, value=mark_tokens
            )
            pooled.append(attn_out.squeeze(1))

        return pooled

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, dna_seq, rna_feat):
        # dna_seq:  [B, 4, L] (if float) or [B, L] (if uint8/long indices)
        # rna_feat: [B, RNA_INPUT_DIM]
        # Returns:  (scalar [B, NUM_MARKS], track_logits [B, NUM_MARKS, NUM_BINS])
        B = dna_seq.size(0)

        # ─── GPU-Accelerated One-Hot Encoding ───
        if dna_seq.dtype == torch.uint8 or dna_seq.dtype == torch.long:
            import torch.nn.functional as F
            dna_seq = F.one_hot(dna_seq.long(), num_classes=5)[:, :, :4].permute(0, 2, 1).to(torch.float32)

        # ─── RNA Branch ───
        rna_emb = self.rna_mlp(rna_feat)  # [B, RNA_HIDDEN_DIM]

        # ─── DNA Branch - Multi-Resolution CNN ───
        a = self.cnn_a(dna_seq)   # [B, 256, 512]
        b = self.cnn_b(dna_seq)   # [B, 256, 512]
        c = self.cnn_c(dna_seq)   # [B, 256, 512]
        dna_feat = self.cnn_merge(torch.cat([a, b, c], dim=1))  # [B, 256, 512]
        dna_feat = dna_feat.transpose(1, 2)    # [B, 512, D_MODEL]

        # ─── Cell-Type-Aware Bidirectional Mamba ───
        for i, (fwd_layer, bwd_layer) in enumerate(
            zip(self.mamba_fwd, self.mamba_bwd)
        ):
            gamma = self.film_gammas[i](rna_emb).unsqueeze(1)  # [B, 1, D]
            beta = self.film_betas[i](rna_emb).unsqueeze(1)    # [B, 1, D]
            dna_cond = (1 + gamma) * dna_feat + beta

            fwd_out = checkpoint(fwd_layer, dna_cond, use_reentrant=False)
            bwd_out = checkpoint(
                bwd_layer, dna_cond.flip(1), use_reentrant=False
            ).flip(1)
            dna_feat = dna_feat + fwd_out + bwd_out

        dna_feat = self.mamba_norm(dna_feat)  # [B, NUM_BINS, D_MODEL]

        # ─── Cross-Attention: Position-Specific Cell Conditioning ───
        rna_tokens = self.rna_to_tokens(rna_emb).view(
            B, self.num_rna_tokens, -1
        )  # [B, K, D_MODEL]
        cross_out, _ = self.cross_attn(
            query=dna_feat, key=rna_tokens, value=rna_tokens
        )
        dna_feat = self.cross_attn_norm(dna_feat + cross_out)  # [B, NUM_BINS, D_MODEL]

        # ─── RNA Projection for Heads & Tracks ───
        rna_proj = self.rna_head_proj(rna_emb)  # [B, D_MODEL]

        # ─── Track Head with RNA FiLM ───
        # Modulate the spatial DNA features based on cell type before tracking
        film_weights = self.track_rna_film(rna_proj)      # [B, 2*D_MODEL]
        gamma, beta = film_weights.chunk(2, dim=-1)       # each [B, D_MODEL]
        gamma = gamma.unsqueeze(1)                        # [B, 1, D_MODEL]
        beta = beta.unsqueeze(1)                          # [B, 1, D_MODEL]
        
        # Apply FiLM to the spatial features
        track_feat = dna_feat * (1 + gamma) + beta        # [B, NUM_BINS, D_MODEL]
        
        # Predict spatial tracks
        track_logits = self.track_head(
            track_feat.transpose(1, 2)
        )  # [B, NUM_MARKS, NUM_BINS]

        # ─── Positional Encoding for Pooling ───
        dna_feat = self.pos_enc(dna_feat)  # [B, NUM_BINS, D_MODEL]

        # ─── Scalar Multitask Heads ───
        scalar_outputs = []
        pooled_feats = self._pool_mark_features(dna_feat)
        for mark_feat, head in zip(pooled_feats, self.mark_heads):
            combined = torch.cat([mark_feat, rna_proj], dim=1)  # [B, 2*D_MODEL]
            scalar_outputs.append(head(combined))

        scalar = torch.cat(scalar_outputs, dim=1)  # [B, NUM_MARKS]
        return scalar, track_logits                 # track_logits: [B, NUM_MARKS, NUM_BINS]
