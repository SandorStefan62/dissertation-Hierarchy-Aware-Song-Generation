import torch
import torch.nn as nn


class HierarchicalConditioner(nn.Module):
    """
    Projects pre-computed hierarchical features into a sequence of conditioning
    tokens for MusicGen's cross-attention.

    Receives three scales of MuQ embeddings extracted over a full song:
        global      [B,  1, n_layers, muq_dim] - whole-song mean pool
        contextual  [B, 12, n_layers, muq_dim] - 12 x 30s windows
        local       [B, 36, n_layers, muq_dim] - 36 x 10s windows

    Produces a single conditioning sequence [B, 49, output_dim] with a validity mask.

    Architecture per token:
        1. Learned weighted sum over n_layers > [B, W, muq_dim]
        2. LayerNorm (stabilises raw MuQ hidden states)
        3. + scale type embedding (global / contextual / local)
        4. (optional) + window-offset embedding (-11 ... +11 contextual steps)  # after some experiments, this proved to be not that useful
        5. Shared MLP: muq_dim -> hidden_dim -> output_dim

    The mask marks contextual/local tokens whose window starts beyond the song's
    actual duration as invalid, so the LM's cross-attention ignores padded windows.

    Args:
        n_layers:       number of MuQ layers kept during extraction (default 8)
        muq_dim:        MuQ hidden state dimension (always 1024)
        hidden_dim:     MLP hidden dimension (default 1024)
        output_dim:     conditioning token dimension fed to MusicGen cross-attention
                        - must match MusicGen's internal dim:
                          small: 1024, medium: 1536, large: 2048
        use_offset_emb: add a window-offset positional embedding that encodes how
                        far each token is from the current generation window.
                        Zero-initialised so it starts as a no-op. (default False)
    """

    N_GLOBAL     = 1
    N_CONTEXTUAL = 12
    N_LOCAL      = 36
    N_TOKENS     = N_GLOBAL + N_CONTEXTUAL + N_LOCAL  # 49

    LOCAL_WINDOW_S = 10.0
    CTX_WINDOW_S   = 30.0

    SCALE_GLOBAL      = 0
    SCALE_CONTEXTUAL  = 1
    SCALE_LOCAL       = 2

    # Window-offset embedding range: -11 to +11 contextual steps (30s each).
    N_OFFSETS    = 2 * (N_CONTEXTUAL - 1) + 1   # 23  (-11 … 0 … +11)
    OFFSET_SHIFT = N_CONTEXTUAL - 1              # 11  (add to offset → embedding index)

    def __init__(
        self,
        n_layers:       int  = 8,
        muq_dim:        int  = 1024,
        hidden_dim:     int  = 1024,
        output_dim:     int  = 1024,
        use_offset_emb: bool = False,
    ):
        super().__init__()
        self.n_layers       = n_layers
        self.muq_dim        = muq_dim
        self.output_dim     = output_dim
        self.use_offset_emb = use_offset_emb

        # learned softmax weights over the n_layers axis.
        # initialised to zero → uniform at the start (softmax(0…0) = 1/n).
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

        # normalise raw activations before projection.
        self.layer_norm = nn.LayerNorm(muq_dim)

        # scale-type embedding: 0=global, 1=contextual, 2=local.
        self.scale_emb = nn.Embedding(3, muq_dim)

        if use_offset_emb:
            # window-offset embedding: encodes how far each token is from the
            # current 30s generation window, in 30s steps (-11 … +11).
            # Zero-init: starts as a no-op so it doesn't disturb the pretrained
            # scale_emb + MLP path at the start of training.
            self.offset_emb = nn.Embedding(self.N_OFFSETS, muq_dim)
            # nn.init.zeros_(self.offset_emb.weight)

        # shared MLP applied identically to every token.
        self.mlp = nn.Sequential(
            nn.Linear(muq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _pool_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted combination over the n_layers axis.

        x: [B, W, n_layers, muq_dim]
        returns: [B, W, muq_dim]
        """
        weights = torch.softmax(self.layer_weights, dim=0)  # [n_layers]
        return (x * weights[None, None, :, None]).sum(dim=2)

    def _build_mask(self, duration_s: torch.Tensor) -> torch.Tensor:
        """Boolean mask [B, 49]: True = valid token, False = beyond song duration.

        Global token is always valid.
        Contextual token i is valid when duration_s > i * CTX_WINDOW_S.
        Local token i is valid when duration_s > i * LOCAL_WINDOW_S.
        """
        B      = duration_s.shape[0]
        device = duration_s.device
        mask   = torch.ones(B, self.N_TOKENS, dtype=torch.bool, device=device)

        # contextual
        ctx_starts = torch.arange(self.N_CONTEXTUAL, device=device) * self.CTX_WINDOW_S
        ctx_valid  = duration_s.unsqueeze(1) > ctx_starts.unsqueeze(0)   # [B, 12]
        mask[:, self.N_GLOBAL : self.N_GLOBAL + self.N_CONTEXTUAL] = ctx_valid

        # local
        loc_starts = torch.arange(self.N_LOCAL, device=device) * self.LOCAL_WINDOW_S
        loc_valid  = duration_s.unsqueeze(1) > loc_starts.unsqueeze(0)   # [B, 36]
        mask[:, self.N_GLOBAL + self.N_CONTEXTUAL :] = loc_valid

        return mask

    def _build_offset_indices(
        self, window_start_s: torch.Tensor  # [B]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Offset embedding indices for global, contextual, local tokens.

        Offset = token's contextual-window index minus the current window index.
        Clamped to [-N_CONTEXTUAL+1, N_CONTEXTUAL-1], shifted to [0, N_OFFSETS-1].
        """
        B      = window_start_s.shape[0]
        device = window_start_s.device

        cur_ctx = (window_start_s / self.CTX_WINDOW_S).long()               # [B]

        ctx_i   = torch.arange(self.N_CONTEXTUAL, device=device)            # [12]
        c_off   = ctx_i.unsqueeze(0) - cur_ctx.unsqueeze(1)                 # [B, 12]
        c_off   = c_off.clamp(-(self.N_CONTEXTUAL - 1), self.N_CONTEXTUAL - 1)
        c_idx   = c_off + self.OFFSET_SHIFT                                 # [B, 12]

        loc_i   = torch.arange(self.N_LOCAL, device=device) // 3            # [36]
        l_off   = loc_i.unsqueeze(0) - cur_ctx.unsqueeze(1)                 # [B, 36]
        l_off   = l_off.clamp(-(self.N_CONTEXTUAL - 1), self.N_CONTEXTUAL - 1)
        l_idx   = l_off + self.OFFSET_SHIFT                                 # [B, 36]

        g_idx   = torch.full((B, 1), self.OFFSET_SHIFT, device=device, dtype=torch.long)

        return g_idx, c_idx, l_idx

    def forward(
        self,
        local:          torch.Tensor,  # [B, 36, n_layers, muq_dim]
        contextual:     torch.Tensor,  # [B, 12, n_layers, muq_dim]
        global_emb:     torch.Tensor,  # [B,  1, n_layers, muq_dim]
        duration_s:     torch.Tensor,  # [B]
        window_start_s: torch.Tensor,  # [B] start of the current generation window
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens  [B, 49, output_dim]
            mask    [B, 49] bool - False for tokens beyond the song's real duration
        """
        device = local.device

        g = self._pool_layers(global_emb) # [B,  1, muq_dim]
        c = self._pool_layers(contextual) # [B, 12, muq_dim]
        l = self._pool_layers(local)      # [B, 36, muq_dim]

        g = self.layer_norm(g)
        c = self.layer_norm(c)
        l = self.layer_norm(l)

        g = g + self.scale_emb(torch.tensor(self.SCALE_GLOBAL,     device=device))
        c = c + self.scale_emb(torch.tensor(self.SCALE_CONTEXTUAL, device=device))
        l = l + self.scale_emb(torch.tensor(self.SCALE_LOCAL,      device=device))

        if self.use_offset_emb:
            g_idx, c_idx, l_idx = self._build_offset_indices(window_start_s)
            g = g + self.offset_emb(g_idx)
            c = c + self.offset_emb(c_idx)
            l = l + self.offset_emb(l_idx)

        tokens = torch.cat([g, c, l], dim=1)    # [B, 49, muq_dim]
        tokens = self.mlp(tokens)               # [B, 49, output_dim]
        mask   = self._build_mask(duration_s)

        return tokens, mask


if __name__ == "__main__":
    from torchinfo import summary

    dummy_local        = torch.rand(1, 36, 8, 1024)
    dummy_contextual   = torch.rand(1, 12, 8, 1024)
    dummy_global       = torch.rand(1,  1, 8, 1024)
    dummy_duration     = torch.tensor([200.0])
    dummy_window_start = torch.tensor([60.0])

    for use_offset in [False, True]:
        print(f"\nuse_offset_emb={use_offset}")
        model = HierarchicalConditioner(use_offset_emb=use_offset)
        summary(
            model,
            input_data=(dummy_local, dummy_contextual, dummy_global, dummy_duration, dummy_window_start),
            depth=3,
            col_names=["input_size", "output_size", "num_params", "trainable"]
        )