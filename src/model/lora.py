"""
LoRA for audiocraft's StreamingMultiheadAttention.

audiocraft uses a fused in_proj_weight [3*dim, dim] (raw nn.Parameter, not
nn.Linear), so PEFT's standard target_modules approach won't find it.

This module uses forward hooks on the cross-attention module to add the LoRA
delta directly to the projected k/v outputs, without touching in_proj_weight:

    # standard projection (frozen):
    k = F.linear(key, W_k)                    # [B, T, dim]
    # LoRA delta added in hook:
    k = k + (key @ A_k.T) @ B_k.T * scaling   # two small GEMMs on [B, T, rank]

This is faster than the parametrize approach, which materialises the full
[3*dim, dim] weight matrix on every forward pass.

Usage
```python
    from src.model.lora import apply_lora, lora_params, lora_state_dict, load_lora_state_dict

    for p in lm.parameters():
        p.requires_grad_(False)

    apply_lora(lm, rank=16, alpha=16, targets=("k", "v"))

    params = list(conditioner.parameters()) + lora_params(lm)
    optimizer = AdamW(params, lr=1e-4)

    state = lora_state_dict(lm)
    load_lora_state_dict(lm, state)
```
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionLoRA(nn.Module):
    """LoRA adapter for one StreamingMultiheadAttention (cross-attention).

    Registered as a forward hook. On each forward call it adds the LoRA
    delta directly to the q/k/v outputs rather than materialising W + ΔW.

    For target 't' in {q, k, v}:
        delta_t = (x_t @ A_t.T) @ B_t.T * scaling   shape [B, T, dim]

    A: [rank, dim]  kaiming_uniform init (same as nn.Linear)
    B: [dim, rank]  zero init delta = 0 at start, no disturbance to pretrained weights

    Args:
        dim:     embedding dimension (1024 for musicgen-small)
        rank:    LoRA rank
        alpha:   LoRA alpha; effective scaling = alpha / rank
        targets: which projections to adapt, subset of ('q', 'k', 'v')
    """

    TARGET_SLICES = {"q": 0, "k": 1, "v": 2}

    def __init__(
        self,
        dim:     int,
        rank:    int,
        alpha:   float,
        targets: tuple[str, ...] = ("k", "v"),
    ):
        super().__init__()
        self.dim     = dim
        self.rank    = rank
        self.scaling = alpha / rank
        self.targets = targets

        for t in targets:
            A = nn.Parameter(torch.empty(rank, dim))
            B = nn.Parameter(torch.zeros(dim, rank))
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            self.register_parameter(f"lora_A_{t}", A)
            self.register_parameter(f"lora_B_{t}", B)

    def pre_forward_hook(self, module, args, kwargs):
        """torch pre-hook: mutates (query, key, value) before attention runs."""
        query, key, value = args[0], args[1], args[2]
        rest = args[3:]

        for t, x_orig in [("q", query), ("k", key), ("v", value)]:
            if t not in self.targets:
                continue
            A = getattr(self, f"lora_A_{t}")
            B = getattr(self, f"lora_B_{t}")
            # [B, T, dim] -> [B, T, rank] -> [B, T, dim]
            delta = (x_orig.to(A.dtype) @ A.T) @ B.T * self.scaling
            if t == "q":
                query = query + delta.to(query.dtype)
            elif t == "k":
                key = key + delta.to(key.dtype)
            else:
                value = value + delta.to(value.dtype)

        return (query, key, value) + rest, kwargs


def remove_lora(lm) -> None:
    for layer in lm.transformer.layers:
        ca = layer.cross_attention
        if hasattr(ca, "lora"):
            ca._modules.pop("lora", None)
        handles_to_remove = [
            h for h in ca._forward_pre_hooks.values()
            if hasattr(h, "__self__") and isinstance(h.__self__, CrossAttentionLoRA)
        ]
        for h in handles_to_remove:
            h.remove()



def apply_lora(
    lm,
    rank:          int              = 4,
    alpha:         float            = 8.0,
    targets:       tuple[str, ...]  = ("k", "v"),
    layer_indices: list[int] | None = None,
) -> None:
    """Attach LoRA pre-hooks to cross-attention in each transformer layer.

    After this call:
    - All LM parameters remain frozen (requires_grad=False)
    - Each cross-attention has a CrossAttentionLoRA submodule with trainable A/B
    - A forward pre-hook injects the LoRA delta into q/k/v before attention

    note: the hook fires before the attention module's internal W_Q/W_K/W_V
    projections, so the LoRA delta is added to the input rather than the
    projected key/value. Gradients still flow correctly through A and B.

    Args:
        lm:            MusicGen language model
        rank:          LoRA rank
        alpha:         LoRA alpha scaling (effective scale = alpha / rank)
        targets:       which QKV projections to adapt
        layer_indices: transformer layers to apply LoRA to; None = all
    """
    layers = lm.transformer.layers
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    for i in layer_indices:
        ca  = layers[i].cross_attention
        dim = ca.in_proj_weight.shape[1]

        adapter = CrossAttentionLoRA(dim=dim, rank=rank, alpha=alpha, targets=targets)
        # store on the cross-attention module so it's part of the module tree
        # (parameters show up in lm.named_modules, state_dict works correctly)
        ca.add_module("lora", adapter)
        adapter.to(ca.in_proj_weight.device)

        ca.register_forward_pre_hook(adapter.pre_forward_hook, with_kwargs=True)

    n_params = sum(p.numel() for p in lora_params(lm))
    print(
        f"[LoRA] Applied to {len(layer_indices)} cross-attention layers | "
        f"targets={targets} | rank={rank} | alpha={alpha} | "
        f"trainable params: {n_params:,}"
    )


def lora_params(lm) -> list[nn.Parameter]:
    params = []
    for module in lm.modules():
        if isinstance(module, CrossAttentionLoRA):
            params.extend(module.parameters())
    return params


def lora_state_dict(lm) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in lm.named_modules():
        if isinstance(module, CrossAttentionLoRA):
            for t in module.targets:
                for ab in ("A", "B"):
                    key = f"lora_{ab}_{t}"
                    state[f"{name}.{key}"] = getattr(module, key).data.clone()
    return state


def load_lora_state_dict(lm, state: dict[str, torch.Tensor]) -> None:
    """Restore LoRA tensors from a saved state dict.

    Handles both key formats:
      new (hook-based): transformer.layers.N.cross_attention.lora.lora_A_k
      old (parametrize): transformer.layers.N.cross_attention.parametrizations.in_proj_weight.0.lora_A_k
    """
    # remap old parametrize keys to new hook-based keys
    remapped = {}
    for k, v in state.items():
        new_k = k.replace(
            "cross_attention.parametrizations.in_proj_weight.0.",
            "cross_attention.lora.",
        )
        remapped[new_k] = v

    module_map: dict[str, CrossAttentionLoRA] = {
        name: m for name, m in lm.named_modules()
        if isinstance(m, CrossAttentionLoRA)
    }

    loaded = 0
    for full_key, tensor in remapped.items():
        for mod_name, mod in module_map.items():
            prefix = mod_name + "."
            if full_key.startswith(prefix):
                attr = full_key[len(prefix):]   # e.g. "lora_A_k"
                if hasattr(mod, attr):
                    getattr(mod, attr).data.copy_(tensor)
                    loaded += 1
                    break

    print(f"[LoRA] Loaded {loaded}/{len(state)} LoRA tensors from checkpoint")
