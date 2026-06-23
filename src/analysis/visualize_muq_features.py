import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


TOKEN_LABELS = (["global"] + [f"ctx_{i*30}s" for i in range(12)] + [f"loc_{i*10}s" for i in range(36)])

SCALE_COLORS = (
        ["#e07b54"]         # global
    +   ["#5a9e6f"] * 12    # contextual
    +   ["#5b7ec9"] * 36    # local
)


def pool_layers(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return x.float().mean(dim=1)
    return (x.float() * weights[None, :, None]).sum(dim=1)


def load_features(pt_path: Path):
    feat = torch.load(pt_path, weights_only=False)
    local      = feat["local"]      # [36, n_layers, 1024]
    contextual = feat["contextual"] # [12, n_layers, 1024]
    global_emb = feat["global"]     # [ 1, n_layers, 1024]
    duration_s = float(feat["duration_s"])
    return local, contextual, global_emb, duration_s


def get_pooled_tokens(local, contextual, global_emb, layer_weights=None):
    g = pool_layers(global_emb, layer_weights)   # [1,  1024]
    c = pool_layers(contextual, layer_weights)   # [12, 1024]
    l = pool_layers(local,      layer_weights)   # [36, 1024]
    return torch.cat([g, c, l], dim=0)           # [49, 1024]


def build_validity_mask(duration_s: float) -> np.ndarray:
    mask = np.ones(49, dtype=bool)
    for i in range(12):
        if duration_s <= i * 30.0:
            mask[1 + i] = False
    for i in range(36):
        if duration_s <= i * 10.0:
            mask[13 + i] = False
    return mask


def plot_cosine_similarity(tokens: torch.Tensor, valid_mask: np.ndarray, out_path: Path, title: str):
    normed = F.normalize(tokens, dim=-1)
    sim    = (normed @ normed.T).numpy() # [49, 49]

    # gray out invalid (padded) tokens
    for i in range(49):
        if not valid_mask[i]:
            sim[i, :] = np.nan
            sim[:, i] = np.nan

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(sim, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="cosine similarity")

    # scale separators
    for pos in [0.5, 12.5]:
        ax.axvline(pos, color="black", linewidth=1.5)
        ax.axhline(pos, color="black", linewidth=1.5)

    ax.set_xticks(range(49))
    ax.set_yticks(range(49))
    ax.set_xticklabels(TOKEN_LABELS, rotation=90, fontsize=5)
    ax.set_yticklabels(TOKEN_LABELS, fontsize=5)
    ax.set_title(title, fontsize=11)

    legend = [
        mpatches.Patch(color="#e07b54", label="global"),
        mpatches.Patch(color="#5a9e6f", label="contextual"),
        mpatches.Patch(color="#5b7ec9", label="local"),
        mpatches.Patch(color="lightgrey", label="padded (beyond duration)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_token_norms(tokens: torch.Tensor, valid_mask: np.ndarray, duration_s: float, out_path: Path, title: str):
    norms = tokens.norm(dim=-1).numpy()    # [49]

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = [SCALE_COLORS[i] if valid_mask[i] else "lightgrey" for i in range(49)]
    ax.bar(range(49), norms, color=colors, width=0.8)

    ax.axvline(0.5,  color="black", linewidth=1.2, linestyle="--")
    ax.axvline(12.5, color="black", linewidth=1.2, linestyle="--")

    ax.set_xticks(range(49))
    ax.set_xticklabels(TOKEN_LABELS, rotation=90, fontsize=6)
    ax.set_ylabel("L2 norm", fontsize=9)
    ax.set_title(f"{title}  |  duration={duration_s:.1f}s", fontsize=10)

    legend = [
        mpatches.Patch(color="#e07b54", label="global"),
        mpatches.Patch(color="#5a9e6f", label="contextual"),
        mpatches.Patch(color="#5b7ec9", label="local"),
        mpatches.Patch(color="lightgrey", label="padded"),
    ]
    ax.legend(handles=legend, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_layer_weights(checkpoint_path: Path, out_path: Path, title: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw  = ckpt["conditioner"]["layer_weights"]           # [n_layers]
    weights = torch.softmax(raw.float(), dim=0).numpy()

    n = len(weights)
    fig, ax = plt.subplots(figsize=(max(6, n), 4))
    ax.bar(range(n), weights, color="steelblue")
    ax.axhline(1 / n, color="red", linestyle="--", linewidth=0.9,
               label=f"uniform (1/{n}={1/n:.3f})")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"layer {4+i}" for i in range(n)], fontsize=9)
    ax.set_ylabel("softmax weight", fontsize=9)
    ax.set_title(f"{title}  |  learned MuQ layer importance", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_pca(tokens: torch.Tensor, valid_mask: np.ndarray, out_path: Path, title: str):
    from torch.linalg import svd

    valid_idx = np.where(valid_mask)[0]
    X = tokens[valid_idx].float()                         # [N_valid, 1024]
    X = X - X.mean(dim=0, keepdim=True)
    _, _, Vh = svd(X, full_matrices=False)
    pc = (X @ Vh[:2].T).numpy()                           # [N_valid, 2]

    fig, ax = plt.subplots(figsize=(8, 7))
    for idx, vi in enumerate(valid_idx):
        color = SCALE_COLORS[vi]
        ax.scatter(pc[idx, 0], pc[idx, 1], c=color, s=60, zorder=3)
        ax.annotate(TOKEN_LABELS[vi], (pc[idx, 0], pc[idx, 1]),
                    fontsize=5, alpha=0.75, textcoords="offset points", xytext=(3, 3))

    legend = [
        mpatches.Patch(color="#e07b54", label="global"),
        mpatches.Patch(color="#5a9e6f", label="contextual"),
        mpatches.Patch(color="#5b7ec9", label="local"),
    ]
    ax.legend(handles=legend, fontsize=9)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.set_title(f"{title}  |  PCA of MuQ tokens", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--feature-pt",  type=Path, required=True)
    parser.add_argument("--checkpoint",  type=Path, default=None, help="conditioner checkpoint to read learned layer weights")
    parser.add_argument("--out-dir",     type=Path, default=Path("experiments/muq_features"))
    args = parser.parse_args()

    local, contextual, global_emb, duration_s = load_features(args.feature_pt)
    song_name = args.feature_pt.stem

    # load learned layer weights if checkpoint provided, else uniform
    layer_weights = None
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        raw  = ckpt["conditioner"]["layer_weights"]
        layer_weights = torch.softmax(raw.float(), dim=0)

    tokens     = get_pooled_tokens(local, contextual, global_emb, layer_weights)  # [49, 1024]
    valid_mask = build_validity_mask(duration_s)

    ckpt_tag = f"_epoch{torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('epoch','?')}" \
               if args.checkpoint else "_uniform_pool"
    title = f"{song_name}{ckpt_tag}"

    out = args.out_dir / song_name
    plot_cosine_similarity(tokens, valid_mask, out / "cosine_similarity.png", title)
    plot_token_norms(tokens, valid_mask, duration_s, out / "token_norms.png", title)
    plot_pca(tokens, valid_mask, out / "pca_tokens.png", title)

    if args.checkpoint is not None:
        plot_layer_weights(args.checkpoint, out / "layer_weights.png", title)

