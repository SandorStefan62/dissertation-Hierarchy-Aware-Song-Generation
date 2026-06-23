"""
Compare MuQ features of generated songs against the originals.

Tests Implemented: 

1. Temporal similarity
    For each song in a run, loads the conditioned / unconditioned / noise MuQ
    feature .pt files and computes cosine similarity against the original at three
    granularities:

    global     - single global token  [1,  L, 1024]
    contextual - 12 contextual tokens [12, L, 1024]
    local      - 36 local tokens      [36, L, 1024]

    Each tensor is averaged across the L MuQ layers before computing similarity,
    giving a single embedding per granularity level.

    Similarity per granularity:
    global:     cosine(gen_global, orig_global)
    contextual: mean cosine over the 12 matched token pairs
    local:      mean cosine over the 36 matched token pairs

2. PCA comparison
    For each song in a run, loads the conditioned / unconditioned / noise MuQ
    feature .pt files and plots the generated features in a PCA plot.

3. Euclidean distance
    For each song in a run, loads the conditioned / unconditioned / noise MuQ
    feature .pt files and computes the Euclidean distance against the original at three
    granularities:

    global     - single global token  [1, L, 1024]
    contextual - 12 contextual tokens [12, L, 1024]
    local      - 36 local tokens      [36, L, 1024]

    Euclidean per granularity:
    global:     euclidean(gen_global, orig_global)
    contextual: mean euclidean over the 12 matched token pairs
    local:      mean euclidean over the 36 matched token pairs

3. Borda count ranking
    For each song in a run, loads the conditioned / unconditioned / noise MuQ
    feature .pt files and computes Borda-count ranking, combining both
    cosine similarity (high=good) and euclidean distance (low=good).

Usage:
    python src/scripts/analyze_generations.py --muq-dir <path-to-generated-muq-features> --original-dir <path-to-original-muq-features>
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


VARIANTS = ("conditioned", "unconditioned", "noise")

CTX_WINDOW_S = 30
LOC_WINDOW_S = 10
N_CTX        = 12
N_LOC        = 36

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        False,
    "font.size":        10,
}

_SCALE_COLORS = (
    ["#e07b54"]        # global  (1)
  + ["#5a9e6f"] * 12  # contextual (12)
  + ["#5b7ec9"] * 36  # local (36)
)
_TOKEN_LABELS = (
    ["global"]
  + [f"ctx_{i*30}s" for i in range(12)]
  + [f"loc_{i*10}s"  for i in range(36)]
)


def load_feat(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return torch.load(path, weights_only=False, map_location="cpu")

def mean_over_layers(t: torch.Tensor) -> torch.Tensor:
    return t.mean(dim=1)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def euc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.dist(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def token_mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    sims = F.cosine_similarity(a, b, dim=-1)  # [N]
    return sims.mean().item()


def token_mean_euc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b, dim=-1).mean().item()  # [N] -> scalar


def neighbor_overlap(orig: torch.Tensor, gen: torch.Tensor, top_n: int) -> float:
    """Mean neighbor-set overlap between generated and original tokens.

    For each position i:
      orig_neighbors_i: TOP_N nearest orig tokens to orig[i], excluding i itself
      gen_neighbors_i:  TOP_N nearest orig tokens to gen[i], all positions valid
      overlap_i = |intersection| / top_n

    orig, gen: [N, 1024]  (already layer-averaged, valid positions only)
    """
    N = orig.shape[0]
    if N <= top_n:
        return float("nan")

    orig_norm = F.normalize(orig.float(), dim=-1)
    gen_norm  = F.normalize(gen.float(),  dim=-1)

    orig_orig_sim = orig_norm @ orig_norm.T # [N, N]
    gen_orig_sim  = gen_norm  @ orig_norm.T # [N, N]
    overlaps = []
    for i in range(N):
        o_sims = orig_orig_sim[i].clone()
        o_sims[i] = -float("inf")  # exclude self
        orig_nbrs = set(o_sims.topk(top_n).indices.tolist())
        gen_nbrs  = set(gen_orig_sim[i].topk(top_n).indices.tolist())
        overlaps.append(len(orig_nbrs & gen_nbrs) / top_n)

    return float(np.mean(overlaps))


def compute_combined_rank(
    cos_sims: list[float], 
    euc_dists: list[float], 
    target_idx: int
) -> tuple[int, int, int]:
    """Borda-count rank combining cosine (high = good) and euclidean (low = good)
    
    Zero-similarity (zero-padded) entries are excluded form rankings.

    Returns:
        (combined_rank, cos_rank, euc_rank) - all 1-based
    """
    valid = [i for i, c in enumerate(cos_sims) if c > 0]
    if not valid:
        return 1, 1, 1
    
    cos_order = sorted(valid, key=lambda i: cos_sims[i], reverse=True)
    euc_order = sorted(valid, key=lambda i: euc_dists[i])
    cos_rank = {i: r + 1 for r, i in enumerate(cos_order)}
    euc_rank = {i: r + 1 for r, i in enumerate(euc_order)}

    borda_order = sorted(valid, key=lambda i: cos_rank[i] + euc_rank[i])
    combined_rank = {i: r + 1 for r, i in enumerate(borda_order)}

    fallback = len(valid) + 1
    return (
        combined_rank.get(target_idx, fallback),
        cos_rank.get(target_idx, fallback),
        euc_rank.get(target_idx, fallback)
    )

@dataclass
class SongSims:
    song:          str
    variant:       str
    global_cos:    float
    global_euc:    float
    ctx_cos:       float
    ctx_euc:       float
    ctx_overlap:   float  # mean neighbor-set overlap across 12 contextual tokens
    local_cos:     float
    local_euc:     float
    local_overlap: float  # mean neighbor-set overlap across 36 local tokens


@dataclass
class RankResult:
    song:        str
    variant:     str
    combined:    int  # Borda over all three levels concatenated
    global_rank: int  # Borda using global embedding only
    ctx_rank:    int  # Borda using contextual embedding only
    local_rank:  int  # Borda using local embedding only
    n:           int  # pool size


def analyze_song(song_stem: str, gen_dir: Path, orig_dir: Path, top_n: int = 4) -> list[SongSims]:
    orig_path = orig_dir / f"{song_stem}.pt"
    orig = load_feat(orig_path)
    if orig is None:
        print(f"  [skip] {song_stem} - original features not found")
        return []

    orig_g = mean_over_layers(orig["global"])     # [1, 1024]
    orig_c = mean_over_layers(orig["contextual"]) # [12, 1024]
    orig_l = mean_over_layers(orig["local"])      # [36, 1024]

    results = []
    for variant in VARIANTS:
        gen_path = gen_dir / f"{song_stem}_{variant}.pt"
        gen = load_feat(gen_path)
        if gen is None:
            continue

        gen_g = mean_over_layers(gen["global"])
        gen_c = mean_over_layers(gen["contextual"])
        gen_l = mean_over_layers(gen["local"])

        results.append(SongSims(
            song=song_stem,
            variant=variant,
            global_cos=cos_sim(gen_g, orig_g),
            global_euc=euc(gen_g, orig_g),
            ctx_cos=token_mean_cos(gen_c, orig_c),
            ctx_euc=token_mean_euc(gen_c, orig_c),
            ctx_overlap=neighbor_overlap(orig_c, gen_c, top_n),
            local_cos=token_mean_cos(gen_l, orig_l),
            local_euc=token_mean_euc(gen_l, orig_l),
            local_overlap=neighbor_overlap(orig_l, gen_l, top_n),
        ))
    return results


def _level_embs(feat: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = mean_over_layers(feat["global"]).mean(dim=0).float()
    c = mean_over_layers(feat["contextual"]).mean(dim=0).float()
    l = mean_over_layers(feat["local"]).mean(dim=0).float()
    return g, c, l


def _borda(gen: torch.Tensor, pool: list[torch.Tensor], target_idx: int) -> int:
    cos_sims  = [F.cosine_similarity(gen.unsqueeze(0), o.unsqueeze(0)).item() for o in pool]
    euc_dists = [torch.dist(gen, o).item() for o in pool]
    rank, _, _ = compute_combined_rank(cos_sims, euc_dists, target_idx)
    return rank


def rank_against_pool(
    song_stems: list[str],
    muq_dir:    Path,
    orig_dir:   Path,
) -> list[RankResult]:
    """For each generated song, rank all originals by Borda similarity.

    Computes one combined rank (all levels concatenated) and three
    per-level ranks (global / contextual / local independently).
    """
    orig: dict[str, tuple] = {}
    for stem in song_stems:
        feat = load_feat(orig_dir / f"{stem}.pt")
        if feat is not None:
            orig[stem] = _level_embs(feat)

    pool_stems = list(orig.keys())
    orig_flat = [torch.cat(list(orig[t]), dim=0) for t in pool_stems]
    orig_g    = [orig[t][0] for t in pool_stems]
    orig_c    = [orig[t][1] for t in pool_stems]
    orig_l    = [orig[t][2] for t in pool_stems]

    results = []
    for stem in pool_stems:
        for variant in VARIANTS:
            gen_feat = load_feat(muq_dir / f"{stem}_{variant}.pt")
            if gen_feat is None:
                continue
            gg, gc, gl = _level_embs(gen_feat)
            target_idx = pool_stems.index(stem)
            results.append(RankResult(
                song=stem, variant=variant,
                combined=_borda(torch.cat([gg, gc, gl]), orig_flat, target_idx),
                global_rank=_borda(gg, orig_g, target_idx),
                ctx_rank=_borda(gc, orig_c, target_idx),
                local_rank=_borda(gl, orig_l, target_idx),
                n=len(pool_stems),
            ))
    return results


def print_ranking(ranks: list[RankResult], run_tag: str, top_n: int) -> None:
    n_songs = len({r.song for r in ranks})
    pool    = ranks[0].n if ranks else 0
    print(f"\n{'='*60}")
    print(f"Pool ranking  |  run: {run_tag}  |  N={n_songs}  |  pool={pool}")
    print(f"{'='*60}")

    variants_present = [v for v in VARIANTS if any(r.variant == v for r in ranks)]
    levels = [
        ("combined", "combined"),
        ("global",   "global_rank"),
        ("ctx",      "ctx_rank"),
        ("local",    "local_rank"),
    ]

    for v in variants_present:
        vranks = [r for r in ranks if r.variant == v]
        if not vranks:
            continue
        print(f"  {v}")
        for label, attr in levels:
            rs     = [getattr(r, attr) for r in vranks]
            hit1   = np.mean([r == 1     for r in rs])
            hitn   = np.mean([r <= top_n for r in rs])
            print(f"    {label:<10}  Hit@1={hit1:<10.2%}  Hit@{top_n}={hitn:<10.2%}")


_GEN_MARKERS = {
    "conditioned":   ("*", "#e07b54", 200),
    "unconditioned": ("v", "#5b7ec9",  80),
    "noise":         ("^", "#5a9e6f",  80),
}

def plot_pca(
    song_stem: str,
    gen_dir:   Path,
    orig_dir:  Path,
    out_path:  Path,
) -> None:
    orig = load_feat(orig_dir / f"{song_stem}.pt")
    if orig is None:
        return

    duration_s  = float(orig["duration_s"])
    orig_tokens = torch.cat([
        mean_over_layers(orig["global"]),      # [1,  1024]
        mean_over_layers(orig["contextual"]),  # [12, 1024]
        mean_over_layers(orig["local"]),       # [36, 1024]
    ], dim=0).float()  # [49, 1024]

    # validity mask: tokens whose window start is within the song
    mask = np.ones(49, dtype=bool)
    for i in range(12):
        if duration_s <= i * CTX_WINDOW_S:
            mask[1 + i] = False
    for i in range(36):
        if duration_s <= i * LOC_WINDOW_S:
            mask[13 + i] = False

    valid_idx = np.where(mask)[0]

    # fit PCA on valid original tokens
    X        = orig_tokens[valid_idx]
    X_mean   = X.mean(dim=0, keepdim=True)
    X_c      = X - X_mean
    _, _, Vh = torch.linalg.svd(X_c, full_matrices=False)
    pc2      = Vh[:2]  # [2, 1024]

    def project(tokens: torch.Tensor) -> np.ndarray:
        """[N, 1024] -> [N, 2] projected into the original-token PCA space."""
        return ((tokens.float() - X_mean) @ pc2.T).numpy()

    orig_proj = project(X_c + X_mean) # [N_valid, 2]

    # load all generated variants
    gen_proj: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        gen = load_feat(gen_dir / f"{song_stem}_{variant}.pt")
        if gen is None:
            continue
        gen_tokens = torch.cat([
            mean_over_layers(gen["global"]),
            mean_over_layers(gen["contextual"]),
            mean_over_layers(gen["local"]),
        ], dim=0).float() # [49, 1024]
        gen_proj[variant] = project(gen_tokens) # [49, 2]

    from matplotlib.lines import Line2D

    ctx_valid_idx = valid_idx[(valid_idx >= 1) & (valid_idx <= 12)]

    # build a lookup: vi -> row index in orig_proj (for line drawing)
    vi_to_orig_row = {vi: k for k, vi in enumerate(valid_idx)}

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_title(f"{song_stem}  (PCA of MuQ tokens, {duration_s:.0f}s)")

        # original tokens (global + contextual + local, fixed colours by scale)
        for k, vi in enumerate(valid_idx):
            px, py = orig_proj[k]
            ax.scatter(px, py, c=_SCALE_COLORS[vi], s=55, zorder=3, alpha=0.85)
            ax.annotate(_TOKEN_LABELS[vi], (px, py), fontsize=5, alpha=0.7, textcoords="offset points", xytext=(3, 3))

        # generated contextual tokens coloured by time; lines for conditioned
        for variant, proj in gen_proj.items():
            marker, _, size = _GEN_MARKERS[variant]

            if variant == "unconditioned":
                continue

            if variant == "noise":
                continue

            for vi in ctx_valid_idx:
                color  = "orange"
                px, py = proj[vi]
                ax.scatter(px, py, marker=marker, c=[color], s=size, zorder=4, alpha=0.75, edgecolors="white", linewidths=0.4)

                if variant == "conditioned" and vi in vi_to_orig_row:
                    orig_row    = vi_to_orig_row[vi]
                    tx, ty      = orig_proj[orig_row]
                    ax.plot([px, tx], [py, ty], color=color, linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_SCALE_COLORS[0],  markersize=8, label="original global"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_SCALE_COLORS[1],  markersize=8, label="original contextual"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_SCALE_COLORS[13], markersize=8, label="original local"),
        ]
        for variant, (marker, color, _) in _GEN_MARKERS.items():
            if variant == "unconditioned":
                continue

            if variant == "noise":
                continue

            if variant in gen_proj:
                legend_handles.append(Line2D([0], [0], marker=marker, color="w", markerfacecolor=color, markersize=8, label=f"generated contextual"))
        legend_handles.append(Line2D([0], [0], color="grey", linewidth=0.8, linestyle="--", label="cond->orig link"))
        ax.legend(handles=legend_handles, fontsize=7, loc="best")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  [plot] {out_path.name}")


def print_song(sims: list[SongSims]) -> None:
    if not sims:
        return
    print(f"\nSong: {sims[0].song}")
    header = f"  {'level':<10}  " + "  ".join(f"{v:<20}" for v in VARIANTS)
    print(header)
    for level, attr in [("global", "global_cos"), ("contextual", "ctx_cos"), ("local", "local_cos")]:
        row = f"  {level:<10}  "
        for v in VARIANTS:
            match = next((s for s in sims if s.variant == v), None)
            val = f"{getattr(match, attr):.4f}" if match else "  N/A  "
            row += f"{val:<20}  "
        print(row)


def plot_aggregate_summary(
    all_sims: list[SongSims],
    ranks:    list[RankResult],
    run_tag:  str,
    out_path: Path,
    hit_n:    int = 3,
) -> None:
    variants_present = [v for v in VARIANTS if any(s.variant == v for s in all_sims)]
    n_songs = len({s.song for s in all_sims})

    COLORS = {"conditioned": "#e07b54", "unconditioned": "#5b7ec9", "noise": "#5a9e6f"}
    SHORT  = {"conditioned": "Conditioned", "unconditioned": "Unconditioned", "noise": "Noise"}

    scale_keys = [
        ("Global",     "global_cos"),
        ("Contextual", "ctx_cos"),
        ("Local",      "local_cos"),
    ]
    rank_levels = [
        ("Combined", "combined"),
        ("Global",   "global_rank"),
        ("Ctx",      "ctx_rank"),
        ("Local",    "local_rank"),
    ]

    n_var = len(variants_present)

    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{run_tag}  —  N={n_songs} songs", fontsize=11)

        # cosine similarity
        n_scales = len(scale_keys)
        width    = 0.22
        x        = np.arange(n_scales)
        offsets  = (np.arange(n_var) - (n_var - 1) / 2) * width

        for vi, v in enumerate(variants_present):
            means, stds = [], []
            for _, attr in scale_keys:
                vals = [getattr(s, attr) for s in all_sims if s.variant == v]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax1.bar(x + offsets[vi], means, width, yerr=stds, capsize=3, color=COLORS[v], alpha=0.85, label=SHORT[v], error_kw={"elinewidth": 1.2, "ecolor": "grey", "alpha": 0.6})

        ax1.set_xticks(x)
        ax1.set_xticklabels([s for s, _ in scale_keys])
        ax1.set_ylabel("Cosine similarity (mean ± std)")
        ax1.set_title("Similarity to original")
        ax1.legend(fontsize=9)
        ax1.set_ylim(0, max(np.mean([getattr(s, a) for s in all_sims if s.variant == "conditioned"]) + 0.12 for _, a in scale_keys))

        # stacked Hit@1 / Hit@K by level
        n_levels = len(rank_levels)
        bw       = 0.22
        xr       = np.arange(n_levels)
        offr     = (np.arange(n_var) - (n_var - 1) / 2) * bw

        for vi, v in enumerate(variants_present):
            vranks = [r for r in ranks if r.variant == v]
            h1, h_extra = [], []
            for _, attr in rank_levels:
                rs = [getattr(r, attr) for r in vranks]
                v1 = np.mean([r == 1     for r in rs]) * 100 if rs else 0.0
                vn = np.mean([r <= hit_n for r in rs]) * 100 if rs else 0.0
                h1.append(v1)
                h_extra.append(vn - v1)

            ax2.bar(xr + offr[vi], h1, bw, color=COLORS[v], alpha=0.9, label=f"{SHORT[v]}  Hit@1")
            ax2.bar(xr + offr[vi], h_extra, bw, bottom=h1, color=COLORS[v], alpha=0.4, hatch="///", edgecolor=COLORS[v], label=f"{SHORT[v]}  Hit@{hit_n}")

        pool_size = ranks[0].n if ranks else "?"
        ax2.set_xticks(xr)
        ax2.set_xticklabels([s for s, _ in rank_levels])
        ax2.set_ylabel("Hit rate (%)")
        ax2.set_title(f"Pool ranking  (pool = {pool_size})")
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=8, ncol=2, loc="upper right")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out_path}")


def print_aggregate(all_sims: list[SongSims], run_tag: str, top_n: int = 4) -> None:
    n_songs = len({s.song for s in all_sims})
    print(f"\n{'='*60}")
    print(f"Aggregate  |  run: {run_tag}  |  N={n_songs} songs")
    print(f"{'='*60}")

    variants_present = [v for v in VARIANTS if any(s.variant == v for s in all_sims)]

    for level, cos_attr, euc_attr, overlap_attr in [
        ("global",     "global_cos", "global_euc", None),
        ("contextual", "ctx_cos",    "ctx_euc",    "ctx_overlap"),
        ("local",      "local_cos",  "local_euc",  "local_overlap"),
    ]:
        print(f"  {level}")
        for v in variants_present:
            vs = [s for s in all_sims if s.variant == v]
            if not vs:
                continue
            cos_vals = [getattr(s, cos_attr) for s in vs]
            euc_vals = [getattr(s, euc_attr) for s in vs]
            line = (f"    {v:<16}  cos: {np.mean(cos_vals):.4f} +/- {np.std(cos_vals):.4f}"
                    f"  |  euc: {np.mean(euc_vals):.4f} +/- {np.std(euc_vals):.4f}")
            if overlap_attr:
                ov_vals = [getattr(s, overlap_attr) for s in vs if not np.isnan(getattr(s, overlap_attr))]
                if ov_vals:
                    line += f"  |  nbr@{top_n}: {(np.mean(ov_vals) * 100):.2f}% +/- {(np.std(ov_vals) * 100):.2f}%"
            print(line)

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--muq-dir", type=Path, nargs="+", required=True, help="Directory containing generated MuQ .pt files")
    parser.add_argument("--orig-dir", type=Path, default=Path("datasets/processed/HX/muq_features"), help="Directory with original MuQ .pt files (default: datasets/processed/HX/muq_features)")
    parser.add_argument("--out-dir", type=Path, default="experiments/gen_analysis", help="Output directory of plots.")
    parser.add_argument("--top-n", type=int, default=4, help="Top n neighbours to measure positional accuracy.")
    parser.add_argument("--hit-n", type=int, default=3, help="Top n rank to measure hit@n.")
    parser.add_argument("--quiet", action="store_true", help="Print only aggregates if true.")
    parser.add_argument("--generate-plots", action="store_true", help="Generate plots if true.")
    parser.add_argument("--summary-plot", action="store_true", help="Save a two-panel aggregate summary figure (cosine + Hit@K).")
    parser.add_argument("--songs", type=str, nargs="*", default=None, help="Specific song stems to analyze (default: all found in --muq-dir)")
    args = parser.parse_args()

    for muq_dir in args.muq_dir:
        if not muq_dir.exists():
            print(f"[warn] {muq_dir} does not exist, skipping")
            continue

        run_tag  = muq_dir.name
        out_path = args.out_dir / run_tag
        out_path.mkdir(parents=True, exist_ok=True)

        # discover songs from conditioned files (they always exist)
        cond_files = sorted(muq_dir.glob("*_conditioned.pt"))
        if not cond_files:
            print(f"[warn] no *_conditioned.pt files found in {muq_dir}")
            continue

        song_stems = [f.name.replace("_conditioned.pt", "") for f in cond_files]
        if args.songs:
            song_stems = [s for s in song_stems if s in args.songs]

        print(f"\nRun: {run_tag}  ({len(song_stems)} songs)")

        all_sims: list[SongSims] = []
        for stem in song_stems:
            sims = analyze_song(stem, muq_dir, args.orig_dir, top_n=args.top_n)
            if not args.quiet:
                print_song(sims)
            
            if args.generate_plots:
                plot_pca(stem, muq_dir, args.orig_dir, out_path / f"{stem}.png")
            all_sims.extend(sims)

        if all_sims:
            print_aggregate(all_sims, run_tag, top_n=args.top_n)

        ranks = rank_against_pool(song_stems, muq_dir, args.orig_dir)
        if ranks:
            print_ranking(ranks, run_tag, args.hit_n)

        if args.summary_plot and all_sims and ranks:
            plot_aggregate_summary(
                all_sims, ranks, run_tag,
                out_path / "summary.png",
                hit_n=args.hit_n,
            )
