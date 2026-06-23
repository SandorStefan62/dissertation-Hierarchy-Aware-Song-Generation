import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


IGNORE_LABELS = {"end"}   # structural markers that don't name a section type


def load_label_segments(label_path: Path) -> list[tuple[float, str]]:
    """Parse '<timestamp> <label>' label file.

    Returns list of (start_s, label) sorted by start time.
    """
    segments = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        segments.append((t, parts[1]))
    return sorted(segments, key=lambda x: x[0])


def dominant_label(
        segments: list[tuple[float, str]],
        win_start: float,
        win_end: float,
        min_coverage: float = 0.3
) -> str | None:
    """Return the label covering the most time in [win_start, win_end].

    Returns None if:
      - no segment overlaps the window
      - total labelled coverage < min_coverage * window_duration
      - all overlapping labels are in IGNORE_LABELS
    """
    win_dur = win_end - win_start
    coverage: dict[str, float] = defaultdict(float)

    for i, (seg_start, label) in enumerate(segments):
        if label in IGNORE_LABELS:
            continue
        seg_end = segments[i + 1][0] if i + 1 < len(segments) else win_end + 1.0
        # overlap with [win_start, win_end]
        ov_start = max(seg_start, win_start)
        ov_end = min(seg_end, win_end)
        if ov_end > ov_start:
            coverage[label] += ov_end - ov_start

    if not coverage:
        return None
    if sum(coverage.values()) < min_coverage * win_dur:
        return None

    return max(coverage, key=coverage.__getitem__)


def pool_layers(feat: torch.Tensor) -> torch.Tensor:
    """Weighted or uniform mean over the n_layers axis.

    feat: [W, n_layers, 1024]  (W windows)
    Returns: [W, 1024]
    """
    return feat.float().mean(dim=1)


LOCAL_WINDOW_S  = 10.0
N_LOCAL         = 36
CTX_WINDOW_S    = 30.0
N_CTX           = 12


def build_dataset(
    song_ids:           list[str],
    feature_dir:        Path,
    label_dir:          Path,
    scale:              str,                        # contextual | local_mean | global
    exclude_silence:    bool = True,
    min_coverage:       float = 0.3,
    mean_locals:        bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, sample_song_ids) arrays for logistic regression.

    Each sample is one 30s contextual window that has a valid dominant label.
    
    X: [N, 1024] float32
    y: [N]       str labels
    sample_song_ids: [N]  str - which song each sample came from
    """
    X_rows, y_rows, sid_rows = [], [], []

    for sid in song_ids:
        feat_path  = feature_dir / f"{sid}.pt"
        label_path = label_dir   / f"{sid}.txt"

        if not feat_path.exists() or not label_path.exists():
            continue

        feat     = torch.load(feat_path, weights_only=False)
        segments = load_label_segments(label_path)
        duration = float(feat["duration_s"])

        # choose feature tensoir based on scale
        if scale == "contextual":
            # [12, n_layers, 1024]
            tokens = pool_layers(feat["contextual"])  # [12, 1024]
            window_starts = [i * CTX_WINDOW_S for i in range(N_CTX)]

        elif scale == "local":
            if mean_locals:
                # local: [36, n_layers, 1024] - group 3 locals per ctx window
                local_pooled = pool_layers(feat["local"])  # [36, 1024]
                tokens = torch.stack([
                    local_pooled[3 * i : 3 * i + 3].mean(dim=0)
                    for i in range(N_CTX)
                ])  # [12, 1024]
                window_starts = [i * CTX_WINDOW_S for i in range(N_CTX)]
            else:
                # local: [36, n_layers, 1024]
                # local_pooled = pool_layers(feat["local"])
                # tokens = torch.stack([
                #     local_pooled[i * LOCAL_WINDOW_S : (i + 1) * LOCAL_WINDOW_S].mean(dim=0)
                #     for i in range(N_LOCAL)
                # ])  # [36, 1024]
                # window_starts = [i * LOCAL_WINDOW_S for i in range(N_LOCAL)]
                # local: [36, n_layers, 1024]
                tokens = pool_layers(feat["local"])
                window_starts = [i * LOCAL_WINDOW_S for i in range(N_LOCAL)]

        elif scale == "global":
            # [1, n_layers, 1024] > [1024], replicate for each ctx window
            g = pool_layers(feat["global"])  # [1, 1024]
            tokens = g.expand(N_CTX, -1)                    # [12, 1024]
            window_starts = [i * CTX_WINDOW_S for i in range(N_CTX)]

        else:
            raise ValueError(f"Unknown scale: {scale!r}")

        for i, win_start in enumerate(window_starts):
            # skip windows beyond song duration
            if win_start >= duration:
                break

            if scale == "contextual":
                win_end = win_start + CTX_WINDOW_S
            elif scale == "local":
                if mean_locals:
                    win_end = win_start + CTX_WINDOW_S
                else:
                    win_end = win_start + LOCAL_WINDOW_S
            elif scale == "global":
                win_end = duration

            label = dominant_label(segments, win_start, win_end,min_coverage=min_coverage)

            if label is None:
                continue
            if exclude_silence and label == "silence":
                continue

            X_rows.append(tokens[i].numpy())
            y_rows.append(label)
            sid_rows.append(sid)

    if not X_rows:
        raise RuntimeError(f"No samples found for scale={scale!r}. "
                           "Check feature_dir and label_dir paths.")

    return np.array(X_rows, dtype=np.float32), np.array(y_rows), sid_rows


def run_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    C: float = 1.0,
) -> dict:
    """Train LogisticRegression and return metrics dict."""
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_val, y_pred, labels=clf.classes_)

    return {
        "clf":      clf,
        "y_pred":   y_pred,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "report":   report,
        "cm":       cm,
        "classes":  clf.classes_,
    }


def print_results(
        scale: str, results: dict,
        y_val: np.ndarray, 
        n_train: int, 
        n_val: int
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Scale: {scale.upper()}")
    print(f"{'=' * 60}")
    print(f"Train samples: {n_train}   Val samples: {n_val}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Macro F1:  {results['macro_f1']:.4f}")
    print("\nPer-class F1:")
    report = results["report"]
    for cls in results["classes"]:
        r = report.get(cls, {})
        n = int(np.sum(y_val == cls))
        print(f"  {cls:<14s}  F1={r.get('f1-score', 0):.3f}  "
              f"P={r.get('precision', 0):.3f}  R={r.get('recall', 0):.3f}  "
              f"(n={n})")


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: Path,
    results: dict | None = None,
    n_train: int = 0,
    n_val: int = 0,
    scale: str = "",
) -> None:
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        print("[pca] matplotlib or sklearn not available - skipping")
        return

    print(f"  Computing PCA for {X.shape[0]} samples…", flush=True)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    classes = sorted(set(y))
    cmap    = plt.cm.get_cmap("tab10", len(classes))
    label2c = {c: cmap(i) for i, c in enumerate(classes)}

    fig, (ax_scatter, ax_stats) = plt.subplots(
        1, 2, figsize=(15, 7),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    for cls in classes:
        mask = y == cls
        ax_scatter.scatter(Z[mask, 0], Z[mask, 1], c=[label2c[cls]], label=cls, s=12, alpha=0.7)
    ax_scatter.legend(markerscale=2, fontsize=9)
    ax_scatter.set_title(title)
    ax_scatter.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax_scatter.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")

    ax_stats.axis("off")
    if results is not None:
        report = results["report"]
        train_suffix = f"  (~{n_train // 12} songs)" if scale == "global" else ""
        val_suffix   = f"  (~{n_val   // 12} songs)" if scale == "global" else ""
        lines = [
            f"Train samples: {n_train}{train_suffix}",
            f"Val samples:   {n_val}{val_suffix}",
            "",
            f"Accuracy:  {results['accuracy']:.4f}",
            f"Macro F1:  {results['macro_f1']:.4f}",
            "",
            f"{'Class':<14s}  {'F1':>6s}  {'P':>6s}  {'R':>6s}  {'n':>5s}",
            "-" * 46,
        ]
        for cls in results["classes"]:
            r = report.get(cls, {})
            n = int(np.sum(y == cls))
            lines.append(
                f"{cls:<14s}  {r.get('f1-score', 0):>6.3f}  "
                f"{r.get('precision', 0):>6.3f}  {r.get('recall', 0):>6.3f}  {n:>5d}"
            )
        ax_stats.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax_stats.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved > {out_path}")

def load_song_ids(txt_path: Path) -> list[str]:
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--feature-dir", type=Path, help="Directory of {song_id}.pt MuQ feature files")
    parser.add_argument("--label-dir", type=Path, help="Directory of {song_id}.txt label files")
    parser.add_argument("--train-txt", type=Path, help="Newline-separated train song IDs")
    parser.add_argument("--val-txt", type=Path, help="Newline-separated val song IDs")
    parser.add_argument("--scale", nargs="+", choices=["contextual", "local", "global"], default=["contextual", "local", "global"], help="Feature scales to probe (default: all three)")
    parser.add_argument("--mean-locals", action="store_true", help="Mean local windows to contextual size")
    parser.add_argument("--no-silence", action="store_true", help="Exclude windows dominated by silence")
    parser.add_argument("--min-coverage", type=float, default=0.3, help="Min fraction of window that must be labelled (default 0.3)")
    parser.add_argument("--C", type=float, default=1.0, help="LogisticRegression regularisation strength")
    parser.add_argument("--pca", action="store_true", help="Generate PCA plots of val features coloured by label")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/probe_results"), help="Where to save t-SNE plots (created if needed)")

    args = parser.parse_args()

    train_ids = load_song_ids(args.train_txt)
    val_ids   = load_song_ids(args.val_txt)
    print(f"Train songs: {len(train_ids)}   Val songs: {len(val_ids)}")

    if args.pca:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for scale in args.scale:
        print(f"\nBuilding {scale} features…", flush=True)

        X_tr, y_tr, _ = build_dataset(
            train_ids, args.feature_dir, args.label_dir, scale,
            exclude_silence=args.no_silence,
            min_coverage=args.min_coverage,
        )
        X_va, y_va, _ = build_dataset(
            val_ids, args.feature_dir, args.label_dir, scale,
            exclude_silence=args.no_silence,
            min_coverage=args.min_coverage,
        )

        results = run_probe(X_tr, y_tr, X_va, y_va, C=args.C)
        all_results[scale] = results
        print_results(scale, results, y_va, len(X_tr), len(X_va))

        if args.pca:
            plot_pca(
                X_va, y_va,
                title=f"PCA val features - {scale}",
                out_path=args.out_dir / f"pca_{scale}.png",
                results=results,
                n_train=len(X_tr),
                n_val=len(X_va),
                scale=scale,
            )

    # summary table
    if len(args.scale) > 1:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"{'Scale':<16s}  {'Accuracy':>10s}  {'Macro F1':>10s}")
        for scale in args.scale:
            r = all_results[scale]
            print(f"  {scale:<14s}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}")