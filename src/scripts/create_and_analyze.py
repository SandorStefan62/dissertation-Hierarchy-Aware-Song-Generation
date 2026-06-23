import sys
import torch
import numpy as np
import scipy.io.wavfile as wavfile

from argparse import ArgumentParser
from pathlib import Path
from muq import MuQ

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "audiocraft"))

from src.analysis.generate import load_model, load_features, generate_song
from src.mfms.get_embeddings import extract_features
from src.data_loaders.songformdb_dataset import load_split
from src.scripts.analyze_generations import (
    analyze_song, print_song, print_aggregate,
    rank_against_pool, print_ranking, plot_pca,
)

MUQ_SR = 24000


def _save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    pcm = (wav.squeeze(0).numpy() * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm)


def _extract_muq(wav_path: Path, out_path: Path, muq_model, device, n_layers: int) -> None:
    feat = extract_features(wav_path, muq_model, device, sr=MUQ_SR, n_layers=n_layers)
    torch.save(feat, out_path)
    print(f"  [muq]  {out_path.name}")


def _generate_variant(
    song_stem: str,
    variant: str,
    null_cond: bool,
    noise_cond: bool,
    lm, musicgen, conditioner,
    local, contextual, global_emb, dur_tensor,
    audio_out_dir: Path,
    muq_out_dir:   Path,
    muq_model,
    device, amp_dtype, sr: int,
    top_k: int, temperature: float, overlap_s: float,
    n_muq_layers: int,
) -> None:
    wav_path = audio_out_dir / f"{song_stem}_{variant}.wav"
    muq_path = muq_out_dir   / f"{song_stem}_{variant}.pt"

    if not wav_path.exists():
        print(f"  [gen]  {song_stem}_{variant}")
        wav = generate_song(
            lm, musicgen, conditioner,
            local, contextual, global_emb, dur_tensor,
            device, amp_dtype,
            top_k=top_k, temperature=temperature, overlap_s=overlap_s,
            null_cond=null_cond, noise_cond=noise_cond,
        )
        _save_wav(wav_path, wav, sr)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"  [skip] {song_stem}_{variant}  (audio exists)")

    if not muq_path.exists():
        _extract_muq(wav_path, muq_path, muq_model, device, n_muq_layers)
    else:
        print(f"  [skip] {song_stem}_{variant}  (muq exists)")


def step_generate_and_extract(
    song_ids:      list[str],
    feature_dir:   Path,
    audio_out_dir: Path,
    muq_out_dir:   Path,
    musicgen, lm, conditioner,
    muq_model,
    device, amp_dtype, sr: int,
    top_k: int, temperature: float, overlap_s: float,
    uncond: bool, noise: bool,
    n_muq_layers: int,
) -> list[str]:
    """Generate audio and extract MuQ for all songs. Returns stems that were processed."""
    processed = []
    for sid in song_ids:
        feature_pt = feature_dir / f"{sid}.pt"
        if not feature_pt.exists():
            print(f"[skip] {sid} - feature file not found")
            continue

        feat = torch.load(feature_pt, weights_only=False)
        duration_s = float(feat["duration_s"])
        print(f"\n[song] {sid}  duration={duration_s:.1f}s")

        local, contextual, global_emb, dur_tensor = load_features(feature_pt, device)

        variants = [("conditioned", False, False)]
        if uncond:
            variants.append(("unconditioned", True, False))
        if noise:
            variants.append(("noise", False, True))

        for variant, null_cond, noise_cond in variants:
            _generate_variant(
                sid, variant, null_cond, noise_cond,
                lm, musicgen, conditioner,
                local, contextual, global_emb, dur_tensor,
                audio_out_dir, muq_out_dir,
                muq_model, device, amp_dtype, sr,
                top_k, temperature, overlap_s, n_muq_layers,
            )

        processed.append(sid)

    return processed


def step_analyze(
    song_stems:  list[str],
    muq_out_dir: Path,
    orig_dir:    Path,
    analysis_dir: Path,
    top_n: int,
    hit_n: int,
    quiet: bool,
    generate_plots: bool,
) -> None:
    run_tag = muq_out_dir.name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_sims = []
    for stem in song_stems:
        sims = analyze_song(stem, muq_out_dir, orig_dir, top_n=top_n)
        if not quiet:
            print_song(sims)
        if generate_plots:
            plot_pca(stem, muq_out_dir, orig_dir, analysis_dir / f"{stem}.png")
        all_sims.extend(sims)

    if all_sims:
        print_aggregate(all_sims, run_tag, top_n=top_n)

    ranks = rank_against_pool(song_stems, muq_out_dir, orig_dir)
    if ranks:
        print_ranking(ranks, run_tag, hit_n)



if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)

    # required
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained conditioner checkpoint .pt")
    parser.add_argument("--split-txt", type=Path, required=True, help="Newline-separated song IDs to process (e.g. test.txt)")
    parser.add_argument("--feature-dir", type=Path, required=True, help="Directory of original MuQ .pt files used as conditioning input")

    # optional paths
    parser.add_argument("--orig-dir", type=Path, default=None, help="Original MuQ features for analysis. Defaults to --feature-dir.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/generated"), help="Root output directory for audio and generated MuQ features")
    parser.add_argument("--analysis-dir", type=Path, default=Path("experiments/gen_analysis"), help="Root output directory for analysis results and plots")
    parser.add_argument("--run-tag", type=str,  default=None, help="Sub-directory name for this run (defaults to checkpoint stem)")

    # generation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k",  type=int, default=250)
    parser.add_argument("--overlap-s", type=float, default=2.0, help="Seconds of previous segment used as autoregressive prompt (default 2.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uncond", action="store_true", help="Also generate unconditioned baseline")
    parser.add_argument("--noise", action="store_true", help="Also generate noise-conditioned baseline")
    parser.add_argument("--n-muq-layers", type=int, default=8, help="Number of MuQ layers to extract (default 8)")

    # analysis
    parser.add_argument("--top-n", type=int, default=4, help="Neighbour set size for overlap metric")
    parser.add_argument("--hit-n", type=int, default=3, help="K for Hit@K in pool ranking")
    parser.add_argument("--quiet", action="store_true", help="Print only aggregate results, not per-song detail")
    parser.add_argument("--generate-plots", action="store_true", help="Save PCA plots per song")

    # flow control
    parser.add_argument("--analyze-only", action="store_true", help="Skip generation and MuQ extraction; run analysis on existing files")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    run_tag      = args.run_tag or args.checkpoint.stem
    orig_dir     = args.orig_dir or args.feature_dir
    audio_out    = args.out_dir / "audio"        / run_tag
    muq_out      = args.out_dir / "muq_features" / run_tag
    analysis_out = args.analysis_dir / run_tag

    audio_out.mkdir(parents=True, exist_ok=True)
    muq_out.mkdir(parents=True, exist_ok=True)

    song_ids = load_split(args.split_txt)
    # filter to songs that have features on disk
    song_ids = [sid for sid in song_ids if (args.feature_dir / f"{sid}.pt").exists()]

    print(f"Run tag:      {run_tag}")
    print(f"Split:        {args.split_txt.name}  ({len(song_ids)} songs with features)")
    print(f"Audio out:    {audio_out}")
    print(f"MuQ out:      {muq_out}")
    print(f"Analysis out: {analysis_out}")
    print(f"Original MuQ: {orig_dir}")

    if not args.analyze_only:
        print("\nLoading models")
        musicgen, lm, conditioner = load_model(args.checkpoint, device)
        sr = musicgen.compression_model.sample_rate

        print("Loading MuQ for extraction ...")
        muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

        print("\nStep 1+2: Generate audio and extract MuQ")
        song_ids = step_generate_and_extract(
            song_ids, args.feature_dir,
            audio_out, muq_out,
            musicgen, lm, conditioner, muq_model,
            device, amp_dtype, sr,
            top_k=args.top_k, temperature=args.temperature, overlap_s=args.overlap_s,
            uncond=args.uncond, noise=args.noise,
            n_muq_layers=args.n_muq_layers,
        )

    print("\nStep 3: Analyse")
    step_analyze(
        song_ids, muq_out, orig_dir, analysis_out,
        top_n=args.top_n, hit_n=args.hit_n,
        quiet=args.quiet, generate_plots=args.generate_plots,
    )

    print("\nDone.")
