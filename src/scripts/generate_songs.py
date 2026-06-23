"""
Generate full-length MusicGen output for one or more songs, conditioned on
MuQ hierarchical features.

Each song is generated as a single continuous piece via autoregressive
continuation: 30s segments are chained using a short overlap prompt so the
output is musically coherent rather than a sequence of independent clips.

Conditioning tokens are computed once from the full-song MuQ features and
reused for every segment

To enable temporal awareness, train with --offset-emb.

Outputs (under --out-dir):
    audio/<run_tag>/<song_stem>_conditioned.wav
    audio/<run_tag>/<song_stem>_unconditioned.wav       (with --uncond)
    audio/<run_tag>/<song_stem>_noise.wav               (with --noise)
    muq_features/<run_tag>/<song_stem>_conditioned.pt   (with --extract-muq)
    muq_features/<run_tag>/<song_stem>_unconditioned.pt (with --uncond --extract-muq)
    muq_features/<run_tag>/<song_stem>_noise.pt         (with --noise --extract-muq)

Usage - single song:
    python src/scripts/generate_predictions.py --checkpoint <path-to-checkpoint> --feature-pt <path-to-muq-features-pt> --uncond --noise --extract-muq

Usage - full split:
    python src/scripts/generate_predictions.py --checkpoint <path-to-checkpoint> --split-txt <path-to-split-txt> --feature-dir <path-to-muq-features-dir> --uncond --extract-muq
"""

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

MUQ_SR = 24000



def process_song(
    feature_pt:    Path,
    musicgen,
    lm,
    conditioner,
    muq_model,
    audio_out_dir: Path,
    muq_out_dir:   Path,
    device:        torch.device,
    amp_dtype,
    sr:            int,
    top_k:         int,
    temperature:   float,
    overlap_s:     float,
    uncond:        bool,
    noise:         bool,
    extract_muq:   bool,
    n_muq_layers:  int,
    seed:          int = 42,
) -> None:
    """Generate a full-length conditioned (and optionally baseline) song.

    Skips any output file that already exists.
    Baselines (uncond, noise) use a fixed seed and RNG restore so the
    conditioned generation stream is not affected.
    """
    if not feature_pt.exists():
        print(f"[skip] {feature_pt.name} - feature file not found")
        return

    song_stem  = feature_pt.stem
    feat       = torch.load(feature_pt, weights_only=False)
    duration_s = float(feat["duration_s"])
    print(f"\n[song] {song_stem}  duration={duration_s:.1f}s")

    local, contextual, global_emb, dur_tensor = load_features(feature_pt, device)

    # conditioned
    wav_cond_path = audio_out_dir / f"{song_stem}_conditioned.wav"
    muq_cond_path = muq_out_dir   / f"{song_stem}_conditioned.pt"

    if not wav_cond_path.exists():
        print(f"  [gen]  {song_stem}_conditioned")
        wav = generate_song(
            lm, musicgen, conditioner,
            local, contextual, global_emb, dur_tensor,
            device, amp_dtype,
            top_k=top_k, temperature=temperature, overlap_s=overlap_s,
            null_cond=False,
        )
        _save_wav(wav_cond_path, wav, sr)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"  [skip] {song_stem}_conditioned")

    if extract_muq and not muq_cond_path.exists():
        _extract_and_save(wav_cond_path, muq_cond_path, muq_model, device, n_muq_layers)

    # unconditioned baseline
    if uncond:
        wav_null_path = audio_out_dir / f"{song_stem}_unconditioned.wav"
        muq_null_path = muq_out_dir   / f"{song_stem}_unconditioned.pt"

        if not wav_null_path.exists():
            print(f"  [gen]  {song_stem}_unconditioned")
            wav = generate_song(
                lm, musicgen, conditioner,
                local, contextual, global_emb, dur_tensor,
                device, amp_dtype,
                top_k=top_k, temperature=temperature, overlap_s=overlap_s,
                null_cond=True,
                noise_cond=False,
            )
            _save_wav(wav_null_path, wav, sr)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"  [skip] {song_stem}_unconditioned")

        if extract_muq and not muq_null_path.exists():
            _extract_and_save(wav_null_path, muq_null_path, muq_model, device, n_muq_layers)

    # noise baseline
    if noise:
        wav_noise_path = audio_out_dir / f"{song_stem}_noise.wav"
        muq_noise_path = muq_out_dir   / f"{song_stem}_noise.pt"

        if not wav_noise_path.exists():
            print(f"  [gen]  {song_stem}_noise")

            wav = generate_song(
                lm, musicgen, conditioner,
                local, contextual, global_emb, dur_tensor,
                device, amp_dtype,
                top_k=top_k, temperature=temperature, overlap_s=overlap_s,
                null_cond=False,
                noise_cond=True,
            )

            _save_wav(wav_noise_path, wav, sr)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"  [skip] {song_stem}_noise")

        if extract_muq and not muq_noise_path.exists():
            _extract_and_save(wav_noise_path, muq_noise_path, muq_model, device, n_muq_layers)


def _save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    pcm = (wav.squeeze(0).numpy() * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm)


def _extract_and_save(
    wav_path: Path, out_path: Path, muq_model, device, n_layers: int
) -> None:
    feat = extract_features(wav_path, muq_model, device, sr=MUQ_SR, n_layers=n_layers)
    torch.save(feat, out_path)
    print(f"         muq -> {out_path.name}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--feature-pt", type=Path, help="MuQ feature .pt file for a single song")
    source.add_argument("--split-txt", type=Path, help="Newline-separated list of song IDs (e.g. val.txt)")
    parser.add_argument("--feature-dir", type=Path, default=None, help="Directory of MuQ feature .pt files. Required with --split-txt.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/generated"))
    parser.add_argument("--run-tag", type=str,  default=None, help="Sub-directory tag (defaults to checkpoint filename stem)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--overlap-s", type=float, default=2.0, help="Seconds of previous segment used as prompt for continuation (default 2.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uncond", action="store_true", help="Also generate unconditioned baseline")
    parser.add_argument("--noise", action="store_true", help="Also generate noise-conditioned baseline")
    parser.add_argument("--extract-muq",  action="store_true", help="Extract MuQ features from each generated song")
    parser.add_argument("--n-muq-layers", type=int, default=8, help="MuQ layers to extract (default 8)")
    args = parser.parse_args()

    if args.split_txt is not None and args.feature_dir is None:
        parser.error("--feature-dir is required when using --split-txt")

    torch.manual_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    run_tag = args.run_tag or args.checkpoint.stem

    audio_out_dir = args.out_dir / "audio"        / run_tag
    muq_out_dir   = args.out_dir / "muq_features" / run_tag
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    if args.extract_muq:
        muq_out_dir.mkdir(parents=True, exist_ok=True)

    musicgen, lm, conditioner = load_model(args.checkpoint, device)
    sr = musicgen.compression_model.sample_rate

    muq_model = None
    if args.extract_muq:
        print("Loading MuQ for feature extraction ...")
        muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

    feature_pts = ([args.feature_pt] if args.feature_pt is not None else [args.feature_dir / f"{sid}.pt" for sid in load_split(args.split_txt)])
    if args.split_txt is not None:
        print(f"Split: {args.split_txt.name}  ({len(feature_pts)} songs)")

    shared = dict(
        musicgen=musicgen, lm=lm, conditioner=conditioner, muq_model=muq_model,
        audio_out_dir=audio_out_dir, muq_out_dir=muq_out_dir,
        device=device, amp_dtype=amp_dtype, sr=sr,
        top_k=args.top_k, temperature=args.temperature, overlap_s=args.overlap_s,
        uncond=args.uncond, noise=args.noise,
        extract_muq=args.extract_muq, n_muq_layers=args.n_muq_layers,
        seed=args.seed,
    )

    for feature_pt in feature_pts:
        process_song(feature_pt, **shared)

    print("\nDone.")
