import librosa
import numpy as np
import sys
import torch

from argparse import ArgumentParser
from muq import MuQ
from pathlib import Path
from tqdm import tqdm


TARGET_DURATION = 360  # seconds - cap; songs longer than this are truncated
LOCAL_WINDOW_S = 10    # seconds per local window
LOCAL_N_WINDOWS = 36   # 36 x 10s = 360s
CTX_WINDOW_S = 30      # seconds per contextual window
CTX_N_WINDOWS = 12     # 12 x 30s = 360s

assert TARGET_DURATION % LOCAL_WINDOW_S == 0, "TARGET_DURATION must be divisible by LOCAL_WINDOW_S"
assert TARGET_DURATION % CTX_WINDOW_S == 0, "TARGET_DURATION must be divisible by CTX_WINDOW_S"
assert LOCAL_N_WINDOWS == TARGET_DURATION // LOCAL_WINDOW_S
assert CTX_N_WINDOWS == TARGET_DURATION // CTX_WINDOW_S

sys.path.append(str(Path(__file__).resolve().parent.parent))


def _load_audio(song_path: Path, sr: int) -> tuple[np.ndarray, float]:
    """Load audio, truncating to TARGET_DURATION if needed. No padding.

    Returns the waveform (actual length) and duration in seconds.
    MuQ is run on audio - windows beyond duration_s are
    filled with zeros in extract_features.
    """
    wav, _ = librosa.load(song_path, sr=sr)
    duration_s = len(wav) / sr
    target_samples = TARGET_DURATION * sr
    if len(wav) > target_samples:
        wav = wav[:target_samples]
        duration_s = float(TARGET_DURATION)
    return wav, duration_s

#TODO: Find another foundation model to extract features with. FUTURE EXPERIMENT.

def extract_features(song_path: Path, muq: MuQ, device: torch.device, sr: int = 24000, n_layers: int = 8) -> dict[str, torch.Tensor]:
    """
    Extracts multi-scale MuQ features for a song.

    MuQ is run only on actual audio (no silence padding). Windows that fall
    beyond the song's real duration return zero vectors. duration_s is stored
    so the conditioner can create an attention mask during training.

    Returns a dict with:
        "local"       - [LOCAL_N_WINDOWS, n_layers, 1024]  (36 x 10s windows)
        "contextual"  - [CTX_N_WINDOWS,   n_layers, 1024]  (12 x 30s windows)
        "global"      - [1,               n_layers, 1024]  (full-song mean pool)
        "duration_s"  - float, actual song duration        (seconds)
    """
    assert song_path.exists(), f"Audio file not found: {song_path}"
    assert sr > 0
    assert 1 <= n_layers <= 13, f"n_layers must be between 1 and 13, got {n_layers}"

    wav, duration_s = _load_audio(song_path, sr)
    assert len(wav) > 0, f"Audio file is empty: {song_path}"

    audio = torch.tensor(wav).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = muq(audio, output_hidden_states=True)

    assert len(feats.hidden_states) > 0, "MuQ returned no hidden states"

    # Stack all layers then keep the last n_layers (upper layers carry
    # higher-level semantic and structural information)
    hidden = torch.stack([h.squeeze(0) for h in feats.hidden_states])  # [total_layers, T, 1024]
    assert hidden.ndim == 3 and hidden.shape[2] == 1024, f"Unexpected hidden state shape: {hidden.shape}"
    hidden = hidden[-n_layers:]  # [n_layers, T, 1024]

    _, T, D = hidden.shape
    fps = T / duration_s  # frames per second based on actual audio length
    assert int(LOCAL_WINDOW_S * fps) > 0, f"fps={fps:.2f} too low to resolve {LOCAL_WINDOW_S}s windows"

    zero_window = torch.zeros(n_layers, D, device=hidden.device)

    def pool_window(start_s: float, end_s: float) -> torch.Tensor:
        """Mean-pool hidden states over [start_s, end_s). Returns zeros if beyond song."""
        if start_s >= duration_s:
            return zero_window
        s = int(start_s * fps)
        e = int(min(end_s, duration_s) * fps)
        if e <= s:
            return zero_window
        return hidden[:, s:e, :].mean(dim=1)  # [n_layers, D]                   # maybe investigating another approach compared to means is worth doing here?

    global_emb = hidden.mean(dim=1).unsqueeze(0)  # [1, n_layers, 1024]
    contextual_emb = torch.stack([
        pool_window(i * CTX_WINDOW_S, (i + 1) * CTX_WINDOW_S)
        for i in range(CTX_N_WINDOWS)
    ])  # [12, n_layers, 1024]
    local_emb = torch.stack([
        pool_window(i * LOCAL_WINDOW_S, (i + 1) * LOCAL_WINDOW_S)
        for i in range(LOCAL_N_WINDOWS)
    ])  # [36, n_layers, 1024]

    return {
        "local": local_emb.cpu(),
        "contextual": contextual_emb.cpu(),
        "global": global_emb.cpu(),
        "duration_s": duration_s,
    }


def process_directory(wav_dir: Path, muq: MuQ, device: torch.device, output_dir: Path, sr: int = 24000, n_layers: int = 6) -> None:
    wav_paths = sorted(wav_dir.glob("*.wav"))
    assert len(wav_paths) > 0, f"No .wav files found in {wav_dir}"

    skipped, processed, failed = 0, 0, 0

    for wav_path in tqdm(wav_paths, desc="Processing audio files", total=len(wav_paths)):
        out_path = output_dir / f"{wav_path.stem}.pt"
        if out_path.exists():
            tqdm.write(f"[skip] - {wav_path.name}")
            skipped += 1
            continue
        try:
            features = extract_features(wav_path, muq, device, sr=sr, n_layers=n_layers)
            torch.save(features, out_path)
            tqdm.write(f"[done]  {wav_path.name}  >  {out_path.name}")
            processed += 1
        except Exception as e:
            tqdm.write(f"[fail]  {wav_path.name}  -  {e}")
            failed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wav-file", type=Path, required=False, help="Path to a single .wav file to process.")
    parser.add_argument("--wav-dir", type=Path, required=False, help="Path to a directory of .wav files to process.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--n-layers", type=int, default=8) # default is 8 because, as per paper Layer-wise Investigation of Large-Scale Self-Supervised Music Representation Models they are most significant layers for structure analysis (I've chosen all > 76 accuracy).
    args = parser.parse_args()

    output_dir = args.output_dir / "muq_features"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wav_file is not None:
        features = extract_features(args.wav_file, muq=muq, device=device, sr=args.sr, n_layers=args.n_layers)
        torch.save(features, output_dir / f"{args.wav_file.stem}.pt")
    elif args.wav_dir is not None:
        process_directory(wav_dir=args.wav_dir, muq=muq, device=device, output_dir=output_dir, sr=args.sr, n_layers=args.n_layers)