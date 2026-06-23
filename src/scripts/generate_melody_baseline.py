import sys
import torch
import librosa
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "audiocraft"))

from audiocraft.models.musicgen import MusicGen


SONGS = [
    "HX_0082_dragosteadintei",
    "HX_0586_bed",
    "HX_0530_whatthehell",
    "HX_0172_marrythenight",
    "HX_0492_sm",
    "HX_0619_closer",
    "HX_0679_firsttime",
]

ORIG_DIR      = Path("datasets/processed/HX/audio")
OUT_DIR       = Path("experiments/generated/audio/musicgen-melody")
MAX_DURATION  = 120.0   # cap at 2 minutes
EXTEND_STRIDE = 18.0   # overlap stride for sliding window (must be < max_duration=30s)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading facebook/musicgen-melody ...")
    model = MusicGen.get_pretrained("facebook/musicgen-melody", device=device)
    model.extend_stride = EXTEND_STRIDE

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for song_stem in SONGS:
        orig_path = ORIG_DIR / f"{song_stem}.wav"
        if not orig_path.exists():
            print(f"\n[skip] {song_stem} - not found at {orig_path}")
            continue

        out_path = OUT_DIR / f"{song_stem}.wav"
        if out_path.exists():
            print(f"\n[skip] {song_stem} - already exists at {out_path}")
            continue

        y, _ = librosa.load(orig_path, sr=model.sample_rate, mono=True)
        melody_wav = torch.from_numpy(y).unsqueeze(0)   # [1, T]

        song_duration_s = y.shape[-1] / model.sample_rate
        gen_duration    = min(song_duration_s, MAX_DURATION)
        print(f"\n{song_stem}  (original {song_duration_s:.1f}s  →  generating {gen_duration:.1f}s)")

        model.set_generation_params(duration=gen_duration)

        with torch.no_grad():
            wav = model.generate_with_chroma(
                descriptions=[""],
                melody_wavs=melody_wav,
                melody_sample_rate=model.sample_rate,
                progress=True,
            )   # [1, 1, T_samples]

        pcm = (wav[0, 0].cpu().numpy() * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(str(out_path), model.sample_rate, pcm)
        print(f"  saved → {out_path}")

    print("\nDone.")
