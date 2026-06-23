import librosa
import sys
import torch

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "audiocraft"))

from audiocraft.models.musicgen import MusicGen

TARGET_DURATION = 360  # seconds - matches get_embeddings.py



def process_directory(audio_dir: Path, output_dir: Path, model_name: str) -> None:
    wav_paths = sorted(audio_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    
    output_dir = output_dir / f"{model_name.replace('/', '-')}"

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} compression model on {device} ...")
    musicgen = MusicGen.get_pretrained(model_name, device=device)
    compression_model = musicgen.compression_model.eval()
    for p in compression_model.parameters():
        p.requires_grad_(False)

    sr = compression_model.sample_rate
    frame_rate = compression_model.frame_rate
    print(f"EnCodec  sample_rate={sr}  frame_rate={frame_rate}")

    skipped, processed, failed = 0, 0, 0

    for wav_path in tqdm(wav_paths, desc="Encoding"):
        out_path = output_dir / f"{wav_path.stem}.pt"
        if out_path.exists():
            tqdm.write(f"[skip] {wav_path.name}")
            skipped += 1
            continue

        try:
            wav, _ = librosa.load(wav_path, sr=sr, mono=True, duration=TARGET_DURATION)         # [T] float32
            wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)                             # [1, 1, T]

            # compress_model.encode expects [B, C, T]
            with torch.no_grad():
                codes, _ = compression_model.encode(wav_t.to(device))                           # [1, K, T_codes]

            codes = codes.squeeze(0).cpu()                                                      # [K, T_codes]
            torch.save({"codes": codes, "frame_rate": float(frame_rate)}, out_path)
            tqdm.write(f"[done] {wav_path.name}  codes={tuple(codes.shape)}")
            processed += 1

        except Exception as e:
            tqdm.write(f"[fail] {wav_path.name}  {e}")
            failed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio-dir",  type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model",      default="facebook/musicgen-small")
    args = parser.parse_args()
    process_directory(args.audio_dir, args.output_dir, args.model)
