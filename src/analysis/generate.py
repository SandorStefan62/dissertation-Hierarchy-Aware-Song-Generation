import argparse
import sys
import torch
import scipy.io.wavfile as wavfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "audiocraft"))

from audiocraft.models.musicgen import MusicGen
from src.model.hierarchical_conditioner import HierarchicalConditioner
from src.model.lora import apply_lora, load_lora_state_dict


def patch_fuser(lm, cond_name: str = "muq") -> None:
    lm.fuser.fuse2cond.setdefault("cross", []).append(cond_name)
    lm.fuser.cond2fuse[cond_name] = "cross"


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})

    # read training config from checkpoint
    model_name       = saved_args.get("model",            "facebook/musicgen-small")
    n_layers         = saved_args.get("n_layers",         8)
    hidden_dim       = saved_args.get("hidden_dim",       1024)
    n_heads          = saved_args.get("n_heads",          8)
    n_encoder_layers = saved_args.get("n_encoder_layers", 2)
    lora_rank        = saved_args.get("lora_rank",        0)
    lora_alpha       = saved_args.get("lora_alpha",       8.0)
    lora_targets     = tuple(saved_args.get("lora_targets", "k,v").split(","))
    use_offset       = saved_args.get("offset_emb",       False)

    print(f"Loading {model_name} ...")
    musicgen = MusicGen.get_pretrained(model_name, device=device)
    lm = musicgen.lm

    patch_fuser(lm)
    for p in lm.parameters():
        p.requires_grad_(False)

    if lora_rank > 0:
        apply_lora(lm, rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
        load_lora_state_dict(lm, ckpt["lora"])
        print(f"LoRA loaded (rank={lora_rank}, targets={lora_targets})")

    conditioner = HierarchicalConditioner(
        n_layers=n_layers,
        muq_dim=1024,
        hidden_dim=hidden_dim,
        output_dim=lm.dim,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        use_offset_emb=use_offset,
    ).to(device)
    conditioner.load_state_dict(ckpt["conditioner"])
    conditioner.eval()
    lm.eval()

    print(f"Checkpoint loaded (epoch {ckpt.get('epoch', '?')})")
    return musicgen, lm, conditioner


def load_features(feature_pt: Path, device: torch.device):
    feat = torch.load(feature_pt, map_location="cpu", weights_only=False)
    return (
        feat["local"].unsqueeze(0).to(device),       # [1, 36, n_layers, 1024]
        feat["contextual"].unsqueeze(0).to(device),  # [1, 12, n_layers, 1024]
        feat["global"].unsqueeze(0).to(device),      # [1,  1, n_layers, 1024]
        torch.tensor([float(feat["duration_s"])], device=device),
    )


def generate_tokens(lm, cond_tokens: torch.Tensor, cond_mask: torch.Tensor,
                    duration_s: float = 30.0, use_sampling: bool = True,
                    top_k: int = 250, temperature: float = 1.0,
                    prompt: torch.Tensor | None = None,
                    text_prompt: str | None = None) -> torch.Tensor:
    """Run one segment of LM generation with the given conditioning tokens.

    Args:
        prompt: optional [1, K, T_prompt] code tensor from a previous segment
                for continuation generation. The output contains only the newly
                generated tokens (prompt is NOT included in the return value).

    Returns codes [1, K, T] where T = duration_s * 50.
    """
    from audiocraft.modules.conditioners import ConditioningAttributes

    max_new_tokens = int(duration_s * 50)  # EnCodec frame rate = 50 Hz
    B_muq = cond_tokens.shape[0]

    original_forward = lm.forward

    def patched_forward(sequence, conditions, condition_tensors=None, **kwargs):
        """Patches original forward method to merge custom muq conditioning with original conditioning signals."""
        merged = dict(condition_tensors) if condition_tensors else {}
        B_seq = sequence.shape[0]
        if B_seq != B_muq:
            tokens = cond_tokens.expand(B_seq, -1, -1)
            mask   = cond_mask.expand(B_seq, -1)
        else:
            tokens, mask = cond_tokens, cond_mask
        merged["muq"] = (tokens, mask)
        return original_forward(sequence, conditions, condition_tensors=merged, **kwargs)

    device    = next(lm.parameters()).device
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    lm.forward = patched_forward
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
            out = lm.generate(
                prompt=prompt,
                max_gen_len=max_new_tokens,
                use_sampling=use_sampling,
                top_k=top_k,
                temp=temperature,
            )
    finally:
        lm.forward = original_forward

    # lm.generate returns prompt + new tokens when prompt is given; strip prompt
    if prompt is not None:
        out = out[:, :, prompt.shape[2]:]

    return out  # [1, K, T]


def generate_song(lm, musicgen, conditioner,
                  local: torch.Tensor, contextual: torch.Tensor,
                  global_emb: torch.Tensor, duration_s: torch.Tensor,
                  device: torch.device, amp_dtype,
                  top_k: int = 250, temperature: float = 1.0,
                  segment_s: float = 30.0, overlap_s: float = 2.0,
                  null_cond: bool = False, noise_cond: bool = False) -> torch.Tensor:
    """Generate a full song via autoregressive continuation.

    The conditioning tokens are computed once from the full-song MuQ features
    and reused for every segment. Each segment is seeded by the last overlap_s
    seconds of the previous segment so the generation is musically continuous.

    Args:
        duration_s:  target duration tensor [1] (seconds)
        overlap_s:   seconds of previous audio fed as prompt to the next segment
        null_cond:   if True use zero conditioning (unconditioned baseline)

    Returns wav [1, T_samples] float32 on CPU.
    """
    total_s      = float(duration_s.item())
    segment_frames  = int(segment_s  * 50)
    overlap_frames  = int(overlap_s  * 50)
    n_segments   = max(1, int(total_s // segment_s))

    n_tok = conditioner.N_TOKENS
    o_dim = conditioner.output_dim

    if not null_cond and not noise_cond:
        print("         Computing conditioning tokens ...")
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            cond_tokens, cond_mask = conditioner(
                local, contextual, global_emb, duration_s,
                torch.zeros(1, device=device),  # window_start_s unused without offset emb
            )

    if null_cond:
        cond_tokens = torch.zeros(1, n_tok, o_dim, device=device)
        cond_mask   = torch.zeros(1, n_tok, device=device)

    if noise_cond:
        cond_tokens = torch.randn(1, n_tok, o_dim, device=device)
        cond_mask   = torch.ones(1, n_tok, device=device)

    print("         Conditioning tokens generated.")

    print(f"         Generating {n_segments} segments ...")
    all_codes: torch.Tensor | None = None
    for seg_idx in range(n_segments):
        prompt = (all_codes[:, :, -overlap_frames:] if all_codes is not None and overlap_frames > 0 else None)
        new_codes = generate_tokens(
            lm, cond_tokens, cond_mask,
            duration_s=segment_s, top_k=top_k, temperature=temperature,
            prompt=prompt,
        )
        all_codes = (torch.cat([all_codes, new_codes], dim=2) if all_codes is not None else new_codes)

    return decode_to_audio(musicgen, all_codes)  # [1, T_samples]


def decode_to_audio(musicgen, codes: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        wav = musicgen.compression_model.decode(codes, None)
    return wav.squeeze(0).cpu()  # [1, T_samples]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-pt", type=Path, required=True, help="Path to a single song's .pt MuQ feature file")
    parser.add_argument("--window-start", type=float, default=0.0, help="Window start in seconds (default 0)")
    parser.add_argument("--duration", type=float, default=30.0, help="Generation duration in seconds (default 30)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/generated"))
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    musicgen, lm, conditioner = load_model(args.checkpoint, device)

    local, contextual, global_emb, duration_s = load_features(args.feature_pt, device)
    window_start_s = torch.tensor([args.window_start], device=device)
    song_name = args.feature_pt.stem

    print(f"\nSong: {song_name}  |  duration={duration_s.item():.1f}s  |  window_start={args.window_start}s")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("\nGenerating conditioned ...")
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        cond_tokens, cond_mask = conditioner(
            local, contextual, global_emb, duration_s, window_start_s
        )
    codes_conditioned = generate_tokens(
        lm, cond_tokens, cond_mask,
        duration_s=args.duration, top_k=args.top_k, temperature=args.temperature,
        text_prompt=args.text_prompt,
    )

    # unconditioned generation (no MuQ conditioning)
    print("Generating unconditioned ...")
    null_tokens = torch.zeros_like(cond_tokens)
    null_mask   = torch.zeros(cond_mask.shape, dtype=torch.bool, device=device)
    codes_unconditioned = generate_tokens(
        lm, null_tokens, null_mask,
        duration_s=args.duration, top_k=args.top_k, temperature=args.temperature,
        text_prompt=args.text_prompt,
    )

    print("\nDecoding ...")
    wav_cond   = decode_to_audio(musicgen, codes_conditioned)
    wav_uncond = decode_to_audio(musicgen, codes_unconditioned)

    sr = musicgen.compression_model.sample_rate
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{song_name}_w{int(args.window_start)}-{int(args.window_start + args.duration)}"
    out_cond   = args.out_dir / f"{tag}_conditioned.wav"
    out_uncond = args.out_dir / f"{tag}_unconditioned.wav"

    # wav tensors are [1, T_samples] float32 in [-1, 1]; scipy needs int16
    def save_wav(path, wav_tensor, sample_rate):
        pcm = (wav_tensor.squeeze(0).numpy() * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(str(path), sample_rate, pcm)

    save_wav(out_cond,   wav_cond,   sr)
    save_wav(out_uncond, wav_uncond, sr)

    print(f"\nSaved:")
    print(f"  conditioned   -> {out_cond}")
    print(f"  unconditioned -> {out_uncond}")
    print(f"\nListen and compare whether the conditioned output reflects the")
    print(f"musical character of '{song_name}'.")
