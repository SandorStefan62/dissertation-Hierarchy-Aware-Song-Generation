import librosa
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset


class SongWindowDataset(Dataset):
    """
    Flat dataset of fixed-length audio windows paired with pre-computed MuQ features.

    Each item is one window_s-second chunk of a song together with the song's full
    hierarchical MuQ feature dict (local/contextual/global).

    When codes_dir is provided, items return pre-encoded EnCodec codes (sliced to
    the window) instead of raw audio. This eliminates per-step EnCodec inference
    during training - run precompute_codes.py once to populate codes_dir.

    Args:
        feature_dir:  directory of {song_id}.pt MuQ feature files
        audio_dir:    directory of {song_id}.wav audio files
        codes_dir:    directory of {song_id}.pt pre-encoded codes files (optional)
        window_s:     window length in seconds (default 30 - MusicGen's context)
        stride_s:     step between window starts (default = window_s, non-overlapping)
        audio_sr:     sample rate for audio loading (default 32000 - EnCodec)
        min_coverage: fraction of window that must be real audio (default 0.5)
        song_ids:     optional list of song IDs; if None uses all .pt files
        cache:        load all features + codes into RAM at init (default True).
                      On Linux, forked DataLoader workers share this memory via
                      copy-on-write, so RAM cost is ~1x regardless of num_workers.
                      Disable if you are RAM-constrained (~2.1 MB/song needed).

    Returns per item (codes mode):
        local           [36, n_layers, 1024]
        contextual      [12, n_layers, 1024]
        global          [1,  n_layers, 1024]
        duration_s      scalar tensor
        codes           [K, T_window] - pre-encoded EnCodec codes
        window_start_s  scalar tensor
        song_id         str

    Returns per item (audio fallback mode, codes_dir=None):
        ... same but "audio" [window_s * audio_sr] instead of "codes"
    """

    def __init__(
        self,
        feature_dir: Path,
        audio_dir: Path,
        codes_dir: Path | None = None,
        window_s: float = 30.0,
        stride_s: float | None = None,
        audio_sr: int = 32000,
        min_coverage: float = 0.5,
        song_ids: list[str] | None = None,
        cache: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.audio_dir = Path(audio_dir)
        self.codes_dir = Path(codes_dir) if codes_dir is not None else None
        self.window_s = window_s
        self.stride_s = stride_s if stride_s is not None else window_s
        self.audio_sr = audio_sr
        self.target_samples = int(window_s * audio_sr)
        min_real_s = window_s * min_coverage

        if song_ids is not None:
            pt_paths = [self.feature_dir / f"{sid}.pt" for sid in song_ids]
            pt_paths = [p for p in pt_paths if p.exists()]
        else:
            pt_paths = sorted(self.feature_dir.glob("*.pt"))

        if len(pt_paths) == 0:
            raise FileNotFoundError(f"No .pt feature files found in {self.feature_dir}")

        # in-memory caches: populated at init, shared COW with DataLoader workers on Linux.
        self._feat_cache:  dict[str, dict] | None = {} if cache else None
        self._codes_cache: dict[str, dict] | None = {} if (cache and codes_dir) else None

        self.index: list[tuple[str, Path, float]] = []  # (song_id, audio_path, window_start_s)
        missing = 0

        for pt_path in pt_paths:
            audio_path = self.audio_dir / f"{pt_path.stem}.wav"
            if not audio_path.exists():
                missing += 1
                continue

            if self.codes_dir is not None:
                codes_path = self.codes_dir / f"{pt_path.stem}.pt"
                if not codes_path.exists():
                    missing += 1
                    continue

            feat = torch.load(pt_path, weights_only=False)
            duration_s: float = float(feat["duration_s"])

            if self._feat_cache is not None:
                self._feat_cache[pt_path.stem] = feat
            if self._codes_cache is not None:
                self._codes_cache[pt_path.stem] = torch.load(codes_path, weights_only=True)

            start = 0.0
            while start < duration_s - min_real_s:
                self.index.append((pt_path.stem, audio_path, start))
                start += self.stride_s

        if missing > 0:
            print(f"[SongWindowDataset] warning: {missing} songs skipped (missing audio or codes)")

    def __len__(self) -> int:
        return len(self.index)

    def _load_feat(self, song_id: str) -> dict:
        if self._feat_cache is not None:
            return self._feat_cache[song_id]
        return torch.load(self.feature_dir / f"{song_id}.pt", weights_only=False)

    def _load_codes(self, song_id: str) -> dict:
        if self._codes_cache is not None:
            return self._codes_cache[song_id]
        return torch.load(self.codes_dir / f"{song_id}.pt", weights_only=True)

    def __getitem__(self, idx: int) -> dict:
        song_id, audio_path, window_start_s = self.index[idx]
        features = self._load_feat(song_id)

        if self.codes_dir is not None:
            codes_data = self._load_codes(song_id)
            codes: torch.Tensor = codes_data["codes"]       # [K, T_full]
            frame_rate: float   = codes_data["frame_rate"]

            start_frame = int(window_start_s * frame_rate)
            end_frame   = int((window_start_s + self.window_s) * frame_rate)
            T_target    = end_frame - start_frame

            window_codes = codes[:, start_frame:end_frame]  # [K, T_window]
            if window_codes.shape[1] < T_target:
                window_codes = F.pad(window_codes, (0, T_target - window_codes.shape[1]))

            return {
                "local":          features["local"],
                "contextual":     features["contextual"],
                "global":         features["global"],
                "duration_s":     torch.tensor(float(features["duration_s"])),
                "codes":          window_codes,
                "window_start_s": torch.tensor(window_start_s),
                "song_id":        song_id,
            }

        # audio fallback (codes_dir not set)
        wav, _ = librosa.load(
            audio_path,
            sr=self.audio_sr,
            offset=window_start_s,
            duration=self.window_s,
            mono=True,
        )
        if len(wav) < self.target_samples:
            wav = np.pad(wav, (0, self.target_samples - len(wav)))

        return {
            "local":          features["local"],
            "contextual":     features["contextual"],
            "global":         features["global"],
            "duration_s":     torch.tensor(float(features["duration_s"])),
            "audio":          torch.from_numpy(wav),
            "window_start_s": torch.tensor(window_start_s),
            "song_id":        song_id,
        }


def load_split(split_txt: Path) -> list[str]:
    return [line.strip() for line in split_txt.read_text().splitlines() if line.strip()]
