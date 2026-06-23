import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))   # append repo dir to script
print(Path(__file__).parent.parent.parent)


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False

def _wsl_host_free_gb() -> float | None:
    """Query the Windows host for free space on the drive holding the WSL virtual disk.

    WSL2 stores its filesystem inside a .vhdx file on a Windows drive. shutil.disk_usage
    reports the virtual disk's capacity, not the real Windows drive's free space, so we
    go to the source: find the distro's base path from the registry, extract the drive
    letter, and ask Windows directly.
    """
    distro = os.environ.get("WSL_DISTRO_NAME", "")
    if not distro:
        return None
    try:
        ps_basepath = (
            f"(Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss"
            f" | Where-Object {{$_.GetValue('DistributionName') -eq '{distro}'}}"
            f").GetValue('BasePath')"
        )
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_basepath],
            capture_output=True, text=True, timeout=10,
        )
        base_path = r.stdout.strip()
        if not base_path or len(base_path) < 3:
            return None

        drive = base_path[0].upper()  # "C:\Users\..." -> "C"
        ps_free = f'(New-Object System.IO.DriveInfo("{drive}")).AvailableFreeSpace'
        r2 = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_free],
            capture_output=True, text=True, timeout=10,
        )
        return int(r2.stdout.strip()) / (1024 ** 3)
    except Exception:
        return None

def _free_space_gb(path: Path) -> float:
    if _is_wsl():
        host_gb = _wsl_host_free_gb()
        if host_gb is not None:
            return host_gb
        # Registry lookup failed - fall through with a warning
        print("Warning: could not query Windows host disk space; falling back to WSL virtual disk size.")
    return shutil.disk_usage(path).free / (1024 ** 3)

def _yt_dlp_available() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def _load_log(log_csv: Path) -> dict[str, str]:
    """Returns {song_id: status} from existing log. Status: 'ok' | 'fail' | 'skip'."""
    if not log_csv.exists():
        return {}
    with open(log_csv, newline="") as f:
        return {row["id"]: row["status"] for row in csv.DictReader(f)}

def _append_log(log_csv: Path, song_id: str, status: str, note: str = "") -> None:
    write_header = not log_csv.exists()
    with open(log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "status", "note"])
        if write_header:
            writer.writeheader()
        writer.writerow({"id": song_id, "status": status, "note": note})

def _validate_duration(wav_path: Path, expected_s: float, tolerance_s: float) -> bool:
    try:
        info = sf.info(str(wav_path))
        actual = info.frames / info.samplerate
        return abs(actual - expected_s) <= tolerance_s
    except Exception:
        return False

def _download_song(
    song_id: str,
    youtube_url: str,
    expected_duration: float,
    output_dir: Path,
    log_csv: Path,
    duration_tolerance: float,
    stop_event: threading.Event,
    min_free_gb: float,
) -> str:
    """Downloads one song. Returns status string: 'ok' | 'fail' | 'skip' | 'abort'."""
    if stop_event.is_set():
        return "abort"

    if _free_space_gb(output_dir) < min_free_gb:
        stop_event.set()
        return "abort"

    out_path = output_dir / f"{song_id}.wav"

    if out_path.exists():
        if _validate_duration(out_path, expected_duration, duration_tolerance):
            _append_log(log_csv, song_id, "skip", "already exists")
            return "skip"
        else:
            # Exists but duration is wrong - re-download
            out_path.unlink()

    # yt-dlp: download best audio, convert to wav via ffmpeg
    # --no-playlist: never expand playlists
    # -o: output template - yt-dlp adds .wav extension automatically
    tmp_stem = output_dir / song_id
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",       # best quality
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--output", str(tmp_stem) + ".%(ext)s",
        youtube_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            note = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            _append_log(log_csv, song_id, "fail", note[:200])
            return "fail"
    except subprocess.TimeoutExpired:
        _append_log(log_csv, song_id, "fail", "timeout")
        return "fail"
    except Exception as e:
        _append_log(log_csv, song_id, "fail", str(e)[:200])
        return "fail"

    if not out_path.exists():
        _append_log(log_csv, song_id, "fail", "wav not produced")
        return "fail"

    if not _validate_duration(out_path, expected_duration, duration_tolerance):
        actual_s = sf.info(str(out_path)).frames / sf.info(str(out_path)).samplerate
        note = f"duration mismatch: expected {expected_duration:.1f}s got {actual_s:.1f}s"
        _append_log(log_csv, song_id, "fail", note)
        out_path.unlink()
        return "fail"

    _append_log(log_csv, song_id, "ok")
    return "ok"

def load_unique_songs(jsonl_path: Path) -> dict[str, dict]:
    """Parse JSONL, deduplicate by song ID. Returns {id: {youtube_url, duration}}."""
    songs = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            song_id = row["id"]
            if song_id not in songs:
                songs[song_id] = {
                    "youtube_url": row["youtube_url"],
                    "duration": row["duration"],
                }
    return songs

def download_all(
    jsonl_path: Path,
    output_dir: Path,
    log_csv: Path,
    workers: int = 4,
    duration_tolerance: float = 10.0,
    min_free_gb: float = 20.0,
) -> None:
    assert _yt_dlp_available(), (
        "yt-dlp not found. Install with: pip install yt-dlp"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    free_gb = _free_space_gb(output_dir)
    print(f"Free disk space    : {free_gb:.1f} GB  (minimum kept: {min_free_gb:.1f} GB)")
    if free_gb <= min_free_gb:
        print(f"ERROR: already below the {min_free_gb} GB threshold. Aborting.")
        return

    songs = load_unique_songs(jsonl_path)
    existing_log = _load_log(log_csv)

    # Skip songs already marked 'ok' or 'skip' in a previous run
    pending = {
        sid: meta
        for sid, meta in songs.items()
        if existing_log.get(sid) not in ("ok", "skip")
    }

    already_done = len(songs) - len(pending)
    print(f"Total unique songs : {len(songs)}")
    print(f"Already done       : {already_done}")
    print(f"To download        : {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    stop_event = threading.Event()
    counts = {"ok": 0, "fail": 0, "skip": 0, "abort": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_song,
                sid,
                meta["youtube_url"],
                meta["duration"],
                output_dir,
                log_csv,
                duration_tolerance,
                stop_event,
                min_free_gb,
            ): sid
            for sid, meta in pending.items()
        }

        with tqdm(total=len(futures), desc="Downloading", unit="song") as pbar:
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    status = future.result()
                except Exception as e:
                    status = "fail"
                    _append_log(log_csv, sid, "fail", str(e)[:200])

                if status == "abort" and counts["abort"] == 0:
                    remaining = sum(1 for f in futures if not f.done())
                    tqdm.write(
                        f"\nDisk space below {min_free_gb:.0f} GB - stopping. "
                        f"{remaining} songs cancelled. Resume the script to continue when space is freed."
                    )
                    for f in futures:
                        f.cancel()

                counts[status] += 1
                pbar.set_postfix(
                    ok=counts["ok"], fail=counts["fail"],
                    skip=counts["skip"], abort=counts["abort"],
                )
                pbar.update(1)

    total = sum(counts.values())
    print(f"\nDone: {counts['ok']} ok, {counts['skip']} skipped, {counts['fail']} failed, {counts['abort']} aborted  (of {total})")
    print(f"Free disk space    : {_free_space_gb(output_dir):.1f} GB remaining")
    print(f"Log saved to: {log_csv}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Download SongFormDB-HX WAVs from YouTube via yt-dlp")
    parser.add_argument("--jsonl", type=Path, help="Path to SongFormDB-HX.jsonl")
    parser.add_argument("--output-dir", type=Path, help="Directory to save .wav files")
    parser.add_argument("--log-csv", type=Path, help="CSV file for download log (used for resuming)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers (default: 4)")
    parser.add_argument("--duration-tolerance", type=float, default=10.0, help="Max acceptable duration mismatch in seconds (default: 10)")
    parser.add_argument("--min-free-gb", type=float, default=20.0, help="Stop downloading if free disk space drops below this threshold in GB (default: 20)")
    args = parser.parse_args()

    download_all(
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        log_csv=args.log_csv,
        workers=args.workers,
        duration_tolerance=args.duration_tolerance,
        min_free_gb=args.min_free_gb,
    )

    # note: if any song downloads fail, follow the gan reconstruction tutorial from the author's huggingface page.