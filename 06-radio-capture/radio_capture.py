"""
radio_capture.py
================
Captures live radio streams with a two-gate Voice Activity Detection (VAD)
pre-screening so that recordings containing only music or adverts are skipped.

VAD workflow
------------
Before committing to a full capture the code:
  1. Pulls a short sample (default 30 s) from the stream into memory as WAV.

  Gate 1 — silero-VAD (speech presence)
  2. Runs silero-VAD on the sample and computes the fraction of frames that
     are classified as speech. Skips if below speech_ratio_threshold.

  Gate 2 — spectral music rejection (librosa)
  3. Computes spectral flatness and beat tempo on the same sample.
     Music (including music with vocals) has a tonal/harmonic spectrum
     (low spectral flatness) and a regular rhythmic pulse (detected tempo).
     Skips if the sample looks like music even when Gate 1 passes.

  4. If either gate fails, waits a random interval and retries.
  5. Only when both gates pass does the full capture start.

Dependencies (Colab)
--------------------
pip install soundfile librosa
# silero-vad is downloaded automatically from torch hub on first use.
# torch / torchaudio / librosa / numpy / pandas are pre-installed in Colab.
"""
from __future__ import annotations

import csv
import io
import random
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module still loads when they are
# absent (VAD will simply be unavailable).
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

try:
    import librosa
    import numpy as np
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "timestamp_utc",
    "language",
    "station_id",
    "label",
    "url",
    "requested_duration_sec",
    "actual_duration_sec",
    "bytes",
    "status",
    "error",
    "vad_speech_ratio",
    "vad_music_score",
    "path",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadioStation:
    language: str
    station_id: str
    label: str
    url: str


@dataclass(frozen=True)
class VadConfig:
    """Controls the two-gate Voice Activity Detection pre-screening step.

    Gate 1 — silero-VAD
    -------------------
    speech_ratio_threshold:
        Minimum fraction of sample frames that silero-VAD must classify as
        speech before Gate 2 is evaluated.
        Music-heavy segments typically score < 0.2; dialogue typically > 0.5.
        NOTE: Music with strong vocals (pop, gospel, R&B) can score 0.4–0.9
        on silero alone — use music_rejection_enabled (Gate 2) to catch these.

    Gate 2 — spectral music rejection (librosa)
    -------------------------------------------
    music_rejection_enabled:
        Set to False to disable the spectral music gate entirely.
    max_music_flatness:
        If mean spectral flatness is below this value the sample is classified
        as music and rejected. Speech has a noise-like (high-flatness) spectrum
        (~0.07–0.25); music with instruments/vocals is tonal (low-flatness,
        ~0.01–0.07).
    music_tempo_min_bpm:
        If a regular beat is detected above this tempo the sample is likely
        music (nearly all music genres exceed 70 BPM; speech has no pulse).
    max_flatness_with_tempo:
        Combined rule: reject when flatness < this value AND detected tempo
        exceeds music_tempo_min_bpm (catches vocal music whose flatness sits
        in the ambiguous 0.05–0.10 zone).

    Other
    -----
    enabled:
        Set to False to bypass VAD entirely and always record.
    sample_duration_seconds:
        Length of the audio snippet sampled from the stream for analysis.
    max_retries:
        How many sampling attempts before giving up and skipping the station.
    min_retry_delay_seconds / max_retry_delay_seconds:
        Wait between retries is chosen uniformly at random from this range
        so each attempt samples a different moment in the broadcast.
    silero_threshold:
        Per-frame confidence cutoff for silero-VAD. Lower = more sensitive.
    sample_rate:
        Sample rate (Hz) for VAD analysis. silero-VAD supports 8 000 / 16 000.
    """
    enabled: bool = True
    sample_duration_seconds: int = 30
    speech_ratio_threshold: float = 0.45
    max_retries: int = 3
    min_retry_delay_seconds: int = 30
    max_retry_delay_seconds: int = 120
    silero_threshold: float = 0.5
    sample_rate: int = 16_000
    # Gate 2 — music rejection
    music_rejection_enabled: bool = True
    max_music_flatness: float = 0.05
    music_tempo_min_bpm: float = 70.0
    max_flatness_with_tempo: float = 0.10


@dataclass(frozen=True)
class RadioCaptureConfig:
    base_output_dir: Path
    duration_seconds: int = 2 * 60   # 2-minute intervals by default
    audio_codec: str = "mp3"
    audio_bitrate: str = "128k"
    ffmpeg_loglevel: str = "warning"
    station_subfolder: str = "etop-radio-audios"
    manifest_filename: str = "radio_capture_manifest.csv"
    vad: VadConfig = field(default_factory=VadConfig)


# ---------------------------------------------------------------------------
# Colab / environment helpers
# ---------------------------------------------------------------------------

def mount_drive_if_colab(mount_point: str = "/content/drive") -> None:
    try:
        from google.colab import drive
    except ImportError:
        return
    drive.mount(mount_point)


def ensure_ffmpeg_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required tools: {missing_str}. "
            "In Colab, install with: !apt-get -qq install ffmpeg"
        )


# ---------------------------------------------------------------------------
# File / path helpers
# ---------------------------------------------------------------------------

def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value).strip("_")


def probe_duration_seconds(path: Path) -> float | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = completed.stdout.strip()
        return float(output) if output else None
    except Exception:
        return None


def build_output_path(config: RadioCaptureConfig, station: RadioStation) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{safe_name(station.language)}-{safe_name(station.label)}_"
        f"{ts}.{config.audio_codec}"
    )
    return (
        config.base_output_dir
        / safe_name(station.language).lower()
        / config.station_subfolder
        / file_name
    )


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def build_ffmpeg_command(
    station_url: str,
    out_path: Path,
    config: RadioCaptureConfig,
) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-loglevel", config.ffmpeg_loglevel,
        "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_at_eof", "1",
        "-reconnect_on_network_error", "1",
        "-reconnect_delay_max", "5",
        "-i", station_url,
        "-t", str(config.duration_seconds),
        "-vn", "-c:a", "libmp3lame", "-b:a", config.audio_bitrate,
        str(out_path),
    ]


def run_ffmpeg(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command))
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


# ---------------------------------------------------------------------------
# VAD helpers — Gate 1: silero-VAD (speech presence)
# ---------------------------------------------------------------------------

_silero_model_cache: tuple | None = None  # (model, utils)


def _load_silero_vad() -> tuple:
    """Load silero-VAD from torch hub (cached after first call)."""
    global _silero_model_cache
    if _silero_model_cache is None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for VAD. Install it with: pip install torch"
            )
        print("  VAD: loading silero-VAD model from torch hub…")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _silero_model_cache = (model, utils)
    return _silero_model_cache


def sample_audio_bytes(
    station_url: str,
    duration_seconds: int,
    sample_rate: int,
) -> bytes | None:
    """Capture *duration_seconds* of audio from *station_url* as a WAV blob.

    Uses ffmpeg piping to stdout — nothing is written to disk.
    Returns mono, *sample_rate* Hz, 16-bit PCM WAV, or None on failure.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-loglevel", "error",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_at_eof", "1",
        "-reconnect_on_network_error", "1",
        "-reconnect_delay_max", "5",
        "-i", station_url,
        "-t", str(duration_seconds),
        "-vn",
        "-ac", "1",               # mono
        "-ar", str(sample_rate),  # resample
        "-f", "wav",
        "pipe:1",                 # stdout
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=duration_seconds + 30,
        )
        if result.returncode == 0 and len(result.stdout) > 44:  # >WAV header
            return result.stdout
        return None
    except Exception as exc:
        print(f"  VAD: sample capture error — {exc}")
        return None


def compute_speech_ratio(
    wav_bytes: bytes,
    vad_cfg: VadConfig,
) -> float:
    """Return the fraction of audio frames silero-VAD labels as speech [0, 1].

    Returns 0.0 on any error (fail-safe: won't accidentally allow a capture).
    """
    if not _SF_AVAILABLE:
        raise RuntimeError(
            "soundfile is required for VAD. Install it with: pip install soundfile"
        )

    model, utils = _load_silero_vad()
    get_speech_timestamps = utils[0]

    try:
        audio_array, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as exc:
        print(f"  VAD: could not decode WAV — {exc}")
        return 0.0

    if file_sr != vad_cfg.sample_rate:
        try:
            import torchaudio.functional as F_torchaudio
            tensor = torch.tensor(audio_array).unsqueeze(0)
            tensor = F_torchaudio.resample(tensor, file_sr, vad_cfg.sample_rate)
            audio_array = tensor.squeeze(0).numpy()
        except Exception:
            pass

    audio_tensor = torch.tensor(audio_array)
    total_frames = len(audio_array)
    if total_frames == 0:
        return 0.0

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=vad_cfg.silero_threshold,
        sampling_rate=vad_cfg.sample_rate,
    )
    speech_frames = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    return speech_frames / total_frames


# ---------------------------------------------------------------------------
# VAD helpers — Gate 2: spectral music rejection (librosa)
# ---------------------------------------------------------------------------

def compute_music_score(
    wav_bytes: bytes,
    vad_cfg: VadConfig,
) -> float:
    """Return a musiciness score in [0.0, 1.0] using librosa spectral features.

    Two independent signals are combined:

    1. Spectral flatness — a tonal/harmonic spectrum (music) has very low
       flatness (~0.01–0.07); spoken dialogue is noise-like (high flatness,
       ~0.07–0.25). Music with sung vocals sits at ~0.03–0.08.

    2. Beat/tempo — music has a regular rhythmic pulse; conversation does not.
       ``librosa.beat.beat_track`` estimates tempo in BPM.

    Decision rules:
      is_music = (flatness < max_music_flatness)
                 OR (flatness < max_flatness_with_tempo AND tempo > music_tempo_min_bpm)

    Returns 1.0 if music is detected, 0.0 otherwise.
    Returns 0.0 (fail-open) on any error so music rejection never accidentally
    blocks a real dialogue capture due to a librosa failure.
    """
    if not vad_cfg.music_rejection_enabled:
        return 0.0
    if not _LIBROSA_AVAILABLE:
        print("  VAD Gate2: librosa not available — music check skipped.")
        return 0.0
    if not _SF_AVAILABLE:
        return 0.0

    try:
        audio_array, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as exc:
        print(f"  VAD Gate2: could not decode WAV — {exc}")
        return 0.0

    try:
        sr = vad_cfg.sample_rate

        # Spectral flatness: shape (1, n_frames) — mean across all frames
        flatness = librosa.feature.spectral_flatness(y=audio_array)
        mean_flatness = float(np.mean(flatness))

        # Beat / tempo detection
        tempo_bpm, _ = librosa.beat.beat_track(y=audio_array, sr=sr)
        tempo_bpm = float(tempo_bpm)

        is_music_by_flatness = mean_flatness < vad_cfg.max_music_flatness
        is_music_by_tempo = (
            mean_flatness < vad_cfg.max_flatness_with_tempo
            and tempo_bpm > vad_cfg.music_tempo_min_bpm
        )
        is_music = is_music_by_flatness or is_music_by_tempo

        print(
            f"  VAD Gate2: flatness={mean_flatness:.4f}  "
            f"tempo={tempo_bpm:.1f} BPM  "
            f"music={'YES — rejected' if is_music else 'no'}"
        )
        return 1.0 if is_music else 0.0

    except Exception as exc:
        print(f"  VAD Gate2: error computing music score — {exc}")
        return 0.0


def classify_audio(
    wav_bytes: bytes,
    vad_cfg: VadConfig,
) -> tuple[float, float, bool]:
    """Run both VAD gates on a WAV clip.

    Returns
    -------
    (speech_ratio, music_score, is_music)
    """
    speech_ratio = compute_speech_ratio(wav_bytes, vad_cfg)
    music_score = compute_music_score(wav_bytes, vad_cfg)
    return speech_ratio, music_score, music_score >= 1.0


def stream_has_dialogue(
    station_url: str,
    vad_cfg: VadConfig,
) -> tuple[bool, float, float]:
    """Sample the live stream and decide whether dialogue is being broadcast.

    Gate 1: silero speech ratio must meet speech_ratio_threshold.
    Gate 2: spectral analysis must NOT classify the clip as music.

    Retries up to *vad_cfg.max_retries* times with a random delay between
    attempts so each attempt samples a different moment in the broadcast.

    Returns
    -------
    (has_dialogue, best_speech_ratio, best_music_score)
    """
    best_ratio = 0.0
    best_music_score = 0.0

    for attempt in range(1, vad_cfg.max_retries + 1):
        print(
            f"  VAD check {attempt}/{vad_cfg.max_retries}: "
            f"sampling {vad_cfg.sample_duration_seconds}s from stream…"
        )

        wav_bytes = sample_audio_bytes(
            station_url=station_url,
            duration_seconds=vad_cfg.sample_duration_seconds,
            sample_rate=vad_cfg.sample_rate,
        )

        if wav_bytes is None:
            print("  VAD: failed to capture sample — stream may be unavailable.")
        else:
            speech_ratio, music_score, is_music = classify_audio(wav_bytes, vad_cfg)
            best_ratio = max(best_ratio, speech_ratio)
            best_music_score = max(best_music_score, music_score)

            print(
                f"  VAD Gate1: speech ratio = {speech_ratio:.1%}  "
                f"(threshold = {vad_cfg.speech_ratio_threshold:.1%})"
            )

            if is_music:
                print("  VAD: music detected by spectral analysis — skipping.")
            elif speech_ratio < vad_cfg.speech_ratio_threshold:
                print("  VAD: insufficient speech (likely silence / advert).")
            else:
                print("  VAD: dialogue detected — proceeding with capture.")
                return True, best_ratio, best_music_score

        if attempt < vad_cfg.max_retries:
            wait = random.randint(
                vad_cfg.min_retry_delay_seconds,
                vad_cfg.max_retry_delay_seconds,
            )
            print(f"  VAD: waiting {wait}s before next sample…")
            time.sleep(wait)

    print(
        f"  VAD: no dialogue detected after {vad_cfg.max_retries} attempts — "
        "skipping capture."
    )
    return False, best_ratio, best_music_score


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        file_obj.flush()


# ---------------------------------------------------------------------------
# Core capture logic
# ---------------------------------------------------------------------------

def capture_station(
    station: RadioStation,
    config: RadioCaptureConfig,
) -> tuple[dict[str, object], bool]:
    """Capture audio from *station*, gated by two-stage VAD.

    Returns
    -------
    (manifest_row, stop_requested)
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    status = "unknown"
    error = ""
    stop_requested = False
    vad_speech_ratio: float | None = None
    vad_music_score: float | None = None

    print(f"\n=== Station: {station.language} / {station.label} ===")

    # ------------------------------------------------------------------
    # VAD pre-screening
    # ------------------------------------------------------------------
    vad_cfg = config.vad
    if vad_cfg.enabled:
        if not _TORCH_AVAILABLE or not _SF_AVAILABLE:
            missing = [
                pkg for pkg, ok in [("torch", _TORCH_AVAILABLE), ("soundfile", _SF_AVAILABLE)]
                if not ok
            ]
            print(
                f"  VAD: disabled — missing packages: {', '.join(missing)}. "
                "Install them or set VadConfig(enabled=False) to suppress this warning."
            )
        else:
            try:
                has_dialogue, vad_speech_ratio, vad_music_score = stream_has_dialogue(
                    station.url, vad_cfg
                )
            except KeyboardInterrupt:
                return (
                    {
                        "timestamp_utc": ts,
                        "language": station.language,
                        "station_id": station.station_id,
                        "label": station.label,
                        "url": station.url,
                        "requested_duration_sec": config.duration_seconds,
                        "actual_duration_sec": None,
                        "bytes": 0,
                        "status": "interrupted",
                        "error": "KeyboardInterrupt during VAD",
                        "vad_speech_ratio": vad_speech_ratio,
                        "vad_music_score": vad_music_score,
                        "path": "",
                    },
                    True,
                )

            if not has_dialogue:
                return (
                    {
                        "timestamp_utc": ts,
                        "language": station.language,
                        "station_id": station.station_id,
                        "label": station.label,
                        "url": station.url,
                        "requested_duration_sec": config.duration_seconds,
                        "actual_duration_sec": None,
                        "bytes": 0,
                        "status": "vad_skipped",
                        "error": "No dialogue detected by VAD",
                        "vad_speech_ratio": vad_speech_ratio,
                        "vad_music_score": vad_music_score,
                        "path": "",
                    },
                    False,
                )

    # ------------------------------------------------------------------
    # Full capture
    # ------------------------------------------------------------------
    print(f"=== Recording: {station.language} / {station.label} ===")
    out_path = build_output_path(config=config, station=station)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_command(station_url=station.url, out_path=out_path, config=config)

    try:
        run_ffmpeg(cmd)
        status = "ok"
    except KeyboardInterrupt:
        status = "interrupted"
        error = "KeyboardInterrupt"
        stop_requested = True
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    bytes_written = out_path.stat().st_size if out_path.exists() else 0
    actual_duration = probe_duration_seconds(out_path)

    row = {
        "timestamp_utc": ts,
        "language": station.language,
        "station_id": station.station_id,
        "label": station.label,
        "url": station.url,
        "requested_duration_sec": config.duration_seconds,
        "actual_duration_sec": actual_duration,
        "bytes": bytes_written,
        "status": status,
        "error": error,
        "vad_speech_ratio": vad_speech_ratio,
        "vad_music_score": vad_music_score,
        "path": str(out_path),
    }
    return row, stop_requested


def capture_radio_streams(
    stations: Iterable[RadioStation],
    config: RadioCaptureConfig,
) -> tuple[list[dict[str, object]], Path]:
    ensure_ffmpeg_tools()

    manifest_path = config.base_output_dir / config.manifest_filename
    rows: list[dict[str, object]] = []

    for station in stations:
        row, stop_requested = capture_station(station=station, config=config)
        append_manifest_row(manifest_path=manifest_path, row=row)
        rows.append(row)

        if stop_requested:
            print("Capture interrupted by user. Stopping cleanly.")
            break

    return rows, manifest_path


# ---------------------------------------------------------------------------
# Playback helper
# ---------------------------------------------------------------------------

def listen_to_captures(
    manifest_path: str | Path,
    n: int = 1,
    status_filter: str = "ok",
) -> None:
    """Play back the most recent *n* successful captures inline in a notebook.

    Parameters
    ----------
    manifest_path:
        Path to the CSV manifest written by ``capture_radio_streams``.
    n:
        Number of recordings to play back, starting from the most recent.
        Pass n=-1 to play all matching rows.
    status_filter:
        Only rows whose ``status`` column matches this value are shown.
        Default is ``"ok"`` (successful captures only).

    Example
    -------
    listen_to_captures(manifest_path, n=3)
    """
    try:
        import pandas as pd
        from IPython.display import Audio, display
    except ImportError as exc:
        raise ImportError(
            "listen_to_captures requires pandas and IPython. "
            "In Colab both are pre-installed."
        ) from exc

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(
        manifest_path,
        header=None,
        names=MANIFEST_FIELDS,
        engine="python",
    )

    # Drop the header row if the CSV was written with one
    if not df.empty and list(df.iloc[0].values) == MANIFEST_FIELDS:
        df = df.iloc[1:].reset_index(drop=True)

    filtered = df[df["status"] == status_filter]

    if filtered.empty:
        print(f"No captures with status='{status_filter}' found in {manifest_path}")
        return

    rows_to_play = filtered.tail(n) if n != -1 else filtered
    rows_to_play = rows_to_play[::-1]  # most recent first

    for _, row in rows_to_play.iterrows():
        audio_path = Path(str(row["path"]))
        if not audio_path.exists():
            print(f"File not found, skipping: {audio_path}")
            continue
        label = (
            f"{row.get('label', '')} | {row.get('timestamp_utc', '')} | "
            f"{row.get('actual_duration_sec', '?')}s"
        )
        speech_pct = row.get("vad_speech_ratio")
        music_score = row.get("vad_music_score")
        if speech_pct not in (None, ""):
            try:
                label += f" | speech={float(speech_pct):.1%}"
            except (ValueError, TypeError):
                pass
        if music_score not in (None, ""):
            try:
                label += f" | music_score={float(music_score):.2f}"
            except (ValueError, TypeError):
                pass
        print(label)
        display(Audio(str(audio_path)))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_capture_summary(rows: list[dict[str, object]], manifest_path: Path) -> None:
    total = len(rows)
    ok_count = sum(row.get("status") == "ok" for row in rows)
    fail_count = sum(row.get("status") == "failed" for row in rows)
    skipped_count = sum(row.get("status") == "vad_skipped" for row in rows)
    interrupt_count = sum(row.get("status") == "interrupted" for row in rows)

    print("\n" + "=" * 50)
    print("RADIO CAPTURE SUMMARY")
    print("=" * 50)
    print(f"Total stations processed : {total}")
    print(f"  Successful             : {ok_count}")
    print(f"  Failed                 : {fail_count}")
    print(f"  VAD-skipped (no speech): {skipped_count}")
    print(f"  Interrupted            : {interrupt_count}")
    print(f"Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_radio_capture() -> None:
    mount_drive_if_colab()

    config = RadioCaptureConfig(
        base_output_dir=Path(
            "/content/drive/path_to_output"
        ),
        duration_seconds=2 * 60,      # 2-minute intervals
        station_subfolder="etop-radio-audios",
        vad=VadConfig(
            enabled=True,
            sample_duration_seconds=30,
            speech_ratio_threshold=0.45,   # raised from 0.35
            max_retries=3,
            min_retry_delay_seconds=30,
            max_retry_delay_seconds=120,
            music_rejection_enabled=True,  # Gate 2: spectral music rejection
            max_music_flatness=0.05,
            music_tempo_min_bpm=70.0,
            max_flatness_with_tempo=0.10,
        ),
    )

    stations = [
        RadioStation(
            language="Ateso",
            station_id="etop-fm-99-4",
            label="Etop",
            url="https://radio.garden/api/ara/content/listen/Mva5mveI/channel.mp3",
        ),
    ]

    rows, manifest_path = capture_radio_streams(stations=stations, config=config)
    print_capture_summary(rows=rows, manifest_path=manifest_path)

    # Play back the most recent successful capture in the notebook
    listen_to_captures(manifest_path, n=1)


if __name__ == "__main__":
    main_radio_capture()
