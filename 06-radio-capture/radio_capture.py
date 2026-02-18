"""
radio_capture.py
================
Captures live radio streams with optional Voice Activity Detection (VAD)
pre-screening so that recordings containing only music or adverts are skipped.

VAD workflow
------------
Before committing to a full (e.g. 1-hour) capture the code:
  1. Pulls a short sample (default 30 s) from the stream into memory as WAV.
  2. Runs silero-VAD on the sample and computes the fraction of frames that
     are classified as speech.
  3. If the ratio is below `VadConfig.speech_ratio_threshold` it waits a
     random interval and retries (up to `max_retries` times).
  4. Only when sufficient speech is detected does it start the full capture.

Dependencies (Colab)
--------------------
pip install soundfile
# silero-vad is downloaded automatically from torch hub on first use.
# torch / torchaudio are pre-installed in Colab.
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
    """Controls the Voice Activity Detection pre-screening step.

    Attributes
    ----------
    enabled:
        Set to False to bypass VAD entirely and always record.
    sample_duration_seconds:
        Length of the audio snippet sampled from the stream for analysis.
        Longer samples give more reliable estimates but add latency.
    speech_ratio_threshold:
        Minimum fraction of sample frames that silero-VAD must classify
        as speech before the full capture is started.
        Music-heavy segments typically score < 0.2; dialogue typically > 0.4.
    max_retries:
        How many sampling attempts to make before giving up on this
        scheduling slot and skipping the station.
    min_retry_delay_seconds / max_retry_delay_seconds:
        The actual wait between retries is chosen uniformly at random
        from this range (adds jitter to avoid always hitting the same
        part of the broadcast schedule).
    silero_threshold:
        Frame-level confidence threshold passed to silero-VAD's
        ``get_speech_timestamps``.  Lower = more sensitive but more
        false positives.
    sample_rate:
        Sample rate (Hz) used when resampling the snippet for VAD.
        silero-VAD supports 8 000 Hz and 16 000 Hz.
    """
    enabled: bool = True
    sample_duration_seconds: int = 30
    speech_ratio_threshold: float = 0.35
    max_retries: int = 3
    min_retry_delay_seconds: int = 30
    max_retry_delay_seconds: int = 120
    silero_threshold: float = 0.5
    sample_rate: int = 16_000


@dataclass(frozen=True)
class RadioCaptureConfig:
    base_output_dir: Path
    duration_seconds: int = 60 * 60
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
# VAD helpers
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
    The output is mono, resampled to *sample_rate* Hz, 16-bit PCM WAV.
    Returns None if ffmpeg fails or produces no output.
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
    """Return the fraction of audio frames silero-VAD labels as speech.

    Parameters
    ----------
    wav_bytes:
        Raw WAV file content (bytes).
    vad_cfg:
        VAD configuration (used for sample_rate and silero_threshold).

    Returns
    -------
    float in [0.0, 1.0] where 1.0 means the entire clip is speech.
    Returns 0.0 on any error.
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

    # Resample if the actual file SR doesn't match (shouldn't happen since
    # ffmpeg already resampled, but guard anyway).
    if file_sr != vad_cfg.sample_rate:
        try:
            import torchaudio.functional as F_torchaudio
            tensor = torch.tensor(audio_array).unsqueeze(0)
            tensor = F_torchaudio.resample(tensor, file_sr, vad_cfg.sample_rate)
            audio_array = tensor.squeeze(0).numpy()
        except Exception:
            pass  # proceed with whatever rate we have

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


def stream_has_dialogue(
    station_url: str,
    vad_cfg: VadConfig,
) -> tuple[bool, float]:
    """Sample the live stream and decide whether dialogue is being broadcast.

    Retries up to *vad_cfg.max_retries* times, each time waiting a random
    delay in [min_retry_delay_seconds, max_retry_delay_seconds] before
    re-sampling (mimics tuning in at different random moments).

    Returns
    -------
    (has_dialogue, best_speech_ratio)
        has_dialogue — True if threshold was met on any attempt.
        best_speech_ratio — highest speech ratio observed across all attempts.
    """
    best_ratio = 0.0

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
            ratio = compute_speech_ratio(wav_bytes, vad_cfg)
            best_ratio = max(best_ratio, ratio)
            print(
                f"  VAD: speech ratio = {ratio:.1%}  "
                f"(threshold = {vad_cfg.speech_ratio_threshold:.1%})"
            )

            if ratio >= vad_cfg.speech_ratio_threshold:
                print("  VAD: dialogue detected — proceeding with capture.")
                return True, best_ratio

            print("  VAD: likely music / advert.")

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
    return False, best_ratio


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
    """Capture audio from *station*, optionally gated by VAD.

    Returns
    -------
    (manifest_row, stop_requested)
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    status = "unknown"
    error = ""
    stop_requested = False
    vad_speech_ratio: float | None = None

    print(f"\n=== Station: {station.language} / {station.label} ===")

    # ------------------------------------------------------------------
    # VAD pre-screening
    # ------------------------------------------------------------------
    vad_cfg = config.vad
    if vad_cfg.enabled:
        if not _TORCH_AVAILABLE or not _SF_AVAILABLE:
            missing = []
            if not _TORCH_AVAILABLE:
                missing.append("torch")
            if not _SF_AVAILABLE:
                missing.append("soundfile")
            print(
                f"  VAD: disabled — missing packages: {', '.join(missing)}. "
                "Install them or set VadConfig(enabled=False) to suppress this warning."
            )
        else:
            try:
                has_dialogue, vad_speech_ratio = stream_has_dialogue(station.url, vad_cfg)
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
                        "path": "",
                    },
                    True,
                )

            if not has_dialogue:
                row = {
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
                    "path": "",
                }
                return row, False

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
        duration_seconds=60 * 60,
        station_subfolder="etop-radio-audios",
        vad=VadConfig(
            enabled=True,
            sample_duration_seconds=30,   # sample 30 s before deciding
            speech_ratio_threshold=0.35,  # 35 % of sample must be speech
            max_retries=3,
            min_retry_delay_seconds=30,   # wait 30–120 s between retries
            max_retry_delay_seconds=120,
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


if __name__ == "__main__":
    main_radio_capture()
