# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark for Qwen3-Omni: speed measurement and WER evaluation.

Combines benchmark_omni_tts_speed.py and voice_clone_omni_wer.py into a
two-phase pipeline: generate audio while the server is running, then
transcribe without the server to avoid GPU OOM.

The benchmark always persists generated WAVs to disk so the follow-up
transcribe phase can reuse them without regenerating audio.  Callers who
only need speed numbers may still pass ``--generate-only`` and discard the
resulting ``audio/`` directory.

Usage (run from project root so ``benchmarks`` is on sys.path; use
``python -m benchmarks.eval.benchmark_omni_seedtts`` if invoking via a
subprocess from another directory):
    # Full pipeline (generate + transcribe)
    python -m benchmarks.eval.benchmark_omni_seedtts \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_en \
        --model qwen3-omni --port 8000 --max-samples 50

    # Phase 1: generate audio only (server must be running)
    python -m benchmarks.eval.benchmark_omni_seedtts \
        --generate-only \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_en \
        --model qwen3-omni --port 8000 --max-samples 50

    # Phase 2: transcribe + WER only (server not needed)
    python -m benchmarks.eval.benchmark_omni_seedtts \
        --transcribe-only \
        --meta seedtts_testset/en/meta.lst \
        --output-dir results/qwen3_omni_en \
        --model qwen3-omni --lang en --device cuda:0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from dataclasses import dataclass

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import (
    get_wav_duration,
    save_json_results,
    wait_for_service,
)
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import (
    VoiceCloneOmni,
    build_base_url,
    build_speed_results,
    print_speed_summary,
    run_seedtts_transcribe,
    save_generated_audio_metadata,
    save_speed_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_PREVIEW_LENGTH = 60


@dataclass
class OmniSeedttsBenchmarkConfig:
    model: str
    meta: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    lang: str = "en"
    speaker: str = "Ethan"
    voice_clone: bool = False
    output_dir: str = "results/omni_seedtts"
    max_samples: int | None = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    warmup: int = 1
    max_concurrency: int = 1
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    # Transcribe phase
    device: str = "cuda:0"


def _build_results_config(
    config: OmniSeedttsBenchmarkConfig,
    *,
    base_url: str,
) -> dict:
    return {
        "model": config.model,
        "base_url": base_url,
        "meta": config.meta,
        "voice_clone": config.voice_clone,
        "lang": config.lang,
        "speaker": config.speaker,
        "max_samples": config.max_samples,
        "max_new_tokens": config.max_new_tokens,
        "warmup": config.warmup,
        "max_concurrency": config.max_concurrency,
        "request_rate": config.request_rate,
    }


def make_send_fn(
    model_name: str,
    api_url: str,
    *,
    lang: str,
    voice_clone: bool,
    speaker: str,
    max_tokens: int,
    temperature: float,
    save_audio_dir: str,
) -> SendFn:
    """Return a SendFn that calls Qwen3-Omni via VoiceCloneOmni and saves WAV."""
    task = VoiceCloneOmni()

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:TEXT_PREVIEW_LENGTH],
        )
        start_time = time.perf_counter()
        try:
            wav_bytes, _, usage = await task.generate_speech(
                session,
                api_url,
                model_name,
                sample,
                lang,
                speaker=speaker,
                max_tokens=max_tokens,
                temperature=temperature,
                voice_clone=voice_clone,
            )
            result.audio_duration_s = get_wav_duration(wav_bytes)
            elapsed = time.perf_counter() - start_time
            if result.audio_duration_s > 0:
                result.is_success = True
                result.rtf = elapsed / result.audio_duration_s
            else:
                result.error = f"Invalid audio ({len(wav_bytes)} bytes)"

            if usage:
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)

            # note (Chenyang): engine_time_s should be the time taken by
            # the engine. Current omni chat completions has no X-Engine-Time
            # header, so we use request elapsed time as engine_time_s proxy.
            # This shall largely affect the results at high concurrency,
            # since the wait time is included in the request elapsed time.

            result.engine_time_s = elapsed
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s

            wav_path = os.path.join(save_audio_dir, f"{result.request_id}.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            result.wav_path = wav_path
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


async def run_omni_seedtts_benchmark(
    config: OmniSeedttsBenchmarkConfig,
) -> dict:
    """Generate audio and measure speed. Always saves audio for WER use.

    Returns a dict with keys: summary, per_request, config.
    """
    if not os.path.isfile(config.meta):
        raise FileNotFoundError(f"Meta file not found: {config.meta}")

    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_seedtts_samples(config.meta, config.max_samples)
    logger.info(f"Prepared {len(samples)} requests")

    save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
    os.makedirs(save_audio_dir, exist_ok=True)

    send_fn = make_send_fn(
        config.model,
        api_url,
        lang=config.lang,
        voice_clone=config.voice_clone,
        speaker=config.speaker,
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
        save_audio_dir=save_audio_dir,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, send_fn)

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    save_speed_results(outputs, metrics, results_config, config.output_dir)
    save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


def evaluate_generated_audio(config: OmniSeedttsBenchmarkConfig) -> dict:
    """Transcribe previously saved audio with ASR and compute WER + ASR speed.

    note (Chenyang): Server need not be running.

    Returns a dict with keys: wer_summary, asr_speed, per_sample.
    """
    wer_config = {
        "model": config.model,
        "speaker": config.speaker,
        "voice_clone": config.voice_clone,
        "meta": config.meta,
        "max_samples": config.max_samples,
    }
    return run_seedtts_transcribe(
        config,
        wer_config=wer_config,
        log_per_sample=True,
    )


def _config_from_args(args: argparse.Namespace) -> OmniSeedttsBenchmarkConfig:
    # ``--no-ref-audio`` is kept as a legacy alias so existing automation and
    # shell history keep working after the script merge.  ``--voice-clone``
    # remains the canonical flag.  If neither is passed the dataclass default
    # (``voice_clone=False``) applies.
    voice_clone = args.voice_clone and not args.no_ref_audio
    device = args.device if args.device is not None else args.asr_device
    return OmniSeedttsBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        meta=args.meta,
        lang=args.lang,
        speaker=args.speaker,
        voice_clone=voice_clone,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup=args.warmup,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        device=device,
    )


async def benchmark(config: OmniSeedttsBenchmarkConfig) -> dict:
    results = await run_omni_seedtts_benchmark(config)
    print_speed_summary(
        results["summary"], config.model, concurrency=config.max_concurrency
    )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SeedTTS benchmark for Qwen3-Omni: speed and WER evaluation."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-omni",
        help="Model name for the API request.",
    )
    # ``--testset`` is a legacy alias for ``--meta`` (kept for shell history).
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="seedtts_testset/en/meta.lst",
        help="Path to a meta.lst file (seed-tts-eval format).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Language for prompt construction and ASR.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ethan",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Speaker voice for TTS.",
    )
    # Voice-clone toggle: ``--voice-clone`` and ``--no-ref-audio`` are
    # complementary flags.  They map to a single ``voice_clone`` bool with the
    # dataclass default ``False`` (plain TTS, no reference audio).
    voice_clone_group = parser.add_mutually_exclusive_group()
    voice_clone_group.add_argument(
        "--voice-clone",
        dest="voice_clone",
        action="store_true",
        help="Pass ref_audio via 'audios' field for voice cloning.",
    )
    voice_clone_group.add_argument(
        "--no-ref-audio",
        dest="no_ref_audio",
        action="store_true",
        help="Legacy alias: disable voice cloning (equivalent to omitting "
        "--voice-clone; kept for backward-compatible shell history).",
    )
    parser.set_defaults(voice_clone=False, no_ref_audio=False)
    parser.add_argument("--output-dir", type=str, default="results/omni_seedtts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Legacy flag kept for backward compatibility. The unified "
        "benchmark always saves generated WAVs so the transcribe phase can "
        "reuse them; passing this flag is a no-op.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for ASR model (transcribe phase).",
    )
    parser.add_argument(
        "--asr-device",
        dest="asr_device",
        type=str,
        default="cuda:0",
        help="Legacy alias for --device (ASR transcription device).",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds to wait for server readiness.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Only synthesize audio and measure speed; skip WER transcription.",
    )
    mode.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only run ASR transcription and WER on existing output-dir.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _config_from_args(args)

    if args.save_audio:
        logger.info("--save-audio is a no-op: the unified benchmark always saves WAVs.")

    if args.transcribe_only:
        evaluate_generated_audio(config)
        return

    wait_for_service(build_base_url(config), timeout=args.server_timeout)
    gen_results = asyncio.run(benchmark(config))

    if args.generate_only:
        return

    accuracy_results = evaluate_generated_audio(config)
    combined = {
        "generation": {
            "speed": gen_results["summary"],
            "config": gen_results["config"],
            "per_request": gen_results["per_request"],
        },
        "accuracy": {
            "asr_speed": accuracy_results["asr_speed"],
            "wer": accuracy_results["wer_summary"],
        },
    }
    save_json_results(combined, config.output_dir, "eval_results.json")


if __name__ == "__main__":
    main()
