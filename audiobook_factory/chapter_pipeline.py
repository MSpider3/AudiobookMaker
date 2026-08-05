"""
audiobook_factory/chapter_pipeline.py
======================================
Three-stage overlapped pipeline for high-performance audiobook synthesis.

Stage A: Text preparation and static contiguous chunk distribution (CPU thread)
Stage B: Dedicated per-GPU synthesis worker threads with batch execution
Stage C: Streaming partial mastering and async disk I/O concatenation (CPU thread)
"""
from __future__ import annotations

from asyncio import CancelledError
import atexit
import concurrent.futures
from dataclasses import dataclass, field
import io
import logging
import os
import queue
import threading
from typing import Callable, Any, TYPE_CHECKING, List

import gc
import torch
from audiobook_factory.gpu_pool import GPUDetector, ProviderPool
from audiobook_factory.pipeline import AudiobookConfig, CancelToken, _cleanup_chunk_files, _chunk, _check_rust

if TYPE_CHECKING:
    from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

__all__ = ["run_chapter_pipeline", "_validate_chunk_file"]

_MAX_IN_MEMORY_CHUNK_SECONDS: float = 30.0
_PARTIAL_FLUSH_CHUNK_COUNT: int = 20
_ACQUIRE_POLL_TIMEOUT_SEC: float = 0.5
_MINIMUM_CHUNK_WAV_BYTES: int = 1000
# Minimum valid WAV file size. Files smaller than this are corrupted.


def _validate_chunk_file(path: str) -> bool:
    """Returns True if the chunk WAV file exists and is not corrupted.

    Args:
        path: File path to validate.

    Returns:
        True if file exists and is >= _MINIMUM_CHUNK_WAV_BYTES bytes.
    """
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < _MINIMUM_CHUNK_WAV_BYTES:
        logger.warning(
            "Chunk file too small (%d bytes, minimum %d): %s. "
            "Treating as corrupted and re-synthesizing.",
            size, _MINIMUM_CHUNK_WAV_BYTES, path,
        )
        return False
    return True


_disk_io_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="audiobookmaker_disk_io",
)
atexit.register(_disk_io_executor.shutdown, wait=True)


@dataclass
class _SynthResult:
    chunk_index: int
    audio: bytes | str
    duration: float


@dataclass
class _StageError:
    exception: BaseException


@dataclass
class _ProgressState:
    """Thread-safe shared progress counter for Stage B workers.

    Attributes:
        total: Total chunk count, set once before threads start.
        callback: Optional callback invoked after each increment.
        _done: Internal counter protected by _lock.
        _lock: Mutex for thread-safe increment.
    """

    total: int
    callback: Callable[[int, int], None] | None
    _done: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def increment(self) -> None:
        """Increments done count and calls callback if registered.

        Thread-safe.
        """
        with self._lock:
            self._done += 1
            done = self._done
        if self.callback is not None:
            self.callback(done, self.total)


def _concat_partial(chunk_paths: list[str], out_path: str, config: AudiobookConfig) -> None:
    """Concatenates chunk WAV files with pause padding into an intermediate partial WAV file without loudnorm."""
    valid_paths = [p for p in chunk_paths if p and os.path.exists(p) and os.path.getsize(p) >= 100]
    if not valid_paths:
        return
    import numpy as np
    import soundfile as sf

    pause_samples = np.zeros(int(config.pause * config.sample_rate), dtype=np.float32)
    segments = []
    for i, p in enumerate(valid_paths):
        try:
            data, sr = sf.read(p, dtype="float32")
            if len(data) > 0:
                segments.append(data)
                if i < len(valid_paths) - 1:
                    segments.append(pause_samples)
        except Exception as exc:
            logger.warning("Failed to read chunk %s during partial concat: %s", p, exc)
    if segments:
        raw = np.concatenate(segments)
        sf.write(out_path, raw, config.sample_rate)


def _master_final(partial_paths: list[str], out_path: str, config: AudiobookConfig) -> None:
    """Final mastering pass applying loudness normalization (LUFS/true_peak)."""
    valid_paths = [p for p in partial_paths if p and os.path.exists(p) and os.path.getsize(p) >= 100]
    if not valid_paths:
        return
    if _check_rust():
        import audiobook_rust

        bitrate_kbps = getattr(config, "bitrate_kbps", 64)
        audiobook_rust.master_audio(
            valid_paths,
            out_path,
            0.0,  # pause already inserted between chunks during partial concat
            int(config.sample_rate),
            float(config.lufs),
            float(config.true_peak),
            int(bitrate_kbps),
        )
    else:
        import numpy as np
        import soundfile as sf

        segments = []
        for p in valid_paths:
            try:
                data, sr = sf.read(p, dtype="float32")
                if len(data) > 0:
                    segments.append(data)
            except Exception as exc:
                logger.warning("Failed to read partial %s during final mastering: %s", p, exc)
        if segments:
            raw = np.concatenate(segments)
            sf.write(out_path, raw, config.sample_rate)


def _flush_accumulated_batch(
    accumulated: list[tuple[int, list[str]]],
    provider: BaseTTSProvider,
    voice_ref: bytes,
    config: AudiobookConfig,
    out_dir: str,
    master_queue: queue.Queue,
    cancel_token: CancelToken,
    progress_state: _ProgressState,
    chapter_index: int,
    chunk_completed_cb: Callable[[int], None] | None = None,
) -> None:
    """Synthesizes a batch of accumulated chunks in a single forward pass.

    Processes all accumulated (chunk_index, sentences) pairs together,
    using provider.synthesize_batch() for batched GPU tensor execution.
    Puts one _SynthResult per chunk into master_queue.

    Thread-safe for master_queue writes. Not thread-safe for provider — 
    caller must ensure exclusive provider ownership.

    Args:
        accumulated: List of (chunk_index, sentences) to process as a batch.
        provider: Provider instance exclusively owned by the calling thread.
        voice_ref: Voice reference WAV bytes.
        config: AudiobookConfig.
        out_dir: Chapter temp directory for chunk spill file storage.
        master_queue: Output queue to Stage C.
        cancel_token: Cancellation token.
        progress_state: Shared progress counter for progress_callback.
        chapter_index: Zero-based chapter index used for filename scoping.
        chunk_completed_cb: Optional callback invoked after each chunk is synthesized.
    """
    texts = [" ".join(sentences) for (_, sentences) in accumulated]

    logger.debug(
        "[flush_batch] voice_ref type=%s len=%s device=%s batch_size=%d",
        type(voice_ref).__name__,
        len(voice_ref) if isinstance(voice_ref, bytes) else "N/A",
        getattr(provider, "device", "unknown"),
        len(accumulated),
    )

    try:
        results: list[tuple[bytes | str, float]] = provider.synthesize_batch(
            texts=texts,
            voice_ref=voice_ref,
            return_bytes=True,
        )
    except CancelledError:
        master_queue.put(None)
        raise
    except Exception as exc:
        master_queue.put(_StageError(exc))
        return

    for (chunk_index, _), (audio_bytes, duration) in zip(accumulated, results):
        path = os.path.join(out_dir, f"chunk_ch_{chapter_index}_{chunk_index}.wav")
        if isinstance(audio_bytes, bytes):
            if len(audio_bytes) < 100:
                master_queue.put(_StageError(RuntimeError(f"Chunk {chunk_index} audio synthesis returned empty data ({len(audio_bytes)} bytes)")))
                continue
            try:
                with open(path, "wb") as fh:
                    fh.write(audio_bytes)
                result = _SynthResult(chunk_index, audio=path, duration=duration)
            except OSError as exc:
                master_queue.put(_StageError(exc))
                continue
        elif isinstance(audio_bytes, str):
            if not os.path.exists(audio_bytes) or os.path.getsize(audio_bytes) < 100:
                master_queue.put(_StageError(RuntimeError(f"Chunk {chunk_index} audio file invalid or empty")))
                continue
            result = _SynthResult(chunk_index, audio=audio_bytes, duration=duration)
        else:
            master_queue.put(_StageError(RuntimeError(f"Unexpected audio type: {type(audio_bytes)}")))
            continue

        master_queue.put(result)
        progress_state.increment()
        if chunk_completed_cb is not None:
            try:
                chunk_completed_cb(chunk_index)
            except Exception as cb_exc:
                logger.warning("chunk_completed_cb failed for chunk %d: %s", chunk_index, cb_exc)


def _stage_b_device_worker(
    device: str,
    provider: BaseTTSProvider,
    device_synth_queue: queue.Queue,
    master_queue: queue.Queue,
    voice_ref: bytes,
    config: AudiobookConfig,
    out_dir: str,
    cancel_token: CancelToken,
    progress_state: _ProgressState,
    chapter_index: int,
    chunk_completed_cb: Callable[[int], None] | None = None,
) -> None:
    """Dedicated synthesis thread for one GPU device.

    Consumes all chunks assigned to `device` from `device_synth_queue`,
    synthesizes audio using `provider` (already bound to `device`), and
    puts _SynthResult or _StageError items into `master_queue`.

    Sentinel behavior: on receiving None from device_synth_queue, flushes any
    remaining accumulated batch, puts one None into master_queue, and returns.

    Args:
        device: The CUDA device string this thread owns ("cuda:0", "cuda:1").
        provider: The BaseTTSProvider instance bound to this device.
        device_synth_queue: This device's exclusive input chunk queue.
        master_queue: Shared output queue to Stage C.
        voice_ref: Voice reference WAV bytes.
        config: AudiobookConfig for batch size and model settings.
        out_dir: Chapter temp directory for chunk file storage.
        cancel_token: Cooperative cancellation token.
        progress_state: Shared mutable counter for progress_callback tracking.
        chapter_index: Zero-based chapter index.
        chunk_completed_cb: Optional callback for progress JSON chunk update.
    """
    accumulated: list[tuple[int, list[str]]] = []

    try:
        while True:
            try:
                item = device_synth_queue.get(timeout=_ACQUIRE_POLL_TIMEOUT_SEC)
            except queue.Empty:
                if cancel_token.is_cancelled:
                    master_queue.put(None)
                    return
                continue

            if item is None:
                # Flush any remaining accumulated batch before exiting
                if accumulated:
                    _flush_accumulated_batch(
                        accumulated, provider, voice_ref, config,
                        out_dir, master_queue, cancel_token, progress_state, chapter_index,
                        chunk_completed_cb
                    )
                master_queue.put(None)
                return

            if cancel_token.is_cancelled:
                master_queue.put(None)
                return

            accumulated.append(item)

            batch_size = GPUDetector.suggest_batch_size(device, config.max_len)
            if len(accumulated) >= batch_size:
                _flush_accumulated_batch(
                    accumulated, provider, voice_ref, config,
                    out_dir, master_queue, cancel_token, progress_state, chapter_index,
                    chunk_completed_cb
                )
                accumulated.clear()
    except CancelledError:
        logger.debug("Stage B worker on %s cancelled cleanly.", device)
        return


def run_chapter_pipeline(
    sentences: list[str],
    voice_ref: bytes,
    out_wav_path: str,
    out_dir: str,
    chapter_index: int,
    config: AudiobookConfig,
    pool: ProviderPool,
    cancel_token: CancelToken,
    log_callback: Callable[[str], None],
    progress_callback: Callable[[int, int], None] | None = None,
    pinned_device: str | None = None,
    completed_chunks: list[int] | None = None,
    chunk_completed_cb: Callable[[int], None] | None = None,
) -> list[float]:
    """Synthesizes, masters, and writes one chapter using a 3-stage pipeline.

    Runs concurrent threads:
      Stage A: text preparation & static contiguous chunk distribution → device_synth_queues
      Stage B: dedicated per-GPU synthesis worker threads → master_queue
      Stage C: streaming partial mastering & async disk I/O → final WAV output

    Args:
        sentences: Pre-split sentence list for this chapter.
        voice_ref: Preprocessed voice reference audio bytes.
        out_wav_path: Full path where the final mastered WAV must be written.
        out_dir: Directory for temporary chunk files.
        chapter_index: Zero-based chapter index used for device affinity and logging.
        config: AudiobookConfig for this generation job.
        pool: The ProviderPool to acquire GPU providers from.
        cancel_token: CancelToken for cooperative cancellation.
        log_callback: Log callback for logging output.
        progress_callback: Optional progress callback receiving (chunks_done, total_chunks).
        pinned_device: Optional CUDA device string to lock all Stage B work to a single GPU.
        completed_chunks: Optional list of already-completed chunk indices for resume.
        chunk_completed_cb: Optional callback invoked after each chunk synthesis succeeds.

    Returns:
        List of float durations in seconds, one per synthesized sentence/chunk.

    Raises:
        CancelledError: If cancel_token.is_cancelled becomes True.
        RuntimeError: If any stage fails in a non-recoverable way.
    """
    if cancel_token.is_cancelled:
        raise CancelledError("Chapter pipeline cancelled before start.")

    os.makedirs(out_dir, exist_ok=True)

    # ── Pre-compute chunk list upfront ────────────────────────────────────────
    all_chunks: list[list[str]] = []
    for sent in sentences:
        for chunk in _chunk(sent, config.max_len):
            all_chunks.append([chunk])

    total_chunks = len(all_chunks)
    if total_chunks == 0:
        log_callback(f"  [Ch{chapter_index}] No text chunks to synthesize.")
        return []

    # ── Queues & Progress State ───────────────────────────────────────────────
    master_queue: queue.Queue[_SynthResult | _StageError | None] = queue.Queue()
    progress_state = _ProgressState(total=total_chunks, callback=progress_callback)

    # ── Pre-filter Cached Chunks vs Pending Chunks ─────────────────────────────
    cached_results: dict[int, _SynthResult] = {}
    pending_items: list[tuple[int, list[str]]] = []

    if getattr(config, "resume_incomplete_chunks", True) and completed_chunks:
        import soundfile
        for idx, chunk_list in enumerate(all_chunks):
            chunk_path = os.path.join(out_dir, f"chunk_ch_{chapter_index}_{idx}.wav")
            if idx in completed_chunks and _validate_chunk_file(chunk_path):
                try:
                    info = soundfile.info(chunk_path)
                    cached_results[idx] = _SynthResult(idx, audio=chunk_path, duration=info.duration)
                    progress_state.increment()
                    continue
                except Exception as exc:
                    logger.warning("Failed to read cached chunk %s: %s — re-synthesizing.", chunk_path, exc)
            pending_items.append((idx, chunk_list))
    else:
        pending_items = list(enumerate(all_chunks))

    cached_count = len(cached_results)
    pending_count = len(pending_items)

    # ── Determine active devices & Stage B worker count ───────────────────────
    if pinned_device is not None:
        active_devices = [pinned_device]
    else:
        active_devices = pool.devices

    stage_b_thread_count = len(active_devices)

    log_callback(
        f"  [Ch{chapter_index}] Synthesizing {total_chunks} chunk(s) "
        f"({cached_count} cached, {pending_count} pending) via 3-stage pipeline "
        f"({stage_b_thread_count} Stage B worker(s) on {', '.join(active_devices)})..."
    )

    device_synth_queues: dict[str, queue.Queue[tuple[int, list[str]] | None]] = {
        dev: queue.Queue() for dev in active_devices
    }

    # ── Stage A: Text Preparation & Static Contiguous Distribution ────────────
    def _stage_a_worker():
        try:
            total = len(pending_items)
            num_devs = len(active_devices)

            if pinned_device is not None:
                # Path A: All pending chunks go to pinned_device
                for item in pending_items:
                    if cancel_token.is_cancelled:
                        break
                    device_synth_queues[pinned_device].put(item)
                device_synth_queues[pinned_device].put(None)
            else:
                # Path B: Static contiguous split across all active devices
                counts = [(total + num_devs - 1 - i) // num_devs for i in range(num_devs)]
                offset = 0
                for i, dev in enumerate(active_devices):
                    chunk_slice = pending_items[offset : offset + counts[i]]
                    for item in chunk_slice:
                        if cancel_token.is_cancelled:
                            break
                        device_synth_queues[dev].put(item)
                    device_synth_queues[dev].put(None)  # Sentinel per device queue
                    offset += counts[i]
        except Exception as exc:
            logger.error("Stage A error on chapter %d: %s", chapter_index, exc)

    thread_a = threading.Thread(target=_stage_a_worker, name=f"StageA-Ch{chapter_index}", daemon=False)

    # ── Stage B & Provider Pre-acquisition ───────────────────────────────────
    device_providers: dict[str, BaseTTSProvider] = {}
    stage_b_threads: list[threading.Thread] = []

    durations_res: list[float] = [0.0] * total_chunks
    stage_c_exception: BaseException | None = None
    _chapter_succeeded: bool = False

    try:
        for device in active_devices:
            provider = pool.acquire(cancel_token=cancel_token, preferred_device=device)
            device_providers[device] = provider

        for device in active_devices:
            t = threading.Thread(
                target=_stage_b_device_worker,
                args=(
                    device,
                    device_providers[device],
                    device_synth_queues[device],
                    master_queue,
                    voice_ref,
                    config,
                    out_dir,
                    cancel_token,
                    progress_state,
                    chapter_index,
                    chunk_completed_cb,
                ),
                name=f"StageB-{device}-Ch{chapter_index}",
                daemon=False,
            )
            stage_b_threads.append(t)

        # ── Stage C: Mastering Thread ─────────────────────────────────────────
        def _stage_c_worker():
            nonlocal stage_c_exception
            received_chunks: dict[int, _SynthResult] = dict(cached_results)
            stage_b_active_count = stage_b_thread_count
            next_expected_index = 0
            accumulated_audio_paths: list[str] = []
            partial_files: list[str] = []
            temp_files_to_clean: list[str] = []
            pending_flushes: list[concurrent.futures.Future] = []
            _pipeline_failed = False
            pipeline_exc: BaseException | None = None

            try:
                # Drain pre-populated contiguous cached chunks upfront
                while next_expected_index in received_chunks:
                    res = received_chunks.pop(next_expected_index)
                    durations_res[res.chunk_index] = res.duration
                    if isinstance(res.audio, bytes):
                        chunk_tmp = os.path.join(out_dir, f"partial_chunk_{chapter_index}_{res.chunk_index}.wav")
                        with open(chunk_tmp, "wb") as f_tmp:
                            f_tmp.write(res.audio)
                        accumulated_audio_paths.append(chunk_tmp)
                        temp_files_to_clean.append(chunk_tmp)
                    else:
                        accumulated_audio_paths.append(res.audio)

                    if len(accumulated_audio_paths) >= _PARTIAL_FLUSH_CHUNK_COUNT:
                        p_path = os.path.join(out_dir, f"partial_{chapter_index}_{len(partial_files)}.wav")
                        paths_to_flush = list(accumulated_audio_paths)
                        future = _disk_io_executor.submit(_concat_partial, paths_to_flush, p_path, config)
                        pending_flushes.append(future)
                        partial_files.append(p_path)
                        accumulated_audio_paths.clear()

                    next_expected_index += 1

                while stage_b_active_count > 0:
                    if cancel_token.is_cancelled:
                        break
                    try:
                        item = master_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if item is None:
                        stage_b_active_count -= 1
                        continue
                    elif isinstance(item, _StageError):
                        _pipeline_failed = True
                        pipeline_exc = item.exception
                        stage_b_active_count -= 1
                        continue
                    elif isinstance(item, _SynthResult):
                        received_chunks[item.chunk_index] = item
                        if isinstance(item.audio, str):
                            temp_files_to_clean.append(item.audio)

                        while next_expected_index in received_chunks:
                            res = received_chunks.pop(next_expected_index)
                            durations_res[res.chunk_index] = res.duration

                            if isinstance(res.audio, bytes):
                                chunk_tmp = os.path.join(out_dir, f"partial_chunk_{chapter_index}_{res.chunk_index}.wav")
                                with open(chunk_tmp, "wb") as f_tmp:
                                    f_tmp.write(res.audio)
                                accumulated_audio_paths.append(chunk_tmp)
                                temp_files_to_clean.append(chunk_tmp)
                            else:
                                accumulated_audio_paths.append(res.audio)

                            if len(accumulated_audio_paths) >= _PARTIAL_FLUSH_CHUNK_COUNT:
                                p_path = os.path.join(out_dir, f"partial_{chapter_index}_{len(partial_files)}.wav")
                                paths_to_flush = list(accumulated_audio_paths)
                                future = _disk_io_executor.submit(_concat_partial, paths_to_flush, p_path, config)
                                pending_flushes.append(future)
                                partial_files.append(p_path)
                                accumulated_audio_paths.clear()

                            next_expected_index += 1

                # Flush remaining accumulated paths to final partial file
                if accumulated_audio_paths and not cancel_token.is_cancelled and not _pipeline_failed:
                    p_path = os.path.join(out_dir, f"partial_{chapter_index}_{len(partial_files)}.wav")
                    paths_to_flush = list(accumulated_audio_paths)
                    future = _disk_io_executor.submit(_concat_partial, paths_to_flush, p_path, config)
                    pending_flushes.append(future)
                    partial_files.append(p_path)
                    accumulated_audio_paths.clear()

                # Await all async partial WAV writes before final mastering pass
                for f in pending_flushes:
                    try:
                        f.result(timeout=60.0)
                    except Exception as exc:
                        raise RuntimeError(f"Async partial WAV write failed: {exc}") from exc

                # Final master concatenation into out_wav_path with LUFS & dBTP normalization
                if partial_files and not cancel_token.is_cancelled and not _pipeline_failed:
                    _master_final(partial_files, out_wav_path, config)

                if not os.path.exists(out_wav_path) or os.path.getsize(out_wav_path) < 100:
                    if not cancel_token.is_cancelled and not _pipeline_failed:
                        raise RuntimeError(f"Mastered chapter audio file was not created or empty at {out_wav_path}")

            except Exception as exc:
                stage_c_exception = exc
            finally:
                _cleanup_chunk_files(partial_files)
                _cleanup_chunk_files(temp_files_to_clean)

                if _pipeline_failed and pipeline_exc:
                    stage_c_exception = RuntimeError(f"Stage B failure: {pipeline_exc}")

        thread_c = threading.Thread(target=_stage_c_worker, name=f"StageC-Ch{chapter_index}", daemon=False)

        # ── Start Threads ─────────────────────────────────────────────────────
        thread_a.start()
        for t in stage_b_threads:
            t.start()
        thread_c.start()

        # ── Join Threads with Timeouts in Strict Order ─────────────────────────
        thread_a.join(timeout=30.0)
        for t in stage_b_threads:
            t.join(timeout=30.0)
        thread_c.join(timeout=30.0)

        if stage_c_exception is None and not cancel_token.is_cancelled and os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 0:
            _chapter_succeeded = True

    finally:
        # Always release providers after all threads are joined
        for device, provider in device_providers.items():
            pool.release(provider)

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        if _chapter_succeeded:
            chunk_files_to_clean = [
                os.path.join(out_dir, f"chunk_ch_{chapter_index}_{i}.wav")
                for i in range(total_chunks)
            ]
            _cleanup_chunk_files(chunk_files_to_clean)

    if cancel_token.is_cancelled:
        raise CancelledError("Chapter pipeline cancelled.")

    if stage_c_exception is not None:
        raise stage_c_exception

    return durations_res
