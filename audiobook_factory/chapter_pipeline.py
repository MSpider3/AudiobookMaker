"""
audiobook_factory/chapter_pipeline.py
======================================
Per-chapter synthesis pipeline.

Stage A: plans the chapter's chunks and fills one shared work queue, longest
         chunk first.
Stage B: one worker thread per GPU pulls batches off that queue, synthesizes
         them, verifies each chunk and writes it to the chunk cache.
Stage C: reassembles chunks in reading order, inserts pauses and masters the
         chapter to a loudness-normalised WAV.

The queue is shared rather than split per device, so the GPUs balance
themselves: a faster card simply takes more batches, and if one card fails
its batch goes back on the queue for the others.
"""
from __future__ import annotations

from asyncio import CancelledError
import atexit
import concurrent.futures
import dataclasses
from dataclasses import dataclass, field
import logging
import os
import queue
import threading
from typing import Callable, TYPE_CHECKING

import gc
import torch
from audiobook_factory.gpu_pool import GPUDetector, ProviderPool
from audiobook_factory.pipeline import AudiobookConfig, CancelToken, _cleanup_chunk_files, _chunk, _check_rust

if TYPE_CHECKING:
    from audiobook_factory.chunk_verifier import ChunkVerifier
    from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

__all__ = ["run_chapter_pipeline", "_validate_chunk_file"]

_PARTIAL_FLUSH_CHUNK_COUNT: int = 20
_MINIMUM_CHUNK_WAV_BYTES: int = 1000
# Minimum valid WAV file size. Files smaller than this are corrupted.
_SEQUENTIAL_BATCH_SIZE: int = 4
# Batch size for providers that synthesize one text at a time anyway: small
# enough that progress, cancellation and the chunk cache stay current.
_FATAL_VERDICT_SCORE: float = 10.0
# ChunkVerdict.score at or above this means the chunk has no usable audio.


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
            try:
                self.callback(done, self.total)
            except TypeError:
                try:
                    self.callback(done / self.total if self.total > 0 else 0.0)
                except Exception as exc:
                    logger.warning("Progress callback failed: %s", exc)


def _concat_partial(
    chunk_paths: list[str],
    out_path: str,
    config: AudiobookConfig,
    pauses: list[float] | None = None,
) -> None:
    """Concatenates chunk WAV files with pause padding into an intermediate partial WAV file without loudnorm.

    Args:
        chunk_paths: Chunk WAV files in reading order.
        out_path: Destination WAV.
        config: AudiobookConfig; ``config.pause`` is the default silence.
        pauses: Seconds of silence after each chunk, aligned with
            ``chunk_paths``. Defaults to ``config.pause`` for every chunk.
    """
    if pauses is None or len(pauses) != len(chunk_paths):
        pauses = [float(config.pause)] * len(chunk_paths)
    valid = [
        (p, pause) for p, pause in zip(chunk_paths, pauses)
        if p and os.path.exists(p) and os.path.getsize(p) >= 100
    ]
    if not valid:
        return
    valid_paths = [p for p, _ in valid]
    valid_pauses = [pause for _, pause in valid]
    import numpy as np
    import soundfile as sf

    # The provider decides the rate of its chunks; config.sample_rate is the
    # *output* rate and is applied when the chapter is encoded. Stamping the
    # partial with config.sample_rate would change pitch and speed whenever
    # the two differ.
    sr = 0
    fade_len = 0

    segments = []
    for i, p in enumerate(valid_paths):
        try:
            data, chunk_sr = sf.read(p, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if len(data) > 0:
                if sr == 0:
                    sr = int(chunk_sr)
                    fade_len = min(int(0.005 * sr), 120)  # 5ms micro-fade to eliminate digital clicks/pops
                # Apply 5ms micro fade-in and fade-out to prevent boundary clicks/pops
                if len(data) > 2 * fade_len and fade_len > 0:
                    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    data = data.copy()
                    data[:fade_len] *= fade_in
                    data[-fade_len:] *= fade_out
                segments.append(data)
                segments.append(np.zeros(int(max(0.0, valid_pauses[i]) * sr), dtype=np.float32))
        except Exception as exc:
            logger.warning("Failed to read chunk %s during partial concat: %s", p, exc)
    if segments:
        raw = np.concatenate(segments)
        sf.write(out_path, raw, sr)


def _master_final(partial_paths: list[str], out_path: str, config: AudiobookConfig) -> bool:
    """Final mastering pass applying loudness normalization (LUFS/true_peak).

    Returns:
        True when the written WAV is loudness-normalised. False when
        normalisation was unavailable and the audio was written as-is, so the
        caller can normalise while encoding instead.
    """
    valid_paths = [p for p in partial_paths if p and os.path.exists(p) and os.path.getsize(p) >= 100]
    if not valid_paths:
        return False
    normalized = True
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
        sr = int(config.sample_rate)
        for p in valid_paths:
            try:
                data, sr = sf.read(p, dtype="float32")
                if len(data) > 0:
                    segments.append(data)
            except Exception as exc:
                logger.warning("Failed to read partial %s during final mastering: %s", p, exc)
        if segments:
            raw = np.concatenate(segments)
            # Apply EBU R128 loudness normalization & true peak limiting in pure-Python
            try:
                import pyloudnorm as pyln
                meter = pyln.Meter(int(sr))
                input_loudness = meter.integrated_loudness(raw)
                if not np.isneginf(input_loudness) and not np.isnan(input_loudness):
                    target_lufs = float(config.lufs)
                    gain_db = target_lufs - input_loudness
                    target_tp_linear = 10.0 ** (float(config.true_peak) / 20.0)
                    gain_linear = 10.0 ** (gain_db / 20.0)
                    peak = float(np.max(np.abs(raw)))
                    if peak > 0 and (peak * gain_linear) > target_tp_linear:
                        gain_linear = target_tp_linear / peak
                    raw = raw * gain_linear
            except Exception as norm_err:
                normalized = False
                logger.warning("pyloudnorm mastering normalization fallback error: %s", norm_err)

            sf.write(out_path, raw, int(sr))
    return normalized


def _batch_size_for(device: str, provider: BaseTTSProvider, config: AudiobookConfig) -> int:
    """Chooses how many chunks a device synthesizes per provider call.

    Measured once per chapter, right after the previous chapter released its
    memory, so PyTorch's own allocator cache cannot shrink later batches.
    """
    explicit = int(getattr(config, "batch_size", 0) or 0)
    if explicit > 0:
        return explicit
    info = provider.info() if hasattr(provider, "info") else None
    if info is not None and not info.supports_batch:
        return _SEQUENTIAL_BATCH_SIZE
    return GPUDetector.suggest_batch_size(
        device, config.max_len, getattr(config, "vram_headroom_gb", 2.0)
    )


def _as_wav_bytes(audio: bytes | str, chunk_index: int) -> bytes:
    """Returns a provider result as WAV bytes, whether it handed back bytes or a path."""
    if isinstance(audio, (bytes, bytearray)):
        data = bytes(audio)
    elif isinstance(audio, str):
        with open(audio, "rb") as fh:
            data = fh.read()
    else:
        raise RuntimeError(f"Unexpected audio type for chunk {chunk_index}: {type(audio).__name__}")
    if len(data) < 100:
        raise RuntimeError(
            f"Chunk {chunk_index} audio synthesis returned empty data ({len(data)} bytes)"
        )
    return data


def _write_chunk_atomically(path: str, data: bytes) -> None:
    """Writes a chunk WAV so that a file on disk is always a complete chunk.

    The chunk cache treats an existing file as finished work, so a crash
    mid-write must never leave a truncated file under the final name.
    """
    tmp_path = f"{path}.part"
    with open(tmp_path, "wb") as fh:
        fh.write(data)
    os.replace(tmp_path, path)


def _verify_and_repair(
    provider: BaseTTSProvider,
    verifier: "ChunkVerifier",
    chunk_index: int,
    text: str,
    audio: bytes,
    duration: float,
    voice_ref: bytes,
    config: AudiobookConfig,
    cancel_token: CancelToken,
) -> tuple[bytes, float, str | None]:
    """Checks one chunk and re-synthesizes it while it fails.

    Returns:
        ``(audio, duration, flag)``. ``flag`` is None when the chunk passed
        (possibly after a retry), otherwise the reason the best attempt was
        still rejected.

    Raises:
        RuntimeError: If every attempt produced no usable audio at all.
    """
    verdict = verifier.check(text, audio, duration)
    if verdict.ok:
        return audio, duration, None

    best = (verdict.score, audio, duration, verdict.reason)
    retries = max(0, int(getattr(config, "verify_max_retries", 2)))
    for attempt in range(1, retries + 1):
        if cancel_token.is_cancelled:
            break
        logger.info(
            "[verify] Chunk %d rejected (%s); re-synthesizing (%d/%d).",
            chunk_index, best[3], attempt, retries,
        )
        try:
            seed = int(getattr(config, "seed", -1) if getattr(config, "seed", -1) is not None else -1)
            if seed >= 0:
                # A fixed seed would reproduce the same bad take.
                provider.config = dataclasses.replace(config, seed=seed + attempt)
            new_audio, new_duration = provider.synthesize(text, voice_ref, return_bytes=True)
            new_audio = _as_wav_bytes(new_audio, chunk_index)
        except CancelledError:
            raise
        except Exception as exc:
            logger.warning("[verify] Re-synthesis of chunk %d failed: %s", chunk_index, exc)
            continue
        finally:
            provider.config = config
        new_verdict = verifier.check(text, new_audio, new_duration)
        if new_verdict.ok:
            return new_audio, new_duration, None
        if new_verdict.score < best[0]:
            best = (new_verdict.score, new_audio, new_duration, new_verdict.reason)

    if best[0] >= _FATAL_VERDICT_SCORE:
        raise RuntimeError(
            f"Chunk {chunk_index} has no usable audio after {retries + 1} attempt(s): {best[3]}"
        )
    return best[1], best[2], best[3]


def _synthesize_batch(
    batch: list[tuple[int, str]],
    provider: BaseTTSProvider,
    voice_ref: bytes,
    config: AudiobookConfig,
    out_dir: str,
    master_queue: queue.Queue,
    cancel_token: CancelToken,
    progress_state: _ProgressState,
    chapter_index: int,
    verifier: "ChunkVerifier | None" = None,
    chunk_completed_cb: Callable[[int], None] | None = None,
    chunks_completed_cb: Callable[[list[int]], None] | None = None,
    chunk_flagged_cb: Callable[[int, str], None] | None = None,
) -> None:
    """Synthesizes one batch, verifies it and hands each chunk to Stage C.

    Not thread-safe for the provider — the caller owns it exclusively.

    Args:
        batch: ``(chunk_index, text)`` pairs to synthesize together.
        provider: Provider instance exclusively owned by the calling thread.
        voice_ref: Voice reference WAV bytes.
        config: AudiobookConfig.
        out_dir: Chapter temp directory holding the chunk cache.
        master_queue: Output queue to Stage C.
        cancel_token: Cancellation token.
        progress_state: Shared progress counter.
        chapter_index: Chapter number used in chunk filenames.
        verifier: Optional chunk verifier.
        chunk_completed_cb: Optional per-chunk callback (legacy).
        chunks_completed_cb: Optional callback receiving every chunk index
            finished by this batch, called once.
        chunk_flagged_cb: Optional callback for chunks kept despite failing
            verification.

    Raises:
        Exception: Anything the provider raises; the caller decides what to
            do with the unfinished chunks.
    """
    texts = [text for _, text in batch]
    results = provider.synthesize_batch(texts=texts, voice_ref=voice_ref, return_bytes=True)
    if len(results) != len(batch):
        raise RuntimeError(
            f"{provider.get_name()} returned {len(results)} results for a batch of {len(batch)}."
        )

    finished: list[int] = []
    try:
        for (chunk_index, text), (audio, duration) in zip(batch, results):
            data = _as_wav_bytes(audio, chunk_index)
            if verifier is not None and verifier.enabled:
                data, duration, flag = _verify_and_repair(
                    provider, verifier, chunk_index, text, data, duration,
                    voice_ref, config, cancel_token,
                )
                if flag and chunk_flagged_cb is not None:
                    chunk_flagged_cb(chunk_index, flag)
            path = os.path.join(out_dir, f"chunk_ch_{chapter_index}_{chunk_index}.wav")
            _write_chunk_atomically(path, data)
            master_queue.put(_SynthResult(chunk_index, audio=path, duration=duration))
            finished.append(chunk_index)
            progress_state.increment()
            if chunk_completed_cb is not None:
                try:
                    chunk_completed_cb(chunk_index)
                except Exception as cb_exc:
                    logger.warning("chunk_completed_cb failed for chunk %d: %s", chunk_index, cb_exc)
    finally:
        if finished and chunks_completed_cb is not None:
            try:
                chunks_completed_cb(finished)
            except Exception as cb_exc:
                logger.warning("chunks_completed_cb failed: %s", cb_exc)


def _stage_b_device_worker(
    device: str,
    provider: BaseTTSProvider,
    work_queue: queue.Queue,
    master_queue: queue.Queue,
    voice_ref: bytes,
    config: AudiobookConfig,
    out_dir: str,
    cancel_token: CancelToken,
    progress_state: _ProgressState,
    chapter_index: int,
    worker_errors: list[BaseException],
    verifier: "ChunkVerifier | None" = None,
    chunk_completed_cb: Callable[[int], None] | None = None,
    chunks_completed_cb: Callable[[list[int]], None] | None = None,
    chunk_flagged_cb: Callable[[int, str], None] | None = None,
    device_stats: dict[str, int] | None = None,
) -> None:
    """Dedicated synthesis thread for one GPU device.

    Pulls batches from the shared ``work_queue`` until it is empty. If a
    batch fails, its unfinished chunks go back on the queue for the other
    devices and this worker stops; the chapter only fails if chunks are still
    missing once every worker has stopped. Always puts exactly one ``None``
    sentinel on ``master_queue`` when it exits.

    Args:
        device: The device string this thread owns ("cuda:0", "cuda:1").
        provider: The BaseTTSProvider instance bound to this device.
        work_queue: Shared queue of ``(chunk_index, text)`` items.
        master_queue: Shared output queue to Stage C.
        voice_ref: Voice reference WAV bytes.
        config: AudiobookConfig for batch size and model settings.
        out_dir: Chapter temp directory for chunk file storage.
        cancel_token: Cooperative cancellation token.
        progress_state: Shared mutable counter for progress_callback tracking.
        chapter_index: Chapter number.
        worker_errors: Shared list collecting the error of each failed worker.
        verifier: Optional chunk verifier.
        chunk_completed_cb: Optional per-chunk callback.
        chunks_completed_cb: Optional per-batch callback.
        chunk_flagged_cb: Optional callback for chunks that failed verification.
        device_stats: Optional dict receiving the number of chunks this
            device synthesized, keyed by device.
    """
    try:
        batch_size = max(1, _batch_size_for(device, provider, config))
        while not cancel_token.is_cancelled:
            batch: list[tuple[int, str]] = []
            while len(batch) < batch_size:
                try:
                    batch.append(work_queue.get_nowait())
                except queue.Empty:
                    break
            if not batch:
                break

            written: set[int] = set()

            def _note_written(indices: list[int]) -> None:
                written.update(indices)
                if device_stats is not None:
                    device_stats[device] = device_stats.get(device, 0) + len(indices)
                if chunks_completed_cb is not None:
                    chunks_completed_cb(indices)

            try:
                _synthesize_batch(
                    batch, provider, voice_ref, config, out_dir, master_queue,
                    cancel_token, progress_state, chapter_index, verifier,
                    chunk_completed_cb, _note_written, chunk_flagged_cb,
                )
            except CancelledError:
                logger.debug("Stage B worker on %s cancelled cleanly.", device)
                break
            except Exception as exc:
                unfinished = [item for item in batch if item[0] not in written]
                logger.error(
                    "Stage B worker on %s failed (%s: %s); returning %d chunk(s) to the queue.",
                    device, type(exc).__name__, exc, len(unfinished),
                )
                worker_errors.append(exc)
                for item in unfinished:
                    work_queue.put(item)
                break
    except Exception as exc:
        logger.exception("Stage B worker on %s crashed", device)
        worker_errors.append(exc)
    finally:
        # Stage C waits for one sentinel per worker.
        master_queue.put(None)


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
    *,
    chunk_pauses: list[float] | None = None,
    verifier: "ChunkVerifier | None" = None,
    chunks_completed_cb: Callable[[list[int]], None] | None = None,
    chunk_flagged_cb: Callable[[int, str], None] | None = None,
    master_info: dict | None = None,
) -> list[float]:
    """Synthesizes, masters, and writes one chapter.

    Args:
        sentences: The chapter's chunk texts in reading order. An entry
            longer than ``config.max_len`` is split further.
        voice_ref: Preprocessed voice reference audio bytes.
        out_wav_path: Full path where the final mastered WAV must be written.
        out_dir: Directory for temporary chunk files.
        chapter_index: Chapter number used for chunk filenames and logging.
        config: AudiobookConfig for this generation job.
        pool: The ProviderPool to acquire GPU providers from.
        cancel_token: CancelToken for cooperative cancellation.
        log_callback: Log callback for logging output.
        progress_callback: Optional progress callback receiving (chunks_done, total_chunks).
        pinned_device: Optional device string to lock all synthesis to a single GPU.
        completed_chunks: Chunk indices that may be reused from the cache
            (each must still be a valid file). ``None`` reuses every valid
            chunk file found in ``out_dir``; ``[]`` reuses nothing.
        chunk_completed_cb: Optional callback invoked after each chunk succeeds.
        chunk_pauses: Seconds of silence after each entry of ``sentences``.
            Defaults to ``config.pause`` for all.
        verifier: Optional ChunkVerifier applied to every synthesized chunk.
        chunks_completed_cb: Optional callback invoked once per batch with
            the chunk indices it finished.
        chunk_flagged_cb: Optional callback for chunks kept despite failing
            verification, receiving (chunk_index, reason).
        master_info: Optional dict; ``master_info["normalized"]`` is set to
            whether the written WAV is loudness-normalised.

    Returns:
        List of float durations in seconds, one per chunk.

    Raises:
        CancelledError: If cancel_token.is_cancelled becomes True.
        RuntimeError: If any stage fails in a non-recoverable way.
    """
    if cancel_token.is_cancelled:
        raise CancelledError("Chapter pipeline cancelled before start.")

    os.makedirs(out_dir, exist_ok=True)

    # ── Stage A: plan chunks ──────────────────────────────────────────────────
    all_chunks: list[str] = []
    pauses: list[float] = []
    for position, sent in enumerate(sentences):
        pieces = [piece for piece in _chunk(sent, config.max_len) if piece and piece.strip()]
        trailing = float(config.pause)
        if chunk_pauses is not None and position < len(chunk_pauses):
            trailing = float(chunk_pauses[position])
        for piece_index, piece in enumerate(pieces):
            all_chunks.append(piece)
            pauses.append(trailing if piece_index == len(pieces) - 1 else float(config.pause))

    total_chunks = len(all_chunks)
    if total_chunks == 0:
        log_callback(f"  [Ch{chapter_index}] No text chunks to synthesize.")
        return []

    master_queue: queue.Queue[_SynthResult | None] = queue.Queue()
    progress_state = _ProgressState(total=total_chunks, callback=progress_callback)

    # ── Reuse chunks already in the cache ─────────────────────────────────────
    cached_results: dict[int, _SynthResult] = {}
    pending_items: list[tuple[int, str]] = []
    reusable = None if completed_chunks is None else set(completed_chunks)
    resume = getattr(config, "resume_incomplete_chunks", True) and reusable != set()

    for idx, chunk_text in enumerate(all_chunks):
        if resume and (reusable is None or idx in reusable):
            chunk_path = os.path.join(out_dir, f"chunk_ch_{chapter_index}_{idx}.wav")
            if os.path.exists(chunk_path) and _validate_chunk_file(chunk_path):
                try:
                    import soundfile
                    info = soundfile.info(chunk_path)
                    cached_results[idx] = _SynthResult(idx, audio=chunk_path, duration=info.duration)
                    progress_state.increment()
                    continue
                except Exception as exc:
                    logger.warning("Failed to read cached chunk %s: %s — re-synthesizing.", chunk_path, exc)
        pending_items.append((idx, chunk_text))

    cached_count = len(cached_results)
    pending_count = len(pending_items)

    # Longest first: batches are then made of similar-length chunks (little
    # padding, no short chunk waiting on a long one), and the heaviest batch
    # runs first so an out-of-memory condition shows up immediately.
    work_queue: queue.Queue[tuple[int, str]] = queue.Queue()
    for item in sorted(pending_items, key=lambda entry: len(entry[1]), reverse=True):
        work_queue.put(item)

    # ── Determine active devices & Stage B worker count ───────────────────────
    if pinned_device is not None:
        active_devices = [pinned_device]
    else:
        active_devices = pool.devices

    stage_b_thread_count = len(active_devices)

    log_callback(
        f"  [Ch{chapter_index}] Synthesizing {total_chunks} chunk(s) "
        f"({cached_count} cached, {pending_count} pending) "
        f"on {stage_b_thread_count} device(s): {', '.join(active_devices)}..."
    )

    device_providers: dict[str, BaseTTSProvider] = {}
    stage_b_threads: list[threading.Thread] = []
    worker_errors: list[BaseException] = []
    device_stats: dict[str, int] = {device: 0 for device in active_devices}

    durations_res: list[float] = [0.0] * total_chunks
    stage_c_exception: BaseException | None = None
    _chapter_succeeded: bool = False

    try:
        for device in active_devices:
            provider = pool.acquire(cancel_token=cancel_token, preferred_device=device)
            # Pools outlive a single run, so a pooled provider still carries
            # the config of whichever job created it. It is exclusively ours
            # until release(), so point it at this job's settings.
            provider.config = config
            device_providers[device] = provider

        for device in active_devices:
            t = threading.Thread(
                target=_stage_b_device_worker,
                args=(
                    device,
                    device_providers[device],
                    work_queue,
                    master_queue,
                    voice_ref,
                    config,
                    out_dir,
                    cancel_token,
                    progress_state,
                    chapter_index,
                    worker_errors,
                    verifier,
                    chunk_completed_cb,
                    chunks_completed_cb,
                    chunk_flagged_cb,
                    device_stats,
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
            accumulated_pauses: list[float] = []
            partial_files: list[str] = []
            pending_flushes: list[concurrent.futures.Future] = []

            def _flush_partial() -> None:
                p_path = os.path.join(out_dir, f"partial_{chapter_index}_{len(partial_files)}.wav")
                future = _disk_io_executor.submit(
                    _concat_partial, list(accumulated_audio_paths), p_path, config,
                    list(accumulated_pauses),
                )
                pending_flushes.append(future)
                partial_files.append(p_path)
                accumulated_audio_paths.clear()
                accumulated_pauses.clear()

            def _drain_in_order() -> None:
                nonlocal next_expected_index
                while next_expected_index in received_chunks:
                    res = received_chunks.pop(next_expected_index)
                    durations_res[res.chunk_index] = res.duration
                    accumulated_audio_paths.append(res.audio)
                    accumulated_pauses.append(pauses[res.chunk_index])
                    if len(accumulated_audio_paths) >= _PARTIAL_FLUSH_CHUNK_COUNT:
                        _flush_partial()
                    next_expected_index += 1

            try:
                _drain_in_order()

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
                    received_chunks[item.chunk_index] = item
                    _drain_in_order()

                if not cancel_token.is_cancelled:
                    # Workers may have exited between our last get() and their sentinel.
                    while True:
                        try:
                            item = master_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is not None:
                            received_chunks[item.chunk_index] = item
                    _drain_in_order()

                    if next_expected_index < total_chunks:
                        cause = f": {worker_errors[-1]}" if worker_errors else ""
                        raise RuntimeError(
                            f"Stage B failure — only {next_expected_index} of {total_chunks} "
                            f"chunks were synthesized for chapter {chapter_index}{cause}"
                        )

                    if accumulated_audio_paths:
                        _flush_partial()

                # Await all async partial WAV writes before final mastering pass
                for f in pending_flushes:
                    try:
                        f.result(timeout=60.0)
                    except Exception as exc:
                        raise RuntimeError(f"Async partial WAV write failed: {exc}") from exc

                if partial_files and not cancel_token.is_cancelled:
                    normalized = _master_final(partial_files, out_wav_path, config)
                    if master_info is not None:
                        master_info["normalized"] = normalized

                    if not os.path.exists(out_wav_path) or os.path.getsize(out_wav_path) < 100:
                        raise RuntimeError(f"Mastered chapter audio file was not created or empty at {out_wav_path}")

            except Exception as exc:
                stage_c_exception = exc
            finally:
                _cleanup_chunk_files(partial_files)

        thread_c = threading.Thread(target=_stage_c_worker, name=f"StageC-Ch{chapter_index}", daemon=False)

        # ── Start Threads ─────────────────────────────────────────────────────
        for t in stage_b_threads:
            t.start()
        thread_c.start()

        # ── Join Threads in Strict Order ─────────────────────────
        for t in stage_b_threads:
            t.join()
        thread_c.join()

        if stage_c_exception is None and not cancel_token.is_cancelled and os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 0:
            _chapter_succeeded = True
            if pending_count and len(active_devices) > 1:
                log_callback(
                    f"  [Ch{chapter_index}] Device share: "
                    + ", ".join(f"{dev} {count} chunk(s)" for dev, count in device_stats.items())
                )
            if worker_errors:
                log_callback(
                    f"  [Ch{chapter_index}] ⚠ {len(worker_errors)} device worker(s) failed "
                    f"({worker_errors[-1]}); the remaining device(s) finished the chapter."
                )

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
