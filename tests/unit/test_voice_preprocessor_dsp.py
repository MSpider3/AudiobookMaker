"""
test_voice_preprocessor_dsp.py
==============================
Signal-level tests for the reference-voice preprocessing pipeline.

Every test builds a synthetic signal with a known property (a soft word
onset, a click between two silences, a known noise floor, a quiet and a loud
take of the same performance) and asserts on something measurable in the
output. The reference clip is what the cloning model copies for the whole
book, so these are the properties that must not regress:

* speech is never cut: onsets, tails and segment ends survive
* results do not depend on the recording level
* the level step lands on the loudness target, under the peak ceiling, last
* a step that could not run is reported and never cached
* a request that would damage the voice (formant shift) is not applied
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import sys
import threading

import numpy as np
import pyloudnorm
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import audiobook_factory.voice_preprocessor as vp

_SR: int = 24000
_PEAK: float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic signals
# ══════════════════════════════════════════════════════════════════════════════

def _ramp(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)


def _burst(dur: float, f0: float = 150.0, amp: float = _PEAK, sr: int = _SR,
           ramp_s: float = 0.02) -> np.ndarray:
    """A harmonic 'vowel' with raised-cosine onset and release."""
    t = np.arange(int(round(dur * sr))) / sr
    x = sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (5, 0.2)))
    x = x / np.max(np.abs(x)) * amp
    n = int(round(ramp_s * sr))
    if n:
        x[:n] *= _ramp(n)
        x[-n:] *= _ramp(n)[::-1]
    return x


def _tone(dur: float, freq: float, amp: float, sr: int = _SR, ramp_s: float = 0.005) -> np.ndarray:
    t = np.arange(int(round(dur * sr))) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    n = int(round(ramp_s * sr))
    if n:
        x[:n] *= _ramp(n)
        x[-n:] *= _ramp(n)[::-1]
    return x


def _silence(dur: float, sr: int = _SR) -> np.ndarray:
    return np.zeros(int(round(dur * sr)))


def _noise(n: int, rms: float, seed: int = 0) -> np.ndarray:
    return rms * np.random.default_rng(seed).standard_normal(n)


def _speech(sr: int = _SR, lead: float = 1.0, trail: float = 1.0, bursts: int = 6,
            burst_s: float = 0.6, pause_s: float = 0.3, amp: float = _PEAK,
            f0: float = 150.0) -> np.ndarray:
    """Speech-like bursts separated by pauses, with silence at both ends."""
    parts = [_silence(lead, sr)]
    for i in range(bursts):
        parts.append(_burst(burst_s, f0=f0 * (1.0 + 0.04 * (i % 3)), amp=amp, sr=sr))
        if i < bursts - 1:
            parts.append(_silence(pause_s, sr))
    parts.append(_silence(trail, sr))
    return np.concatenate(parts)


def _wav(x: np.ndarray, sr: int = _SR, subtype: str = "FLOAT") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def _read(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float64")
    return data, sr


def _band_energy(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Time-domain energy inside a band; comparable across signal lengths."""
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(2.0 * spectrum[(freqs >= lo) & (freqs <= hi)].sum() / len(x))


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-20)


def _lufs(x: np.ndarray, sr: int) -> float:
    return float(pyloudnorm.Meter(sr).integrated_loudness(np.asarray(x, dtype=np.float64)))


def _active_runs(x: np.ndarray, sr: int, level: float) -> list[tuple[int, int]]:
    """Sample ranges where the 10 ms envelope exceeds ``level`` (gaps under 50 ms closed)."""
    hop = sr // 100
    n = len(x) // hop
    env = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
    runs: list[list[int]] = []
    for i in np.flatnonzero(env > level):
        if runs and i - runs[-1][1] <= 5:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return [(a * hop, b * hop) for a, b in runs]


def _config(**overrides) -> "vp.PreprocessConfig":
    """A config with only the named steps on.

    Unknown fields are dropped so the same test can be pointed at an older
    module and fail on its behaviour rather than on a constructor error.
    """
    base = dict(
        noise_reduce=False, noise_gate=False, highpass_filter=False,
        silence_removal=False, normalize_volume=False, formant_shift=False,
        resample=False, trim_silence=False, edge_fade_ms=0, select_best_window=False,
    )
    base.update(overrides)
    known = {f.name for f in dataclasses.fields(vp.PreprocessConfig)}
    return vp.PreprocessConfig(**{k: v for k, v in base.items() if k in known})


def _run(x: np.ndarray, config: "vp.PreprocessConfig", sr: int = _SR) -> tuple[np.ndarray, int]:
    return _read(vp.preprocess(_wav(x, sr), config, use_cache=False))


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Keep every test's cache inside its own temp directory."""
    cache_dir = tmp_path / "voice_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(vp, "_get_cache_dir", lambda: str(cache_dir))
    return cache_dir


# ══════════════════════════════════════════════════════════════════════════════
# Defect 1 and 6 — silence handling
# ══════════════════════════════════════════════════════════════════════════════

class TestSilenceHandling:

    @pytest.mark.parametrize("min_segment_ms", [100, 300])
    def test_click_between_two_silences_is_discarded(self, min_segment_ms):
        click = _tone(0.02, 3000.0, 0.8, ramp_s=0.002)
        x = np.concatenate([_silence(1.0), click, _silence(1.0), _burst(1.0), _silence(1.0)])

        y = vp._remove_silence(x, _SR, -40.0, min_segment_ms, 500)

        kept = _band_energy(y, _SR, 2500, 3500) / _band_energy(x, _SR, 2500, 3500)
        assert kept < 1e-6, f"{kept:.3f} of the click's energy survived"
        assert np.sum(y ** 2) > 0.999 * np.sum(_burst(1.0) ** 2), "the speech was cut"

    def test_soft_word_onset_keeps_its_pre_roll(self):
        # A fricative 46 dB below the vowel peak: under the -40 dB threshold,
        # so only pre-roll can save it.
        fricative = _tone(0.06, 5000.0, _PEAK * 10 ** (-46 / 20))
        x = np.concatenate([_silence(1.0), fricative, _burst(1.0), _silence(1.0)])

        y = vp._remove_silence(x, _SR, -40.0, 300, 500)

        kept = _band_energy(y, _SR, 4500, 5500) / _band_energy(x, _SR, 4500, 5500)
        assert kept > 0.99, f"only {kept:.3f} of the soft onset survived"

    def test_segment_end_survives_with_zero_silence_kept(self):
        # Bursts that stop at full level: any sample lost at the end shows.
        burst = _burst(0.5, ramp_s=0.0)
        x = np.concatenate([_silence(1.0), burst, _silence(1.0), burst, _silence(1.0)])

        y = vp._remove_silence(x, _SR, -40.0, 300, 0)

        kept = np.sum(y ** 2) / np.sum(x ** 2)
        assert kept > 0.9999, f"{(1 - kept) * 100:.2f}% of the speech energy was cut"
        assert len(y) < len(x) - 2 * _SR, "the pauses were not shortened"

    def test_short_isolated_word_is_not_a_click(self):
        # 150 ms of voice between two pauses is a word ("Oh."), not a click.
        word = _burst(0.15)
        x = np.concatenate([_silence(0.5), _burst(0.8), _silence(0.8), word,
                            _silence(0.8), _burst(0.8), _silence(0.5)])

        y = vp._remove_silence(x, _SR, -40.0, 100, 300)

        assert np.sum(y ** 2) > 0.9999 * np.sum(x ** 2)

    def test_default_trims_only_the_ends(self):
        x = np.concatenate([_silence(1.0), _burst(1.0), _silence(2.0), _burst(1.0), _silence(1.5)])
        cfg = _config(trim_silence=True)
        edge = getattr(cfg, "edge_silence_ms", 150) / 1000.0

        y, sr = _run(x, cfg)

        assert sr == _SR
        assert abs(len(y) / sr - (4.0 + 2 * edge)) < 0.03, "ends were not trimmed to the edge pad"
        runs = _active_runs(y, sr, 0.01)
        assert len(runs) == 2
        pause = (runs[1][0] - runs[0][1]) / sr
        assert abs(pause - 2.0) < 0.03, f"the internal pause changed to {pause:.2f}s"

    def test_trimmed_onset_matches_the_input_sample_for_sample(self):
        # 80 ms soft onset; after trimming it must be bit-identical apart from
        # 16-bit rounding, i.e. preserved to well within a millisecond.
        x = np.concatenate([_silence(1.0), _burst(1.0, ramp_s=0.08), _silence(1.0)])
        x = x + _noise(len(x), 1e-4)
        y, sr = _run(x, _config(trim_silence=True))

        onset = _SR  # the burst starts at 1.0 s
        probe = x[onset - int(0.05 * _SR):onset + int(0.15 * _SR)]
        errors = [
            float(np.max(np.abs(y[lag:lag + len(probe)] - probe)))
            for lag in range(0, len(y) - len(probe))
        ]
        assert min(errors) < 1e-3, "the onset waveform is not present unaltered in the output"

    def test_edges_fade_to_silence(self):
        x = _speech() + _noise(len(_speech()), 0.003)
        cfg = _config(trim_silence=True, edge_fade_ms=10)

        y, _ = _run(x, cfg)

        assert abs(y[0]) < 1e-4 and abs(y[-1]) < 1e-4, "the clip does not start and end at zero"
        assert np.max(np.abs(np.diff(y[:120]))) < 0.01, "there is a step at the start"

    def test_pause_shortening_is_opt_in_and_never_cuts_speech(self):
        x = np.concatenate([_silence(0.5), _burst(1.0), _silence(2.0), _burst(1.0), _silence(0.5)])
        x = x + _noise(len(x), 0.001)

        untouched, _ = _run(x, _config())
        y, sr = _run(x, _config(silence_removal=True, max_silence_kept_ms=500))

        assert len(untouched) == len(x), "pauses were edited although silence_removal is off"
        runs = _active_runs(y, sr, 0.01)
        assert len(runs) == 2
        pause = (runs[1][0] - runs[0][1]) / sr
        assert abs(pause - 0.5) < 0.03, f"the 2 s pause became {pause:.2f}s, expected 0.5s"
        speech_in = sum(np.sum(x[a:b] ** 2) for a, b in _active_runs(x, _SR, 0.01))
        speech_out = sum(np.sum(y[a:b] ** 2) for a, b in runs)
        assert speech_out > 0.999 * speech_in
        # The joint must not click: nothing in the shortened pause exceeds the noise.
        joint = y[runs[0][1] + 240:runs[1][0] - 240]
        assert np.max(np.abs(joint)) < 0.008

    def test_pauses_shorter_than_the_limit_are_left_alone(self):
        x = _speech(lead=0.3, trail=0.3, pause_s=0.3)

        y, _ = _run(x, _config(silence_removal=True, max_silence_kept_ms=500))

        assert len(y) == len(x)

    def test_one_loud_click_does_not_turn_quiet_speech_into_silence(self):
        # Thresholds are relative to the peak; a stray click 39 dB above the
        # voice must not push the whole voice under the threshold.
        click = _tone(0.002, 4000.0, 0.9, ramp_s=0.0005)
        quiet = _speech(amp=0.01, lead=0.5, trail=0.5)
        x = np.concatenate([_silence(0.3), click, _silence(1.0), quiet])

        y, _ = _run(x, _config(silence_removal=True, trim_silence=True))

        kept = _band_energy(y, _SR, 100, 900) / _band_energy(quiet, _SR, 100, 900)
        assert kept > 0.99, f"only {kept:.2f} of the quiet speech survived"


# ══════════════════════════════════════════════════════════════════════════════
# Defect 2 — noise gate
# ══════════════════════════════════════════════════════════════════════════════

class TestNoiseGate:

    @staticmethod
    def _take() -> np.ndarray:
        # A word, then a soft tail 30 dB down, with room tone throughout.
        soft_tail = _burst(0.3, amp=_PEAK * 10 ** (-30 / 20))
        x = np.concatenate([_silence(0.5), _burst(1.0), soft_tail, _silence(0.8)])
        return x + _noise(len(x), _PEAK * 10 ** (-55 / 20))

    def test_result_does_not_depend_on_recording_level(self):
        x = self._take()
        cfg = _config(noise_gate=True, noise_gate_threshold_db=-45.0)

        loud, _ = _run(x, cfg)
        quiet, _ = _run(x * 0.05, cfg)

        assert np.max(np.abs(loud - quiet / 0.05)) < 2e-3
        tail = slice(int(1.55 * _SR), int(1.75 * _SR))
        ratio = _rms_db(quiet[tail]) - _rms_db(x[tail] * 0.05)
        assert abs(ratio) < 0.5, f"the soft tail of the quiet take changed by {ratio:.1f} dB"

    def test_gate_attenuates_and_never_mutes(self):
        x = self._take()
        cfg = _config(noise_gate=True, noise_gate_threshold_db=-35.0)
        depth = getattr(cfg, "noise_gate_range_db", 12.0)

        y, _ = _run(x, cfg)

        pause = slice(int(0.05 * _SR), int(0.4 * _SR))
        change = _rms_db(y[pause]) - _rms_db(x[pause])
        assert abs(change + depth) < 1.5, f"room tone changed by {change:.1f} dB, expected -{depth}"
        assert np.mean(y[pause] == 0.0) < 0.2, "the gate muted to digital silence"

    def test_word_onset_and_body_pass_unchanged(self):
        x = np.concatenate([_silence(0.5), _burst(1.0, ramp_s=0.08), _silence(0.5)])
        x = x + _noise(len(x), _PEAK * 10 ** (-60 / 20))

        y, _ = _run(x, _config(noise_gate=True, noise_gate_threshold_db=-45.0))

        # From 3 ms into the onset (about -45 dB) to the end of the word.
        word = slice(int(0.503 * _SR), int(1.5 * _SR))
        assert np.max(np.abs(y[word] - x[word])) < 0.01 * _PEAK

    def test_gain_moves_smoothly(self):
        carrier = _tone(3.0, 1000.0, 1e-3, ramp_s=0.0)
        x = carrier.copy()
        x[_SR:2 * _SR] += _burst(1.0, ramp_s=0.0)   # abrupt word on a steady bed

        y = vp._noise_gate(x, _SR, -45.0, 12.0)

        closed = np.r_[0:int(0.9 * _SR), int(2.2 * _SR):3 * _SR]
        usable = closed[np.abs(carrier[closed]) > 5e-4]
        gain = y[usable] / x[usable]
        assert gain.min() > 10 ** (-12.5 / 20), "attenuation exceeds the configured range"
        assert np.max(np.abs(np.diff(y[int(1.9 * _SR):]))) < 0.2, "the release steps"
        # After the word the gate holds, then releases gradually.
        after = y[int(2.02 * _SR):int(2.06 * _SR)] / x[int(2.02 * _SR):int(2.06 * _SR)]
        assert np.nanmedian(after) > 0.95, "the gate closed without a hold"


# ══════════════════════════════════════════════════════════════════════════════
# Defect 3 — caching
# ══════════════════════════════════════════════════════════════════════════════

class TestCaching:

    @staticmethod
    def _noisy_take() -> bytes:
        x = _speech()
        return _wav(x + _noise(len(x), 0.02))

    def test_result_of_a_skipped_step_is_not_cached(self, monkeypatch):
        data = self._noisy_take()
        cfg = _config(noise_reduce=True, noise_reduce_strength=1.0)
        reference = vp.preprocess(data, cfg, use_cache=False)

        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "noisereduce", None)   # "not installed"
            without_library = vp.preprocess(data, cfg, use_cache=True)

        after_install = vp.preprocess(data, cfg, use_cache=True)

        assert without_library != reference, "the simulated missing library had no effect"
        assert after_install == reference, "the cache served audio processed without noise reduction"

    def test_skipped_step_is_reported(self, monkeypatch, _isolated_cache):
        monkeypatch.setitem(sys.modules, "noisereduce", None)

        _, report = vp.preprocess_with_report(
            self._noisy_take(), _config(noise_reduce=True), use_cache=True
        )

        assert any(s.startswith("noise_reduce") for s in report.steps_skipped)
        assert any("noise_reduce" in w for w in report.warnings)
        assert os.listdir(_isolated_cache) == []

    def test_cache_round_trip(self, _isolated_cache):
        data = self._noisy_take()
        cfg = vp.PreprocessConfig()

        first, first_report = vp.preprocess_with_report(data, cfg)
        second, second_report = vp.preprocess_with_report(data, cfg)

        assert first == second
        assert not first_report.from_cache and second_report.from_cache
        assert second_report.loudness_lufs == pytest.approx(first_report.loudness_lufs, abs=0.05)
        assert len(os.listdir(_isolated_cache)) == 1

    def test_cache_key_ignores_int_versus_float(self):
        data = self._noisy_take()
        as_int = vp._get_cache_path(data, vp.PreprocessConfig(highpass_cutoff_hz=80))
        as_float = vp._get_cache_path(data, vp.PreprocessConfig(highpass_cutoff_hz=80.0))
        other = vp._get_cache_path(data, vp.PreprocessConfig(highpass_cutoff_hz=90))
        assert as_int == as_float != other

    def test_cache_directory_is_bounded(self, monkeypatch, _isolated_cache):
        monkeypatch.setattr(vp, "_CACHE_MAX_ENTRIES", 3)
        x = _speech()
        for i in range(6):
            vp.preprocess(_wav(x * (0.5 + 0.05 * i)), _config(normalize_volume=True))

        entries = os.listdir(_isolated_cache)
        assert len(entries) == 3
        assert all(name.endswith(vp._CACHE_ENTRY_SUFFIX) for name in entries)

    def test_concurrent_callers_get_identical_valid_audio(self, _isolated_cache):
        data = self._noisy_take()
        cfg = _config(normalize_volume=True, highpass_filter=True)
        results: list[bytes] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                results.append(vp.preprocess(data, cfg))
            except BaseException as exc:   # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(results)) == 1
        entries = os.listdir(_isolated_cache)
        assert len(entries) == 1 and not entries[0].endswith(".tmp")
        with open(os.path.join(_isolated_cache, entries[0]), "rb") as f:
            assert f.read() == results[0]

    def test_corrupt_cache_entry_is_ignored(self):
        data = self._noisy_take()
        cfg = _config(normalize_volume=True)
        good = vp.preprocess(data, cfg)
        with open(vp._get_cache_path(data, cfg), "wb") as f:
            f.write(b"truncated")

        assert vp.preprocess(data, cfg) == good

    def test_unwritable_cache_does_not_break_preprocessing(self, monkeypatch):
        def deny() -> str:
            raise PermissionError("read-only install")

        monkeypatch.setattr(vp, "_get_cache_dir", deny)

        out = vp.preprocess(self._noisy_take(), _config(normalize_volume=True), use_cache=True)

        assert len(_read(out)[0]) > 0

    def test_clear_cache(self, _isolated_cache):
        vp.preprocess(self._noisy_take(), _config(normalize_volume=True))
        assert vp.clear_cache() == 1
        assert os.listdir(_isolated_cache) == []


# ══════════════════════════════════════════════════════════════════════════════
# Defect 4 — formant shift
# ══════════════════════════════════════════════════════════════════════════════

class TestFormantShift:

    @pytest.mark.parametrize("timbre", [0.0, 2.0, 16.0])
    def test_formant_request_cannot_damage_the_clip(self, timbre):
        x = _speech()
        x = x + _noise(len(x), 0.005)
        plain = vp.preprocess(_wav(x), _config(normalize_volume=True), use_cache=False)

        shifted = vp.preprocess(
            _wav(x),
            _config(normalize_volume=True, formant_shift=True, formant_timbre=timbre),
            use_cache=False,
        )

        y, _ = _read(shifted)
        assert np.mean(np.abs(y) >= 0.999) == 0.0, "the output is clipped"
        assert _rms_db(y) > -40.0, "the output is (nearly) silent"
        assert shifted == plain, "the formant step altered the voice"

    def test_ignored_formant_request_is_reported(self):
        _, report = vp.preprocess_with_report(
            _wav(_speech()), _config(formant_shift=True, formant_timbre=2.0), use_cache=False
        )
        assert any(s.startswith("formant_shift") for s in report.steps_skipped)
        assert any("Formant" in w for w in report.warnings)


# ══════════════════════════════════════════════════════════════════════════════
# Defect 5 — level
# ══════════════════════════════════════════════════════════════════════════════

class TestLevel:

    @pytest.mark.parametrize("gain", [1.0, 0.02])
    def test_default_reaches_the_loudness_target_at_any_input_level(self, gain):
        cfg = vp.PreprocessConfig()
        target = getattr(cfg, "loudness_target_lufs", -20.0)
        ceiling = getattr(cfg, "true_peak_ceiling_dbfs", -1.0)

        y, sr = _read(vp.preprocess(_wav(_speech() * gain), cfg, use_cache=False))

        assert -23.5 <= target <= -19.5
        assert abs(_lufs(y, sr) - target) < 1.0
        assert 20 * np.log10(np.max(np.abs(y))) <= ceiling + 0.05

    def test_peak_ceiling_wins_and_dynamics_are_untouched(self):
        # Sparse spikes: far more peak than loudness, so the ceiling binds.
        x = _speech(amp=0.02)
        x[::_SR // 2] += 0.9
        cfg = _config(normalize_volume=True)

        y, sr = _run(x, cfg)

        ceiling = 10 ** (cfg.true_peak_ceiling_dbfs / 20)
        assert np.max(np.abs(y)) <= ceiling * 1.01
        assert _lufs(y, sr) < cfg.loudness_target_lufs - 1.0
        scale = np.dot(y, x) / np.dot(x, x)
        assert np.max(np.abs(y - scale * x)) < 1e-3, "the clip was limited instead of scaled"

    def test_level_is_applied_after_every_other_step(self):
        x = _speech(sr=48000) + _noise(len(_speech(sr=48000)), 0.004)
        cfg = vp.PreprocessConfig(
            noise_reduce=True, noise_gate=True, highpass_filter=True, silence_removal=True,
            normalize_volume=True, resample=True, target_sample_rate=24000,
        )

        y, sr = _read(vp.preprocess(_wav(x, 48000), cfg, use_cache=False))

        assert sr == 24000
        assert abs(_lufs(y, sr) - cfg.loudness_target_lufs) < 0.5

    @pytest.mark.parametrize("sr", [16000, 24000, 44100, 48000])
    def test_numpy_loudness_matches_pyloudnorm(self, sr):
        x = _speech(sr=sr) + _noise(len(_speech(sr=sr)), 0.003)
        for gain in (1.0, 0.1):
            assert vp._loudness_numpy(x * gain, sr) == pytest.approx(_lufs(x * gain, sr), abs=0.2)

    def test_target_is_reached_without_pyloudnorm(self, monkeypatch):
        cfg = vp.PreprocessConfig()
        monkeypatch.setitem(sys.modules, "pyloudnorm", None)

        y, sr = _read(vp.preprocess(_wav(_speech() * 0.1), cfg, use_cache=False))

        assert abs(_lufs(y, sr) - cfg.loudness_target_lufs) < 1.0

    def test_peak_mode_is_still_available(self):
        cfg = _config(normalize_volume=True, normalize_mode="peak", normalize_target_dbfs=-6.0)

        y, _ = _run(_speech() * 0.1, cfg)

        assert 20 * np.log10(np.max(np.abs(y))) == pytest.approx(-6.0, abs=0.05)

    def test_unknown_normalize_mode_is_rejected(self):
        with pytest.raises(ValueError):
            _run(_speech(), _config(normalize_volume=True, normalize_mode="rms"))

    def test_output_is_never_clipped_even_with_normalization_off(self):
        x = _speech() * 3.0   # float file peaking at +3.5 dBFS

        y, _ = _run(x, _config())

        assert np.max(np.abs(y)) < 1.0
        scale = np.dot(y, x) / np.dot(x, x)
        assert np.max(np.abs(y - scale * x)) < 1e-3, "the waveform was clipped, not scaled"


# ══════════════════════════════════════════════════════════════════════════════
# Defect 7 — sample rate and channels
# ══════════════════════════════════════════════════════════════════════════════

class TestSampleRateAndChannels:

    def test_default_brings_a_48k_clip_to_24k_mono(self):
        x = _speech(sr=48000)

        y, sr = _read(vp.preprocess(_wav(np.stack([x, x], axis=1), 48000), use_cache=False))

        assert sr == 24000 and y.ndim == 1

    def test_never_upsamples_unless_asked(self):
        x = _speech(sr=16000)

        _, kept = _run(x, _config(resample=True, target_sample_rate=24000), sr=16000)
        _, raised = _run(
            x, _config(resample=True, target_sample_rate=24000, allow_upsample=True), sr=16000
        )

        assert kept == 16000
        assert raised == 24000

    @pytest.mark.parametrize("backend", ["soxr", "scipy"])
    def test_resampler_is_transparent_in_band_and_does_not_alias(self, backend, monkeypatch):
        if backend == "scipy":
            monkeypatch.setitem(sys.modules, "soxr", None)   # exercise the fallback
        sr = 48000
        in_band = _tone(2.0, 1000.0, 0.25, sr=sr, ramp_s=0.05)
        # Just above the new Nyquist, where a short anti-alias filter leaks:
        # 13 kHz folds to 11 kHz.
        above_nyquist = _tone(2.0, 13000.0, 0.25, sr=sr, ramp_s=0.05)

        y, out_sr = _run(in_band + above_nyquist, _config(resample=True, target_sample_rate=24000), sr=sr)

        assert out_sr == 24000
        # Mean power, not energy: the two signals have different sample counts.
        reference = _band_energy(in_band, sr, 900, 1100) / len(in_band)
        kept = _band_energy(y, out_sr, 900, 1100) / len(y) / reference
        alias = _band_energy(y, out_sr, 10500, 11500) / len(y) / reference
        assert abs(10 * np.log10(kept)) < 0.05
        assert 10 * np.log10(alias + 1e-30) < -80.0

    def test_dead_channel_does_not_halve_or_dirty_the_voice(self):
        voice = _speech()
        dead = _noise(len(voice), 1e-4)   # an unconnected input's hiss

        y, _ = _run(np.stack([voice, dead], axis=1), _config())

        assert np.max(np.abs(y - voice)) < 1e-3

    def test_out_of_phase_channels_do_not_cancel(self):
        voice = _speech()

        y, _ = _run(np.stack([voice, -voice], axis=1), _config())

        assert _rms_db(y) == pytest.approx(_rms_db(voice), abs=0.1)

    def test_ordinary_stereo_is_averaged(self):
        voice = _speech()
        left = voice + _noise(len(voice), 0.01, seed=1)
        right = voice + _noise(len(voice), 0.01, seed=2)

        y, _ = _run(np.stack([left, right], axis=1), _config())

        assert np.max(np.abs(y - 0.5 * (left + right))) < 1e-3


# ══════════════════════════════════════════════════════════════════════════════
# Defect 8 — defaults and caller compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestDefaultsAndCompatibility:

    def test_defaults_favour_clone_quality(self):
        cfg = vp.PreprocessConfig()
        # On: safe repairs.
        assert cfg.highpass_filter and cfg.noise_reduce and cfg.normalize_volume
        assert cfg.trim_silence and cfg.resample
        assert cfg.noise_reduce_strength <= 0.25
        assert cfg.normalize_mode == "loudness"
        assert cfg.target_sample_rate == 24000 and not cfg.allow_upsample
        # Off: anything that edits the performance.
        assert not cfg.noise_gate
        assert not cfg.silence_removal
        assert not cfg.formant_shift
        assert not cfg.select_best_window

    def test_keyword_set_used_by_the_ui_and_api_is_still_accepted(self):
        cfg = vp.PreprocessConfig(
            noise_reduce=True, noise_reduce_strength=0.5,
            noise_gate=True, noise_gate_threshold_db=-40,
            highpass_filter=True, highpass_cutoff_hz=80.0,
            silence_removal=True, silence_threshold_db=-40,
            min_segment_ms=300, max_silence_kept_ms=500,
            normalize_volume=True, normalize_target_dbfs=-3,
            formant_shift=False, formant_quefrency=1.0, formant_timbre=1.0,
            resample=False, target_sample_rate=44100,
        )
        x = _speech() + _noise(len(_speech()), 0.003)

        out = vp.preprocess(_wav(x), cfg, use_cache=False)

        y, sr = _read(out)
        assert isinstance(out, bytes) and sr == _SR and y.ndim == 1
        assert sf.info(io.BytesIO(out)).subtype == "PCM_16"
        assert abs(_lufs(y, sr) - cfg.loudness_target_lufs) < 1.0

    def test_private_entry_point_still_returns_wav_bytes(self):
        out = vp._run_preprocessing_pipeline(_wav(_speech()), vp.PreprocessConfig())
        assert out[:4] == b"RIFF"


# ══════════════════════════════════════════════════════════════════════════════
# Robustness
# ══════════════════════════════════════════════════════════════════════════════

class TestRobustness:

    def test_nan_and_inf_samples_are_repaired(self):
        x = _speech()
        x[30000] = np.nan
        x[30001] = np.inf

        out, report = vp.preprocess_with_report(_wav(x), use_cache=False)

        y, _ = _read(out)
        assert np.all(np.isfinite(y))
        assert len(np.unique(y)) > 1000, "the output collapsed to a constant"
        assert any("NaN" in w for w in report.warnings)

    @pytest.mark.parametrize("n_samples", [1, 10, 200, 4000])
    def test_very_short_clips_do_not_crash(self, n_samples):
        x = _burst(1.0)[:n_samples] if n_samples > 1 else np.array([0.2])

        out, report = vp.preprocess_with_report(_wav(x), use_cache=False)

        y, _ = _read(out)
        assert np.all(np.isfinite(y)) and len(y) > 0
        assert report.warnings, "a clip this short must be flagged"

    def test_silent_clip_is_returned_and_flagged(self):
        out, report = vp.preprocess_with_report(_wav(_silence(2.0)), use_cache=False)

        y, _ = _read(out)
        assert not np.any(y)
        assert any("No speech" in w for w in report.warnings)

    def test_bad_inputs_raise_clear_errors(self, tmp_path):
        with pytest.raises(ValueError, match="no samples"):
            vp.preprocess(_wav(np.zeros(0)), use_cache=False)
        with pytest.raises(ValueError, match="decode") as excinfo:
            vp.preprocess(b"this is not audio", use_cache=False)
        assert "BytesIO" not in str(excinfo.value), "internal object repr leaked to the user"
        with pytest.raises(FileNotFoundError):
            vp.preprocess(str(tmp_path / "missing.wav"), use_cache=False)
        with pytest.raises(TypeError):
            vp.preprocess(12345, use_cache=False)

    def test_overlong_recording_is_refused_before_it_is_loaded(self, monkeypatch):
        monkeypatch.setattr(vp, "_MAX_INPUT_SECONDS", 5.0)

        with pytest.raises(ValueError, match="minutes long"):
            vp.preprocess(_wav(_speech(bursts=8)), use_cache=False)   # 8.9 s
        with pytest.raises(ValueError, match="minutes long"):
            vp.analyze_voice(_wav(_speech(bursts=8)))

    def test_path_and_bytes_inputs_agree(self, tmp_path):
        data = _wav(_speech())
        path = tmp_path / "voice.wav"
        path.write_bytes(data)

        assert vp.preprocess(str(path), use_cache=False) == vp.preprocess(data, use_cache=False)
        assert vp.preprocess(path, use_cache=False) == vp.preprocess(data, use_cache=False)

    def test_dc_offset_is_removed(self):
        y, _ = _run(_speech(lead=0.2, trail=0.2) + 0.2, _config())
        assert abs(np.mean(y)) < 1e-3

    def test_highpass_removes_rumble_and_keeps_the_fundamental(self):
        rumble = _tone(3.0, 30.0, 0.3, ramp_s=0.1)
        fundamental = _tone(3.0, 100.0, 0.3, ramp_s=0.1)

        y, _ = _run(rumble + fundamental, _config(highpass_filter=True, highpass_cutoff_hz=80))

        assert 10 * np.log10(_band_energy(y, _SR, 20, 40) / _band_energy(rumble, _SR, 20, 40)) < -40
        kept = 10 * np.log10(_band_energy(y, _SR, 90, 110) / _band_energy(fundamental, _SR, 90, 110))
        assert kept > -1.0, f"the 100 Hz fundamental lost {-kept:.1f} dB"

    def test_highpass_is_stable_at_high_sample_rates(self):
        sr = 192000
        x = _tone(1.0, 200.0, 0.3, sr=sr, ramp_s=0.05)

        y, _ = _run(x, _config(highpass_filter=True), sr=sr)

        assert np.max(np.abs(y - x)) < 5e-4

    def test_noise_reduction_lowers_the_floor_without_touching_the_voice(self):
        clean = _speech()
        x = clean + _noise(len(clean), 0.005)
        voiced = np.abs(clean) > 0
        mid_pause = slice(int(0.2 * _SR), int(0.8 * _SR))

        y, _ = _run(x, _config(noise_reduce=True, noise_reduce_strength=0.5))

        assert _rms_db(y[mid_pause]) - _rms_db(x[mid_pause]) < -4.0
        gain = np.dot(y[voiced], clean[voiced]) / np.dot(clean[voiced], clean[voiced])
        assert abs(20 * np.log10(gain)) < 0.2, "the voice itself was attenuated"

    def test_noise_profile_ignores_the_users_trim_threshold(self):
        # A trim threshold of -10 dB would class most of the voice as
        # "silence"; the noise profile must not be learned from it.
        clean = _speech()
        x = clean + _noise(len(clean), 0.005)
        voiced = np.abs(clean) > 0

        y, _ = _run(x, _config(noise_reduce=True, noise_reduce_strength=1.0,
                               silence_threshold_db=-10.0))
        reference, _ = _run(x, _config(noise_reduce=True, noise_reduce_strength=1.0))

        assert np.array_equal(y, reference)
        gain = np.dot(y[voiced], clean[voiced]) / np.dot(clean[voiced], clean[voiced])
        assert abs(20 * np.log10(gain)) < 0.5

    def test_noise_reduction_leaves_a_clean_clip_alone(self):
        x = _speech() + _noise(len(_speech()), 1e-5)

        out, report = vp.preprocess_with_report(
            _wav(x), _config(noise_reduce=True, noise_reduce_strength=1.0), use_cache=False
        )

        assert any(s.startswith("noise_reduce") for s in report.steps_skipped)
        assert np.max(np.abs(_read(out)[0] - x)) < 1e-3

    def test_log_callback_receives_progress_and_warnings(self):
        lines: list[str] = []

        vp.preprocess(_wav(_speech()[:_SR * 3]), log_fn=lines.append, use_cache=False)

        assert any("Loaded audio" in line for line in lines)
        assert any("WARNING" in line for line in lines)   # 3 s clip is too short
        assert lines[-1].endswith("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis report
# ══════════════════════════════════════════════════════════════════════════════

class TestVoiceReport:

    def test_measurements_match_a_signal_with_known_properties(self):
        noise_rms = 0.003
        clean = _speech(bursts=8)              # 1.0 + 8*0.6 + 7*0.3 + 1.0 = 8.9 s
        x = clean + _noise(len(clean), noise_rms)

        report = vp.analyze_voice(_wav(x))

        assert report.duration_s == pytest.approx(8.9, abs=0.01)
        assert report.sample_rate == _SR and report.channels == 1
        assert report.peak_dbfs == pytest.approx(20 * np.log10(np.max(np.abs(x))), abs=0.1)
        assert report.true_peak_dbfs >= report.peak_dbfs - 1e-6
        assert report.loudness_lufs == pytest.approx(_lufs(x, _SR), abs=0.3)
        assert report.noise_floor_dbfs == pytest.approx(20 * np.log10(noise_rms), abs=2.0)
        expected_snr = _rms_db(clean[np.abs(clean) > 0]) - 20 * np.log10(noise_rms)
        assert report.snr_db == pytest.approx(expected_snr, abs=3.0)
        assert report.speech_ratio == pytest.approx(8 * 0.6 / 8.9, abs=0.08)
        assert report.clipped_ratio == 0.0
        assert report.warnings == []

    def test_dc_offset_does_not_pose_as_noise(self):
        clean = _speech(bursts=8)
        x = clean + _noise(len(clean), 0.003)

        plain = vp.analyze_voice(_wav(x))
        offset = vp.analyze_voice(_wav(x + 0.1))

        assert offset.noise_floor_dbfs == pytest.approx(plain.noise_floor_dbfs, abs=0.5)
        assert offset.snr_db == pytest.approx(plain.snr_db, abs=0.5)

    def test_report_is_json_serialisable(self):
        _, report = vp.preprocess_with_report(_wav(_speech()), use_cache=False)

        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["source"]["duration_s"] == pytest.approx(report.source.duration_s)
        assert isinstance(payload["warnings"], list)

    @pytest.mark.parametrize("build, expected", [
        (lambda: _speech(bursts=3), "only"),                               # 3.4 s
        (lambda: _speech(bursts=40), "cloning works best"),                # 37 s
        (lambda: np.clip(_speech(bursts=8), -0.3, 0.3), "clipped"),
        (lambda: _speech(bursts=8) + _noise(len(_speech(bursts=8)), 0.06), "background noise"),
        (lambda: _speech(bursts=2, lead=4.0, trail=4.0), "is speech"),     # 1.2 s in 9.5 s
        (lambda: _speech(bursts=8) * 0.001, "very quiet"),
    ])
    def test_each_problem_raises_its_warning(self, build, expected):
        report = vp.analyze_voice(_wav(build()))
        assert any(expected in w for w in report.warnings), report.warnings

    def test_clipping_is_measured(self):
        x = np.clip(_speech(bursts=8), -0.3, 0.3)

        report = vp.analyze_voice(_wav(x, subtype="PCM_16"))

        flat = np.mean(np.abs(x) >= 0.3)
        assert report.clipped_ratio == pytest.approx(flat, rel=0.2)

    def test_clean_sine_is_not_reported_as_clipped(self):
        report = vp.analyze_voice(_wav(_tone(6.0, 440.0, 0.99), subtype="PCM_16"))
        assert report.clipped_ratio < 1e-4

    def test_preprocess_with_report_describes_the_output_and_the_source(self):
        x = np.clip(_speech(sr=48000, bursts=8, lead=2.0), -0.4, 0.4)
        data = _wav(np.stack([x, x], axis=1), 48000)
        cfg = vp.PreprocessConfig()

        out, report = vp.preprocess_with_report(data, cfg, use_cache=False)

        assert out == vp.preprocess(data, cfg, use_cache=False)
        y, sr = _read(out)
        assert report.sample_rate == sr == 24000 and report.channels == 1
        assert report.duration_s == pytest.approx(len(y) / sr, abs=1e-6)
        assert report.loudness_lufs == pytest.approx(cfg.loudness_target_lufs, abs=0.3)
        assert report.source.sample_rate == 48000 and report.source.channels == 2
        assert report.source.duration_s == pytest.approx(len(x) / 48000, abs=1e-6)
        assert report.source.clipped_ratio > 0.01
        assert any("clipped" in w for w in report.warnings), "source clipping must be surfaced"
        for step in ("highpass_filter", "trim_silence", "resample", "normalize_volume"):
            assert step in report.steps_applied

    def test_clean_processed_clip_has_no_warnings(self):
        x = _speech(bursts=8) + _noise(len(_speech(bursts=8)), 0.004)

        _, report = vp.preprocess_with_report(_wav(x), use_cache=False)

        assert report.warnings == []
        assert report.steps_skipped == []
        assert "noise_reduce" in report.steps_applied


# ══════════════════════════════════════════════════════════════════════════════
# Best-window selection
# ══════════════════════════════════════════════════════════════════════════════

class TestBestWindow:

    @staticmethod
    def _sparse(seconds: float, f0: float) -> np.ndarray:
        """One short word every three seconds."""
        unit = np.concatenate([_burst(0.5, f0=f0), _silence(2.5)])
        return np.tile(unit, int(seconds / 3.0))

    @staticmethod
    def _dense(f0: float) -> np.ndarray:
        """Twelve seconds of fluent speech."""
        return _speech(lead=0.5, trail=0.5, bursts=13, burst_s=0.6, pause_s=0.25, f0=f0)

    @staticmethod
    def _dominant_f0(y: np.ndarray, sr: int) -> float:
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        band = (freqs > 60) & (freqs < 250)
        return float(freqs[band][np.argmax(spectrum[band])])

    def test_selects_the_densest_stretch_of_a_long_recording(self):
        x = np.concatenate([self._sparse(30.0, 100.0), self._dense(200.0), self._sparse(18.0, 100.0)])
        x = x + _noise(len(x), 0.001)
        cfg = _config(select_best_window=True, best_window_seconds=10.0, trim_silence=True)

        out, report = vp.preprocess_with_report(_wav(x), cfg, use_cache=False)

        y, sr = _read(out)
        assert "select_best_window" in report.steps_applied
        assert 8.0 <= len(y) / sr <= 10.0 + 0.01
        assert 190.0 < self._dominant_f0(y, sr) < 230.0, "the window is not in the fluent part"
        assert report.speech_ratio > 0.6

    def test_window_starts_and_ends_between_words(self):
        x = np.concatenate([self._sparse(12.0, 100.0), self._dense(200.0), self._sparse(12.0, 100.0)])
        cfg = _config(select_best_window=True, best_window_seconds=10.0)

        y, sr = _run(x, cfg)

        runs = _active_runs(y, sr, 0.01)
        assert runs[0][0] > 0.05 * sr and runs[-1][1] < len(y) - 0.05 * sr
        durations = [(b - a) / sr for a, b in runs]
        assert min(durations) > 0.55, "a word was cut by the window edge"

    def test_prefers_clean_speech_over_clipped_speech(self):
        clipped = np.clip(self._dense(100.0) * 2.0, -_PEAK, _PEAK)
        x = np.concatenate([clipped, _silence(3.0), self._dense(200.0), _silence(1.0)])
        cfg = _config(select_best_window=True, best_window_seconds=10.0)

        y, sr = _run(x, cfg)

        assert 190.0 < self._dominant_f0(y, sr) < 230.0, "the clipped take was chosen"

    def test_is_off_by_default_and_the_long_clip_is_flagged(self):
        x = np.concatenate([self._dense(150.0)] * 3)

        out, report = vp.preprocess_with_report(_wav(x), use_cache=False)

        assert len(_read(out)[0]) / 24000 > 30.0
        assert "select_best_window" not in report.steps_applied
        assert any("best-window" in w for w in report.warnings)

    def test_clip_shorter_than_the_window_is_kept_whole(self):
        x = self._dense(150.0)
        cfg = _config(select_best_window=True, best_window_seconds=20.0)

        out, report = vp.preprocess_with_report(_wav(x), cfg, use_cache=False)

        assert len(_read(out)[0]) == len(x)
        assert any(s.startswith("select_best_window") for s in report.steps_skipped)
