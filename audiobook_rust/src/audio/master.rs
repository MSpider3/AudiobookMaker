use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use rayon::prelude::*;
use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use ebur128::{EbuR128, Mode};
use mp3lame_encoder::{Builder, MonoPcm, Bitrate, FlushNoGap};

/// Read a single WAV file's samples and convert to Mono float32 format.
fn read_wav_samples(path: &str) -> Result<(Vec<f32>, u32), String> {
    let mut reader = WavReader::open(path)
        .map_err(|e| format!("Failed to open WAV {}: {}", path, e))?;
    
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    
    let mut raw_samples = Vec::new();
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                raw_samples.push(sample.map_err(|e| e.to_string())?);
            }
        }
        SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            for sample in reader.samples::<i32>() {
                let s = sample.map_err(|e| e.to_string())?;
                raw_samples.push(s as f32 / max_val);
            }
        }
    }
    
    // In case the input WAV is multi-channel, convert to Mono by averaging channels
    let channels = spec.channels as usize;
    if channels > 1 {
        let mut mono_samples = Vec::with_capacity(raw_samples.len() / channels);
        for chunk in raw_samples.chunks_exact(channels) {
            let sum: f32 = chunk.iter().sum();
            mono_samples.push(sum / (channels as f32));
        }
        Ok((mono_samples, sample_rate))
    } else {
        Ok((raw_samples, sample_rate))
    }
}

/// Helper to write WAV file from float samples.
fn write_wav_file(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| format!("Failed to create WAV: {}", e))?;
        
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val = (clamped * 32767.0) as i16;
        writer.write_sample(val)
            .map_err(|e| format!("Failed to write WAV sample: {}", e))?;
    }
    writer.finalize().map_err(|e| format!("Failed to finalize WAV: {}", e))?;
    Ok(())
}

/// Helper to encode float samples to MP3 using LAME.
fn write_mp3_file(path: &str, samples: &[f32], sample_rate: u32, bitrate_kbps: u32) -> Result<(), String> {
    // Select matching Bitrate variant
    let l_bitrate = match bitrate_kbps {
        32 => Bitrate::Kbps32,
        64 => Bitrate::Kbps64,
        96 => Bitrate::Kbps96,
        128 => Bitrate::Kbps128,
        160 => Bitrate::Kbps160,
        192 => Bitrate::Kbps192,
        256 => Bitrate::Kbps256,
        320 => Bitrate::Kbps320,
        _ => {
            if bitrate_kbps < 48 { Bitrate::Kbps32 }
            else if bitrate_kbps < 80 { Bitrate::Kbps64 }
            else if bitrate_kbps < 112 { Bitrate::Kbps96 }
            else if bitrate_kbps < 144 { Bitrate::Kbps128 }
            else if bitrate_kbps < 176 { Bitrate::Kbps160 }
            else if bitrate_kbps < 224 { Bitrate::Kbps192 }
            else if bitrate_kbps < 288 { Bitrate::Kbps256 }
            else { Bitrate::Kbps320 }
        }
    };

    let mut builder = Builder::new()
        .ok_or_else(|| "Create LAME builder failed".to_string())?;

    builder.set_num_channels(1)
        .map_err(|e| format!("Set LAME channels failed: {:?}", e))?;

    builder.set_sample_rate(sample_rate)
        .map_err(|e| format!("Set LAME sample rate failed: {:?}", e))?;

    builder.set_brate(l_bitrate)
        .map_err(|e| format!("Set LAME bitrate failed: {:?}", e))?;

    let mut mp3_encoder = builder.build()
        .map_err(|e| format!("Initialize LAME encoder failed: {:?}", e))?;

    // Since LAME expects float samples in MonoPcm, let's wrap them
    let input = MonoPcm(&samples);
    
    // Allocate buffer for mp3 output (max recommended size)
    let max_size = mp3lame_encoder::max_required_buffer_size(samples.len());
    let mut mp3_out = Vec::with_capacity(max_size);
    
    let encoded_size = mp3_encoder.encode(input, mp3_out.spare_capacity_mut())
        .map_err(|e| format!("LAME encoding failed: {:?}", e))?;
    
    unsafe {
        mp3_out.set_len(encoded_size);
    }

    let mut flush_buf = Vec::with_capacity(7200);
    let flushed_size = mp3_encoder.flush::<FlushNoGap>(flush_buf.spare_capacity_mut())
        .map_err(|e| format!("LAME flush failed: {:?}", e))?;
    
    unsafe {
        flush_buf.set_len(flushed_size);
    }
    
    mp3_out.extend_from_slice(&flush_buf);

    let mut file = File::create(path).map_err(|e| format!("Failed to create MP3 file: {}", e))?;
    file.write_all(&mp3_out).map_err(|e| format!("Failed to write MP3 data: {}", e))?;
    
    Ok(())
}

// ── Loudness normalisation with a look-ahead peak limiter ───────────────────
//
// Same algorithm and constants as audiobook_factory/loudness.py (the
// pure-Python path); keep the two in step.

/// One limiter gain value per millisecond.
const LIMITER_BLOCK_SECONDS: f64 = 0.001;
/// Sliding minimum over +/- this many blocks.
const LIMITER_HOLD_BLOCKS: usize = 8;
/// Box filter over +/- this many blocks, applied `LIMITER_SMOOTH_PASSES` times.
const LIMITER_SMOOTH_BLOCKS: usize = 3;
const LIMITER_SMOOTH_PASSES: usize = 2;
/// The limiter controls sample peaks; inter-sample peaks sit a little higher.
const LIMITER_MARGIN_DB: f64 = 0.3;
/// Below this shortfall the gain is simply capped: not worth limiting.
const LIMITER_MIN_SHORTFALL_DB: f64 = 0.5;
/// Never push peaks further than this into the limiter.
const MAX_LIMITER_REDUCTION_DB: f64 = 9.0;
/// Limiting removes a little loudness; correct once if it is more than this.
const LOUDNESS_RETRY_THRESHOLD_LU: f64 = 0.1;
/// Samples handed to the loudness meter per call while measuring a limited signal.
const MEASURE_CHUNK_SAMPLES: usize = 1 << 16;

fn db_to_linear(db: f64) -> f64 {
    10.0f64.powf(db / 20.0)
}

fn limiter_block_len(sample_rate: u32) -> usize {
    ((sample_rate as f64 * LIMITER_BLOCK_SECONDS).round() as usize).max(1)
}

/// Peak absolute value of each limiter block.
fn block_peaks(samples: &[f32], block: usize) -> Vec<f32> {
    samples
        .par_chunks(block)
        .map(|chunk| chunk.iter().fold(0.0f32, |peak, s| peak.max(s.abs())))
        .collect()
}

/// Block gain curve that keeps `block_peaks * gain` at or below `ceiling`.
/// Returns None when nothing is above the ceiling.
fn limiter_curve(peaks: &[f32], gain: f64, ceiling: f64) -> Option<Vec<f64>> {
    let blocks = peaks.len();
    let required: Vec<f64> = peaks
        .iter()
        .map(|&p| (ceiling / (p as f64 * gain).max(1e-12)).min(1.0))
        .collect();
    if required.iter().all(|&r| r >= 1.0) {
        return None;
    }

    // Hold: centred sliding minimum, edges repeated.
    let hold = LIMITER_HOLD_BLOCKS;
    let mut curve = vec![1.0f64; blocks];
    for (i, slot) in curve.iter_mut().enumerate() {
        let lo = i.saturating_sub(hold);
        let hi = (i + hold).min(blocks - 1);
        *slot = required[lo..=hi].iter().cloned().fold(f64::INFINITY, f64::min);
    }

    // Smooth: box filter with repeated edges, so the gain never steps.
    let smooth = LIMITER_SMOOTH_BLOCKS as isize;
    let width = (2 * smooth + 1) as f64;
    for _ in 0..LIMITER_SMOOTH_PASSES {
        let source = curve.clone();
        for (i, slot) in curve.iter_mut().enumerate() {
            let mut total = 0.0;
            for offset in -smooth..=smooth {
                let index = (i as isize + offset).clamp(0, blocks as isize - 1) as usize;
                total += source[index];
            }
            *slot = total / width;
        }
    }
    Some(curve)
}

/// Gain for sample `index`: the block curve interpolated between block centres.
#[inline]
fn curve_gain_at(curve: &[f64], block: usize, index: usize) -> f64 {
    let position = (index as f64 + 0.5) / block as f64 - 0.5;
    if position <= 0.0 {
        return curve[0];
    }
    let left = position.floor() as usize;
    if left + 1 >= curve.len() {
        return curve[curve.len() - 1];
    }
    let fraction = position - left as f64;
    curve[left] + (curve[left + 1] - curve[left]) * fraction
}

/// Integrated loudness of `samples * gain * curve` without storing the result.
fn limited_loudness(
    samples: &[f32],
    sample_rate: u32,
    gain: f64,
    curve: Option<&[f64]>,
    block: usize,
) -> Result<f64, String> {
    let mut meter = EbuR128::new(1, sample_rate, Mode::I)
        .map_err(|e| format!("Failed to create EBU R128 state: {:?}", e))?;
    let mut buffer = vec![0.0f32; MEASURE_CHUNK_SAMPLES];
    for (chunk_index, chunk) in samples.chunks(MEASURE_CHUNK_SAMPLES).enumerate() {
        let offset = chunk_index * MEASURE_CHUNK_SAMPLES;
        for (i, (out, &sample)) in buffer.iter_mut().zip(chunk.iter()).enumerate() {
            let limit = curve.map_or(1.0, |c| curve_gain_at(c, block, offset + i));
            *out = (sample as f64 * gain * limit) as f32;
        }
        meter
            .add_frames_f32(&buffer[..chunk.len()])
            .map_err(|e| format!("EBU R128 analysis failed: {:?}", e))?;
    }
    meter
        .loudness_global()
        .map_err(|e| format!("EBU R128 global loudness check failed: {:?}", e))
}

/// Applies `gain` and the limiter `curve` in place.
fn apply_gain(samples: &mut [f32], gain: f64, curve: Option<&[f64]>, block: usize) {
    match curve {
        None => {
            let gain = gain as f32;
            samples.par_iter_mut().for_each(|s| *s *= gain);
        }
        Some(curve) => {
            samples.par_iter_mut().enumerate().for_each(|(i, s)| {
                *s = (*s as f64 * gain * curve_gain_at(curve, block, i)) as f32;
            });
        }
    }
}

/// Brings `samples` to `target_lufs` without exceeding `target_tp_db`.
///
/// A static gain is used when the peaks allow it. Otherwise the full gain is
/// applied and the peaks that would cross the ceiling are limited, instead of
/// leaving the chapter quieter than asked.
fn normalize_loudness(
    samples: &mut [f32],
    sample_rate: u32,
    target_lufs: f64,
    target_tp_db: f64,
) -> Result<(), String> {
    let mut ebu = EbuR128::new(1, sample_rate, Mode::I | Mode::TRUE_PEAK)
        .map_err(|e| format!("Failed to create EBU R128 state: {:?}", e))?;
    ebu.add_frames_f32(samples)
        .map_err(|e| format!("EBU R128 analysis failed: {:?}", e))?;
    let global_lufs = ebu.loudness_global()
        .map_err(|e| format!("EBU R128 global loudness check failed: {:?}", e))?;
    if !(global_lufs.is_normal() && global_lufs > -100.0) {
        return Ok(()); // silence: nothing to normalise
    }

    // ebur128 reports true peak as a linear amplitude, not dBTP.
    let peak_linear = ebu.true_peak(0)
        .map_err(|e| format!("EBU R128 true peak check failed: {:?}", e))?;
    let peak_db = 20.0 * peak_linear.max(1e-10).log10();

    let mut gain_db = target_lufs - global_lufs;
    let headroom_db = target_tp_db - peak_db;
    let shortfall_db = gain_db - headroom_db;
    if shortfall_db <= LIMITER_MIN_SHORTFALL_DB {
        apply_gain(samples, db_to_linear(gain_db.min(headroom_db)), None, 1);
        return Ok(());
    }

    let max_gain_db = headroom_db + MAX_LIMITER_REDUCTION_DB;
    let reached = gain_db <= max_gain_db;
    gain_db = gain_db.min(max_gain_db);
    let ceiling = db_to_linear(target_tp_db - LIMITER_MARGIN_DB);
    let block = limiter_block_len(sample_rate);
    let peaks = block_peaks(samples, block);

    let mut curve = limiter_curve(&peaks, db_to_linear(gain_db), ceiling);
    let mut output_lufs =
        limited_loudness(samples, sample_rate, db_to_linear(gain_db), curve.as_deref(), block)?;
    let missing = target_lufs - output_lufs;
    if reached && output_lufs.is_normal() && missing > LOUDNESS_RETRY_THRESHOLD_LU {
        // The limiter took a little loudness away; ask for that much more.
        gain_db = (gain_db + missing).min(max_gain_db);
        curve = limiter_curve(&peaks, db_to_linear(gain_db), ceiling);
        output_lufs =
            limited_loudness(samples, sample_rate, db_to_linear(gain_db), curve.as_deref(), block)?;
    }
    apply_gain(samples, db_to_linear(gain_db), curve.as_deref(), block);

    // Inter-sample peaks can still sit above the ceiling; trim if they do.
    let mut check = EbuR128::new(1, sample_rate, Mode::TRUE_PEAK)
        .map_err(|e| format!("Failed to create EBU R128 state: {:?}", e))?;
    check.add_frames_f32(samples)
        .map_err(|e| format!("EBU R128 analysis failed: {:?}", e))?;
    let limited_peak = check.true_peak(0)
        .map_err(|e| format!("EBU R128 true peak check failed: {:?}", e))?;
    let over_db = 20.0 * limited_peak.max(1e-10).log10() - target_tp_db;
    if over_db > 0.0 {
        apply_gain(samples, db_to_linear(-over_db), None, 1);
        output_lufs -= over_db;
    }
    if !reached {
        println!(
            "[Master Rust] Peaks are {:.1} dB above what a {:.1} LUFS target allows; mastered to {:.1} LUFS to avoid audible limiting.",
            shortfall_db, target_lufs, output_lufs
        );
    }
    Ok(())
}

/// Master a list of chunk WAV files into a single, loudness-normalized destination file.
/// Performs fast parallel decoding of WAV files under CPU (via rayon).
pub fn master_audio_rust(
    chunk_paths: Vec<String>,
    out_path: String,
    pause_sec: f64,
    default_sample_rate: u32,
    target_lufs: f64,
    target_tp_db: f64,
    bitrate_kbps: u32,
) -> Result<(), String> {
    if chunk_paths.is_empty() {
        return Err("No chunk paths provided for mastering.".to_string());
    }

    // Decode WAV chunks in parallel using Rayon (bypassing python GIL)
    let decoded_results: Vec<Result<(Vec<f32>, u32), String>> = chunk_paths
        .par_iter()
        .map(|path| read_wav_samples(path))
        .collect();

    // Verify all decoded successfully and determine sample rate
    let mut sample_rate = default_sample_rate;
    let mut chunks = Vec::with_capacity(decoded_results.len());

    for (p, res) in chunk_paths.iter().zip(decoded_results.into_iter()) {
        match res {
            Ok((samples, rate)) => {
                sample_rate = rate; // use the rate from the files
                chunks.push(samples);
            }
            Err(e) => {
                println!("[Master Rust] Warning: Failed to read chunk {}: {}", p, e);
                // We keep going but skip this corrupt chunk
            }
        }
    }

    if chunks.is_empty() {
        return Err("No valid WAV files were decoded.".to_string());
    }

    // Concatenate all chunks adding silence pause between them
    let pause_len = (pause_sec * sample_rate as f64) as usize;
    let pause_samples = vec![0.0f32; pause_len];
    
    let total_len: usize = chunks.iter().map(|c| c.len()).sum::<usize>() 
        + (chunks.len() - 1) * pause_len;
        
    let mut concatenated = Vec::with_capacity(total_len);
    
    for (i, chunk) in chunks.into_iter().enumerate() {
        concatenated.extend_from_slice(&chunk);
        if i < chunk_paths.len() - 1 {
            concatenated.extend_from_slice(&pause_samples);
        }
    }

    normalize_loudness(&mut concatenated, sample_rate, target_lufs, target_tp_db)?;

    // Write final output file based on file extension
    let is_mp3 = out_path.to_ascii_lowercase().ends_with(".mp3");
    if is_mp3 {
        write_mp3_file(&out_path, &concatenated, sample_rate, bitrate_kbps)?;
    } else {
        // Fallback to WAV format
        write_wav_file(&out_path, &concatenated, sample_rate)?;
    }

    Ok(())
}
