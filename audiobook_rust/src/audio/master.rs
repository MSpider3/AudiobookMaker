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

    // Measure integrated loudness (EBU R128)
    let mut ebu = EbuR128::new(1, sample_rate, Mode::I | Mode::TRUE_PEAK)
        .map_err(|e| format!("Failed to create EBU R128 state: {:?}", e))?;
        
    ebu.add_frames_f32(&concatenated)
        .map_err(|e| format!("EBU R128 analysis failed: {:?}", e))?;
        
    let global_lufs = ebu.loudness_global()
        .map_err(|e| format!("EBU R128 global loudness check failed: {:?}", e))?;

    // Perform loudness normalization via linear gain scale
    if global_lufs.is_normal() && global_lufs > -100.0 {
        let gain_db = target_lufs - global_lufs;
        let mut gain = 10.0f32.powf((gain_db / 20.0) as f32);
        
        // Scan for True Peak to avoid clipping
        let peak_db = ebu.true_peak(0)
            .map_err(|e| format!("EBU R128 true peak check failed: {:?}", e))?;
            
        let peak_after_gain_db = peak_db + gain_db;
        if peak_after_gain_db > target_tp_db {
            // Cap gain so we do not exceed hard true_peak target
            let safe_gain_db = target_tp_db - peak_db;
            gain = 10.0f32.powf((safe_gain_db / 20.0) as f32);
            println!(
                "[Master Rust] Loudnorm required gain ({:.2} dB) truncated to safe gain ({:.2} dB) to respect True Peak target ({:.2} dBTP)",
                gain_db, safe_gain_db, target_tp_db
            );
        }
        
        // Apply gain in-place
        for sample in concatenated.iter_mut() {
            *sample *= gain;
        }
    }

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
