import os
import numpy as np
import torch
import soundfile as sf
import queue
# --- Qwen3 TTS Custom Imports ---
from qwen_tts import Qwen3TTSModel 

# --- Global variables for the worker process (used for multiprocessing) ---
tts_model_global = None
# Global Voice Prompt Cache for Single-Source Contextual Mode
global_voice_prompt = None

TRIM_THRESHOLD = 0.04 

# --- Configuration for Single-Source High Fidelity ---
# The logic: We use a single high-quality reference with its transcript to allow the model
# to infer prosody from the text context (ICL Mode).
# NOTE: This transcript MUST match the 'genesis_text' used in tts_consumer if generating a new profile.
NARRATOR_TRANSCRIPT = "I am the narrator of this story. I read with a deep, calm, and resonant voice, capturing the mystery and tension of the chronicles."

def tts_consumer(job_queue, results_queue, args):
    """
    Final, lightweight worker. It generates audio using Qwen3-TTS High-Fidelity Mode.
    """
    global tts_model_global, global_voice_prompt
    
    # --- 1. GENESIS STEP: Ensure Narrator Profile Exists ---
    # We use the Qwen3-VoiceDesign model to strictly create the persona ONCE.
    # Then we switch to Base model to clone it efficiently.
    profile_path = os.path.abspath(args.voice_file)
    
    # FORCE RE-GENESIS if the file is tiny (likely corrupted/empty)
    if os.path.exists(profile_path) and os.path.getsize(profile_path) < 10000:
        print(f"    [Worker] Profile found but too small ({os.path.getsize(profile_path)} bytes). Deleting.")
        os.remove(profile_path)

    if not os.path.exists(profile_path):
        print(f"\n    [Worker] Voice Profile not found at: {profile_path}")
        print("    [Worker] initiating 'GENESIS' protocol with Qwen3-VoiceDesign...")
        
        try:
            design_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            design_model = Qwen3TTSModel.from_pretrained(
                design_model_name,
                device_map=args.device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            
            # The Genesis Prompt
            narrator_description = (
                "A mature male voice, mid-40s, with a deep and resonant timbre. "
                "The tone is calm, measured, and highly articulated, with a subtle British 'received pronunciation' accent. "
                "There is a slight rasp or 'vocal fry' that suggests weariness or age. "
                "Pacing is deliberate, with clear pauses for dramatic effect. "
                "The voice feels reliable but carries an undertone of suppressed tension."
            )
            # Must match NARRATOR_TRANSCRIPT
            genesis_text = "I am the narrator of this story. I read with a deep, calm, and resonant voice, capturing the mystery and tension of the chronicles."
            
            print(f"    [Worker] Synthesizing Golden Reference from description...")
            wavs, sr = design_model.generate_voice_design(
                text=genesis_text,
                instruct=narrator_description,
                language="English",
                temperature=0.7, # Lower than default 0.9 for stability
                top_p=0.9
            )
            
            # Save the Golden Reference
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            # Handle list output
            final_wav = wavs[0] if isinstance(wavs, list) else wavs
            sf.write(profile_path, final_wav, sr)
            print(f"    [Worker] GENESIS COMPLETE. Profile saved to: {profile_path}")
            
            # Cleanup to free VRAM for the Base model
            del design_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    [Worker] GENESIS FAILED: {e}")
            return

    # --- 2. PRODUCTION STEP: Initialize Base Model ---
    if tts_model_global is None:
        print("    [Worker Process] Initializing Qwen3-TTS-1.7B-Base (Production Mode)...")
        # CRITICAL: We MUST use the 'Base' model for the Cloning/ICL phase.
        model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        
        try:
            tts_model_global = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=args.device,
                dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2"
            )
            # Fix 'pad_token_id' warning
            if tts_model_global.model.generation_config.pad_token_id is None:
                tts_model_global.model.generation_config.pad_token_id = tts_model_global.model.generation_config.eos_token_id
            
            print("    [Worker Process] Base Model loaded.")
            # Note: We are skipping ICL Caching to ensure stability. 
            # We will use the generated profile as a pure X-Vector reference.
        
        except Exception as e:
             print(f"    [Worker Process] FATAL: Failed to init Qwen3: {e}")
             import traceback
             traceback.print_exc()
             return

    # Generation counter
    gen_count = 0

    while True:
        try:
            job = job_queue.get(timeout=5)
            if job == "STOP":
                print("\n    [Worker Process] STOP signal received. Exiting cleanly.")
                break

            # Job now contains optional speaker_ref
            speaker_ref = args.voice_file # Default
            
            if len(job) == 4:
                idx, sentence_text, output_wav_path, speaker_ref = job
            else:
                idx, sentence_text, output_wav_path = job

            print(f"\r  > [Worker] Generating chunk {idx+1}...", end="", flush=True)

            # --- ROBUST X-VECTOR GENERATION ---
            # We use x_vector_only_mode=True for EVERYTHING.
            # This ensures we rely on the *Sound* of the voice (Timbre) rather than fitting a transcript text.
            # This fixes the "gibberish" or "blanks" caused by imperfect genesis files.
            
            wav_data, sr = tts_model_global.generate_voice_clone(
                text=sentence_text,
                language="English", 
                ref_audio=speaker_ref,
                x_vector_only_mode=True, # Forcing specific timbre cloning
                temperature=0.7, # Good balance for X-Vector
                top_p=0.8
            )
            
            final_audio = wav_data[0] if isinstance(wav_data, (list, tuple)) or (isinstance(wav_data, torch.Tensor) and wav_data.ndim > 1) else wav_data
            
            if isinstance(final_audio, torch.Tensor):
                final_audio = final_audio.cpu().float().numpy()

            sf.write(output_wav_path, final_audio, sr)
            results_queue.put((idx, sentence_text, output_wav_path))

            gen_count += 1
            if gen_count % 10 == 0:
                torch.cuda.empty_cache()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n--- FATAL ERROR in TTS Consumer Process ---: {e}")
            import traceback
            traceback.print_exc()
            break
