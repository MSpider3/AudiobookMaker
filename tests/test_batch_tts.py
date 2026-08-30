import os
import time
import torch
import shutil
from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers import get_tts_provider

def run_test():
    print("=" * 60)
    print("           AudiobookMaker Parallel Batch Verification           ")
    print("=" * 60)

    # 1. Create output folder
    test_out_dir = "./test_output_speed"
    os.makedirs(test_out_dir, exist_ok=True)

    # 2. Configure Settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    config = AudiobookConfig(
        voice_file="./narrator_voice/LOTM_narrator_voice_no_space.wav",
        tts_provider_name="qwen",
        tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device=device,
        worker_count=4,
        parallel_mode="chunks",
        output_dir=test_out_dir
    )

    # 3. Instantiate TTS Provider
    print("Initializing Qwen3-TTS model (this may take a few seconds on first load)...")
    provider = get_tts_provider(config.tts_provider_name, config)
    print("Model initialized successfully!")

    # 4. Prepare test sentences
    texts = [
        "The mysterious world of Tingen was silent under the silver moonlight.",
        "Klein Moretti woke up from a table, feeling a cold sensation on his temple.",
        "Inside the revolver, a single brass bullet sat quietly in the cylinder.",
        "He reached out his hand, touching the sticky red fluid on his forehead."
    ]

    # --- Mode A: Sequential Run ---
    print("\nStarting Sequential Chunk Synthesis...")
    seq_paths = [os.path.join(test_out_dir, f"seq_{i}.wav") for i in range(len(texts))]
    start_seq = time.time()
    for text, path in zip(texts, seq_paths):
        provider.synthesize(text, config.voice_file, path)
    end_seq = time.time()
    seq_time = end_seq - start_seq
    print(f"Sequential run finished in: {seq_time:.2f} seconds")

    # --- Mode B: Batched Run ---
    voice_bytes = b""
    if os.path.exists(config.voice_file):
        with open(config.voice_file, "rb") as vf:
            voice_bytes = vf.read()
    batch_paths = [os.path.join(test_out_dir, f"batch_{i}.wav") for i in range(len(texts))]
    start_batch = time.time()
    batch_results = provider.synthesize_batch(texts, voice_bytes, return_bytes=True)
    for (audio_bytes, dur), bp in zip(batch_results, batch_paths):
        if isinstance(audio_bytes, bytes):
            with open(bp, "wb") as fh:
                fh.write(audio_bytes)
    end_batch = time.time()
    batch_time = end_batch - start_batch
    print(f"Batched run finished in: {batch_time:.2f} seconds")

    # 5. Results & Validation
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("-" * 60)
    print(f"Sequential Mode: {seq_time:.2f} seconds")
    print(f"Batched Mode:    {batch_time:.2f} seconds")
    print(f"Speedup:         {seq_time / batch_time:.2f}x faster!")
    print("=" * 60)

    # Check that files were created and are valid
    print("\nVerifying files...")
    for idx, (sp, bp) in enumerate(zip(seq_paths, batch_paths)):
        seq_exists = os.path.exists(sp) and os.path.getsize(sp) > 0
        batch_exists = os.path.exists(bp) and os.path.getsize(bp) > 0
        print(f"Chunk {idx}:")
        print(f"  Sequential output: {'✅ Valid' if seq_exists else '❌ Invalid'}")
        print(f"  Batched output:    {'✅ Valid' if batch_exists else '❌ Invalid'}")
        if not (seq_exists and batch_exists):
            print("ERROR: One or more files are missing or empty.")
            return

    # Clean up output
    print(f"\nCleaning up {test_out_dir} directory...")
    shutil.rmtree(test_out_dir, ignore_errors=True)
    print("Cleanup done.")
    print("Verification completed successfully!")

if __name__ == "__main__":
    run_test()
