import argparse
import torch

def parse_arguments():
    """
    Defines and parses all command-line arguments for the Audiobook Factory.
    This function is the single source of truth for all user-configurable settings.
    """
    parser = argparse.ArgumentParser(
        description="The Audiobook Factory: A complete EPUB to Audiobook pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Core Arguments ---
    parser.add_argument("-i", "--epub_file", type=str, default="./LOTM/Lord of the Mysteries - Vol. 1 - Clown.epub", help="Path to the input EPUB file.")
    parser.add_argument("-v", "--voice_file", type=str, default="./narrator_voice/LOTM_narrator_voice_no_space.wav", help="Path to the narrator's voice sample WAV file.")
    parser.add_argument("-b", "--book_title", type=str, default="Lord of the Mysteries - Vol. 1 - Clown", help="The name of the book, used for the output folder and filenames.")
    parser.add_argument("--author", type=str, default=None, help="The author's name for metadata. If not provided, will try to extract from EPUB.")
    parser.add_argument("--cover_image", type=str, default=None, help="Path to a custom cover image. If not provided, will try to extract from EPUB.")

    # --- Output Mode ---
    parser.add_argument("--single_file", action="store_true", help="Combine all chapters into a single output file with chapter markers.")
    
    # --- EPUB Processing ---
    parser.add_argument("--skip_start", type=int, default=1, help="Number of 'chapters' to skip at the beginning of the EPUB.")
    parser.add_argument("--skip_end", type=int, default=212, help="Number of 'chapters' to skip at the end of the EPUB.")
    
    # --- Audio Pacing ---
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds of silence between sentences.")
    parser.add_argument("--para_pause", type=float, default=1.2, help="Seconds of silence for paragraph breaks.")
    
    # --- TTS Tuning ---
    parser.add_argument("--max_len", type=int, default=399, help="Maximum character length for a single text chunk.")
    parser.add_argument("--temperature", type=float, default=0.3, help="TTS generation temperature (Lower = More stable/less gibberish).")
    parser.add_argument("--top_p", type=float, default=0.8, help="TTS generation top_p.")
    
    # --- Mastering & Output Arguments ---
    parser.add_argument("--output_format", type=str, default="mp3", choices=['m4b', 'm4a', 'mp4', 'webm', 'mov', 'mp3', 'flac', 'wav', 'ogg', 'aac'], help="The desired output audio format for each chapter.")
    parser.add_argument("--lufs", type=int, default=-18, help="Target loudness in LUFS for audio normalization.")
    parser.add_argument("--true_peak", type=float, default=-1.5, help="Target true peak in dBTP to prevent clipping.")
    
    # --- System & Control ---
    parser.add_argument("--collector_timeout", type=int, default=300, help="Seconds the collector will wait for a result.")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of all chapters, ignoring existing JSON progress.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for TTS ('cuda' or 'cpu').")
    parser.add_argument("--tts_model_name", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="The Qwen3 TTS model to use.")
    
    # --- Advanced Pipeline Configs ---
    parser.add_argument("--enable_booknlp", action="store_true", help="Enable BookNLP for character voice analysis. (First run will download ~2GB models).")
    parser.add_argument("--booknlp_model_size", type=str, default="small", choices=["small", "big"], help="BookNLP model size.")
    parser.add_argument("--docling_ocr", action="store_true", help="Enable OCR for images inside Docling.")
    
    return parser.parse_args()