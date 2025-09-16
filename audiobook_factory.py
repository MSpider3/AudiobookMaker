import os
import shutil
import subprocess
import torch
import numpy as np
import json
from scipy.io.wavfile import write as write_wav
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import time
import mutagen
from mutagen.id3 import ID3, SYLT, Encoding
from mutagen.flac import Picture
from multiprocessing import Process, Queue, set_start_method
import queue
from threading import Thread
import argparse
import re
import soundfile as sf
import pyloudnorm as pyln
# Enable TF32 for matrix multiplication on Ampere GPUs and newer for a speed boost
torch.backends.cuda.matmul.allow_tf32 = True
# Enable benchmark mode to allow cuDNN to find the best algorithm for the hardware
torch.backends.cudnn.benchmark = True
# --- TTS Library Imports ---
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Register custom classes for PyTorch deserialization (needed for some TTS models)
torch.serialization.add_safe_globals([
    XttsConfig, Xtts, XttsAudioConfig, BaseDatasetConfig, XttsArgs
])

# --- Global variables for the worker process (used for multiprocessing) ---
tts_model_global = None
gpt_cond_latent_global = None
speaker_embedding_global = None


# =========================
# === HELPER FUNCTIONS  ===
# =========================

def ms_to_srt_time(ms):
    """Converts milliseconds to SRT timestamp format (HH:MM:SS,ms)."""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def update_progress_file(progress_path, chapter_num, status):
    """
    Updates the progress JSON file for a specific chapter.
    """
    with open(progress_path, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    for chapter in progress_data["chapters"]:
        if chapter["num"] == chapter_num:
            chapter["status"] = status
            break
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=4)

def load_or_create_progress_file(progress_path, chapters_data, book_title):
    """Loads a progress file if it exists, otherwise creates a new one."""
    if os.path.exists(progress_path):
        print("Found existing progress file. Loading state.")
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("No progress file found. Creating a new one.")
        progress_data = {
            "book_title": book_title,
            "chapters": [
                {"num": c["num"], "title": c["title"], "status": "pending"} for c in chapters_data
            ]
        }
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)
        return progress_data

def format_lrc_timestamp(seconds):
    """
    Converts a time in seconds to LRC timestamp format [mm:ss.xx].
    Used for synchronizing lyrics with audio.
    """
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    hundredths = int((seconds - (minutes * 60) - sec) * 100)
    return f"[{minutes:02d}:{sec:02d}.{hundredths:02d}]"

def extract_chapters_from_epub(epub_path, skip_start, skip_end):
    """
    Reads an EPUB, extracts chapters, and also extracts metadata like
    author, title, and the cover image.
    """
    print(f"Reading EPUB file: {epub_path}")
    book = epub.read_epub(epub_path)
    
    # --- Metadata Extraction ---
    metadata = {
        "title": "Unknown Title",
        "author": "Unknown Author",
        "cover_image_data": None
    }
    titles = book.get_metadata('DC', 'title')
    if titles:
        metadata["title"] = titles[0][0]

    # Get author/creator metadata
    creators = book.get_metadata('DC', 'creator')
    if creators:
        metadata["author"] = creators[0][0]

    # --- Cover Image Extraction ---
    cover_item = book.get_item_with_id('cover')
    if cover_item:
        metadata["cover_image_data"] = cover_item.get_content()
    else:
        # Fallback to finding the first image with 'cover' in its properties
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            if 'cover-image' in item.get_properties():
                metadata["cover_image_data"] = item.get_content()
                break

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    total_items = len(items)
    start_skip = min(skip_start, total_items)
    end_skip = min(skip_end, total_items - start_skip)
    content_items = items[start_skip : total_items - end_skip]
    print(f"Found {len(content_items)} content chapters to process.")
    
    chapters_data = []
    for i, item in enumerate(content_items):
        soup = BeautifulSoup(item.get_body_content(), 'html.parser')
        
        title_tag = soup.find('h1') or soup.find('h2') or soup.find('h3')
        title = title_tag.get_text(strip=True) if title_tag else f"Chapter {i + 1}"
        # Sanitize title to be a valid filename
        title = re.sub(r'[.\\/*?:"<>|]', "", title)

        if title_tag:
            title_tag.decompose()

        body = soup.find('body')
        if body:
            text = body.get_text(separator='\n\n', strip=True)
        else:
            text = soup.get_text(separator='\n\n', strip=True)

        text = re.sub(r'\[\d+\]', '', text)

        if text and len(text) > 100:
            chapters_data.append({"num": i + 1, "title": title, "text": text})
            
    return metadata, chapters_data

def robust_sentence_splitter(text, tts_synthesizer, max_len):
    """
    A more advanced sentence splitter that is paragraph-aware and also
    intelligently splits long sentences at natural punctuation boundaries.
    """
    print("Splitting text into intelligent, paragraph-aware chunks...")
    paragraphs = text.split('\n\n')
    final_chunks = []
    
    for para_index, para in enumerate(paragraphs):
        cleaned_para = para.replace('\n', ' ').strip()
        if not cleaned_para: continue
        
        sentences_from_para = tts_synthesizer.split_into_sentences(cleaned_para)
        
        for sent_index, sentence in enumerate(sentences_from_para):
            sanitized_sentence = sentence.strip(" \"'")
            if not sanitized_sentence or not any(c.isalnum() for c in sanitized_sentence): continue

            is_paragraph_end = (sent_index == len(sentences_from_para) - 1)

            if len(sanitized_sentence) <= max_len:
                final_chunks.append({"text": sanitized_sentence, "is_para_end": is_paragraph_end})
            else:
                current_sentence_part = sanitized_sentence
                while len(current_sentence_part) > max_len:
                    split_pos = -1
                    for delimiter in ['.', ',', ';', ':', '—']:
                        pos = current_sentence_part.rfind(delimiter, 0, max_len)
                        if pos > split_pos:
                            split_pos = pos
                    
                    if split_pos == -1:
                        split_pos = current_sentence_part.rfind(' ', 0, max_len)
                    
                    if split_pos == -1:
                        split_pos = max_len - 1

                    final_chunks.append({"text": current_sentence_part[:split_pos+1].strip(), "is_para_end": False})
                    current_sentence_part = current_sentence_part[split_pos+1:].strip()
                
                if current_sentence_part:
                    final_chunks.append({"text": current_sentence_part, "is_para_end": is_paragraph_end})

    return final_chunks

# =====================================
# === TTS WORKER PROCESS FUNCTION   ===
# =====================================

# <<< NEW >>>: A tunable constant for silence trimming.
# Lower values are more aggressive. Default should be fine.
TRIM_THRESHOLD = 0.04 

def tts_consumer(job_queue, results_queue, args):
    """
    Final, lightweight worker. It generates audio, saves it to a temporary
    file, and returns the path. This offloads assembly to FFmpeg for max speed.
    """
    global tts_model_global, gpt_cond_latent_global, speaker_embedding_global

    if tts_model_global is None:
        print("    [Worker Process] Initializing TTS model...")
        tts_model_global = TTS(args.tts_model_name).to(args.device)
        print("    [Worker Process] Computing speaker latents...")
        gpt_cond_latent_global, speaker_embedding_global = tts_model_global.synthesizer.tts_model.get_conditioning_latents(audio_path=args.voice_file)
        print("    [Worker Process] Worker is ready.")

    sample_rate = tts_model_global.synthesizer.output_sample_rate

    while True:
        try:
            job = job_queue.get(timeout=5)
            if job == "STOP":
                print("\n    [Worker Process] STOP signal received. Exiting cleanly.")
                break

            # Job now contains the output path
            idx, sentence_text, output_wav_path = job
            print(f"\r  > [Worker] Generating chunk {idx+1}...", end="", flush=True)

            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                wav_chunk = tts_model_global.synthesizer.tts_model.inference(
                    text=sentence_text, language="en",
                    gpt_cond_latent=gpt_cond_latent_global,
                    speaker_embedding=speaker_embedding_global,
                    enable_text_splitting=False, temperature=args.temperature, top_p=args.top_p
                )
            
            # Convert audio data and save it directly to the file
            sf.write(output_wav_path, wav_chunk['wav'].astype(np.float32), sample_rate)
            
            # Return the path to the saved file
            results_queue.put((idx, sentence_text, output_wav_path))

        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n--- FATAL ERROR in TTS Consumer Process ---: {e}"); 
            break


# ===============================
# === MAIN ORCHESTRATOR LOGIC ===
# ===============================

def main(args):
    """
    Final, high-performance main orchestrator. This version combines robust
    best practices with high-speed, FFmpeg-based mastering. It uses in-memory
    assembly to guarantee perfect subtitle sync and resilient progress tracking
    to handle interruptions gracefully.
    """
    start_time = time.time()

    # --- 1. SETUP & CHECKPOINTING ---
    sanitized_book_title = re.sub(r'[.\\/*?:"<>|]', "", args.book_title)
    book_output_dir = os.path.join(os.getcwd(), sanitized_book_title)
    audio_chapters_dir = os.path.join(book_output_dir, "audio_chapters")
    os.makedirs(audio_chapters_dir, exist_ok=True)

    progress_file_path = os.path.join(book_output_dir, "generation_progress.json")
    epub_metadata, chapters_from_epub = extract_chapters_from_epub(args.epub_file, args.skip_start, args.skip_end)

    if not chapters_from_epub:
        print("No chapters found in EPUB. Exiting.")
        return

    if args.force_reprocess and os.path.exists(progress_file_path):
        print("--force_reprocess flag detected. Deleting old progress file.")
        os.remove(progress_file_path)

    progress_data = load_or_create_progress_file(progress_file_path, chapters_from_epub, sanitized_book_title)

    # --- 2. INITIALIZE TOOLS & WORKER PROCESS ---
    print("Initializing TTS model...")
    tts_model = TTS(args.tts_model_name)
    print("TTS model ready.")

    print("\n--- Starting Persistent Worker Process ---")
    job_queue = Queue(maxsize=64)
    results_queue = Queue(maxsize=64)
    consumer_process = Process(target=tts_consumer, args=(job_queue, results_queue, args))
    consumer_process.start()
    
    finalized_chapters_info = []
    
    if args.single_file:
        mastered_chapters_path = os.path.join(book_output_dir, "mastered_chapters")
        os.makedirs(mastered_chapters_path, exist_ok=True)

    # --- 3. MAIN CHAPTER GENERATION LOOP ---
    for chapter_progress in progress_data["chapters"]:
        # BEST PRACTICE: Robust handling of already completed chapters
        if chapter_progress["status"] == "complete":
            print(f"\n>>> Chapter {chapter_progress['num']}: '{chapter_progress['title']}' is already complete.")
            # BEST PRACTICE: If in single file mode, we still need chapter info for final assembly
            if args.single_file:
                chapter_info = next((c for c in chapters_from_epub if c["num"] == chapter_progress["num"]), None)
                if chapter_info:
                    mastered_wav = os.path.join(mastered_chapters_path, f"master_{chapter_info['num']:04d}.wav")
                    if os.path.exists(mastered_wav):
                        try:
                            info = sf.info(mastered_wav)
                            finalized_chapters_info.append({"title": chapter_info["title"], "duration": info.duration, "path": mastered_wav})
                        except Exception as e:
                            print(f"\n[WARNING] Could not read info from completed chapter {mastered_wav}. It may need reprocessing. Error: {e}")
            continue
            
        chapter_info = next((c for c in chapters_from_epub if c["num"] == chapter_progress["num"]), None)
        if not chapter_info:
            print(f"Warning: Could not find chapter data for chapter number {chapter_progress['num']}. Skipping.")
            continue
            
        chapter_num, chapter_title, chapter_text = chapter_info["num"], chapter_info["title"], chapter_info["text"]
        print(f"\n>>> Processing Chapter {chapter_num}/{len(progress_data['chapters'])}: {chapter_title}")
        chapter_start_time = time.time()
        
        # BEST PRACTICE: Mark as in-progress immediately for better crash recovery
        chapter_progress["status"] = "in_progress"
        with open(progress_file_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)

        # --- 4. GENERATE AUDIO CHUNKS & COLLECT IN MEMORY ---
        temp_chunk_folder = os.path.join(book_output_dir, "temp_audio_chunks")
        if os.path.exists(temp_chunk_folder): shutil.rmtree(temp_chunk_folder)
        os.makedirs(temp_chunk_folder)

        # BEST PRACTICE: Use a more robust sentence splitter
        sentence_chunks = robust_sentence_splitter(f"{chapter_title}\n\n{chapter_text}", tts_model.synthesizer, args.max_len)
        total_chunks = len(sentence_chunks)
        collected_files = [None] * total_chunks

        def collect_results():
            for _ in range(total_chunks):
                try:
                    # Collects file paths now, not raw audio
                    idx, text, path = results_queue.get(timeout=args.collector_timeout)
                    collected_files[idx] = {"text": text, "path": path}
                except queue.Empty:
                    print(f"\n[Collector] CRITICAL: Timed out after {args.collector_timeout}s waiting for audio chunk."); break
        
        collector_thread = Thread(target=collect_results)
        collector_thread.start()
        
        for i, chunk_info in enumerate(sentence_chunks):
            # Create a path for each chunk and send it with the job
            output_wav_path = os.path.join(temp_chunk_folder, f"s_{i:04d}.wav")
            job_queue.put((i, chunk_info["text"], output_wav_path))
            print(f"\r  > [Producer] Sent job {i+1}/{total_chunks} to queue.", end="")
        
        print("\n  > [Producer] All jobs sent. Waiting for results...")
        collector_thread.join()

        # --- FAST ASSEMBLY & SYNCED TIMESTAMP GENERATION ---
        print("\n  > Assembling chapter via FFmpeg and generating synced timestamps...")
        
        lrc_lines, srt_blocks, current_time_sec = [], [], 0.0
        sample_rate = 24000 

        # 1. Read durations from the actual files to build the timeline
        for i, result in enumerate(collected_files):
            if result is None or not os.path.exists(result['path']): continue
            try:
                info = sf.info(result['path'])
                duration_sec = info.duration
            except Exception:
                duration_sec = 0 

            lrc_lines.append(f"{format_lrc_timestamp(current_time_sec)}{result['text']}")
            srt_blocks.append(f"{i+1}\n{ms_to_srt_time(int(current_time_sec*1000))} --> {ms_to_srt_time(int((current_time_sec+duration_sec)*1000))}\n{result['text']}\n")

            current_time_sec += duration_sec
            if i < total_chunks - 1:
                is_para_end = sentence_chunks[i]["is_para_end"]
                current_time_sec += args.para_pause if is_para_end else args.pause
        
        # 2. Create silence files and filelist for FFmpeg
        sent_pause_path = os.path.join(temp_chunk_folder, "pause_sent.wav")
        para_pause_path = os.path.join(temp_chunk_folder, "pause_para.wav")
        sf.write(sent_pause_path, np.zeros(int(args.pause * sample_rate)), sample_rate)
        sf.write(para_pause_path, np.zeros(int(args.para_pause * sample_rate)), sample_rate)

        filelist_path = os.path.join(temp_chunk_folder, "filelist.txt")
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(collected_files):
                if result and os.path.exists(result['path']):
                    f.write(f"file '{os.path.basename(result['path'])}'\n")
                    if i < total_chunks - 1:
                        pause_file = para_pause_path if sentence_chunks[i]["is_para_end"] else sent_pause_path
                        f.write(f"file '{os.path.basename(pause_file)}'\n")

        # 3. Assemble with FFmpeg
        unmastered_wav_path = os.path.join(temp_chunk_folder, "unmastered_chapter.wav")
        ffmpeg_concat_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.basename(filelist_path), '-c', 'copy', os.path.basename(unmastered_wav_path)]
        subprocess.run(ffmpeg_concat_cmd, check=True, capture_output=True, cwd=temp_chunk_folder)

        # --- 6. HIGH-PERFORMANCE FFmpeg-BASED MASTERING ---
        print(f"  > Mastering audio to {args.lufs} LUFS and {args.true_peak} dBTP peak using FFmpeg...")
        master_wav_path = os.path.join(temp_chunk_folder, "master_chapter.wav")
        ffmpeg_master_cmd = [
            'ffmpeg', '-y', '-i', unmastered_wav_path,
            '-af', f'loudnorm=I={args.lufs}:TP={args.true_peak}:LRA=11:print_format=summary',
            master_wav_path
        ]
        try:
            subprocess.run(ffmpeg_master_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] ffmpeg mastering failed: {e.stderr.decode()}")
            # Handle error as needed
            continue
        
        # --- 7. CONDITIONAL FINALIZATION (MULTI-FILE VS. SINGLE-FILE) ---
        sanitized_chapter_title = re.sub(r'[.\\/*?:"<>|]', "", chapter_title)
        base_filename = f"{sanitized_book_title} - {sanitized_chapter_title}"
        
        if not args.single_file:
            # --- MULTIPLE FILES MODE FINALIZATION ---
            final_output_path = os.path.join(audio_chapters_dir, f"{base_filename}.{args.output_format}")
            
            # --- Write the appropriate, perfectly synced subtitle file ---
            if args.output_format == 'mp4':
                chapter_srt_path = os.path.join(audio_chapters_dir, f"{base_filename}.srt")
                with open(chapter_srt_path, 'w', encoding='utf-8') as f: f.write("\n".join(srt_blocks))
            else:
                chapter_lrc_path = os.path.join(audio_chapters_dir, f"{base_filename}.lrc")
                with open(chapter_lrc_path, 'w', encoding='utf-8') as f: f.write("\n".join(lrc_lines))

            cover_art_path = None
            if args.cover_image and os.path.exists(args.cover_image):
                cover_art_path = args.cover_image
            elif epub_metadata["cover_image_data"]:
                cover_art_path = os.path.join(temp_chunk_folder, "cover.jpg")
                with open(cover_art_path, "wb") as f: f.write(epub_metadata["cover_image_data"])
            
            ffmpeg_cmd = None
            
            # --- Build command for audio formats (MP3/FLAC) ---
            if args.output_format in ['mp3', 'flac']:
                ffmpeg_cmd = ['ffmpeg', '-y', '-i', master_wav_path]
                if cover_art_path:
                    ffmpeg_cmd.extend(['-i', cover_art_path, '-map', '0:a', '-map', '1:v', '-disposition:v', 'attached_pic'])
                
                if args.output_format == 'mp3':
                    ffmpeg_cmd.extend(['-c:a', 'libmp3lame', '-b:a', '320k', '-ar', '44100', '-ac', '2', '-id3v2_version', '3', '-c:v', 'copy', final_output_path])
                else: # flac
                    ffmpeg_cmd.extend(['-c:a', 'flac', '-ar', '44100', '-ac', '2', '-c:v', 'copy', final_output_path])
            
            # --- Build command for video format (MP4) ---
            elif args.output_format == 'mp4':
                if not cover_art_path:
                    print("[ERROR] MP4 output requires a cover image, but none was found or provided.")
                else:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y',
                        '-i', master_wav_path,
                        '-i', cover_art_path,
                        '-i', chapter_srt_path,
                        '-map', '0:a', '-map', '1:v', '-map', '2:s',
                        '-c:v', 'libx264', '-preset', 'medium', '-tune', 'stillimage',
                        '-c:a', 'aac', '-b:a', '256k', '-ar', '44100', '-ac', '2', # <<< Corrected audio quality
                        '-c:s', 'mov_text',
                        '-movflags', '+faststart',
                        # <<< Add Metadata
                        '-metadata', f"title={chapter_title}",
                        '-metadata', f"artist={args.author or epub_metadata['author']}",
                        '-metadata', f"album={sanitized_book_title or epub_metadata['title']}",
                        final_output_path
                    ]
            
            # --- <<< NEW >>> Build command for WAV format ---
            elif args.output_format == 'wav':
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', master_wav_path,
                    '-c:a', 'pcm_s32le',  # <<< Set 32-bit sample size
                    '-ar', '44100',         # <<< Set 44.1kHz sample rate
                    '-ac', '2',             # <<< Set stereo channels
                    # <<< Add Metadata
                    '-metadata', f"title={chapter_title}",
                    '-metadata', f"artist={args.author or epub_metadata['author']}",
                    '-metadata', f"album={sanitized_book_title or epub_metadata['title']}",
                    '-metadata', f"track={chapter_num}",
                    final_output_path
                ]

            # --- Execute FFmpeg command ---
            if ffmpeg_cmd:
                try:    
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:    
                    print(f"\n[ERROR] Final FFmpeg conversion failed: {e.stderr.decode()}"); 
                    continue

            # Embed metadata and lyrics into MP3/FLAC
            if args.output_format in ['mp3', 'flac']:
                print(f"  > Embedding metadata and lyrics into {args.output_format.upper()}...")
                try:
                    if args.output_format == 'mp3':
                        try:
                            audio = ID3(final_output_path)
                            audio.add(mutagen.id3.TALB(encoding=3, text=sanitized_book_title or epub_metadata['title']))
                            audio.add(mutagen.id3.TPE1(encoding=3, text=args.author or epub_metadata['author']))
                            audio.add(mutagen.id3.TIT2(encoding=3, text=chapter_title))
                            audio.add(mutagen.id3.TRCK(encoding=3, text=str(chapter_num)))
                            audio.add(mutagen.id3.TCON(encoding=3, text="Audiobook"))
                            lrc_pattern = re.compile(r"\[(\d+):(\d+)\.(\d+)\](.*)")
                            lrc_entries = []
                            for line in lrc_lines:
                                match = lrc_pattern.match(line)
                                if match:
                                    min, sec, cs = int(match.group(1)), int(match.group(2)), int(match.group(3))
                                    timestamp_ms = (min * 60 + sec) * 1000 + cs * 10
                                    lrc_entries.append((match.group(4).strip(), timestamp_ms))
                            audio.add(SYLT(encoding=3, lang='eng', format=2, type=1, desc='Lyrics', text=lrc_entries))
                            audio.save(v2_version=3)
                        except Exception as e:
                            print(f"\n[WARNING] Could not embed MP3 metadata: {e}")
                    elif args.output_format == 'flac':
                        audio = mutagen.flac.FLAC(final_output_path)
                        audio['album'] = args.book_title or epub_metadata['title']
                        audio['artist'] = args.author or epub_metadata['author']
                        audio['title'] = chapter_title
                        audio['tracknumber'] = str(chapter_num)
                        audio['genre'] = "Audiobook"
                        if os.path.exists(chapter_lrc_path):
                            with open(chapter_lrc_path, 'r', encoding='utf-8') as f:
                                audio['lyrics'] = f.read()
                        audio.save()
                except Exception as e:
                    print(f"\n[WARNING] Could not embed metadata: {e}")
        else:
            # --- SINGLE FILE MODE: STAGE MASTERED FILE ---
            final_mastered_wav_path = os.path.join(mastered_chapters_path, f"master_{chapter_num:04d}.wav")
            shutil.move(master_wav_path, final_mastered_wav_path)
            
            # CRITICAL FIX: Get duration from the actual mastered file for accuracy
            try:
                info = sf.info(final_mastered_wav_path)
                duration = info.duration
            except Exception as e:
                print(f"\n[WARNING] Could not read duration from {final_mastered_wav_path}: {e}")
                duration = 0 # Fallback
            
            finalized_chapters_info.append({"title": chapter_title, "duration": duration, "path": final_mastered_wav_path})
            print(f"  > Chapter {chapter_num} mastered and staged for single-file assembly.")

        # --- 8. CLEANUP AND PROGRESS UPDATE ---
        shutil.rmtree(temp_chunk_folder)
        chapter_progress["status"] = "complete"
        with open(progress_file_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)
        
        print(f"--- Chapter {chapter_num} processing complete ---")
        chapter_elapsed_time = time.time() - chapter_start_time
        print(f"  > Chapter completed in {chapter_elapsed_time:.2f} seconds.")

    # --- 9. FINAL ASSEMBLY FOR SINGLE FILE MODE ---
    if args.single_file:
        print("\n\n--- All chapters processed. Assembling single audiobook file... ---")
        
        finalized_chapters_info.sort(key=lambda x: x['path'])
        
        metadata_path = os.path.join(book_output_dir, "chapters_metadata.txt")
        current_time_ms = 0
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            for chapter in finalized_chapters_info:
                start_time = int(current_time_ms)
                end_time = int(current_time_ms + chapter['duration'] * 1000)
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_time}\n")
                f.write(f"END={end_time}\n")
                f.write(f"title={chapter['title']}\n")
                current_time_ms = end_time
        
        concat_list_path = os.path.join(book_output_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for chapter in finalized_chapters_info:
                f.write(f"file '{os.path.relpath(chapter['path'], book_output_dir)}'\n")

        final_output_path = os.path.join(book_output_dir, f"{sanitized_book_title}.{args.output_format}")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.basename(concat_list_path),
            '-i', os.path.basename(metadata_path),
            '-map_metadata', '1'
        ]
        
        cover_art_path = None
        if args.cover_image and os.path.exists(args.cover_image):
            cover_art_path = args.cover_image
        elif epub_metadata["cover_image_data"]:
            cover_art_path = os.path.join(book_output_dir, "cover.jpg")
            with open(cover_art_path, "wb") as f: f.write(epub_metadata["cover_image_data"])
        
        if cover_art_path:
            ffmpeg_cmd.extend(['-i', os.path.basename(cover_art_path), '-map', '0:a', '-map', '2:v', '-disposition:v', 'attached_pic'])

        if args.output_format == 'mp3':
            ffmpeg_cmd.extend(['-c:a', 'libmp3lame', '-b:a', '320k', '-ar', '44100', '-ac', '2', '-id3v2_version', '3', '-c:v', 'copy'])
        # Add other single-file format options here if needed (e.g., m4b)
        
        ffmpeg_cmd.append(os.path.basename(final_output_path))
        
        print("  > Running final FFmpeg command to build the book...")
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, cwd=book_output_dir)
            print(f"  > Successfully created single file: {os.path.basename(final_output_path)}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Final single-file assembly failed: {e.stderr.decode()}")
        
        # Cleanup temporary files for single-file mode
        shutil.rmtree(mastered_chapters_path)
        os.remove(metadata_path)
        os.remove(concat_list_path)
        if cover_art_path and "cover.jpg" in cover_art_path: os.remove(cover_art_path)

    # --- 10. SHUTDOWN AND FINISH ---
    print("\n--- All processing is complete! Shutting down worker process... ---")
    job_queue.put("STOP")
    consumer_process.join()
    consumer_process.close()

    print("\n--- Project Complete! ---")
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/3600:.2f} hours.")
    if args.single_file:
        print(f"Your single-file audiobook is ready in: '{book_output_dir}'")
    else:
        print(f"Your chapterized audiobook is ready in: '{audio_chapters_dir}'")
        
        
# ===============================
# === ENTRY POINT / ARGPARSE  ===
# ===============================

if __name__ == "__main__":
    try: set_start_method('spawn', force=True)
    except RuntimeError: pass

    parser = argparse.ArgumentParser(description="The Audiobook Factory: A complete EPUB to Audiobook pipeline.")
    
    # --- Core Arguments ---
    parser.add_argument("-i", "--epub_file", type=str, default="./LOTM/Lord of the Mysteries - Vol. 1 - Clown.epub", help="Path to the input EPUB file.")
    parser.add_argument("-v", "--voice_file", type=str, default="./narrator_voice/LOTM_narrator_voice_no_space.wav", help="Path to the narrator's voice sample WAV file.")
    parser.add_argument("-b", "--book_title", type=str, default="Lord of the Mysteries - Vol. 1 - Clown", help="The name of the book, used for the output folder and filenames.")
    parser.add_argument("--author", type=str, default=None, help="The author's name for metadata. If not provided, will try to extract from EPUB.")
    parser.add_argument("--cover_image", type=str, default=None, help="Path to a custom cover image. If not provided, will try to extract from EPUB.")

    # --- <<< NEW >>> Output Mode ---
    parser.add_argument("--single_file", action="store_true", help="Combine all chapters into a single output file with chapter markers.")
    
    # --- EPUB Processing ---
    parser.add_argument("--skip_start", type=int, default=6, help="Number of 'chapters' to skip at the beginning of the EPUB.")
    parser.add_argument("--skip_end", type=int, default=231, help="Number of 'chapters' to skip at the end of the EPUB.")
    
    # --- Audio Pacing ---
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds of silence between sentences.")
    parser.add_argument("--para_pause", type=float, default=1.2, help="Seconds of silence for paragraph breaks.")
    
    # --- TTS Tuning ---
    parser.add_argument("--max_len", type=int, default=240, help="Maximum character length for a single text chunk.")
    parser.add_argument("--temperature", type=float, default=0.8, help="TTS generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.8, help="TTS generation top_p.")
    
    # --- Mastering & Output Arguments ---
    parser.add_argument("--output_format", type=str, default="mp3", choices=['mp3', 'wav', 'flac', 'mp4'], help="The desired output audio format for each chapter.")
    parser.add_argument("--lufs", type=int, default=-18, help="Target loudness in LUFS for audio normalization.")
    parser.add_argument("--true_peak", type=float, default=-1.5, help="Target true peak in dBTP to prevent clipping.")
    
    # --- System & Control ---
    parser.add_argument("--collector_timeout", type=int, default=300, help="Seconds the collector will wait for a result.")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of all chapters, ignoring existing JSON progress.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for TTS ('cuda' or 'cpu').")
    parser.add_argument("--tts_model_name", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="The Coqui TTS model to use.")
    
    args = parser.parse_args()

    if not os.path.exists(args.epub_file): print(f"ERROR: EPUB file not found: '{args.epub_file}'"); exit(1)
    if not os.path.exists(args.voice_file): print(f"ERROR: Voice sample not found: '{args.voice_file}'"); exit(1)
        
    main(args)