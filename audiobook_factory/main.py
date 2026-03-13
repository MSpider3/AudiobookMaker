import os
import sys
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

# --- TTS Library Imports ---
# (Removed Coqui TTS imports - migrating to Qwen3)

# --- Global variables for the worker process (used for multiprocessing) ---
# tts_model_global is managed in audio_processor.py for the worker process
# Fix sys.path to allow importing 'audiobook_factory' package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from our custom modules
from audiobook_factory.config import parse_arguments
from audiobook_factory.utils import ms_to_srt_time, update_progress_file, load_or_create_progress_file, format_lrc_timestamp
from audiobook_factory.audio_processor import tts_consumer
from audiobook_factory.ffmpeg_utils import get_format_settings
from audiobook_factory.ebook_importer import EbookImporter
from audiobook_factory.story_analyzer import StoryAnalyzer
from audiobook_factory.text_processing import smart_sentence_splitter, normalize_text

# PyTorch Performance Tuners for Ampere GPUs and newer
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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

    # --- Initialize Modules ---
    importer = EbookImporter(enable_ocr=args.docling_ocr)
    analyzer = StoryAnalyzer(book_output_dir, model_size=args.booknlp_model_size, enable=args.enable_booknlp)
    
    # --- Import EPUB ---
    # We use our new importer which handles metadata + docling text extraction
    epub_metadata = importer.extract_metadata(args.epub_file)
    if not epub_metadata:
        print("Failed to extract metadata. Exiting.")
        return

    # Use Docling to process chapters
    chapters_from_epub = importer.process_epub(args.epub_file)
    
    if not chapters_from_epub:
        print("No chapters found. Exiting.")
        return

    # Filter chapters based on skip args
    total_found = len(chapters_from_epub)
    start_skip = min(args.skip_start, total_found)
    end_skip = min(args.skip_end, total_found - start_skip)
    chapters_to_process = chapters_from_epub[start_skip : total_found - end_skip]
    
    print(f"Processing {len(chapters_to_process)} chapters after skipping (Started with {total_found}).")

    progress_file_path = os.path.join(book_output_dir, "generation_progress.json")
    
    if args.force_reprocess and os.path.exists(progress_file_path):
        print("--force_reprocess flag detected. Deleting old progress file.")
        os.remove(progress_file_path)
    
    # Init progress with the filtered list
    progress_data = load_or_create_progress_file(progress_file_path, chapters_to_process, sanitized_book_title)

    # --- 2. START WORKER PROCESS ---
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
    allowed_chapter_nums = set(c["num"] for c in chapters_to_process)
    
    for chapter_progress in progress_data["chapters"]:
        # Skip chapters not in the current run's scope (respecting skip_start/skip_end)
        if chapter_progress["num"] not in allowed_chapter_nums:
            continue
            
        # BEST PRACTICE: Robust handling of already completed chapters
        if chapter_progress["status"] == "complete":
            print(f"\n>>> Chapter {chapter_progress['num']}: '{chapter_progress['title']}' is already complete.")
            # BEST PRACTICE: If in single file mode, we still need chapter info for final assembly
            if args.single_file:
                # Find original info to get title text etc
                chapter_info = next((c for c in chapters_to_process if c["num"] == chapter_progress["num"]), None)
                if chapter_info:
                    mastered_wav = os.path.join(mastered_chapters_path, f"master_{chapter_info['num']:04d}.wav")
                    if os.path.exists(mastered_wav):
                        try:
                            info = sf.info(mastered_wav)
                            finalized_chapters_info.append({"title": chapter_info["title"], "duration": info.duration, "path": mastered_wav})
                        except Exception as e:
                            print(f"\n[WARNING] Could not read info from completed chapter {mastered_wav}. It may need reprocessing. Error: {e}")
            continue
            
        chapter_info = next((c for c in chapters_to_process if c["num"] == chapter_progress["num"]), None)
        if not chapter_info:
            print(f"Warning: Could not find chapter data for chapter number {chapter_progress['num']}. Skipping.")
            continue
            
        chapter_num, chapter_title, chapter_text = chapter_info["num"], chapter_info["title"], chapter_info["text"]
        print(f"\n>>> Processing Chapter {chapter_num}: {chapter_title}")
        chapter_start_time = time.time()
        
        # BEST PRACTICE: Mark as in-progress immediately for better crash recovery
        chapter_progress["status"] = "in_progress"
        with open(progress_file_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)

        # --- 4. TEXT ANALYSIS & SEGMENTATION ---
        # A. Normalize (De-wrapping, Drop Caps)
        # Note: Docling output is usually cleaner than raw scrape, but normalization helps.
        clean_text = normalize_text(f"{chapter_title}\n\n{chapter_text}")
        
        # B. Analyze (BookNLP -> Speakers)
        # Segments: list of {"text": "...", "speaker": "..."}
        annotated_segments = analyzer.analyze_text(clean_text, chapter_num)
        
        # C. Split for TTS (Max Char Limit)
        # We need to flatten the segments into TTS-ready chunks while preserving speaker info.
        tts_jobs = []
        
        for segment in annotated_segments:
            speaker_name = segment["speaker"]
            seg_text = segment["text"]
            
            # Determine Voice File
            voice_path = args.voice_file # Default Narrator
            
            # Simple directory lookup for character voices
            # Structure: ./voices/CharacterName.wav
            if speaker_name != "Narrator":
                # Sanitize name
                safe_name = re.sub(r'[^\w\s]', '', speaker_name).strip()
                char_voice_candidate = os.path.join(os.getcwd(), "voices", f"{safe_name}.wav")
                if os.path.exists(char_voice_candidate):
                    voice_path = char_voice_candidate
                    # print(f"  [Voice Switch] Using voice for {speaker_name}")

            # Split logic
            chunks = smart_sentence_splitter(seg_text, args.max_len)
            
            for i_chunk, chunk in enumerate(chunks):
                tts_jobs.append({
                    "text": chunk, 
                    "voice": voice_path,
                    "is_para_end": (i_chunk == len(chunks) - 1) # Restore paragraph pauses
                })
                
        # --- 5. GENERATE AUDIO CHUNKS ---
        temp_chunk_folder = os.path.join(book_output_dir, "temp_audio_chunks")
        if os.path.exists(temp_chunk_folder): shutil.rmtree(temp_chunk_folder)
        os.makedirs(temp_chunk_folder)
        
        total_chunks = len(tts_jobs)
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
        
        for i, job_data in enumerate(tts_jobs):
            # Create a path for each chunk and send it with the job
            output_wav_path = os.path.join(temp_chunk_folder, f"s_{i:04d}.wav")
            # Payload: (idx, text, path, voice_ref)
            job_queue.put((i, job_data["text"], output_wav_path, job_data["voice"]))
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
            
            # Simple pause logic since we simplified splitting
            # If next chunk has different voice, maybe add extra pause?
            if i < total_chunks - 1:
                # current voice
                curr_voice = tts_jobs[i]["voice"]
                next_voice = tts_jobs[i+1]["voice"]
                
                # If voice changes, 0.8s pause. Else standard 0.5s
                pause_len = 0.8 if curr_voice != next_voice else args.pause
                current_time_sec += pause_len
        
        # 2. Create silence files and filelist for FFmpeg
        sent_pause_path = os.path.join(temp_chunk_folder, "pause_sent.wav")
        voice_switch_pause_path = os.path.join(temp_chunk_folder, "pause_switch.wav")
        sf.write(sent_pause_path, np.zeros(int(args.pause * sample_rate)), sample_rate)
        sf.write(voice_switch_pause_path, np.zeros(int(0.8 * sample_rate)), sample_rate)

        filelist_path = os.path.join(temp_chunk_folder, "filelist.txt")
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(collected_files):
                if result and os.path.exists(result['path']):
                    f.write(f"file '{os.path.basename(result['path'])}'\n")
                    if i < total_chunks - 1:
                        curr_voice = tts_jobs[i]["voice"]
                        next_voice = tts_jobs[i+1]["voice"]
                        pause_file = voice_switch_pause_path if curr_voice != next_voice else sent_pause_path
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
        
        if args.single_file:
            # --- SINGLE FILE MODE: STAGE MASTERED FILE ---
            final_mastered_wav_path = os.path.join(mastered_chapters_path, f"master_{chapter_num:04d}.wav")
            shutil.move(master_wav_path, final_mastered_wav_path)
            
            try:
                info = sf.info(final_mastered_wav_path)
                finalized_chapters_info.append({"title": chapter_title, "duration": info.duration, "path": final_mastered_wav_path})
                print(f"  > Chapter {chapter_num} mastered and staged for single-file assembly.")
            except Exception as e:
                print(f"\n[WARNING] Could not read duration from {final_mastered_wav_path}: {e}")

        else:
            # --- MULTIPLE FILES MODE FINALIZATION ---
            # CORRECT: base_filename is defined once with the chapter number for proper sorting.
            base_filename = f"{sanitized_book_title} - {sanitized_chapter_title}"
            final_output_path = os.path.join(audio_chapters_dir, f"{base_filename}.{args.output_format}")
            
            # --- Cover Art ---
            cover_art_path = None
            if args.cover_image and os.path.exists(args.cover_image):
                cover_art_path = args.cover_image
            elif epub_metadata["cover_image_data"]:
                cover_art_path = os.path.join(temp_chunk_folder, "cover.jpg")
                with open(cover_art_path, "wb") as f: f.write(epub_metadata["cover_image_data"])

            # --- FFmpeg Command Builder ---
            ffmpeg_cmd = ['ffmpeg', '-y', '-i', master_wav_path]
            
            common_metadata = [
                '-metadata', f"title={chapter_title}",
                '-metadata', f"artist={args.author or epub_metadata['author']}",
                '-metadata', f"album={sanitized_book_title or epub_metadata['title']}",
                '-metadata', f"track={chapter_num}",
                '-metadata', "genre=Audiobook"
            ]
            
            # --- FFmpeg Command Builder (Starts with the master audio file) ---
            ffmpeg_cmd = ['ffmpeg', '-y', '-i', master_wav_path]

            # --- CALL THE SETTINGS FACTORY ---
            audio_settings, video_settings, subtitle_codec, subtitle_type = get_format_settings(args.output_format)
            
            subtitle_path = None # Will hold the path to the .srt, .vtt, or .lrc file
            
            if subtitle_type:
                # Determine the correct file extension and path
                file_extension = f".{subtitle_type}"
                subtitle_path = os.path.join(audio_chapters_dir, f"{base_filename}{file_extension}")
                print(f"  > Generating {subtitle_type.upper()} file...")

                if subtitle_type == 'lrc':
                    # For LRC, we use the lrc_lines generated earlier
                    with open(subtitle_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lrc_lines))
                
                elif subtitle_type == 'srt':
                    # For SRT, we use the srt_blocks generated earlier
                     with open(subtitle_path, 'w', encoding='utf-8') as f:
                        f.write("\n\n".join(srt_blocks))
                
                elif subtitle_type == 'vtt':
                    # For VTT (used by WebM), we convert the SRT blocks
                    vtt_content = "WEBVTT\n\n" + "\n\n".join(
                        block.replace(",", ".") for block in srt_blocks
                    )
                    with open(subtitle_path, 'w', encoding='utf-8') as f:
                        f.write(vtt_content)

            # --- Add Inputs and Map Streams ---
            # Handle Video Containers that embed subtitles (MP4, MOV, WebM)
            if subtitle_type in ['srt', 'vtt']:
                if not cover_art_path:
                    print(f"[WARNING] Video format '{args.output_format}' selected but no cover art found.")
                else:
                    # Add cover art AND subtitles as inputs
                    ffmpeg_cmd.extend(['-i', cover_art_path, '-i', subtitle_path])
                    # Map the streams: 0:audio, 1:video(cover), 2:subtitles
                    ffmpeg_cmd.extend(['-map', '0:a', '-map', '1:v', '-map', '2:s', '-c:s', subtitle_codec])
            
            # Handle Audio Formats that just get a picture attached
            elif cover_art_path:
                # Add cover art as an input
                ffmpeg_cmd.extend(['-i', cover_art_path])
                # Map the streams: 0:audio, 1:video(cover) and mark cover as attached pic
                ffmpeg_cmd.extend(['-map', '0:a', '-map', '1:v', '-disposition:v', 'attached_pic'])

            # --- Assemble the Final Command ---
            ffmpeg_cmd.extend(audio_settings)    # Add audio encoding settings (e.g., -c:a libmp3lame)
            ffmpeg_cmd.extend(video_settings)    # Add video encoding settings (e.g., -c:v copy)
            ffmpeg_cmd.extend(common_metadata)   # Add all metadata tags
            ffmpeg_cmd.append(final_output_path) # Add the final output file path
            
            try:    
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"  > Successfully created chapter file: {os.path.basename(final_output_path)}")
            except subprocess.CalledProcessError as e:    
                print(f"\n[ERROR] FFmpeg conversion failed for chapter {chapter_num}: {e.stderr.decode()}"); 
                continue

        # --- 8. CLEANUP AND PROGRESS UPDATE ---
        shutil.rmtree(temp_chunk_folder)
        chapter_progress["status"] = "complete"
        with open(progress_file_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)
        
        print(f"--- Chapter {chapter_num} processing complete ---")
        chapter_elapsed_time = time.time() - chapter_start_time
        print(f"  > Chapter completed in {chapter_elapsed_time:.2f} seconds.")

    # --- 9. FINAL ASSEMBLY FOR SINGLE FILE MODE ---
    if args.single_file and finalized_chapters_info:
        print("\n\n--- All chapters processed. Assembling single audiobook file... ---")
        
        finalized_chapters_info.sort(key=lambda x: x['path'])
        
        # --- Create FFMETADATA file for chapters ---
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
        
        # --- Create file list for concatenation ---
        concat_list_path = os.path.join(book_output_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for chapter in finalized_chapters_info:
                f.write(f"file '{os.path.relpath(chapter['path'], book_output_dir)}'\n")

        final_output_path = os.path.join(book_output_dir, f"{sanitized_book_title}.{args.output_format}")
        
        # --- Build the FFmpeg command ---
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

        ffmpeg_cmd.extend([
            '-metadata', f"title={sanitized_book_title or epub_metadata['title']}",
            '-metadata', f"artist={args.author or epub_metadata['author']}",
            '-metadata', "genre=Audiobook"
        ])
        
        # --- Call the settings factory ---
        audio_settings, video_settings, _ = get_format_settings(args.output_format)
        
        ffmpeg_cmd.extend(audio_settings)
        if cover_art_path:
            ffmpeg_cmd.extend(video_settings)

        ffmpeg_cmd.append(os.path.basename(final_output_path))
        
        # --- Execute the command ONCE ---
        print("  > Running final FFmpeg command to build the book...")
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, cwd=book_output_dir)
            print(f"  > Successfully created single file: {os.path.basename(final_output_path)}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Final single-file assembly failed: {e.stderr.decode()}")
        
        # --- Cleanup ---
        shutil.rmtree(mastered_chapters_path)
        os.remove(metadata_path)
        os.remove(concat_list_path)
        if cover_art_path and "cover.jpg" in cover_art_path: os.remove(cover_art_path)

    # --- 10. SHUTDOWN AND FINISH ---
    print("\n--- All processing is complete! Shutting down worker process... ---")
    job_queue.put("STOP")
    consumer_process.join()
    consumer_process.close()

    # Clean up temporary directory from file conversion, if it exists
    # we don't need this anymore as EbookImporter handles it or doesn't use it
    
    print("\n--- Project Complete! ---")
    total_time = time.time() - start_time
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/3600:.2f} hours.")
    if args.single_file:
        print(f"Your single-file audiobook is ready in: '{book_output_dir}'")
    else:
        print(f"Your chapterized audiobook is ready in: '{audio_chapters_dir}'")


if __name__ == "__main__":
    try: set_start_method('spawn', force=True)
    except RuntimeError: pass

    args = parse_arguments()

    if not os.path.exists(args.epub_file): print(f"ERROR: EPUB file not found: '{args.epub_file}'"); exit(1)
    if not os.path.exists(args.voice_file): print(f"ERROR: Voice sample not found: '{args.voice_file}'"); exit(1)
        
    main(args)