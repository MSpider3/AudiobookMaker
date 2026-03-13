def get_format_settings(fmt):
    
    audio_settings = ['-ar', '44100', '-ac', '2']
    video_settings = []
    subtitle_codec = 'mov_text'
    subtitle_type = None  # By default, a format has no subtitle/lyric file.

    # --- Lossy Audio Formats ---
    if fmt == 'mp3':
        audio_settings.extend(['-c:a', 'libmp3lame', '-q:a', '0'])
        subtitle_type = 'lrc' # MP3 players widely support sidecar .lrc files
    elif fmt in ['m4a', 'm4b', 'aac']:
        audio_settings.extend(['-c:a', 'aac', '-b:a', '320k'])
        subtitle_type = 'lrc' # M4B/M4A can use sidecar .lrc files
    elif fmt == 'ogg':
        audio_settings.extend(['-c:a', 'libvorbis', '-q:a', '9'])
        subtitle_type = 'lrc' # OGG can use sidecar .lrc files
    elif fmt == 'webm':
        audio_settings = ['-ar', '48000', '-ac', '2', '-c:a', 'libopus', '-b:a', '320k']
        video_settings.extend(['-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0'])
        subtitle_codec = 'webvtt'
        subtitle_type = 'vtt' # WebM requires the .vtt format

    # --- Lossless Audio Formats ---
    elif fmt == 'flac':
        audio_settings.extend(['-c:a', 'flac'])
        subtitle_type = 'lrc' # FLAC players often support sidecar .lrc files
    elif fmt == 'wav':
        audio_settings.extend(['-c:a', 'pcm_s32le'])
        subtitle_type = None # WAV has no standard for synchronized lyrics

    # --- Video Container Formats ---
    if fmt in ['mp4', 'mov']:
        audio_settings.extend(['-c:a', 'aac', '-b:a', '320k'])
        video_settings.extend(['-c:v', 'libx264', '-preset', 'medium', '-tune', 'stillimage'])
        subtitle_type = 'srt' # MP4 and MOV use .srt for embedding

    # --- Set Video Codec for Audio Formats with Pictures ---
    elif fmt in ['mp3', 'flac', 'm4a', 'm4b', 'aac', 'ogg']:
         video_settings.extend(['-c:v', 'copy'])

    return audio_settings, video_settings, subtitle_codec, subtitle_type