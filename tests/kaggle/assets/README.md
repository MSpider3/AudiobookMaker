# Kaggle test assets

Everything `AudiobookMaker_Kaggle_Test.ipynb` reads comes from this folder, so the
notebook needs no uploads.

| Path | What it is |
|---|---|
| `voice/LOTM_narrator_voice.wav` | Narrator reference clip used for every voice-cloning test: 22 s, mono, 24 kHz. |
| `voice/LOTM_narrator_voice.txt` | Transcript of that clip (made with Whisper large-v3-turbo). Engines that clone "in context" need it word for word, so correct it here if you hear a mistake. |
| `books/` | Copies of the dummy books in `tests/fixtures/source_documents/` — one per supported format plus EPUB edge cases — and `expected_chapters.json`, the chapters and phrases each must produce. |

**The narrator clip is a third-party recording.** It is here only as a test reference, it is
not part of AudiobookMaker, and the project licence does not cover it. To test with another
voice, replace the `.wav` and `.txt` (5–30 s of clean speech, any sample rate), or set
`VOICE_FILE` / `VOICE_TRANSCRIPT` in the notebook's settings cell.

`books/` is generated: after regenerating the fixtures run `python tests/kaggle/sync_assets.py`.
`tests/unit/test_kaggle_assets.py` fails if the copies drift.
