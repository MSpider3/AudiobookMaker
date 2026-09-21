# OpenVuln report — MSpider3/AudiobookMaker

## [high] Gradio progress-JSON restore handler (on_progress_upload) presets File/Audio component values with arbitrary server paths, causing unauthenticated arbitrary file download
- key: `BUG-R2-C1-A2-H2`
- disclosure: owner_only
- cwe: CWE-22
- file: `app.py`

# Gradio progress-JSON restore handler (on_progress_upload) presets File/Audio component values with arbitrary server paths, causing unauthenticated arbitrary file download

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C1-A2-H2
- **CWE:** CWE-22
- **CVSS:** 7.5 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N`)
- **EV priority:** P0
- **EV score:** 9
- **PoC status:** reproduced
- **EXP status:** confirmed
- **Affected versions:** audited snapshot v1.3.0 line (commit dc9aed2cac064f436310a35fdd069f34205fa687); the on_progress_upload restore feature is present in current main. Exact introduction version not established. Impact scope depends on the installed gradio version: <= 4.44.1 = whole-filesystem read; >= 5.0.0 = app working directory (project root in the advertised Colab/Kaggle deployment) + system temp dir + gradio cache. Dynamically reproduced on 4.44.1, 5.0.0 and 6.27.0 (poc/poc.md) and business-impact-confirmed on 6.27.0 in a cloud-notebook-faithful replica (exp/exp.md). Note: the shipped app.py passes theme=/css= to Blocks.launch(), kwargs that only exist since gradio 6.0.0, so this exact snapshot launches unmodified only on gradio >= 6.0 (the unpinned requirements.txt currently resolves 6.27.0); the <= 4.44.1 whole-filesystem variant was demonstrated on a copy with only those two launch kwargs removed (handler and components untouched) and equally applies to older app snapshots with a 4.x-era launch call.

## Exploitability rationale

Reachability R:N — the trigger is a single anonymous HTTP upload against the advertised public tunnel (Colab/Kaggle notebooks expose the exact same no-auth Gradio app at a public pinggy.io URL; no authentication exists anywhere in the app); dynamically confirmed end-to-end in a notebook-faithful replica (exp/exp.md: 12 user-content files stolen with 58 anonymous requests, all byte-identical). Exposure E:D — the "Resume from Progress JSON" upload is part of the default UI (Generate tab) and runs in every deployment mode; no option disables it. Certainty C:D — pure logic defect, deterministic single-request trigger, no race or memory layout involved. Impact I:S — complete theft of every user's content on a shared instance, pre-auth: on gradio >= 5.0.0 (current unpinned resolution 6.27.0) the whole application tree is readable — other users' generated audiobooks and progress files (full book text), the fixed-path global progress file temp/generation_progress.json (full text + the cache paths of the victim's original uploads, which the attacker then steals straight out of the gradio upload cache — the "pivot"), the Download-All ZIP, README-instructed narrator_voice/ and Your_Novel/ placements (voice-cloning samples), .git — plus any known-path /tmp artifact of co-tenant processes. The gradio upload cache itself is not enumerable by content-hash guessing (secret-seeded addressing, verified dynamically), but the finding provides the path leak that defeats it. On gradio <= 4.44.1 installs the scope is the entire filesystem of a root-owned notebook VM (escalating toward I:X: /etc/shadow, /root/.cache/huggingface/token, /proc/self/environ with HF tokens and API keys; the secret gradio hash_seed becomes readable there, enabling full known-content cache enumeration), which is why the finding stays P0.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 1400 | `on_progress_upload` |
| `app.py` | 1512 | `on_progress_upload` |
| `app.py` | 1513 | `on_progress_upload` |
| `app.py` | 1556 | `build_app` |
| `app.py` | 1560 | `build_app` |
| `app.py` | 168 | `build_app` |
| `app.py` | 307 | `build_app` |
| `audiobook_factory/progress_io.py` | 64 | `read_progress_file` |

## Background

AudiobookMaker is a self-hosted text-to-speech audiobook generator. Its primary interface is an unauthenticated Gradio web app (app.py) that the README and the bundled Colab/Kaggle notebooks deliberately expose to anonymous visitors through a public SSH tunnel (pinggy.io) pointing at localhost:7860; on those notebooks the Python process runs as root. The "Generate" tab contains a resume feature: the user uploads a previously exported generation_progress.json and the handler on_progress_upload restores ~40 UI components from that JSON (book title, language, TTS settings, and notably the book/voice file components). The progress file is normally produced by the tool itself, so its book_path/voice_file fields are trusted to name server-side files. The feature exists so a user can pick up an interrupted run on the same instance. Because the app has no authentication and no multi-user model, any visitor of a shared instance can upload a hand-crafted progress JSON — the JSON contents are fully attacker-controlled (read_progress_file only parses JSON, it does not validate fields).

## Description

on_progress_upload reads book_path/voice_file verbatim from the uploaded JSON (app.py:1400-1401) and, if os.path.exists() succeeds, pushes the raw string as the value of the book_file gr.File component (output #3, app.py:1512) and voice_studio_upload gr.Audio component (output #4, app.py:1513) via gr.update(value=<path>). The upload-side extension filter on those components (file_types=[".epub", ".mobi", ".pdf", ".docx", ".odt", ".txt"]) only applies to client uploads (it lives in File.preprocess), never to programmatically set output values, so any file with any extension qualifies. Gradio's output postprocessing then turns that value into a client-visible download: File.postprocess wraps the path in a FileData, and move_files_to_cache copies the file from its original server location into the Gradio download cache (shutil.copy2) and attaches a /file=<cache copy> URL, which the unauthenticated /file= route serves to the browser. The copy behavior was verified against the Gradio sources of 4.44.1, 5.0.0, 5.49.1 and 6.27.0. Version-dependent scope: on gradio <= 4.44.1 there is no path restriction at all in output postprocessing, so ANY existing file readable by the process (root on the notebooks) is copied and downloadable — /etc/shadow, /root/.ssh/authorized_keys, /proc/self/environ (containing HF tokens / API keys), other users' data. On gradio >= 5.0.0 the framework's _check_allowed() restricts output file copying to launch(allowed_paths) plus the process working directory plus the system temp directory plus the Gradio cache; paths outside that set abort the event with InvalidPathError. However, in the advertised Colab/Kaggle deployment the process working directory IS the project root (the notebook chdirs into the cloned repository), and the local launch mode passes allowed_paths=[_ROOT] — so on every version the entire project tree is inside the allowed set: audiobook_output/ (every session's generated audiobooks and generation_progress.json files containing the full book text, settings and pronunciation maps), narrator_voice/, Your_Novel/, temp/, .git/ and the source tree, plus /tmp wholesale (including the shared Gradio cache that holds other users' uploaded books and voice samples). The gr.Audio variant behaves identically: with the app's configuration Audio.postprocess keeps the raw path (the ffprobe playability check fails open for non-audio files) and the same move_files_to_cache copy/gate applies.

## Attack

An anonymous visitor of the public notebook tunnel URL (the deployment the project advertises for sharing instances with strangers) opens the Generate tab, uploads the crafted progress JSON into the "Resume from Progress JSON" control (or posts it through the equally unauthenticated queue/REST API), and reads the event response: the returned FileData for the book/voice component contains a /file=... URL of a cache copy of the targeted file, which is then fetched from the same unauthenticated server. One request per file, no victim interaction, no credentials. On gradio <= 4.44.1 installs this yields whole-filesystem read of a root-owned notebook VM (OS secrets, SSH keys, environment tokens); on current installs (>= 5.0.0, unpinned resolution 6.27.0) it yields arbitrary read of the whole project tree — every user's books, voice samples (usable for voice cloning), generated audiobooks and progress data (full book text). A companion file-existence oracle in the same handler ("✅ Found book file at: <path>" vs "⚠️ ... not found") lets the attacker enumerate targets first; on >= 5.0.0 an existing but out-of-scope path additionally produces a distinguishable whole-event abort (success:false, no data), refining the oracle to three outcomes. Dynamically confirmed extra leaks: on launches with show_error=True (the local python app.py mode) the out-of-scope InvalidPathError text — including the app's absolute working directory — reaches the anonymous client (the notebook's debug=True launch suppresses the message text; the abort itself remains observable). Both the queue/SSE flow used by the browser and the simpler named REST endpoint POST /gradio_api/run/on_progress_upload trigger the identical chain anonymously; the gr.Audio output also delivers non-audio files (playability check fails open). All positive cases verified byte-identical via sha256 (victim progress JSON + voice WAV, .git/config with embedded token, /tmp artifact, and on 4.44.1: /root secret file, /etc/shadow and /etc/passwd in a single request; business-impact EXP on 6.27.0 in a notebook-faithful replica: 12 user-content files stolen anonymously, all byte-identical — see exp/exp.md).

### Payload

A small generation_progress.json whose top-level book_path field names an existing server-side file, e.g. {"book_title": "x", "book_path": "/etc/shadow", "chapters": []} (minimum JSON size limits apply; chapters list may be empty). The voice_file field can carry a second target in the same request. On gradio >= 5.0.0 the target must be under the app's working directory, /tmp or the Gradio cache for the copy to succeed — e.g. "audiobook_output/<victim title>/generation_progress.json".

## Data flow

### Step 1 — `app.py:1556-1561`

progress_file_upload.upload(on_progress_upload, inputs=[progress_file_upload], outputs=[progress_upload_status, book_title_box, book_file, voice_studio_upload, ...]) — anonymous, unauthenticated event wiring; book_file (gr.File) is output #3, voice_studio_upload (gr.Audio) output #4.

### Step 2 — `app.py:1400`

book_path = data.get("book_path", "") — raw attacker-controlled string from the uploaded JSON; audiobook_factory/progress_io.py:64-134 (read_progress_file) performs JSON/encoding parsing only, no field validation.

### Step 3 — `app.py:1512-1513`

gr.update(value=book_path) if book_path and os.path.exists(book_path) else gr.update() — the only check is bare existence; the arbitrary server path becomes the value of the gr.File / gr.Audio components.

### Step 4 — `gradio blocks.py (postprocess_update_dict)`

Framework output postprocessing applies File.postprocess / Audio.postprocess to the update value; File.postprocess returns FileData(path=<raw server path>) with no file_types enforcement (the extension filter exists only in upload-side preprocess).

### Step 5 — `gradio processing_utils.py (move_files_to_cache / _move_to_cache)`

For gradio <= 4.44.1: no path restriction on outputs — Block.move_resource_to_block_cache -> save_file_to_cache -> shutil.copy2 copies the targeted file into the Gradio cache and payload.url becomes /file=<cache copy>. For gradio >= 5.0.0 (5.0.0, 5.49.1, 6.27.0 verified): _check_allowed() first requires the path to be inside launch(allowed_paths) or the process working directory or the system temp dir or the Gradio cache; otherwise InvalidPathError aborts the event. Inside the allowed set the identical copy2 + /file= URL behavior runs.

### Step 6 — `gradio routes.py / route_utils.py (/file= route)`

The anonymous client fetches the returned /file=<cache copy> URL; the cache copy is always servable (created/upload dir), so the targeted file's bytes are delivered to the attacker. In the advertised Colab/Kaggle deployment the working directory is the project root (notebook chdirs into the repo), and the local launcher passes allowed_paths=[_ROOT], so the entire project tree is in scope on every gradio version; on <= 4.44.1 the scope is the whole filesystem.

## Fix / patch notes

diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -58,6 +58,17 @@
 # ==============================================================================
 _OUTPUT_DIR = os.path.join(_ROOT, "audiobook_output")
 os.makedirs(_OUTPUT_DIR, exist_ok=True)
+
+
+def _safe_restore_path(p: object) -> bool:
+    """Validate a path from an uploaded progress JSON before presetting a
+    file/audio component with it. Only existing *files* inside the project
+    root are accepted; Gradio copies preset component values into its
+    download cache and serves them to the client, so an unvalidated path
+    would leak arbitrary server files."""
+    try:
+        rp = os.path.realpath(str(p))
+        return os.path.isfile(rp) and os.path.commonpath([rp, _ROOT]) == _ROOT
+    except (OSError, ValueError, TypeError):
+        return False

 # ==============================================================================
 # Helper utilities
@@ -1509,8 +1520,8 @@ def build_app() -> gr.Blocks:

                 return (
                     msg,
-                    gr.update(value=book_path) if book_path and os.path.exists(book_path) else gr.update(),
-                    gr.update(value=voice_file) if voice_file and os.path.exists(voice_file) else gr.update(),
+                    gr.update(value=book_path) if _safe_restore_path(book_path) else gr.update(),
+                    gr.update(value=voice_file) if _safe_restore_path(voice_file) else gr.update(),
                     gr.update(value=author_val),
                     gr.update(value=lang_val),
                     gr.update(value=out_fmt_val),

## References

- https://cwe.mitre.org/data/definitions/22.html
- https://cwe.mitre.org/data/definitions/552.html
- https://www.gradio.app/guides/file-component-behavior
- https://owasp.org/www-community/attacks/Path_Traversal

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] Progress-JSON restore (`on_progress_upload`) presets an attacker-chosen server path into the book file component — arbitrary book-format file content disclosure via Export Config download
- key: `BUG-R2-C1-A2-H3`
- disclosure: owner_only
- cwe: CWE-22
- file: `app.py`

# Progress-JSON restore (`on_progress_upload`) presets an attacker-chosen server path into the book file component — arbitrary book-format file content disclosure via Export Config download

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C1-A2-H3
- **CWE:** CWE-22
- **CVSS:** 7.5 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N`)
- **EV priority:** P0
- **EV score:** 9
- **PoC status:** reproduced
- **EXP status:** confirmed
- **Affected versions:** unknown (progress-JSON restore feature and all consuming handlers present in the audited snapshot; CHANGELOG head v1.3.0). Framework behavior pinned to installed Gradio 6.27.0; gradio is unpinned in requirements.txt, so tree-scope details may differ across installs. Dynamically reproduced on Gradio 6.27.0 in the cloud-notebook launch mode (no allowed_paths): byte-identical .txt disclosure plus boundary abort for out-of-scope paths — see poc/poc.md. Business-impact-confirmed on Gradio 6.27.0 in a cloud-notebook-faithful replica: full EPUB manuscript text + cover image bytes (sha256-verified) and full Word-draft text stolen anonymously through the Export-Config download; co-user generated chapter .txt stolen; .json/.wav rejected at the consuming click; out-of-scope restore abort — see exp/exp.md.

## Exploitability rationale

R:N — the documented cloud deployment (Colab/Kaggle notebooks) exposes the
unauthenticated Gradio UI through a public Pinggy tunnel; any anonymous
visitor of the URL triggers the whole chain (upload JSON → click Export
Config) — dynamically confirmed end-to-end in a notebook-faithful replica
(exp/exp.md: manuscript EPUB + Word draft + co-user chapter text stolen
with 38 anonymous requests, 6 per file). E:D — the Resume-from-Progress-JSON
feature and all three consuming handlers (Preview / Generate / Export
Config) are core, always-on UI paths present in every deployment. C:D —
pure logic flow, no race, no memory layout, no probabilistic step; the
only preconditions (file exists, book-extension name, inside the framework
allowlist trees) are properties of the attacker's target choice, verifiable
in advance via the existence echo. I:S — partial information disclosure:
the full textual content (plus EPUB cover bytes, base64) of files under
the allowlist trees whose names end in .txt/.epub/.mobi/.pdf/.docx/.odt.
In this app's threat model that file class is the core private asset:
the instance owner's manuscripts and drafts in the working tree (the
in-tree Your_Novel/ instruction directs exactly this placement, in .epub)
are reachable unconditionally and with no knowledge beyond common paths;
other users' generated chapter-text exports under audiobook_output/ are
reachable when the book title is known/guessed (public for published
books; confirmable via the existence oracle). Bounded, not prevented:
non-book extensions are rejected at the consuming click, so this leg
alone cannot read the .json progress files (no path pivot) or the .wav
voice samples that the sibling whole-file read (BUG-R2-C1-A2-H2, same
root cause, output-side sink) also exposes — this finding is the
input-side sink and stands on its own: it survives closure of the
output-side download route and already discloses the core asset class.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 1400 | `on_progress_upload` |
| `app.py` | 1512 | `on_progress_upload` |
| `app.py` | 1556 | `build_app` |
| `app.py` | 1593 | `on_export_config` |
| `app.py` | 1638 | `on_export_config` |
| `app.py` | 1742 | `on_export_config` |
| `app.py` | 1824 | `on_export_config` |
| `app.py` | 1831 | `on_export_config` |
| `app.py` | 1019 | `on_preview` |
| `app.py` | 1035 | `on_preview` |
| `app.py` | 1076 | `on_generate` |
| `app.py` | 1216 | `on_generate` |
| `audiobook_factory/text_extractor.py` | 79 | `_detect_type` |
| `audiobook_factory/text_extractor.py` | 386 | `_extract_txt` |
| `audiobook_factory/pipeline.py` | 1038 | `run_pipeline` |

## Background

AudiobookMaker is a single-user TTS audiobook generator with a Gradio web UI
(app.py). It is advertised in two deployment modes: local workstation
(`python app.py`, localhost only) and cloud notebooks (Colab/Kaggle), where
the same unauthenticated UI is exposed to anyone via a public Pinggy SSH
tunnel — the README explicitly tells users to share that link. The UI has no
authentication and no per-user isolation. A core convenience feature,
"Resume from Progress JSON", lets a user upload a previously exported
`generation_progress.json` to restore all settings and file references: the
handler `on_progress_upload` reads top-level `book_path`/`voice_file`
strings from that JSON and presets them into the book/voice file components,
assuming the paths refer to files on the user's own machine. The book file
component's value is then consumed by the Preview / Generate / Export-Config
handlers, which call `extract(path)` — a server-side full-document parser
for .txt/.epub/.mobi/.pdf/.docx/.odt — and the Export-Config handler writes
the extracted chapter text into a downloadable `generation_progress.json`.
The design gap: on a (publicly exposed) server, the restored path is a live
server-side path chosen by an unauthenticated visitor, turning the resume
feature into a file-content read primitive for the app's own download
features.

## Description

`on_progress_upload` (app.py:1380) parses the uploaded JSON with
`read_progress_file` (JSON-syntax checks only) and takes
`book_path = data.get("book_path", "")` (app.py:1400) with no validation.
It returns `gr.update(value=book_path)` gated only by `os.path.exists`
(app.py:1512) into the `book_file` gr.File component (wired at
app.py:1556-1568). On the installed Gradio 6.27.0 the framework's
postprocess copies the referenced file into the Gradio cache
(`save_file_to_cache`, original filename preserved) after a path-allowlist
check (`_check_allowed`: `allowed_paths` ∪ {CWD, system tempdir} ∪ gradio
cache); in cloud mode `allowed_paths` is not set, so the allowed trees are
the process CWD (`/content` or the repo dir on Colab, `/kaggle/working` on
Kaggle) plus `/tmp` and the gradio cache; in local mode the whole project
root is whitelisted (`allowed_paths=[_ROOT]`, app.py:1915). When the
visitor then clicks "Export Config JSON" (or Preview / Generate), the
browser submits the preset file value back; the framework's input-side
checks pass (cache-resident path, original extension retained), and the
handler receives the path via `file_obj.name` (app.py:1593; likewise 1019 /
1076). `extract(path)` (app.py:1638) dispatches by extension
(`_detect_type`, text_extractor.py:79-113): `_extract_txt`
(text_extractor.py:386-398) reads the entire file with
`errors="replace"` (binary bytes lossily decoded, never rejected);
PDF/EPUB/MOBI/DOCX/ODT are parsed in full by their respective parsers
(EPUB and DOCX legs proven end-to-end in exp/exp.md: chapter text
string-equal to the app's own offline extract(), EPUB cover bytes
sha256-identical). `on_export_config` collects the extracted text and
sentences (app.py:1742-1744), embeds the EPUB cover image as base64
(`scan(path)` at app.py:1722, `cover_image_b64` at app.py:1739-1740), and
writes everything into `audiobook_output/<attacker-chosen
title>/generation_progress.json` (app.py:1824), which it returns as a
direct download (app.py:1831 → gr.File at app.py:576) served through the
unauthenticated `/file=` route. The cached-chapters shortcut at app.py:1604
is bypassed by choosing a fresh book title (attacker-controlled in the same
JSON). Additional channels: `on_preview` returns per-chapter
char/word/sentence counts and chapter titles (app.py:1043-1046,
pipeline.py:384-419); `on_generate` with `export_text=True` writes one .txt
per chapter into the output dir and returns them in the download list
(pipeline.py:1038-1045, app.py:1340-1348). Net effect (EXP-confirmed
scope): an anonymous visitor obtains the full textual content (and EPUB
cover bytes) of any existing file inside the allowlist trees whose name
ends in one of the six book extensions — unconditionally the owner's
book documents placed in the working tree (the in-tree Your_Novel/
instruction directs users to keep their .epub books there; the repo ships
Your_Novel/test.txt; Word/.txt drafts anywhere in the tree qualify), and
conditionally (title knowledge, confirmable via the existence oracle)
other users' generated chapter-text exports under audiobook_output/ plus
known-path book-format files under /tmp. The process runs as root on
Colab, so OS permissions are no barrier. Files outside the allowlist
trees (e.g. /etc, /root) are rejected by the framework's path-allowlist
check before the component is preset, and non-book extensions are
rejected by the component's server-side `file_types` check at input
preprocess (so this leg alone cannot read the .json progress files or
.wav voice samples — the sibling whole-file read BUG-R2-C1-A2-H2 exposes
those); these bound, but do not prevent, the disclosure of the app's core
private asset class living exactly inside the bounded scope.

## Attack

Attacker: any anonymous visitor of the public notebook tunnel URL (the
documented sharing deployment; no authentication exists). Step 1: upload
the crafted progress JSON — the handler echoes whether `book_path` exists
and presets the book file component with the server-side copy of that
file. Step 2: click "Export Config JSON" — the server parses the preset
file server-side, writes the extracted full text (plus base64 cover for
EPUB) into `audiobook_output/<title>/generation_progress.json`, and offers
it as a download; the attacker retrieves the foreign file's content.
Cheaper variants: "Preview Chapters" returns per-chapter counts and titles
(content-size oracle); "Generate Audiobook" with export-text enabled
returns per-chapter .txt files (and spoken audio). Targets that matter in
practice (EXP-confirmed): the instance owner's manuscripts and drafts in
the working tree — the in-tree Your_Novel/ instruction directs users to
keep their .epub books there (Colab /content/<repo>/Your_Novel/, Kaggle
/kaggle/working/<repo>/Your_Novel/), and .docx/.txt drafts anywhere in
the tree qualify; other users' generated chapter-text exports under
audiobook_output/<book title>/ when the title is known or guessed
(published-book titles are public; the existence oracle confirms guesses);
known-path book-format files under /tmp. Not reachable through this leg
alone: other users' original UI uploads (secret-seeded gradio cache
paths; the .json progress file that records them is itself rejected by
the extension gate) and .wav voice samples. Two requests per file, fully
deterministic, no user interaction or prior compromise required.

### Payload

A small `generation_progress.json` uploaded through the "Resume from
Progress JSON" control. Key fields: `book_title` (any fresh string, e.g. a
random name, so the output dir does not collide with cached progress),
`book_path` (the absolute server path of the target file, which must exist,
end in .txt/.epub/.mobi/.pdf/.docx/.odt, and lie under the process CWD,
/tmp, the gradio cache, or — local mode only — the project root), and a
minimal `settings` object. Example: `{"book_title": "probe-1",
"book_path": "/content/MyNovelDraft.txt", "settings": {}}`. Optionally
`settings.export_text = true` if the Generate channel is used instead of
Export Config.

## Data flow

### Step 1 — `app.py:1400`

on_progress_upload reads book_path = data.get("book_path", "") from the attacker-uploaded progress JSON; read_progress_file (progress_io.py:64-126) validates JSON syntax only — the value is an arbitrary server path string.

### Step 2 — `app.py:1512`

Handler returns gr.update(value=book_path) gated only by os.path.exists(book_path); wired as the 3rd output to the book_file gr.File component (outputs list app.py:1556-1568).

### Step 3 — `app.py:1556`

progress_file_upload.upload(on_progress_upload, inputs=[progress_file_upload], outputs=[progress_upload_status, book_title_box, book_file, ...]) — the preset lands in the same component later used as input by Preview/Generate/Export handlers.

### Step 4 — `app.py:1593`

Attacker clicks 'Export Config JSON': the browser submits the preset book_file value; the handler takes path = file_obj.name (framework input-side checks pass because the preset was copied server-side into the gradio cache with the original extension preserved; file_types filter passes for the six book extensions).

### Step 5 — `app.py:1604`

_load_cached_chapters_if_available(prog_path, selected_chapters) is bypassed — the attacker controls book_title (restored from the same JSON at app.py:1399/1511), so prog_path points to a fresh, non-existent output dir and extraction proceeds.

### Step 6 — `app.py:1638`

chapters, _ = extract(path, ...) — server-side full-document read of the attacker-chosen file. Dispatch by _detect_type (text_extractor.py:79-113); _extract_txt (text_extractor.py:386-398) reads the whole file with errors="replace"; PDF/EPUB/MOBI/DOCX/ODT parsers read the document in full.

### Step 7 — `app.py:1742`

Extracted chapter text and sentences are collected into chapters_data ({"num", "title", "text", "sentences"}); for EPUB targets the cover image bytes are base64-embedded via scan(path) (app.py:1722) → cover_image_b64 (app.py:1739-1740).

### Step 8 — `app.py:1824`

write_progress_file(prog_path, progress_data) writes audiobook_output/<attacker title>/generation_progress.json containing the foreign file's full extracted text.

### Step 9 — `app.py:1831`

Return gr.update(value=prog_path, visible=True) → export_config_file gr.File (app.py:576); the framework copies the file into the gradio cache and serves it as a download through the unauthenticated /file= route (no auth= is ever set), completing the exfiltration.

### Step 10 — `app.py:1043`

Alternative channels: on_preview (extract at app.py:1035) renders per-chapter chars/words/sentences counts and chapter titles (pipeline.py:384-419) — a content-size/metadata oracle; on_generate (extract at app.py:1216) with export_text=true writes per-chapter .txt files (pipeline.py:1038-1045) returned in download_files (app.py:1340-1348).

## Fix / patch notes

diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1511,8 +1511,12 @@ def build_app():
                 msg,
                 gr.update(value=title) if title else gr.update(),
-                gr.update(value=book_path) if book_path and os.path.exists(book_path) else gr.update(),
-                gr.update(value=voice_file) if voice_file and os.path.exists(voice_file) else gr.update(),
+                # Security: never preset File/Audio components with raw server
+                # paths restored from an uploaded progress JSON. On an
+                # unauthenticated (publicly tunneled) instance this turns any
+                # existing server file into a readable component value that
+                # Preview/Generate/Export ingest server-side (arbitrary file
+                # content disclosure). The recorded path is already shown as
+                # informational text in `msg`; the user re-uploads the book.
+                gr.update(),
+                gr.update(),
                 gr.update(value=author_val),

## References

- https://cwe.mitre.org/data/definitions/22.html
- https://cwe.mitre.org/data/definitions/200.html
- https://cwe.mitre.org/data/definitions/552.html
- https://owasp.org/www-community/attacks/Path_Traversal

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] Gradio progress-cache resume (title-derived audiobook_output/<title>/generation_progress.json) cross-session resource confusion lets an anonymous visitor of the shared instance download another session's complete book content and destroy or replace their resume state and audiobook outputs
- key: `BUG-R2-C1-A2-H5`
- disclosure: owner_only
- cwe: CWE-639
- file: `app.py`

# Gradio progress-cache resume (title-derived audiobook_output/<title>/generation_progress.json) cross-session resource confusion lets an anonymous visitor of the shared instance download another session's complete book content and destroy or replace their resume state and audiobook outputs

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C1-A2-H5
- **CWE:** CWE-639
- **CVSS:** 8.6 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:L/A:L`)
- **EV priority:** P0
- **EV score:** 9
- **PoC status:** reproduced
- **EXP status:** confirmed
- **Affected versions:** v1.3.0 (audited snapshot, commit dc9aed2cac064f436310a35fdd069f34205fa687); earlier versions not examined. Dynamically confirmed end-to-end (gradio 6.27.0, torch 2.14.0+cpu, Python 3.11): oracle probe, no-book export cache-hit, anonymous download of the victim's full book text/cover, and the collateral settings overwrite all reproduced through the real handlers (see poc/poc.md). Business-scenario EXP (see exp/exp.md, 46/46 checks): in the README-advertised shared cloud deployment (notebook launch mode + stub-TTS backend + production Rust mastering) the same confusion additionally delivered the zero-knowledge default-directory grab (empty title), an oracle sweep that enumerated victims and leaked completion counts, the full 40-chapter/62k-char book + byte-identical cover in 3 requests, silent destruction of the victim's resume index (15 already-paid chunks dropped, re-synthesized on resume), cross-session force-delete + re-render of the victim's audiobook (7/7 chapter files overwritten with attacker-voiced audio), and wholesale replacement of a victim's progress file with attacker-authored chapters — 67 anonymous HTTP requests total

## Exploitability rationale

Reachability R:N — the vulnerable handlers are Gradio UI events reachable over HTTP by any anonymous visitor of the README-advertised public notebook URL (Pinggy tunnel in Colab/Kaggle; no auth in either launch mode); no user interaction is required from the victim, whose data is already at rest on disk. Exposure E:D — the title-derived progress cache is the core resume/export workflow of the only UI and exists in every deployment; the vulnerable configuration (one instance shared by several visitors) is the product's advertised usage ("public shareable Gradio links"). Certainty C:D — pure application logic with no framework dependency: once the attacker reproduces the victim's sanitized title (a public, low-entropy string; fixed default "audiobook" when the title box is empty), the cached-chapters load, merge and download are deterministic; a free oracle (check_existing_progress, fired on every title change) confirms candidate titles and leaks completion counts, and the export response itself distinguishes hit from miss. Impact I:S — significant information disclosure: the complete text of another user's book (all chapters, sentences, titles, statuses) plus the preserved cover image (cover_image_b64) is downloaded by the attacker; not code execution. The collateral write-back (victim's settings/book_path/voice_file replaced, per-chapter completed_chunks dropped) adds limited integrity damage to the victim's progress file. EXP-confirmed at business scale (exp/exp.md): the zero-knowledge variant (empty title -> shared default directory audiobook_output/audiobook, where every no-title TXT user lands because TXT scanning auto-fills no title) removes even the title precondition; the oracle sweep enumerated live victims and leaked their completion counts; one no-book request delivered a full 40-chapter book (62,387 chars of text + 761 sentences + byte-identical cover, 140,355-byte file); the destructive variants (force-reprocess delete, planted-progress wholesale replacement, audiobook re-render overwrite of 7/7 victim chapter files) and the quantified resume-state loss (15 already-paid chunks dropped and re-synthesized on the victim's next resume) raised the availability metric to A:L (CVSS 8.6). Total attacker cost for the whole campaign: 67 anonymous HTTP requests, no auth, no cookies, no book upload, no victim interaction.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 60 | `module scope (_OUTPUT_DIR)` |
| `app.py` | 975 | `_load_cached_chapters_if_available` |
| `app.py` | 1117 | `on_generate` |
| `app.py` | 1212 | `on_generate / _runner` |
| `app.py` | 1339 | `on_generate (final download list)` |
| `app.py` | 1510 | `on_progress_upload` |
| `app.py` | 1596 | `on_export_config` |
| `app.py` | 1604 | `on_export_config` |
| `app.py` | 1747 | `on_export_config` |
| `app.py` | 1797 | `on_export_config` |
| `app.py` | 1826 | `on_export_config` |
| `app.py` | 82 | `check_existing_progress` |
| `app.py` | 576 | `build_app (export_config_file component)` |
| `audiobook_factory/pipeline.py` | 496 | `run_pipeline` |
| `AudiobookMaker_Colab.ipynb` | 281 | `launch cell (public tunnel deployment)` |

## Background

AudiobookMaker is an end-to-end AI audiobook generator (Gradio web UI on port 7860, an unauthenticated FastAPI orchestrator on 127.0.0.1:8000, and a headless CLI) intended to run on a local workstation or — as advertised in the README — in Google Colab / Kaggle notebooks whose Gradio UI is published through a public Pinggy SSH tunnel URL that is meant to be shared; there is no authentication and no multi-user model anywhere. In the Book tab a user uploads a book, sets a free-form "Book title", and either generates an audiobook or clicks "Export Config JSON". Both actions write a progress file at audiobook_output/<sanitized title>/generation_progress.json containing the book's settings plus the fully extracted chapter text, sentences and statuses — the "Cached Book Extraction" feature that lets a returning user (or the CLI) resume without re-parsing the book. The security-relevant design fact is that the output directory name is derived solely from the user-supplied title (only the characters \ / * ? : " < > | are stripped), making the progress file a resource in a global, guessable namespace with no ownership binding between a file and the session that created it.

## Description

The two handlers that consume cached chapter text trust any existing file at the title-derived path. In on_export_config the output path is computed from the current title-box value (app.py:1596-1601) and passed to _load_cached_chapters_if_available (app.py:1604 -> 975-1013), which reads the file and — if its chapters list is non-empty and every chapter has non-empty text (app.py:983, always true after the file's owner ran "Export Config JSON" or started a generation, since both write full chapter text: app.py:1806-1824 and pipeline.py:496-537) — returns the file's chapters as ExtractedChapter objects. Crucially, the "Please upload a book file first" guard sits inside `if not chapters:` (app.py:1606-1612), i.e. downstream of the cache hit, so a session that merely reproduces another session's title proceeds with the foreign chapters and no book at all (this handler has no voice/model guard either). Because the file exists, the merge branch then runs (app.py:1747-1805): the merged chapters keep the victim's full text, sentences, titles, numbers and statuses ("text": cd.get("text") or ec.get("text", "") at app.py:1791-1796, where cd is the victim's cached chapter), while the victim's settings, book_path and voice_file are overwritten with the attacker's own values (app.py:1797-1800) and per-chapter completed_chunks keys are dropped; the victim's cover_image_b64 (binary cover art) and any other top-level keys such as generation_summary are preserved (app.py:1801-1803). write_progress_file then overwrites the victim's file (app.py:1804) and the handler returns the very same server path as a File-component value (app.py:1826-1832), which Gradio postprocesses into a browser download — the app's own designed "Download generation_progress.json" feature, so delivery works in every deployment mode (the cloud launch has no allowed_paths, but handler-returned files are copied into the download cache) and in every Gradio version in which the export button works for legitimate users. The attacker therefore receives the victim's complete book content in one request. on_generate is an equivalent, slower channel: same title-derived path (app.py:1117-1121), same cached load (app.py:1210-1213) feeding TTS audio / export_text chapter files, and the victim's progress JSON is appended to the final download list (app.py:1339-1342); its book/voice guards (app.py:1069-1074) are satisfied by any dummy file and a non-"Base" model choice. Target discovery is free: book_title_box.change fires check_existing_progress on every title change (app.py:1374-1378), which confirms whether a title's progress file exists and echoes its book title and completed/total chapter counts (app.py:82-104); book titles are public low-entropy strings and an empty title box maps to the fixed default directory "audiobook" (app.py:1598) — EXP-confirmed as a zero-knowledge attack surface: TXT uploads auto-fill no title (scan returns none), so every no-title user's complete book is cached in that one shared directory, grabbable by an anonymous visitor who submits an empty title and nothing else. The same confusion has three further attacker-reachable sinks through the Generate channel, all EXP-confirmed end-to-end: (a) force_reprocess forwards to run_pipeline, which deletes the title-derived progress file (pipeline.py:489-496) before regenerating it under attacker settings; (b) the Generate handler copies an attacker-uploaded progress JSON over the title-derived file (app.py:1128-1132), replacing the victim's cached book with attacker-authored chapters wholesale; (c) the re-render overwrites the victim's finished chapter audio files in the shared output directory (deterministic make_safe_filename names, sha256-verified 7/7 files replaced with attacker-voiced audio). CVSS v3.1 metric reasoning: AV:N — remotely triggerable through the public tunnel URL; AC:L — no conditions beyond reproducing a guessable/oracle-verifiable (or empty) title, deterministically; PR:N / UI:N — no authentication anywhere and no victim interaction (data is at rest); S:U — the impact stays within the same application/service; C:H — complete loss of confidentiality for the victim's book content (all chapters' full text); I:L — the attacker modifies the victim's progress file (settings/book_path/voice_file replaced, completed_chunks dropped, wholesale content replacement) and their finished audio files, all recoverable by re-running from the victim's own book; A:L — demonstrated destruction of data availability: the victim's resume state (completed_chunks — the paid-for GPU work the chunk-level resume feature exists to save) is silently discarded (EXP: 15 already-synthesized chunks of an 8-chapter book dropped and re-synthesized on the next resume; on real models each chunk is seconds of the scarce Colab/Kaggle GPU quota the product targets), and the force-reprocess variant deletes the whole progress file.

## Attack

Attacker: any anonymous visitor of a shared AudiobookMaker instance — the README-advertised Colab/Kaggle public-link deployment (no authentication; the notebook publishes the UI through a public Pinggy tunnel). Preconditions: (1) another user has processed a book under some title T via "Export Config JSON" or "Generate" (the normal workflow; the progress file then persists on disk indefinitely), and (2) the attacker can reproduce T — satisfied by guessing public book titles, probing with the title-existence oracle that fires on every title change, or using the fixed default directory "audiobook" (empty title; requires zero knowledge because TXT uploads auto-fill no title, so every no-title user shares that directory). Attack: in a fresh browser session (or via Gradio's unauthenticated REST queue API, no browser needed) the attacker sets the Book title to T and clicks "Export Config JSON" with nothing else filled in. Observable result: the UI responds "Config exported! N chapters cached" and offers a download of generation_progress.json whose chapters[] contain the victim's complete book text, sentences, chapter titles/numbers/statuses, plus the victim's preserved cover image (cover_image_b64) and generation summary. Side effect: the victim's progress file on the server is overwritten with the attacker's settings/book_path/voice_file and loses per-chapter completed_chunks (their resume state is damaged). Total cost: 1-2 anonymous HTTP requests per book. EXP-measured at business scale (exp/exp.md): a 25-title oracle sweep (50 requests) enumerated the instance's live victims with completion counts; the empty-title variant stole the default-directory user's complete book in 3 requests; a known/public title delivered a full 40-chapter book (140,355-byte file, 62,387 chars of text, 761 sentences, byte-identical cover) in 3 requests; the same one-request merge silently discarded the victim's 15 already-paid chunks; and the Generate channel (any dummy book file + non-"Base" model) additionally force-deleted the victim's progress file, re-rendered the victim's book under attacker settings, overwrote all 7 of their finished chapter audio files, and — with an uploaded crafted progress JSON — replaced a third victim's 40-chapter book with 2 attacker-authored chapters. Whole campaign: 67 anonymous HTTP requests. Delivery caveat observed on the audited snapshot + gradio 6.27.0: the Generate event's own download list never populates on the live WebSocket path (the backend's status broadcast closes the socket before the files broadcast is forwarded, and listen_ws swallows the normal close — an unrelated shipped regression), so the practically delivered disclosure channel is the Export-Config download; the Generate channel's server-side effects (delete/replace/overwrite) are unaffected by that regression.

### Payload

The entire payload is a string: the victim's book title typed into the "Book title" textbox (or supplied as the book_title field of an uploaded progress JSON, which restores it into the title box). No book upload, no voice, no server path and no malformed data are required. Reconnaissance payloads are simply candidate titles (common book titles, or "audiobook" for victims who left the title empty), each one fired as a title-box change that returns "Existing Progress Found" plus completion counts when the corresponding progress file exists.

## Data flow

### Step 1 — `app.py:1824 / audiobook_factory/pipeline.py:496-537`

Victim session: "Export Config JSON" or "Generate" writes audiobook_output/<sanitized title>/generation_progress.json with every chapter's full text and sentences (the cached-extraction feature). The file persists with no cleanup.

### Step 2 — `app.py:181 / app.py:1399,1510`

Attacker session: the free-form "Book title" gr.Textbox is set to the victim's title — typed directly, or restored from an attacker-authored progress JSON whose book_title field is pushed into the title box by on_progress_upload (gr.update(value=title)).

### Step 3 — `app.py:1596-1601`

on_export_config derives book_out = audiobook_output/<sanitized title> and prog_path = book_out/generation_progress.json from the attacker-supplied title (sanitization strips only the separator characters \ / * ? : " < > | ; empty title maps to the fixed directory "audiobook"). os.makedirs(..., exist_ok=True) adopts the victim's existing directory.

### Step 4 — `app.py:1604 -> app.py:975-1013`

_load_cached_chapters_if_available(prog_path, selected_chapters) reads the victim's file; the chapters list is non-empty and every chapter has non-empty text (gate at app.py:983), and the fresh session's empty chapter selection disables the title filter (selected_titles = None), so the victim's chapters are returned as ExtractedChapter objects.

### Step 5 — `app.py:1606-1612`

The "Please upload a book file first" guard is inside `if not chapters:` and is skipped on the cache hit — the attack proceeds with no book upload and no voice (this handler has no model/voice guard).

### Step 6 — `app.py:1747-1805`

Merge branch: merged chapters preserve the victim's text/sentences/statuses ("text": cd.get("text") or ec.get("text", "") at :1791-1796, cd being the victim's cached chapter); the victim's settings/book_path/voice_file are overwritten with attacker values (:1797-1800); the victim's cover_image_b64 and generation_summary are preserved (:1801-1803); write_progress_file overwrites the victim's file (:1804) — collateral integrity damage.

### Step 7 — `app.py:1826-1832 / app.py:576-580`

The handler returns gr.update(value=prog_path, visible=True) into the non-interactive gr.File "Download generation_progress.json"; Gradio's File-output postprocess copies the server file into the download cache and serves it to the anonymous attacker — the victim's complete book content is downloaded.

### Step 8 — `app.py:1117-1121, 1210-1213, 1339-1342`

Equivalent channel: on_generate derives the same path, performs the same cached load inside _runner (victim text is TTS-generated into downloadable audio / export_text files), and appends the victim's progress JSON to the final download list; the book/voice guards (app.py:1069-1074) are satisfied with any dummy file and a non-"Base" model choice. EXP note (audited snapshot, gradio 6.27.0): the UI-side delivery of that download list is defeated by an unrelated shipped regression (the backend's status broadcast closes the task WebSocket before the completed/files broadcast is forwarded, so out_files stays empty and the browser shows "No output files generated"); the channel's server-side effects are unaffected — the re-rendered victim audio lands in the shared output directory (sha256-verified overwrite of 7/7 chapter files) and the victim's rebuilt progress file carries the attacker's settings. The Generate handler also copies an attacker-uploaded progress JSON over the title-derived file before the runner starts (app.py:1128-1132) — EXP-confirmed wholesale replacement of a victim's cached book — and forwards force_reprocess, whose backend handling deletes the victim's progress file (pipeline.py:489-496, EXP-confirmed via the "Force reprocess enabled. Clearing old progress." log).

### Step 9 — `app.py:82-104 / app.py:1374-1378`

Reconnaissance aid: book_title_box.change fires check_existing_progress on every title change; it resolves the same title-derived path and, when the file exists, echoes the victim's book title and completed/total chapter counts — a free title-enumeration oracle that also works via the unauthenticated REST queue API.

## Fix / patch notes

diff --git a/app.py b/app.py
@@ -60,6 +60,36 @@
 _OUTPUT_DIR = os.path.join(_ROOT, "audiobook_output")
 os.makedirs(_OUTPUT_DIR, exist_ok=True)
 
+# -- Progress-file session ownership (cross-session disclosure fix) ------------
+# audiobook_output/<title>/generation_progress.json is a shared namespace on
+# multi-user instances. It may be consumed as a chapter-text cache (or merged
+# and offered for download) only by the Gradio session that wrote it or that
+# explicitly uploaded a replacement for it.
+_PROGRESS_OWNERS: dict[str, set[str]] = {}
+_PROGRESS_OWNERS_LOCK = threading.Lock()
+
+
+def _register_progress_owner(request, *paths: str) -> None:
+    key = getattr(request, "session_hash", None)
+    if not key:
+        return
+    with _PROGRESS_OWNERS_LOCK:
+        _PROGRESS_OWNERS.setdefault(key, set()).update(
+            os.path.realpath(p) for p in paths if p
+        )
+
+
+def _owns_progress(request, path: str) -> bool:
+    """True if `path` may be consumed by `request`'s session. Non-Gradio
+    callers (request=None, e.g. CLI embeds) keep legacy single-user behavior."""
+    if request is None:
+        return True
+    key = getattr(request, "session_hash", None)
+    if not key:
+        return False
+    with _PROGRESS_OWNERS_LOCK:
+        return os.path.realpath(path) in _PROGRESS_OWNERS.get(key, set())
+
 # ══════════════════════════════════════════════════════════════════════════════
 # Helper utilities
 # ══════════════════════════════════════════════════════════════════════════════
@@ -972,11 +1002,15 @@
                     out.append(title)
             return out or None
 
-        def _load_cached_chapters_if_available(prog_json_path: str, selected_chapters_labels: list[str] | None = None, log_fn=None):
+        def _load_cached_chapters_if_available(prog_json_path: str, selected_chapters_labels: list[str] | None = None, log_fn=None, request=None):
             """If prog_json_path exists and has cached text for chapters, return list of ExtractedChapter objects.
             Otherwise return None."""
             if not os.path.exists(prog_json_path):
                 return None
+            if not _owns_progress(request, prog_json_path):
+                if log_fn:
+                    log_fn("Ignoring cached chapters from a progress file not created by this session.")
+                return None
             try:
                 data = read_progress_file(prog_json_path)
                 ch_list = data.get("chapters", [])
@@ -1064,7 +1098,8 @@
             torch_compile, regen_missing, quantization, resume_incomplete_chunks,
             sample_rate, bitrate_kbps, channels,
             rep_penalty, top_k, speed, nfe_step, seed,
-            progress=gr.Progress(track_tqdm=False)
+            progress=gr.Progress(track_tqdm=False),
+            request: gr.Request = None,
         ):
             if file_obj is None:
                 yield "⚠️ Please upload a book file first.", gr.update(visible=False), gr.update(visible=False), [], None
@@ -1128,9 +1163,13 @@
                     import shutil
                     shutil.copy2(uploaded_progress_path, dest_progress_path)
                     print(f"[UI] Progress file uploaded. Copied to {dest_progress_path}")
+                    _register_progress_owner(request, dest_progress_path)
                 except Exception as e:
                     print(f"[UI] Failed to copy uploaded progress file: {e}")
 
+            _prog_json_pre = os.path.join(book_out, "generation_progress.json")
+            _foreign_progress = os.path.exists(_prog_json_pre) and not _owns_progress(request, _prog_json_pre)
+
             cfg = AudiobookConfig(
                 book_title=book_title,
                 book_path=path,
@@ -1209,7 +1248,7 @@
                 try:
                     # Check if progress JSON already has cached chapter text
                     prog_json_path = os.path.join(book_out, "generation_progress.json")
-                    chapters = _load_cached_chapters_if_available(prog_json_path, selected_chapters, log_q.put)
+                    chapters = _load_cached_chapters_if_available(prog_json_path, selected_chapters, log_q.put, request)
                     
                     if not chapters:
                         # Extract from book file
@@ -1338,7 +1377,10 @@
             if out_files:
                 prog_json_path = os.path.join(book_out, "generation_progress.json")
                 files_to_show = list(out_files)
-                if os.path.exists(prog_json_path):
+                if os.path.exists(prog_json_path) and not _foreign_progress:
+                    # Never hand a progress file created by another session to
+                    # this one, even after a local run touched it.
+                    _register_progress_owner(request, prog_json_path)
                     files_to_show.append(prog_json_path)
                 yield (
                     log_text + "\n✅ Generation complete!",
@@ -1586,6 +1628,7 @@
             torch_compile, regen_missing, quantization, resume_incomplete_chunks,
             sample_rate, bitrate_kbps, channels,
             rep_penalty, top_k, speed, nfe_step, seed,
+            request: gr.Request = None,
         ):
             """Parse the book, cache chapter text, and write a self-contained
             generation_progress.json — without starting TTS generation."""
@@ -1601,7 +1644,7 @@
                 prog_path = os.path.join(book_out, "generation_progress.json")
 
                 # Check if progress JSON already has cached chapter text
-                chapters = _load_cached_chapters_if_available(prog_path, selected_chapters)
+                chapters = _load_cached_chapters_if_available(prog_path, selected_chapters, None, request)
 
                 if not chapters:
                     if not path or not os.path.exists(path):
@@ -1744,6 +1787,13 @@
                     for ch in chapters
                 ]
 
+                if os.path.exists(prog_path) and not _owns_progress(request, prog_path):
+                    return (
+                        "⚠️ A progress file for this title already exists but was not created by this session. "
+                        "Upload it via '🔄 Resume from Progress JSON' to continue with it, or choose a different title.",
+                        gr.update(visible=False),
+                        gr.update(open=True),
+                    )
                 if os.path.exists(prog_path):
                     try:
                         existing = read_progress_file(prog_path)
@@ -1802,6 +1852,7 @@
                         existing["cover_image_b64"] = settings_dict["cover_image_b64"]
                     existing["chapters"] = merged_chapters
                     write_progress_file(prog_path, existing)
+                    _register_progress_owner(request, prog_path)
                 else:
                     progress_data = {
                         "book_title": book_title,
@@ -1822,6 +1873,7 @@
                         ],
                     }
                     write_progress_file(prog_path, progress_data)
+                    _register_progress_owner(request, prog_path)
 
                 return (
                     f"✅ **Config exported!** {len(chapters)} chapters cached.\n\n"

## References

- https://cwe.mitre.org/data/definitions/639.html
- https://cwe.mitre.org/data/definitions/200.html
- https://owasp.org/www-project-top-ten/A01_2021-Broken_Access_Control

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] FastAPI backend unauthenticated TTS config (/api/v1/voice-test, /api/v1/generate) drives VibeVoice provider model loading with trust_remote_code=True into arbitrary Python code execution
- key: `BUG-R2-C2-A1-H1`
- disclosure: owner_only
- cwe: CWE-94
- file: `api/server.py`

# FastAPI backend unauthenticated TTS config (/api/v1/voice-test, /api/v1/generate) drives VibeVoice provider model loading with trust_remote_code=True into arbitrary Python code execution

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C2-A1-H1
- **CWE:** CWE-94
- **CVSS:** 7.8 (`CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H`)
- **EV priority:** P0
- **EV score:** 8
- **PoC status:** reproduced
- **EXP status:** confirmed
- **Affected versions:** unknown (any release shipping the FastAPI server api/server.py together with audiobook_factory/tts_providers/vibevoice_provider.py; current source snapshot dc9aed2 confirmed affected)

## Exploitability rationale

R:L (local/adjacent): the shipped launch mode binds the FastAPI backend to 127.0.0.1:8000 (start_api.py:16) and neither shipped launcher nor notebook ever widens the bind, so the reachability ceiling for the default deployment is the host itself. Dynamically confirmed attacker matrix (real-business impact assessment, 2026-09-15): (1) any local process of ANY OS user — an unprivileged co-tenant account (uid 1000) reached the sink with one unauthenticated POST and executed payload code as the service user (uid 1001), reading the service user's 0600 HF token file and HF_TOKEN env that the attacker account cannot read, running subprocesses and writing files as that user, while the HTTP exchange returned a normal 200 audio/wav; the same leg against a root-run service (the advertised Colab/Kaggle condition) executed as root with root-only file read. (2) Browser-delivered variants, verified server-side: a cross-origin application/json POST is processed (200 + execution, no Access-Control-Allow-Origin in the response so it is unreadable but still fires); the CORS preflight returns 405 with no ACAO, so plain cross-origin JSON is browser-blocked; uvicorn/FastAPI accept an arbitrary Host header (Host: rebind.attacker.example still processed and executed), which is the server-side precondition for DNS rebinding — the remaining browser route on the current fastapi line (0.141.1), subject to browser local-network protections (Chrome LNA / Safari prompts). On fastapi <= 0.130.x installs (requirements.txt is unpinned, so every venv resolved before the 0.132 release keeps this behavior), a POST with NO Content-Type — exactly what a fetch() with a typeless Blob body sends, a non-preflighted fire-and-forget request — is parsed as JSON and executes the payload (verified end-to-end on fastapi 0.115.0 + transformers 4.46.3), making the trigger reachable from any website the operator visits on those installs; text/plain, x-www-form-urlencoded and multipart are rejected on all tested lines. (3) Any deployment that binds the port to a non-loopback interface upgrades reachability to fully remote (CVSS 9.8 tier). In the advertised Colab/Kaggle mode the public Pinggy tunnel forwards port 7860 (Gradio) only, so anonymous tunnel visitors cannot reach this sink directly on the current gradio line — there the realistic attacker is code already running in the VM, which executes as root. E:D (default): the endpoint and the VibeVoice provider are part of the standard API server started by run.sh / start_api.py and the notebooks; the provider is registered in the default factory dispatch; no optional flag or uncommon configuration is needed. C:D (deterministic): the trigger is pure logic — one HTTP request whose config names an attacker-published public HF repo (or, for a local attacker, a planted world-readable directory containing the substring VibeVoice); no races, no memory layout, no timing; the only external dependency of the HF variant (outbound HTTPS to huggingface.co) is required by the app's own default TTS flow, so every functioning deployment satisfies it. Failure handling (tokenizer fallback, fallback_engine swallow) makes execution MORE reliable, not less, and the exchange stays a normal 200 WAV. I:X (execute): arbitrary Python code execution inside the API service process at model-load time — full confidentiality/integrity/availability of everything the service user can touch, plus (dynamically demonstrated) persistent interception of all later synthesis via the first-writer-wins provider pool and in-process patching.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `api/server.py` | 162 | `api_voice_test` |
| `api/server.py` | 113 | `enqueue_generation` |
| `audiobook_factory/pipeline.py` | 273 | `AudiobookConfig.from_dict` |
| `audiobook_factory/pipeline.py` | 1378 | `preview_tts` |
| `audiobook_factory/tts_providers/base_tts_provider.py` | 166 | `get_tts_provider` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 62 | `VibeVoiceTTSProvider._ensure_initialised` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 74 | `VibeVoiceTTSProvider._ensure_initialised` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 78 | `VibeVoiceTTSProvider._ensure_initialised` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 84 | `VibeVoiceTTSProvider._ensure_initialised` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 109 | `VibeVoiceTTSProvider.synthesize` |
| `audiobook_factory/pipeline.py` | 569 | `run_pipeline` |
| `audiobook_factory/gpu_pool.py` | 419 | `GPUPoolManager.get_pool` |
| `audiobook_factory/gpu_pool.py` | 242 | `ProviderPool.__init__` |
| `api/server.py` | 48 | `startup_event._warmup_gpu_pool` |

## Background

AudiobookMaker is a single-user audiobook generation application (book extraction + TTS synthesis) offered in two launch modes: a local workstation mode (run.sh starts a FastAPI backend on 127.0.0.1:8000 via start_api.py plus a Gradio UI on :7860) and a cloud-notebook mode (Colab/Kaggle, exposed through a public SSH tunnel to the Gradio port). The FastAPI backend is an unauthenticated task/preview server: it accepts a free-form `config` JSON dict on /api/v1/generate (full book pipeline) and /api/v1/voice-test (short TTS preview). The dict is materialized via AudiobookConfig.from_dict, which keeps every known field verbatim, including the TTS provider selector (`tts_provider_name`), the HuggingFace model id (`tts_model_name`), the torch `device`, and `voice_file`. Model weights are downloaded on demand from the HuggingFace Hub by design. The "vibevoice" provider wraps the model bezzam/VibeVoice-1.5B-hf and loads it through transformers' AutoProcessor/AutoTokenizer/AutoModelForCausalLM with trust_remote_code=True — the documented opt-in switch that downloads Python files from the target model repo and executes them inside the loading process (the standard HuggingFace custom-code mechanism). Because the service has no authentication and the config reaches the loader without any model-id validation, the "trusted model repo" premise of trust_remote_code is broken: the repo id itself is request-controlled.

## Description

An unauthenticated caller posts a config dict with tts_provider_name="vibevoice" and an attacker-chosen tts_model_name. AudiobookConfig.from_dict performs no value validation (unknown keys are dropped; known keys pass raw), and preview_tts / the generation worker dispatch to VibeVoiceTTSProvider. In _ensure_initialised the only check applied to the model id is a substring containment: `if "VibeVoice" not in model_name: model_name = "bezzam/VibeVoice-1.5B-hf"` — a typo-correction back to the default, not an allowlist; any repo id containing "VibeVoice" (e.g. a public repo named attacker/VibeVoice-evil) passes unchanged. The model is then loaded with trust_remote_code=True at three call sites (AutoProcessor, with AutoTokenizer as failure fallback, and AutoModelForCausalLM), and device_map=None is used on the CPU path (config.device="cpu"), so no GPU is required. On the current dependency line (transformers is unpinned in requirements.txt; 5.17.0 at the time of analysis — and identically on the 4.x line), trust_remote_code=True plus a repo auto_map entry in preprocessor_config.json / tokenizer_config.json / config.json routes through resolve_trust_remote_code (which returns True immediately when the flag is passed explicitly, bypassing the interactive confirmation prompt) into get_class_from_dynamic_module, which downloads the referenced repo Python file into the HF modules cache and executes it via importlib exec_module. Module-level payload code therefore runs at import time, before any class instantiation — which makes the provider's failure handling an amplifier rather than a mitigator: an AutoProcessor failure falls back to AutoTokenizer (also trust_remote_code=True), and a model-load failure is swallowed and replaced by a "fallback_engine" sentinel, so malicious code executes even when the "model" is garbage and the request can still complete normally. Execution also precedes all business validation: VibeVoiceTTSProvider.synthesize calls ensure_ready() before _validate_voice_ref, so no voice file, chapters, or valid audio are needed. A second entry point, /api/v1/generate, reaches the same sink during GPU pool creation: run_pipeline asks GPUPoolManager.get_pool for the attacker-named provider; startup warmup preloads only the default qwen pool, so a vibevoice pool is constructed on first request (ProviderPool instantiates one provider per detected device — ["cpu"] on CPU-only hosts — and warms each via ensure_ready). Pools are registered first-writer-wins keyed by provider name only, so the poisoned provider persists for the process lifetime, and warmup exceptions are caught, which cannot undo an import that already happened. No defense exists on the path: no authentication or middleware on the FastAPI app, no CORS policy, no model allowlist, no HF offline/endpoint pinning anywhere in the project, and preflight checks pass on CPU-only hosts with an empty voice_file. Real-business impact assessment (dynamic, see exp/exp.md) additionally confirmed: the attack is a cross-OS-user privilege pivot (an unprivileged co-tenant account obtained code execution as the service user and read that user's 0600 HF token file + HF_TOKEN env, which the attacker account cannot read, plus subprocess and file-write primitives); against a root-run service (the advertised Colab/Kaggle condition) the payload executed as root and read root-only files; and a single /api/v1/generate request poisons the provider pool persistently — pools are keyed by provider name only and first-writer-wins (gpu_pool.py), so a later operator request naming a different VibeVoice model completes WITHOUT loading it (the operator's model choice is silently ignored), while payload code executed at load time can monkey-patch the provider class so that every later synthesis — including the operator's voice-tests with the legit default model — is intercepted (text recorded) while still returning normal audio and clean task logs.

## Attack

Primary attacker: any process on the host that can reach 127.0.0.1:8000 — an unprivileged co-tenant on a shared workstation or notebook VM, or malware already present with only user-level privileges that pivots to full code execution inside the (GPU/model-laden) service process; dynamically demonstrated as a cross-OS-user pivot (execution as the service user, theft of that user's HF token file and env token, subprocess and file-write primitives) and, against a root-run service, as root. The attack is a single unauthenticated POST; no user interaction, no valid book, no voice sample. Browser-delivered variants (server-side behavior verified dynamically): a cross-origin application/json POST is processed, but its CORS preflight (OPTIONS) returns 405 with no Access-Control-Allow-Origin, so plain cross-origin JSON is browser-blocked; DNS rebinding remains viable on the current fastapi line because uvicorn/FastAPI accept an arbitrary Host header (verified: a POST with Host: rebind.attacker.example was processed and executed), subject to browser local-network protections (Chrome Local Network Access, Safari prompts). On installs whose unpinned requirements resolved fastapi <= 0.130.x (every venv created before the 0.132 release keeps that resolution), a POST with NO Content-Type header — exactly what a fetch() with a typeless Blob body sends, a non-preflighted fire-and-forget request — is parsed as JSON and executes the payload (verified end-to-end on fastapi 0.115.0 + transformers 4.46.3), so any website the operator visits can trigger the execution on those installs; text/plain, x-www-form-urlencoded and multipart/form-data bodies are rejected on all tested fastapi lines. Any deployment that binds the port to a non-loopback interface (e.g. containerized 0.0.0.0 exposure) makes the chain fully remote (CVSS 9.8 tier). In the advertised Colab/Kaggle mode the public Pinggy tunnel forwards the Gradio port (7860) only, so anonymous tunnel visitors cannot reach this sink directly on the current gradio line — there the realistic attacker is any code already running in the VM, which on those notebooks is root. Preconditions for the HF-repo delivery: outbound HTTPS to huggingface.co — which the application itself requires for its default on-demand model downloads, so every functioning deployment satisfies it — plus one freely-published attacker HF repo; a local attacker needs no network at all (planted world-readable directory whose path contains the substring VibeVoice, e.g. under /tmp). Observable impact: arbitrary Python code execution in the API service process (file read/write as the service user, root-level file access in cloud mode, environment/credential theft, subprocess execution, GPU-pool hijacking), persistent interception of all later TTS synthesis after one /api/v1/generate request (first-writer-wins pool + in-process patching), with the HTTP request still able to return 200 and a normal WAV body, leaving no error trace in the exchange.

### Payload

One HTTP request, e.g. POST http://127.0.0.1:8000/api/v1/voice-test with body {"config": {"tts_provider_name": "vibevoice", "tts_model_name": "<attacker>/VibeVoice-evil", "device": "cpu", "voice_file": "/etc/hostname"}, "text": "hi"}. The referenced public HuggingFace repo (freely publishable, name contains the required substring "VibeVoice") contains a small JSON config with an auto_map entry plus the referenced Python file whose module-level code is the payload (write a marker, open a reverse shell, read files, etc.). No weights are needed: the payload executes at import time of the repo Python file, before any model class is instantiated or any load can fail. The same effect is reachable via POST /api/v1/generate with config {"tts_provider_name": "vibevoice", "tts_model_name": "<attacker>/VibeVoice-evil", "device": "cpu"} and "chapters": [] (pool creation warms the provider), which additionally plants the malicious provider in a process-lifetime pool. from_pretrained also accepts local directory paths, so a local path containing "VibeVoice" works identically when combined with any file-write primitive (no network egress needed) — this local-path variant is the one dynamically reproduced in the POC (see poc/poc.md). auto_map format note (confirmed dynamically): transformers v5 (5.17.0, current resolution of the unpinned requirements) expects dotted "module.Class" strings in auto_map values (and a [slow, fast] pair of dotted strings for AutoTokenizer); the v4 line (4.46.3 verified end-to-end) takes the same dotted [slow, fast] pair for AutoTokenizer and additionally needs a "tokenizer_class" key in tokenizer_config.json so AutoTokenizer does not fall back to AutoConfig (every save_pretrained-written 4.x-era custom repo carries it). An attacker simply writes whichever format the installed transformers expects.

## Data flow

### Step 1 — `api/server.py:162-168`

api_voice_test: unauthenticated POST /api/v1/voice-test; VoiceTestRequest.config is a free-form Dict[str, Any] (same for GenerateRequest on /api/v1/generate, api/server.py:113-127); the FastAPI app registers no auth or middleware.

### Step 2 — `audiobook_factory/pipeline.py:273-300`

AudiobookConfig.from_dict drops unknown keys only; tts_provider_name / tts_model_name / device / voice_file are known dataclass fields (pipeline.py:210,213,235,236) and pass attacker values verbatim; _validate_config (pipeline.py:324-337) checks only quantization.

### Step 3 — `audiobook_factory/pipeline.py:1365-1386`

preview_tts requires only non-empty text; _TEMP_DIR is created at import (pipeline.py:63-64); calls get_tts_provider(config.tts_provider_name, config).

### Step 4 — `audiobook_factory/tts_providers/base_tts_provider.py:166-196`

get_tts_provider dispatches the name 'vibevoice' to VibeVoiceTTSProvider(config, device=None) → self._device = attacker-controlled config.device ('cpu').

### Step 5 — `audiobook_factory/tts_providers/vibevoice_provider.py:109-110`

synthesize calls ensure_ready() (model load) BEFORE _validate_voice_ref(voice_ref) — no voice file, chapters, or valid audio are needed for the load to happen.

### Step 6 — `audiobook_factory/tts_providers/vibevoice_provider.py:62-64`

The only model-id check is a substring containment (`if "VibeVoice" not in model_name`) that resets to the default — any attacker repo id containing 'VibeVoice' passes unchanged.

### Step 7 — `audiobook_factory/tts_providers/vibevoice_provider.py:74,78,84-88`

AutoProcessor.from_pretrained(model_name, trust_remote_code=True); on failure falls back to AutoTokenizer.from_pretrained(model_name, trust_remote_code=True); then AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, device_map=None on the CPU path) — no GPU required; a model-load failure is swallowed as 'fallback_engine' (lines 89-92).

### Step 8 — `transformers 5.17.0: models/auto/processing_auto.py:316-327 (and tokenization_auto.py:921-925, auto_factory.py:223-230)`

resolve_trust_remote_code returns True immediately when the flag is explicitly True (the interactive prompt fires only when it is None, dynamic_module_utils.py:712-788); the repo's auto_map entry then routes into get_class_from_dynamic_module.

### Step 9 — `transformers 5.17.0: dynamic_module_utils.py:288-305`

The referenced repo Python file (e.g. tokenization_evil.py via tokenizer_config.json auto_map) is downloaded into the HF modules cache and executed via module_spec.loader.exec_module(module) — arbitrary module-level payload code runs before any class instantiation or load failure.

### Step 10 — `api/server.py:113-127 → api/worker.py:124,140 → audiobook_factory/pipeline.py:569 → audiobook_factory/gpu_pool.py:419,242-245,369,502`

Second entry point: /api/v1/generate → run_pipeline → GPUPoolManager.get_pool('vibevoice', ...) — startup warmup preloaded only the default qwen pool (api/server.py:48-62), so the pool is constructed on first request; ProviderPool instantiates one provider per detected device (['cpu'] on CPU-only hosts, gpu_pool.py:45-60,110-114) and _warmup_provider calls ensure_ready() (the same sink); pools are registered first-writer-wins keyed by provider name only, so the poisoned provider persists for the process lifetime; warmup exceptions are caught (gpu_pool.py:467-476).

## Fix / patch notes

diff --git a/audiobook_factory/tts_providers/vibevoice_provider.py b/audiobook_factory/tts_providers/vibevoice_provider.py
--- a/audiobook_factory/tts_providers/vibevoice_provider.py
+++ b/audiobook_factory/tts_providers/vibevoice_provider.py
@@ -59,6 +59,10 @@ class VibeVoiceTTSProvider(BaseTTSProvider):
             try:
                 import torch
                 from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
-                model_name = getattr(self.config, "tts_model_name", "bezzam/VibeVoice-1.5B-hf")
-                if "VibeVoice" not in model_name:
-                    model_name = "bezzam/VibeVoice-1.5B-hf"
+                _BLESSED_VIBEVOICE_MODEL = "bezzam/VibeVoice-1.5B-hf"
+                model_name = getattr(self.config, "tts_model_name", _BLESSED_VIBEVOICE_MODEL)
+                if model_name != _BLESSED_VIBEVOICE_MODEL:
+                    raise ValueError(
+                        f"Refusing to load unapproved VibeVoice model '{model_name}': "
+                        "this provider enables trust_remote_code, and only the "
+                        "reviewed upstream model id may be loaded."
+                    )

## References

- https://cwe.mitre.org/data/definitions/94.html
- https://cwe.mitre.org/data/definitions/829.html
- https://huggingface.co/docs/transformers/main/en/custom_models

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] FastAPI /api/v1/generate unauthenticated output_dir path control: arbitrary directory creation plus attacker-authored generation_progress.json write/delete at any host path (checkpoint destruction/poisoning of co-located deployments and users)
- key: `BUG-R2-C2-A2-H1`
- disclosure: owner_only
- cwe: CWE-22
- file: `api/server.py`

# FastAPI /api/v1/generate unauthenticated output_dir path control: arbitrary directory creation plus attacker-authored generation_progress.json write/delete at any host path (checkpoint destruction/poisoning of co-located deployments and users)

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C2-A2-H1
- **CWE:** CWE-22
- **CVSS:** 8.8 (`CVSS:3.1/AV:A/AC:L/PR:N/UI:N/S:U/C:L/I:H/A:H`)
- **EV priority:** P1
- **EV score:** 7
- **PoC status:** reproduced
- **EXP status:** confirmed
- **Affected versions:** v0.5.0–v1.3.0 (all releases — the FastAPI /api/v1/generate endpoint, the AudiobookConfig.output_dir field and the output_dir-driven makedirs / generation_progress.json write-delete I/O have shipped since the initial v0.5.0 release; current source snapshot dc9aed2 confirmed affected)

## Exploitability rationale

R:L (local/adjacent) — every shipped launch mode binds the FastAPI backend to 127.0.0.1:8000 (start_api.py:16; run.sh and both Colab/Kaggle notebooks spawn it), so the primitive holder is the in-model local attacker: any process or co-tenant on the host (no authentication exists, and loopback is shared on multi-user hosts, so user B reaches user A's service and writes with A's credentials into paths B cannot touch directly — proven dynamically by the unprivileged pocatck user producing root-owned writes), or any web page rendered in the operator's browser. EXP-settled browser delivery (2026-09-15, both rows executed against the unmodified shipped backend): (a) on FastAPI <= 0.131.x — half the version space of the product's UNPINNED fastapi requirement (requirements.txt:63; strict-content-type default flipped in 0.132.0) — a plain no-preflight cross-origin POST reaches the route: an untyped-Blob no-cors fetch/sendBeacon sends no Content-Type header and the stack parses the body as JSON, planting the checkpoint even with a RELATIVE output_dir ("audiobook_output/<title>") that needs zero knowledge of the victim's install path; (b) on every version including the current 0.141.1, a DNS-rebinding page's same-origin application/json POST reaches the route (uvicorn performs no Host validation — POSTs with Host: evil.attacker.example execute the sinks). The anonymous public notebook-tunnel visitor cannot reach port 8000 directly. Any deployment that widens the bind (container -p / --host 0.0.0.0 — user misconfiguration, nothing shipped binds wide) upgrades the chain to fully remote, CVSS AV:N ~9.8. E:D (default) — the endpoint is part of the standard stack started by run.sh/start_api.py and both notebooks; the three primitives execute before any model/GPU work (only the environment preflight must pass, i.e. any functioning install), so no optional flag or uncommon configuration is needed. C:D (deterministic) — pure filesystem logic (makedirs / path join / os.remove / tmp-file + os.replace atomic write); no races, no memory layout, no timing; the target path and the full JSON content are request-controlled, and later pipeline failures cannot undo writes that already happened (every attack task failed at pool setup AFTER the writes, by design of the payload). I:D (destructive) — arbitrary directory creation anywhere the service user can write, plus deletion and attacker-authored overwrite of any <dir>/generation_progress.json — the product's only resume checkpoint. EXP end-to-end confirmation (2026-09-15): through the real shipped on_generate UI handler, one unauthenticated co-tenant request poisoned the victim's canonical checkpoint; the victim's own UI then announced a normal-looking resume and the victim's next generation silently narrated attacker-chosen chapter text into real mastered MP3s in the victim's output directory (the victim's book file was never re-parsed), and a second request destroyed the resume state forcing full re-synthesis of every chapter. Not code execution by itself: escalation to RCE via planted settings requires one victim interaction and is dominated by the direct unauthenticated config path (BUG-R2-C2-A1-H1) for the same attacker, hence one EV notch below that sibling.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `api/server.py` | 113 | `enqueue_generation` |
| `api/server.py` | 26 | `GenerateRequest` |
| `api/worker.py` | 124 | `_process_single_task` |
| `audiobook_factory/pipeline.py` | 206 | `AudiobookConfig.output_dir` |
| `audiobook_factory/pipeline.py` | 273 | `AudiobookConfig.from_dict` |
| `audiobook_factory/pipeline.py` | 324 | `_validate_config` |
| `audiobook_factory/pipeline.py` | 441 | `run_pipeline.preflight` |
| `audiobook_factory/pipeline.py` | 463 | `run_pipeline` |
| `audiobook_factory/pipeline.py` | 484 | `run_pipeline` |
| `audiobook_factory/pipeline.py` | 487 | `run_pipeline` |
| `audiobook_factory/pipeline.py` | 515 | `run_pipeline` |
| `audiobook_factory/pipeline.py` | 537 | `run_pipeline` |
| `audiobook_factory/pipeline.py` | 559 | `run_pipeline` |
| `audiobook_factory/progress_io.py` | 282 | `_write_unlocked` |
| `audiobook_factory/progress_io.py` | 298 | `_write_unlocked` |
| `audiobook_factory/pipeline.py` | 999 | `_process_chapter_task` |
| `audiobook_factory/pipeline.py` | 1328 | `_process_chapter_task` |
| `app.py` | 60 | `_OUTPUT_DIR` |
| `app.py` | 1117 | `on_generate` |
| `app.py` | 975 | `_load_cached_chapters_if_available` |
| `cli.py` | 250 | `_load_config_json` |
| `start_api.py` | 16 | `__main__` |

## Background

AudiobookMaker is a single-user AI audiobook generator (book extraction + TTS synthesis) shipped as a Gradio web UI plus a detached FastAPI/WebSocket orchestration backend. run.sh starts both: the Gradio UI on localhost:7860 and the backend on 127.0.0.1:8000 (start_api.py); the Colab/Kaggle notebooks spawn the same backend in the VM (where processes run as root). The backend exposes an unauthenticated POST /api/v1/generate whose JSON body contains a free-form "config" dict and a "chapters" list; a background worker materializes the dict into an AudiobookConfig dataclass and runs the shared generation pipeline. One of the config fields is output_dir — the base directory for all generated artifacts. In every other entry point the application itself derives this directory server-side and safely (the Gradio UI computes <project>/audiobook_output/<sanitized book title>, stripping path separators from the title; the CLI defaults to a project-relative path), and chapter/file names produced under it are additionally sanitized by make_safe_filename. The direct API caller sits outside that derivation layer: it supplies output_dir as a raw string. The pipeline's resume mechanism — the generation_progress.json checkpoint file (chapter statuses, cached chapter text, and the full settings dict used to restore a session) — is the artifact this directory hosts, and the product's documented workflows (Gradio progress upload, Colab/Kaggle headless "python cli.py <generation_progress.json>") consume it verbatim. Because the backend has no authentication and no path containment, an untrusted request-controlled base directory turns the pipeline's own bookkeeping I/O into host-wide filesystem primitives.

## Description

The request body's config dict flows unmodified into AudiobookConfig.from_dict, which keeps every known field verbatim (output_dir is a plain str field, default "./output"); unknown keys are dropped and no value validation exists (_validate_config checks only the quantization enum). The preflight gate that runs before the pipeline checks only the environment (Python/torch/CUDA/transformers/ soundfile/FFmpeg); the optional voice-reference check is skipped when voice_file is empty, and no check inspects output_dir — so on any functioning install the pipeline proceeds. run_pipeline then executes three filesystem primitives, all before any GPU pool or model work, i.e. independently of model health: (1) os.makedirs(config.output_dir, exist_ok=True) creates any absolute path with unlimited depth — and runs even in preview_mode, since the preview early-return comes after it; (2) the progress checkpoint <output_dir>/generation_progress.json is written by write_progress_file, whose _write_unlocked re-creates the parent chain (os.makedirs of the dirname) and writes atomically via a .tmp sibling + os.replace — when the target file is absent, read_progress_file raises FileNotFoundError and the pipeline builds the document entirely from the request: book_title, book_path, voice_file (arbitrary strings), the complete settings dict (asdict of the config, including tts_provider_name, tts_model_name, pronunciation_map) and per-chapter title/text/sentences taken verbatim from the chapters payload; when the target already exists, a "dirty" update still injects book_path/voice_file/settings when those keys are missing, and (3) with force_reprocess=true the pipeline first os.remove()s <output_dir>/generation_progress.json (delete-then-write across the same task yields total authorship of an existing victim file) plus the fixed shared <project root>/temp/generation_progress.json. Each chapter task additionally creates <output_dir>/.temp_chunks/abm_chXXX, deletes chunk WAVs under force_reprocess, and shutil.rmtree()s that subtree in its finally block. No containment, allowlist, normalization, authorization, or sandboxing exists anywhere between the HTTP body and these sinks; the only sanitization in the family (make_safe_filename) applies to title-derived filenames, never to the directory. Impact boundaries established during verification: the written/deleted filename is pinned to generation_progress.json (no arbitrary-name file overwrite) and the content is a well-formed JSON document (json.dump with ensure_ascii=False — string values attacker-controlled, no arbitrary binary). The planted artifact is however exactly what the product's resume machinery consumes: the co-located Gradio UI (run.sh starts both) auto-detects audiobook_output/<sanitized title>/generation_progress.json, and on the next generation for that title _load_cached_chapters_if_available silently substitutes the planted chapters[].text/sentences/titles for the victim's book, skipping re-parsing; the documented headless resume (python cli.py <file>) and the UI's progress upload restore the planted settings dict into AudiobookConfig verbatim (feeding the model-loading chains recorded separately). Deleting or poisoning the checkpoint destroys the only resume state (documented recovery is full re-synthesis, hours of GPU work). A failed makedirs propagates to task.error_message which GET /api/v1/tasks/{id} returns unauthenticated — a path-existence/writability oracle.

## Attack

Attacker: any process or co-tenant on the host that can reach the loopback backend (no authentication — on a multi-user host, user B posts to user A's service and the writes execute with A's credentials into paths B cannot write directly), a web page rendered in the operator's browser (EXP-settled delivery: on FastAPI <= 0.131.x installs — the product's fastapi requirement is unpinned — a no-preflight no-cors fetch/sendBeacon with an untyped Blob (no Content-Type header) is accepted and executes the sinks, even with a relative output_dir that requires no knowledge of the victim's install path; on all FastAPI versions a DNS-rebinding page's same-origin application/json POST is accepted since uvicorn performs no Host validation), or any remote client where port 8000 is exposed (container port mapping, custom --host — user misconfiguration, nothing shipped binds wide). Observed effects, all from a single unauthenticated request: (a) create arbitrary directories anywhere the service user can write (root-owned whole filesystem in the Colab/Kaggle notebooks); (b) delete any <dir>/generation_progress.json plus the shared <project root>/temp/generation_progress.json — destroying another deployment's or co-user's resume checkpoint and forcing multi-hour GPU re-synthesis; (c) plant an attacker-authored checkpoint at the predictable documented path audiobook_output/<sanitized title>/generation_progress.json — EXP end-to-end proof (2026-09-15): after the poisoning, the victim's own UI showed a normal-looking resume banner ("Existing Progress Found! Generation will automatically resume"), and the victim's next Generate click logged "Using cached chapter text from progress JSON (skipping book re-parsing)" and narrated the ATTACKER chapters into real mastered MP3s written into the victim's output directory — the victim's uploaded book was never re-parsed; if the victim follows the documented resume workflow (download/re-run the progress JSON via cli.py or the UI progress upload), the planted settings (tts provider and model id, pronunciation map, paths) additionally flow verbatim into AudiobookConfig — the delivery vector for the progress-restore → model-load chains (one victim interaction; the same attacker can also reach those sinks directly per BUG-R2-C2-A1-H1, which requires no interaction); (d) a path/permission error oracle via unauthenticated task status queries. Failed later stages (unknown provider, missing models) do not undo any of these writes.

### Payload

One HTTP request, e.g. POST http://127.0.0.1:8000/api/v1/generate with body {"config": {"output_dir": "/home/victim/audiobook_output/Their Book Title", "book_title": "Their Book Title", "voice_file": "", "force_reprocess": true, "preview_mode": false}, "chapters": [{"num": 1, "title": "Chapter 1", "text": "<attacker-chosen narration text>", "sentences": []}]}. The directory component is fully attacker-chosen (any absolute path; ~ is not expanded but absolute paths need no expansion); the JSON document written at <output_dir>/generation_progress.json carries attacker-authored book_title, book_path, voice_file, the full settings dict and chapter text. Variants: output_dir=/tmp/a/b/c/... with preview_mode=true for pure arbitrary directory creation; force_reprocess=true to delete an existing victim checkpoint first (and the shared <project root>/temp/generation_progress.json) then rewrite it with total authorship; repeated requests with distinct deep paths for directory/inode littering. No valid book, voice file, GPU or model is required — the sinks precede all model work.

## Data flow

### Step 1 — `api/server.py:113-127 (model at 26-27)`

Unauthenticated POST /api/v1/generate: GenerateRequest.config is a free-form Dict[str, Any]; enqueue_generation stores it in Task.config_dict and queues it. No auth dependency, middleware, CORS policy, or body/rate limits exist on the app.

### Step 2 — `api/worker.py:124,139-142`

Worker materializes cfg = AudiobookConfig.from_dict(task.config_dict) and dispatches run_pipeline(cfg, chapters, ...) to the executor; nothing in api/ touches output_dir (grep: zero references outside the payload).

### Step 3 — `audiobook_factory/pipeline.py:273-300 (field at 206)`

from_dict keeps known fields verbatim — output_dir is a plain str field (default "./output"), so the attacker's absolute/escaping path passes unchanged; _validate_config (pipeline.py:324-337) checks only the quantization enum.

### Step 4 — `audiobook_factory/pipeline.py:441-456 → audiobook_factory/preflight.py:255-345`

Preflight checks only the environment (Python/torch/CUDA/transformers/soundfile/FFmpeg); the voice_ref check is skipped when voice_file="" (check_voice_ref=bool(config.voice_file)); no check inspects output_dir, so any functioning install proceeds to the sinks.

### Step 5 — `audiobook_factory/pipeline.py:463`

PRIMITIVE 1 — os.makedirs(config.output_dir, exist_ok=True): arbitrary absolute directory creation, unlimited depth; executes before the preview_mode early-return (466-471), so it needs no chapters, GPU or models.

### Step 6 — `audiobook_factory/pipeline.py:484-496 (_TEMP_DIR at 63-64)`

PRIMITIVE 3 — prog_path_out = join(output_dir, "generation_progress.json"); with force_reprocess=true, os.remove(p) runs on prog_path_out and on the fixed shared <project root>/temp/generation_progress.json (prog_path_tmp), deleting any existing file at those names; OSError is only logged.

### Step 7 — `audiobook_factory/pipeline.py:515-537`

PRIMITIVE 2 — read_progress_file(prog_path_out) raises FileNotFoundError for an absent target; the except branch builds progress_data entirely from the request (book_title/book_path/voice_file strings, settings = asdict(config) incl. tts_provider_name/tts_model_name/pronunciation_map, chapters[].title/text/sentences from the payload) and write_progress_file(prog_path_out, progress_data) plants it at the arbitrary path.

### Step 8 — `audiobook_factory/progress_io.py:282-306 (makedirs at 298)`

_write_unlocked makes the write self-contained: os.makedirs(dirname) re-creates the parent chain, then a <path>.tmp file + os.replace performs the atomic write — no pre-existing directory or file is required.

### Step 9 — `audiobook_factory/pipeline.py:539-559`

Partial authorship of an existing valid target (book_path/voice_file/settings injected when missing — 539-553) and force-sync of the same attacker data to the fixed <project root>/temp/generation_progress.json (559), the cross-user file the notebook workflow directs users to download and re-upload.

### Step 10 — `audiobook_factory/pipeline.py:999-1010,1328`

Per-chapter: makedirs(<output_dir>/.temp_chunks/abm_ch{idx:03d}), chunk_ch_{idx}_*.wav deletion under force_reprocess, and shutil.rmtree of that subtree in the finally block — the target directory's chunk cache is destroyed regardless of task outcome. All primitives above run before the GPU pool setup (pipeline.py:564-572), so they never depend on model health.

### Step 11 — `app.py:60,81-100,975-1015,1117-1120,1212; cli.py:230-275; AudiobookMaker_Colab.ipynb cells 17-18`

Victim-side consumption of the planted artifact: the co-located Gradio UI derives the predictable audiobook_output/<sanitized title>/ layout (path separators stripped from titles only — the directory itself was never attacker-controlled in the UI flow); check_existing_progress auto-detects the checkpoint; _load_cached_chapters_if_available silently substitutes the planted chapter text/sentences/titles for the victim's book on the next generation; the documented headless resume (python cli.py <generation_progress.json>, pointed at audiobook_output/<BookTitle>/generation_progress.json by the notebook) and the UI progress upload restore the planted settings verbatim into AudiobookConfig.

## Fix / patch notes

diff --git a/api/worker.py b/api/worker.py
--- a/api/worker.py
+++ b/api/worker.py
@@ -121,6 +121,26 @@ async def _process_single_task(task_id: str, sem: asyncio.Semaphore) -> None:
         await task.add_log(f"🚀 Starting generation task: {task_id}")
 
         try:
             cfg = AudiobookConfig.from_dict(task.config_dict)
+            # Security: /api/v1/generate is unauthenticated, so output_dir from
+            # the request body must never reach the pipeline's
+            # makedirs/os.remove/write sinks as-is. Contain it inside the
+            # server output base (default <project root>/audiobook_output;
+            # override with the ABM_OUTPUT_BASE environment variable).
+            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+            _base = os.path.realpath(
+                os.environ.get(
+                    "ABM_OUTPUT_BASE", os.path.join(_root, "audiobook_output")
+                )
+            )
+            _raw = os.path.expanduser(str(cfg.output_dir or ""))
+            if not os.path.isabs(_raw):
+                _raw = os.path.join(_base, _raw)
+            _out = os.path.realpath(_raw)
+            if _out != _base and not _out.startswith(_base + os.sep):
+                raise ValueError(
+                    "output_dir must resolve inside the server output base directory"
+                )
+            cfg.output_dir = _out
             chapters = [
                 ExtractedChapter(

## References

- https://cwe.mitre.org/data/definitions/22.html
- https://cwe.mitre.org/data/definitions/73.html
- https://owasp.org/www-community/attacks/Path_Traversal

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] Audiobook generation pipeline chapter audio writer (make_safe_filename extension handling + request-controlled output_dir): path traversal via unsanitized output_format extension to arbitrary-path file overwrite/planting, reachable unauthenticated via /api/v1/generate
- key: `BUG-R2-C2-A2-H3`
- disclosure: owner_only
- cwe: CWE-22
- file: `api/server.py`

# Audiobook generation pipeline chapter audio writer (make_safe_filename extension handling + request-controlled output_dir): path traversal via unsanitized output_format extension to arbitrary-path file overwrite/planting, reachable unauthenticated via /api/v1/generate

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C2-A2-H3
- **CWE:** CWE-22
- **CVSS:** 8.3 (`CVSS:3.1/AV:A/AC:L/PR:N/UI:N/S:U/C:L/I:H/A:H`)
- **EV priority:** P1
- **EV score:** 7
- **PoC status:** reproduced
- **EXP status:** confirmed
- **Affected versions:** unknown (any release shipping the FastAPI server api/server.py together with audiobook_factory/pipeline.py and audiobook_factory/filename_sanitizer.py; current source snapshot dc9aed2 confirmed affected, dynamically reproduced through the unauthenticated /api/v1/generate endpoint — see poc/; business-impact assessment confirmed on the same snapshot in a multi-user workstation replica (co-tenant attacker, unmodified launcher + Rust extension build) — see exp/)

## Exploitability rationale

R:L (local/adjacent): the shipped launch mode binds the FastAPI backend to 127.0.0.1:8000 (start_api.py:16), so the realistic attacker is any process or co-tenant on the host (the endpoint has no authentication), a DNS-rebinding page rendered for the operator (uvicorn performs no Host/Origin validation — re-verified in the EXP: a generate POST with an attacker Host header is served normally), or any deployment that exposes the port (operator change of host, reverse proxy, or container port mapping upgrades reachability to network-remote, CVSS 9.4; the EXP demonstrated the full prepare+write attack over the host's routable IP against a 0.0.0.0-bound instance). The Colab/Kaggle tunnels publish the Gradio port (7860), not the API port, so those modes alone do not expose this path. E:D (default): the generate endpoint, the chapter writer and make_safe_filename are the core output path of every task in both launch modes; no optional flag or uncommon configuration is needed. C:D (deterministic): the trigger is pure logic — sequential HTTP POSTs whose effect (directory name, joined filename, resolved target) is fully computable in advance; no race, no timing, no memory layout. The only environmental precondition is that the server's TTS synthesis works, which is the service's normal operating state (the prepare/makedirs and resume-oracle legs need no TTS at all). I:S (significant): EXP-confirmed on the dc9aed2 snapshot in a multi-user workstation replica — an unprivileged co-tenant (no direct write access to the operator's files) silently replaced the operator's entire 12-chapter audiobook library with metadata-identical attacker audio (title/artist/album mirrored, track spoofed via a multi-chapter task, owner/mode unchanged), planted new chapters/books and bonus tracks (mp3/m4b/flac/ogg/m4a/mp4) into the victim's libraries, had the operator's narrator voice sample read and handed to the TTS provider as the cloning reference for the payload (attacker text in the victim's cloned voice — the app's documented feature), enumerated file existence for arbitrary paths including ones the co-tenant cannot stat (0700 trees, /etc/shadow), and sustained disk consumption in attacker-chosen victim paths with an 8x sample_rate byte-rate amplification (375 KiB/s at 192 kHz wav). Constraints verified as claimed: the final path component must end in a muxer-known extension (.pth drop attempt wrote nothing), and the write scope is the service user's write scope (attacker-owned file untouched). Two EXP refinements: the resume existence oracle is stateful (a probe of a missing path poisons the staged progress; the next probe regenerates — overwriting an existing media target — unless the attacker re-stages, one more unauthenticated request), and bitrate_kbps has no effect on the traversal branch (-b:a only applies to exact output_format matches). No code-execution path is claimed through this branch. The write leg ran with the project's own audiobook_rust extension built as the documented install does; the Gradio-relay delivery of output_format is blocked server-side by Dropdown choices enforcement on the current unpinned gradio 6.27.0 install (latent only on <=4.44.1).

## Code anchors

| File | Line | Function |
|---|---:|---|
| `api/server.py` | 112 | `enqueue_generation` |
| `api/worker.py` | 124 | `_process_single_task` |
| `api/worker.py` | 128 | `_process_single_task` |
| `audiobook_factory/pipeline.py` | 273 | `AudiobookConfig.from_dict` |
| `audiobook_factory/pipeline.py` | 324 | `_validate_config` |
| `audiobook_factory/pipeline.py` | 463 | `run_pipeline` |
| `audiobook_factory/filename_sanitizer.py` | 97 | `make_safe_filename` |
| `audiobook_factory/pipeline.py` | 1180 | `_process_chapter` |
| `audiobook_factory/pipeline.py` | 1182 | `_process_chapter` |
| `audiobook_factory/pipeline.py` | 1194 | `_process_chapter` |
| `audiobook_factory/pipeline.py` | 1228 | `_process_chapter._build_cmd` |
| `audiobook_factory/pipeline.py` | 1301 | `_process_chapter._build_py_cmd` |
| `audiobook_factory/pipeline.py` | 746 | `run_pipeline` |
| `audiobook_factory/pipeline.py` | 639 | `run_pipeline._process` |
| `audiobook_factory/ffmpeg_utils.py` | 43 | `get_format_settings` |

## Background

AudiobookMaker is a self-hosted audiobook generation application (book text extraction + TTS synthesis + audio mastering) offered in two launch modes: a workstation mode (run.sh starts a FastAPI backend on 127.0.0.1:8000 via start_api.py plus a Gradio web UI on port 7860) and cloud-notebook modes (Colab/Kaggle, which run the same backend and publish the Gradio UI through a public tunnel). The FastAPI backend is an unauthenticated task runner: POST /api/v1/generate accepts a raw JSON object with two fields — "config" (an arbitrary key/value dict) and "chapters" (an arbitrary list of chapter objects with attacker-chosen titles and text) — and a background worker converts them into an AudiobookConfig and runs the full pipeline. The pipeline synthesizes each chapter with TTS, masters the audio (a Rust extension when available, otherwise ffmpeg), and writes one output file per chapter into config.output_dir. Output filenames are produced by make_safe_filename() (ported from the epub_to_audiobook project), which sanitizes the chapter title but treats the file extension as a trusted literal. This report concerns what happens when that extension — and the output directory itself — come from the unauthenticated request instead.

## Description

make_safe_filename(title, idx, output_dir, ext) builds chapter filenames as "Chapter {idx} - {sanitized_title}{ext}". The title is properly sanitized (path separators, traversal dots and control characters are removed), but the ext parameter receives only a leading-dot normalization (filename_sanitizer.py:97-100: "if not ext.startswith('.'): ext = '.' + ext") — no character filtering of any kind. The pipeline passes f".{config.output_format}" into this parameter at three sites (chapter audio writer pipeline.py:1180-1181, single-file combine writer pipeline.py:746, and the resume existence probe pipeline.py:639), where config.output_format is a plain request-controlled string: AudiobookConfig.from_dict (pipeline.py:272-300) filters only unknown keys, and _validate_config (pipeline.py:324-334) validates only the "quantization" field. No allowlist for output_format exists anywhere in the repository.
A single request cannot exploit this directly, because the returned name is joined onto config.output_dir with os.path.join and the traversal must climb through the intermediate component "Chapter {idx} - {title}." — if that path component does not exist as a directory, the final open() fails with ENOTDIR. However, the same unauthenticated endpoint provides the missing enabler: run_pipeline unconditionally executes os.makedirs(config.output_dir, exist_ok=True) at pipeline.py:463, before the preview-mode early return, with the same request-controlled string. An attacker therefore issues two POSTs to /api/v1/generate: (1) a "prepare" task whose output_dir is "<base>/Chapter 1 - x." — an arbitrary absolute directory whose name is fully predictable from the filename construction rules ("Chapter 1 - " + sanitized title + the extension's leading dot; the "0001 - " short-prefix fallback for very long extensions is equally predictable) — which creates exactly the intermediate directory component; (2) a "write" task whose output_dir is <base>, whose first chapter title is "x", and whose output_format is "/../../<target>.mp3". The second task's safe_name becomes "Chapter 1 - x./../../<target>.mp3", out_path = os.path.join(<base>, safe_name) — os.path.join performs no normalization — and the kernel resolves the written file to the absolute path <target>.mp3 (each ".." climbs one level; the attacker controls their count and the tail, so any absolute target is reachable, and missing parent directories can be pre-created with additional makedirs tasks).
The write is performed by an ffmpeg subprocess with out_path as the final argv element, in list form (no shell): pipeline.py:1213-1234 when the Rust mastering extension is present, pipeline.py:1283-1310 for the pure-Python fallback. The Rust "fast path" cannot accidentally absorb the write: it requires an exact config.output_format match against "mp3"/"wav" (pipeline.py:1194), so a traversal string always routes through the ffmpeg writer, which writes the same out_path. No -f flag is ever passed and get_format_settings() returns empty settings for unknown format strings (ffmpeg_utils.py:1-43), so ffmpeg infers the muxer from the final path component's extension — the single effective constraint on the primitive: the final component must end in an extension ffmpeg can map to an output muxer (mp3, wav, flac, ogg, m4a, m4b, aac, mp4, mov, webm, mkv, ...). Extensions like .pth, .conf or extensionless names make ffmpeg abort with "Unable to find a suitable output format", which bounds the primitive to media-extension files rather than arbitrary config/executable drops.
The chapter writer requires at least one successfully synthesized TTS chunk (pipeline.py:1176-1178) — the service's normal operating state. The written content is a valid audio file of attacker-chosen text with fully attacker-controlled metadata tags (-metadata title/artist/album/track, pipeline.py:1222-1229), i.e. attacker-authored media suitable for content spoofing. Two sibling sinks share the same extension plumbing: the single-file combine mode writes "Combined_Chapter 0 - <book_title><ext>" through ffmpeg concat (pipeline.py:745-758), and the resume path probes os.path.exists(output_dir + traversal-name) (pipeline.py:639-644) — the latter is an existence oracle for arbitrary paths with no extension constraint at all: when the generation_progress.json in the attacker-chosen output_dir marks a matching chapter as completed (staged by the attacker's own prior run into the same directory), a hit is reported back through output_files in GET /api/v1/tasks/{task_id}.

## Attack

An attacker who can reach the backend port (a local or co-tenant process on the host, a DNS-rebinding page rendered for the operator, or any deployment where port 8000 is exposed — e.g. an operator binding 0.0.0.0 for LAN/remote use) submits the two tasks above. The prepare task completes in under a second (preview mode returns immediately after the makedirs call); the attacker can poll GET /api/v1/tasks/{task_id} to confirm completion before sending the write task. When the write task's chapter finishes synthesizing, the mastering ffmpeg process creates or overwrites the target file at the resolved absolute path. Observable impact: silent destruction or replacement of any media-extension file writable by the service user (finished audiobooks of other runs, web-served media directories, voice reference libraries), planting of attacker-authored audio with spoofed title/artist/album metadata at arbitrary locations, arbitrary directory-tree creation anywhere the service user may write, and — via the resume existence probe — an unrestricted file-existence oracle for any absolute path (finger-printing installed software, probing other users' home directories). The attack leaves the server otherwise operational and produces no errors when the target path is valid, making it suitable for targeted, low-noise tampering. Business-impact assessment (exp/): in a multi-user workstation replica, an unprivileged co-tenant replaced the operator's whole 12-chapter audiobook library with metadata-identical attacker audio (player-invisible substitution; track numbers spoofable via multi-chapter tasks; file owner/mode unchanged), planted new chapters, a new m4b book and flac/ogg/m4a/mp4 tracks into the victim's libraries, drove the operator's own narrator voice sample into the TTS provider as the cloning reference for the attacker's payload text, fingerprinted the host through the existence oracle (including 0700-private paths the co-tenant cannot stat), and sustained amplified disk consumption inside the operator's home; the complete attack also ran against a 0.0.0.0-bound instance over the network. Verified boundaries: no non-media extension drops (muxer constraint) and no writes outside the service user's permission scope.

### Payload

Two unauthenticated POST /api/v1/generate requests. Prepare task: {"config": {"output_dir": "/tmp/abm/Chapter 1 - x.", "preview_mode": true}, "chapters": []} — the directory name must equal the first path component that the write task's generated filename will produce. Write task: {"config": {"output_dir": "/tmp/abm", "output_format": "/../../pwned.mp3"}, "chapters": [{"title": "x", "text": "<any sentence>"}]} — with title "x" and first-chapter index 1, make_safe_filename returns "Chapter 1 - x./../../pwned.mp3" and ffmpeg writes /tmp/pwned.mp3. The number of "../" segments and the tail are chosen so the resolved path equals the target (e.g. output_format "/../../../home/user/media/book.mp3" from /tmp/abm reaches /home/user/media/book.mp3). Any muxer-known extension works (mp3, wav, flac, ogg, m4a, m4b, aac, mp4, mov, webm, mkv, ...).

## Data flow

### Step 1 — `api/server.py:112-127`

POST /api/v1/generate — unauthenticated FastAPI endpoint; the entire request body (config dict + chapter list, including titles) is stored verbatim in a Task and queued.

### Step 2 — `api/worker.py:124-128`

Worker builds AudiobookConfig.from_dict(task.config_dict) — from_dict (pipeline.py:272-300) drops only unknown keys; output_dir and output_format are plain unvalidated str fields (pipeline.py:206-207) — and ExtractedChapter objects with attacker-controlled titles.

### Step 3 — `audiobook_factory/pipeline.py:437`

run_pipeline calls _validate_config (pipeline.py:324-334) — validates only 'quantization'; no check on output_dir/output_format. Preflight (pipeline.py:439-451) is environment-only.

### Step 4 — `audiobook_factory/pipeline.py:463`

Task 1 sink: os.makedirs(config.output_dir, exist_ok=True) with the raw request string — arbitrary (absolute, nested) directory creation, executed before the preview_mode early return (pipeline.py:467-471), so no TTS is required to create the traversal intermediate directory '<base>/Chapter 1 - x.'.

### Step 5 — `audiobook_factory/pipeline.py:1176-1178`

Task 2 gate: chapter write proceeds only if at least one TTS chunk was synthesized (the service's normal operating state).

### Step 6 — `audiobook_factory/pipeline.py:1180-1181`

safe_name = make_safe_filename(chapter.title, idx, config.output_dir, f".{config.output_format}") — the ext argument carries the raw request-controlled output_format.

### Step 7 — `audiobook_factory/filename_sanitizer.py:97-100`

make_safe_filename sanitizes the title base (filename_sanitizer.py:45-77) but applies only a leading-dot normalization to ext — no character filtering; with output_format '/../../<target>.mp3' the returned name is 'Chapter 1 - x./../../<target>.mp3'.

### Step 8 — `audiobook_factory/pipeline.py:1182`

out_path = os.path.join(config.output_dir, safe_name) — os.path.join performs no normalization, so the traversal string survives verbatim and resolves (through the Task-1-created intermediate directory) to an arbitrary absolute path at open() time.

### Step 9 — `audiobook_factory/pipeline.py:1194-1196`

use_pure_rust requires an exact output_format match against 'mp3'/'wav' — a traversal string never matches, so master_target is a temp WAV and the ffmpeg writer is used; the Python fallback (pipeline.py:1283-1310) builds the identical output path.

### Step 10 — `audiobook_factory/pipeline.py:1228/1301 → 1234/1310`

ffmpeg subprocess launched with out_path as the final argv element (list form, no shell) — ffmpeg opens the resolved absolute path for writing; no output-path restriction exists in ffmpeg.

### Step 11 — `audiobook_factory/ffmpeg_utils.py:1-43`

get_format_settings returns empty settings for the unknown format string and no -f flag is passed anywhere, so the muxer is inferred from the final component's extension — the only effective constraint (final component must end in a muxer-known media extension).

### Step 12 — `audiobook_factory/pipeline.py:745-758`

Same extension plumbing in single-file combine mode: full_path = join(output_dir, 'Combined_' + make_safe_filename(book_title, 0, output_dir, f".{output_format}")) written by ffmpeg concat — same traversal with the predictable 'Combined_Chapter 0 - <title>.' intermediate.

### Step 13 — `audiobook_factory/pipeline.py:639-644`

Resume existence probe: existing_path = join(output_dir, make_safe_filename(...)) is tested with os.path.exists when the progress file marks a matching chapter completed; a hit is returned in output_files via GET /api/v1/tasks/{task_id} — an arbitrary-path existence oracle with no extension constraint.

## Fix / patch notes

diff --git a/audiobook_factory/filename_sanitizer.py b/audiobook_factory/filename_sanitizer.py
--- a/audiobook_factory/filename_sanitizer.py
+++ b/audiobook_factory/filename_sanitizer.py
@@ -97,6 +97,13 @@ def make_safe_filename(
     if not ext:
         raise ValueError("ext must be non-empty")
     if not ext.startswith("."):
         ext = "." + ext
+    # ext reaches this function from request-controlled config
+    # (AudiobookConfig.output_format, joined at pipeline.py:1180/746/639).
+    # It must remain a plain suffix: any path character here escapes
+    # output_dir once the returned name is joined (CWE-22).
+    _body = ext[1:]
+    if not _body or any(ch in _FORBIDDEN or ch == "." or ord(ch) < 32 for ch in _body):
+        raise ValueError(f"ext must be a simple extension suffix, got {ext!r}")

 name_max = _detect_name_max(output_dir)
diff --git a/audiobook_factory/pipeline.py b/audiobook_factory/pipeline.py
--- a/audiobook_factory/pipeline.py
+++ b/audiobook_factory/pipeline.py
@@ -330,6 +330,14 @@ def _validate_config(config: AudiobookConfig) -> None:
     if config.quantization not in _VALID_QUANTIZATION_MODES:
         raise ValueError(
             f"Invalid quantization mode '{config.quantization}'. "
             f"Supported options: {sorted(_VALID_QUANTIZATION_MODES)}"
         )
+
+    # output_format flows verbatim into output filenames (make_safe_filename
+    # ext argument) and must never carry path components.
+    if any(sep in config.output_format for sep in ("/", "\\")) or ".." in config.output_format:
+        raise ValueError(
+            f"Invalid output_format {config.output_format!r}: expected a plain "
+            "extension such as 'mp3', 'flac', 'wav', 'm4b'"
+        )

## References

- https://cwe.mitre.org/data/definitions/22.html
- https://cwe.mitre.org/data/definitions/73.html
- https://owasp.org/www-community/attacks/Path_Traversal
- https://docs.python.org/3/library/os.path.html#os.path.join

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] Gradio book-upload auto-scan chain (on_book_upload → _scan_epub → ebooklib read_epub, app.py:656 / text_extractor.py:221): uncapped eager archive decompression lets an anonymous visitor OOM-kill the entire service process with a small crafted .epub
- key: `BUG-R2-C3-A1-H1`
- disclosure: owner_only
- cwe: CWE-409
- file: `app.py`

# Gradio book-upload auto-scan chain (on_book_upload → _scan_epub → ebooklib read_epub, app.py:656 / text_extractor.py:221): uncapped eager archive decompression lets an anonymous visitor OOM-kill the entire service process with a small crafted .epub

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C3-A1-H1
- **CWE:** CWE-409
- **CVSS:** 7.5 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H`)
- **EV priority:** P0
- **EV score:** 8
- **PoC status:** reproduced
- **EXP status:** pending
- **Affected versions:** audited snapshot (commit dc9aed2cac064f436310a35fdd069f34205fa687, v1.3.0 line); the scan/extract chain and the eager ebooklib read are present in current main — exact introduction version not established. Dependencies: any ebooklib with the eager manifest reader (0.20 verified installed; eager full-archive read is the library's long-standing design), any gradio with default max_file_size=None (6.27.0 verified installed; the None→math.inf upload behavior is framework-longstanding).

## Exploitability rationale

Reachability R:N — the trigger is a single anonymous upload against the advertised public tunnel (Colab/Kaggle notebooks expose the exact same no-auth Gradio app at a public pinggy.io URL; no authentication exists anywhere in the app), and the scan runs automatically the moment the upload completes — no button click, no victim interaction. Exposure E:D — the Book-tab upload + auto-scan is THE core flow of the default UI (the status placeholder literally says "Upload a file to begin"); it exists in every deployment mode and no option disables it. Certainty C:D — pure logic defect, deterministic single-request trigger: the attacker fully controls the OPF manifest, the member list and the zip headers, so the total decompressed size (the memory demanded) is attacker-exact; no race or memory layout involved. Impact I:D — availability only, but total and pre-auth: the eager read_epub retains every decompressed archive member simultaneously in the single shared service process (in the cloud notebooks this process IS the interactive kernel), so a ~2–16 MB upload drives RSS into the multi-GB range and the kernel OOM-killer SIGKILLs it — uncatchable by the app's except handlers, killing the session for the owner and every co-visitor at once, repeatable at will (uploads are unlimited, unauthenticated and unrated).

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 168 | `build_app` |
| `app.py` | 635 | `on_book_upload` |
| `app.py` | 656 | `on_book_upload` |
| `app.py` | 756 | `build_app` |
| `app.py` | 1908 | `build_app` |
| `audiobook_factory/text_extractor.py` | 216 | `_scan_epub` |
| `audiobook_factory/text_extractor.py` | 221 | `_scan_epub` |
| `audiobook_factory/text_extractor.py` | 253 | `_scan_epub` |
| `audiobook_factory/extractor_engine.py` | 805 | `ingest_epub` |
| `<site-packages>/ebooklib/epub.py` | 1545 | `EpubReader._load_manifest` |
| `<site-packages>/ebooklib/epub.py` | 1467 | `EpubReader.read_file` |
| `<site-packages>/gradio/blocks.py` | 2812 | `Blocks.launch` |

## Background

AudiobookMaker is a self-hosted text-to-speech audiobook generator. Its primary interface is an unauthenticated Gradio web app (app.py) that the README and the bundled Colab/Kaggle notebooks deliberately expose to anonymous visitors through a public SSH tunnel (pinggy.io) pointing at localhost:7860; on those notebooks the Python process running the web server is the notebook kernel itself. The UI's entry flow is the Book tab: the visitor uploads a book file (EPUB/MOBI/PDF/DOCX/ODT/TXT) into a gr.File component, and the app immediately scans it — chapter list, title, author, cover — so the user can pick chapters before generating audio. The scan is wired to fire automatically when the upload finishes (no button click), and it runs synchronously inside the web-server process. For EPUB/MOBI the scan is implemented with EbookLib's read_epub(), a library that eagerly decompresses the entire archive into memory, because it loads every OPF-manifest item's content up front. Book files are untrusted input by definition — the README's advertised sharing model is strangers pasting their own books into a shared public instance.

## Description

The upload → scan chain performs no resource bounding of any kind, at any layer. (1) Upload size: gradio's only gate is launch(max_file_size), which defaults to None ("no limit is set"); the /gradio_api/upload route substitutes math.inf when it is None, and neither the local launch (app.py:1908-1916) nor either notebook launch cell passes a value — so anonymous uploads are unlimited. The multipart handler streams files to disk, so a compressed bomb is cheap to deliver. (2) File-type gate: gr.File(file_types=[".epub", ...]) is enforced server-side as a pure filename-suffix match (gradio_client is_valid_file) — content is never inspected, so a renamed zip is indistinguishable from a real EPUB; _detect_type likewise routes by extension (.epub — and .mobi — go to the EPUB scan path). (3) The sink: on_book_upload runs synchronously in the web-server process (gradio executes event handlers in-process in a thread pool) and calls scan() → _scan_epub() → epub.read_epub(path) (text_extractor.py:221, no options). (4) The library behavior: EbookLib (0.20 installed) EpubReader._load reads META-INF/container.xml, the OPF, and then _load_manifest iterates EVERY <item> of the attacker-authored OPF manifest and, in every media-type branch — including the catch-all EpubItem branch that accepts any media type such as application/octet-stream — assigns item.content = read_file(href), where read_file is ZipFile.read(): the whole member decompressed into one bytes object. The spine's NCX is read too. All items are retained simultaneously on the returned EpubBook until the handler returns, so peak RSS ≈ the sum of all uncompressed member sizes, which the attacker fixes exactly by writing the manifest and the zip headers (CPython's zipfile truncates member reads at the declared file_size, so the declared values are precisely the memory demanded; truthful headers are written automatically by any zip tool). Deflate amplifies ~1000:1 on compressible data (theoretical max 1032:1), and allowZip64=True permits multi-GB members — a ~16 MB .epub can therefore demand ~16 GB of RAM, a 2–5 MB one 2–5 GB. (5) Failure shapes, both attacker wins: on the advertised cloud deployment (Colab/Kaggle VMs: cgroup memory limit, no swap, heuristic overcommit) the kernel OOM-killer SIGKILLs the process when RSS passes the limit — uncatchable by the except Exception in _scan_epub (text_extractor.py:253), and the process is the notebook kernel, so the whole interactive session dies for the owner and every visitor (restart = re-run cells + reload TTS models); in local mode the python app.py process dies and run.sh has no restart loop (it ends with wait $APP_PID), leaving the UI down. If a host instead refuses the allocation, MemoryError is caught and the handler returns an empty scan — but only after the multi-GB decompression ran to the failure point, and the attacker simply re-uploads (unlimited, unauthenticated, unrated): a repeatable stall. A second one-click reachability leg runs the identical sink: Preview/Generate/Export-config → extract() → _extract_epub → DocumentIngestor.ingest_epub → epub.read_epub (extractor_engine.py:805). No size, ratio, member-count or total cap exists anywhere in the chain — not in the app, not in the launch configuration, and not in ebooklib (whose only relevant option, ignore_ncx, does not bound reads). The minimal fix is a pre-read_epub zip size guard (fix_patch below): CPython's zipfile truncates every member read at the central-directory file_size, so summing the declared member sizes before handing the archive to ebooklib soundly bounds the memory the upload can demand. Note: the two patched files use CRLF line endings in the repository — a byte-exact patch that applies cleanly with strict 'git apply' is included as fix.patch next to this report.

## Attack

An anonymous visitor of the public notebook tunnel URL (the deployment the project advertises for sharing instances with strangers) selects the crafted .epub in the Book tab's file picker — the scan event fires automatically when the upload completes, no button click needed — or posts the same file through the equally unauthenticated REST API (POST /gradio_api/upload, then the upload event's queue endpoint). The server-side handler decompresses every manifest member into memory in the shared service process; within seconds RSS climbs past the host limit and the kernel OOM-killer SIGKILLs the process (the notebook kernel in cloud mode — the owner's session crashes, all concurrent visitors lose the UI, and the owner must re-run the notebook cells and reload the TTS models). Where the allocation is refused instead (MemoryError caught), the multi-GB inflation still stalls the process for its duration and the attacker re-uploads at will — permanent availability loss from a sub-16 MB request, no credentials, no victim interaction, repeatable indefinitely.

### Payload

A small valid zip renamed to .epub (or .mobi): META-INF/container.xml pointing at a minimal OPF whose manifest lists N items of media-type application/octet-stream, each href referring to a zip member of deflated zeros (~1 MB compressed ≈ ~1 GB inflated each at deflate's ~1000:1). N=16 ≈ 16 GB demanded from a ~16 MB upload; smaller targets scale down (a ~2–5 MB bomb exhausts a 2 GiB container; ~13 MB kills a 12.7 GB-class Colab VM). All headers are truthful (written by any zip library), so no validation layer rejects it; the OPF itself stays tiny (the manifest has only N entries).

## Data flow

### Step 1 — `app.py:168-170`

book_file = gr.File(file_types=[".epub", ".mobi", ".pdf", ".docx", ".odt", ".txt"]) — the only input filter on the book upload; enforced server-side (File.preprocess → gradio_client.utils.is_valid_file) as a filename-suffix match, with no content inspection.

### Step 2 — `gradio blocks.py:2812 / routes.py:1941-1948 (installed 6.27.0)`

launch(max_file_size) defaults to None ('If None, no limit is set'); the /gradio_api/upload route substitutes math.inf when it is None; the multipart streamer enforces only this value. Neither app.py:1908-1916 nor the Colab/Kaggle notebook launch cells pass a value — anonymous uploads are unlimited and land on disk.

### Step 3 — `app.py:756-766`

book_file.upload(on_book_upload, ...) — the event fires automatically when the upload completes (browser), and the handler is equally invocable via the unauthenticated REST queue API; no button click and no victim interaction are required.

### Step 4 — `app.py:656`

result: ScanResult = scan(path) — synchronous execution inside the web-server process (gradio runs event handlers in-process in a thread pool; in the cloud notebooks this process is the kernel itself).

### Step 5 — `audiobook_factory/text_extractor.py:79-92,203-205`

_detect_type routes by extension only (.epub → "epub", .mobi → "mobi"); scan() dispatches both to _scan_epub — a renamed zip is never content-checked.

### Step 6 — `audiobook_factory/text_extractor.py:221`

book = epub.read_epub(path) — no options, no zip pre-check, no size/ratio/member cap before handing the archive to the library.

### Step 7 — `<site-packages>/ebooklib/epub.py:1545-1623 (EpubReader._load_manifest)`

for every <item> of the attacker-authored OPF manifest, every media-type branch (NCX/SMIL/nav-xhtml/cover-xhtml/xhtml/images and the catch-all EpubItem branch for any other media type) assigns ei.content = self.read_file(...); read_file is ZipFile.read (epub.py:1467-1481) — the entire member decompressed into one bytes object. The spine's NCX is read too (_load_spine, epub.py:1699-1717).

### Step 8 — `<site-packages>/ebooklib/epub.py:1723-1743 (EpubReader._load_opf_file)`

All items are accumulated on the returned EpubBook (book.add_item(ei)) and retained simultaneously until the handler returns: peak RSS ≈ Σ uncompressed member sizes — attacker-exact, because CPython's ZipExtFile truncates member reads at the declared file_size (zipfile.py:843,1056), so the declared (truthfully written) headers are precisely the memory demanded.

### Step 9 — `audiobook_factory/text_extractor.py:253`

except Exception — catches a raised MemoryError (handler returns an empty scan after the multi-GB inflation already ran; attacker re-uploads at will) but cannot catch the kernel OOM-killer's SIGKILL when RSS passes the host limit (the expected outcome on the advertised cgroup-limited, swapless Colab/Kaggle VMs).

### Step 10 — `audiobook_factory/extractor_engine.py:805`

Second reachability leg: extract() → _extract_epub (text_extractor.py:333,349) → DocumentIngestor.ingest_epub → epub.read_epub(epub_path) — Preview/Generate/Export-config clicks re-run the identical sink even if the auto-scan were removed.

## Fix / patch notes

diff --git a/audiobook_factory/text_extractor.py b/audiobook_factory/text_extractor.py
--- a/audiobook_factory/text_extractor.py
+++ b/audiobook_factory/text_extractor.py
@@ -194,6 +194,31 @@
 # scan() — fast, no OCR, just TOC extraction for the UI
 # ══════════════════════════════════════════════════════════════════════════════
 
+# ── Decompression-bomb guard ────────────────────────────────────────────────
+# CPython's ZipExtFile truncates every member read at the central-directory
+# file_size (zipfile.py: self._left = zipinfo.file_size), so the values the
+# archive declares are a hard upper bound on what any reader — including
+# ebooklib's eager read_epub, which loads EVERY manifest item into memory —
+# can allocate. Rejecting oversized archives before read_epub therefore
+# bounds the memory an uploaded .epub/.mobi can demand.
+_EPUB_MAX_TOTAL_SIZE = 512 * 1024 * 1024   # 512 MiB total uncompressed
+_EPUB_MAX_MEMBERS = 2000
+
+
+def _epub_size_guard(path: str) -> None:
+    """Raise if the archive declares more than the allowed total
+    uncompressed size / member count (decompression-bomb rejection)."""
+    import zipfile
+
+    with zipfile.ZipFile(path, "r") as z:
+        infos = z.infolist()
+    total = sum(i.file_size for i in infos)
+    if total > _EPUB_MAX_TOTAL_SIZE or len(infos) > _EPUB_MAX_MEMBERS:
+        raise ValueError(
+            "Archive too large when decompressed "
+            f"({total} bytes, {len(infos)} members)"
+        )
+
 def scan(path: str) -> ScanResult:
     """
     Fast pre-scan; returns chapter list for EPUB/MOBI or page count for others.
@@ -218,6 +243,7 @@
     from audiobook_factory.extractor_engine import DocumentIngestor  # type: ignore
 
     try:
+        _epub_size_guard(path)
         book = epub.read_epub(path)
         title, author, cover_data = _epub_metadata(book)
 

diff --git a/audiobook_factory/extractor_engine.py b/audiobook_factory/extractor_engine.py
--- a/audiobook_factory/extractor_engine.py
+++ b/audiobook_factory/extractor_engine.py
@@ -802,6 +802,9 @@
         normalizer: TextNormalizer,
     ) -> tuple[list[ChapterItem], list[SkippedItem], list[TocEntry]]:
 
+        from audiobook_factory.text_extractor import _epub_size_guard  # type: ignore
+        _epub_size_guard(epub_path)
+
         book  = epub.read_epub(epub_path)
         items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
         print(f"    Found {len(items)} EPUB document items.")

## References

- https://cwe.mitre.org/data/definitions/409.html
- https://cwe.mitre.org/data/definitions/400.html
- https://owasp.org/www-community/attacks/Zip_bomb
- https://www.gradio.app/guides/file-component-behavior
- https://docs.python.org/3/library/zipfile.html

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] Gradio book-upload DOCX scan path (book_file.upload → on_book_upload → scan → _scan_docx → python-docx Document()): anonymous zero-interaction zip decompression bomb — python-docx eagerly inflates every relationship-reachable archive member (plus lxml trees for XML-typed parts) with no size bound, OOM-killing the shared service process (the notebook kernel itself in the advertised Colab/Kaggle deployment)
- key: `BUG-R2-C3-A1-H2`
- disclosure: owner_only
- cwe: CWE-409
- file: `app.py`

# Gradio book-upload DOCX scan path (book_file.upload → on_book_upload → scan → _scan_docx → python-docx Document()): anonymous zero-interaction zip decompression bomb — python-docx eagerly inflates every relationship-reachable archive member (plus lxml trees for XML-typed parts) with no size bound, OOM-killing the shared service process (the notebook kernel itself in the advertised Colab/Kaggle deployment)

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C3-A1-H2
- **CWE:** CWE-409
- **CVSS:** 7.5 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H`)
- **EV priority:** P1
- **EV score:** 8
- **PoC status:** reproduced
- **EXP status:** pending
- **Affected versions:** any release shipping the Gradio book-upload scan path with python-docx (app.py book_file.upload → on_book_upload → scan() → _scan_docx); requirements.txt declares python-docx unpinned (resolves to 1.2.0, current release, verified installed in the audit environment; the eager OPC relationship walk is stable across the 1.x line); current source snapshot dc9aed2 confirmed affected

## Exploitability rationale

R:N (remote): the README-advertised primary deployment is the Colab/Kaggle notebook, whose launch cell runs the Gradio UI in-process in the notebook kernel and forwards port 7860 through a public anonymous Pinggy tunnel — the anonymous internet visitor is exactly threat-model attacker A-Remote-Web ("can upload any 'book'"). Local-mode deployments (server_name="localhost") degrade reachability to adjacent/local, which only lowers this score. E:D (default): the DOCX upload+scan flow is a core advertised feature, enabled in every shipped launch mode with no flags; requirements.txt ships python-docx (unpinned, resolves to 1.2.0 today) so the sink is installed wherever the feature works. C:D (deterministic): the kill is pure logic — an upload of a crafted zip is decompressed eagerly by python-docx's OPC reader before any validation; no race, no timing, no memory-layout dependence; the only environmental variable is available RAM vs. the bomb's uncompressed size, which the attacker freely scales (deflate zeros expand ~1000:1, so a sub-16 MB upload can demand 10-16+ GB). I:D (disruption): maximal demonstrated impact class is whole-service availability loss — OOM SIGKILL of the process hosting the UI (the notebook kernel in cloud mode: session crash, all in-memory state and running generations lost; the only user-facing surface in local mode), plus unlimited anonymous retries and long inflation stalls for sub-lethal sizes; no code execution, no information disclosure on this path.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 170 | `build_app (book_file gr.File component)` |
| `app.py` | 756 | `build_app (book_file.upload wiring)` |
| `app.py` | 635 | `on_book_upload` |
| `app.py` | 656 | `on_book_upload` |
| `audiobook_factory/text_extractor.py` | 276 | `_scan_docx` |
| `audiobook_factory/text_extractor.py` | 280 | `_scan_docx` |
| `audiobook_factory/text_extractor.py` | 470 | `_extract_docx` |
| `audiobook_factory/text_extractor.py` | 79 | `_detect_type` |
| `app.py` | 1908 | `build_app/demo.launch (local mode, no max_file_size)` |
| `AudiobookMaker_Colab.ipynb` | 16 | `notebook launch cell (cloud mode, in-kernel UI + public tunnel)` |
| `requirements.txt` | 1 | `unpinned python-docx dependency` |

## Background

AudiobookMaker is a single-user audiobook generation application (book extraction + TTS synthesis) offered in two launch modes: a local workstation mode (run.sh starts a FastAPI backend on 127.0.0.1:8000 via start_api.py plus a Gradio UI on :7860) and a cloud-notebook mode (Colab/Kaggle notebooks — the README-advertised primary deployment — which pip-install requirements.txt, launch the Gradio UI *inside the notebook kernel process*, and forward port 7860 through a public anonymous Pinggy SSH tunnel). The UI's first interaction step is uploading a "book": the Book tab's gr.File accepts .epub/.mobi/.pdf/.docx/.odt/.txt, and the upload event immediately runs a metadata pre-scan (on_book_upload → scan(path)) to show type/title/pages — zero interaction beyond the upload itself. For .docx files the scan delegates to python-docx's Document(path), which is expected to read word/document.xml and count words. The scan/extract pipeline is synchronous and runs in the same process that hosts the UI, all user sessions, and (in cloud mode) the notebook kernel.

## Description

The DOCX scan leg hands an attacker-authored zip archive to python-docx 1.2.0 (requirements.txt pins no version; 1.2.0 is the current release and the version actually installed by the shipped install paths). Document(path) → Package.open → PackageReader.from_file performs an *eager, unbounded* whole-package load: _walk_phys_parts walks the OPC relationship graph exactly as authored by the attacker (_rels/.rels and each part's own _rels/*.rels) and calls blob_for(partname) — a full ZipFile.read() of the member, i.e. complete decompression — for every non-external, not-yet-visited relationship target, regardless of relationship type; _load_serialized_parts then retains every blob in a _SerializedPart tuple, and _unmarshal_parts constructs a Part per blob (default Part keeps the raw bytes in _blob; WML-XML content types — document, styles, settings, numbering, headers/footers, comments — additionally get a full lxml element tree via parse_xml → etree.fromstring on top of the retained blob). No layer in the chain bounds anything: the app performs no pre-check, gradio's upload route runs with max_file_size=None → math.inf (neither app.py's launch nor either notebook launch cell overrides it), CPython's zipfile has no decompression-bomb guard, and python-docx's opc layer contains no size/count/ratio check (the only gate, _ContentTypeMap.__getitem__, merely requires a Default/Override entry in the attacker-authored [Content_Types].xml). The single behavioral guard on the path — `except Exception` in _scan_docx — catches MemoryError only when the allocator refuses; with default Linux overcommit the multi-GB allocations succeed and the kernel OOM-killer SIGKILLs the whole process, which no except can intercept. Because deflate-compressed zeros expand ~1032:1 (and zipfile transiently costs ~2x a member's uncompressed size during chunk accumulation + join, with XML-typed members adding the element tree), a .docx of well under 16 MB can demand 10-16+ GB of RSS: the Colab free kernel (~12.7 GB) dies to a ~13 MB upload, larger hosts to proportionally modest uploads (zip64 members allow multi-GB single demands from sub-MB files). In cloud mode the victim process is the notebook kernel itself — the anonymous upload crashes the operator's whole session (all in-memory state, the public UI, any running generation); in local mode it is the app.py process that hosts the only user-facing surface. Sub-lethal bombs still cost seconds of CPU-bound inflation and severe memory pressure, and the attacker can retry unlimited times (every upload is a fresh event; no rate limiting, no lockout, no dedup). A second, click-gated copy of the same sink exists on the extract path (_extract_docx → Document(path) again on Preview/Generate), so hardening only the scan leg would not close the hole. This is threat-model asset A4 (availability) compromised by the in-model anonymous remote web attacker; the ODT leg (odfpy) and EPUB leg (ebooklib) are sibling surfaces of the same ingestion family.

## Attack

Primary attacker: the anonymous internet visitor of the public notebook tunnel URL (Colab/Kaggle cloud mode — the deployment the README advertises with "public shareable Gradio links"); the tunnel is a raw TCP forward with no filtering, the launch cells set no auth and no max_file_size. Secondary attacker (local mode): any co-tenant/process on the workstation or a page rendered in the operator's browser (the UI binds localhost but the app has no CSRF protection story for the upload endpoint). The attack is a single unauthenticated POST to the gradio upload endpoint followed by the automatically-firing upload event — no click, no valid document content, no user interaction beyond the attacker's own upload. Observable impact: the service process (notebook kernel in cloud mode) is OOM-killed (SIGKILL) mid-scan — in Colab/Kaggle the session dies with a crash message and all state is lost; the tunnel URL goes dead until the operator re-runs every notebook cell; repeated uploads keep it dead or, at sub-lethal sizes, pin the shared process in long CPU-bound inflations that starve all concurrent users. Weapon crafting needs only Python's zipfile (write members of zeros with ZIP_DEFLATED/zip64) — no special tooling.

### Payload

A crafted .docx zip: (1) [Content_Types].xml with a Default mapping for the bomb extension (e.g. Extension="bin" ContentType="application/octet-stream") plus an Override typing word/document.xml as application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml; (2) _rels/.rels with an officeDocument relationship to word/document.xml and N additional relationships (any reltype, e.g. image) targeting distinct members media/bomb{i}.bin; (3) word/document.xml — a minimal valid one suffices, because the detonation happens inside Package.open before any paragraph iteration; (4) each media/bomb{i}.bin a deflate-compressed run of zeros (e.g. 500 MB uncompressed ≈ 500 KB compressed). Total: a file of a few MB demanding N x 500 MB of RSS — e.g. 25 members x 512 MB = ~12.8 GB from a ~13 MB upload, sized by the attacker to the target's RAM. Optional amplifiers: type bomb members as WML XML content types with well-formed huge XML (e.g. <r> plus billions of <a/> children) to add the lxml element-tree multiplier; exploit the ~2x transient of zipfile's chunk-join; use zip64 for multi-GB single members. The exact same archive also detonates on the extract path when the attacker (or a victim they influence) clicks Preview/Generate.

## Data flow

### Step 1 — `app.py:168-170`

book_file = gr.File(file_types=[".epub", ".mobi", ".pdf", ".docx", ".odt", ".txt"]) — extension-only filter; .docx explicitly allowed; gradio enforces a filename-suffix match at preprocess (gradio_client.utils.is_valid_file), content is never inspected.

### Step 2 — `app.py:756-768`

book_file.upload(on_book_upload, inputs=[book_file, json_selected_chapters_state], outputs=[...]) — the event fires on upload completion; zero interaction beyond the upload itself.

### Step 3 — `app.py:635-656`

on_book_upload resolves the cached upload path (file_obj.name, the gradio upload-cache copy of the attacker's file) and calls result = scan(path) synchronously in the app process — in cloud mode a worker thread of the notebook kernel process itself.

### Step 4 — `audiobook_factory/text_extractor.py:79-95,199-201`

_detect_type maps the .docx extension to "docx" (magic-byte fallback not consulted) and scan() dispatches to _scan_docx(path).

### Step 5 — `audiobook_factory/text_extractor.py:270-281`

_scan_docx: `import docx2txt` (273) is dead code — never called; the live call is `d = _DocxDoc(path)` (276), i.e. python-docx Document(path). The `except Exception` (280) is the only guard and only catches MemoryError (not SIGKILL).

### Step 6 — `<site-packages>/docx/api.py:29-33 → docx/opc/package.py:124-128 (python-docx 1.2.0)`

Document(docx) → Package.open(docx) → PackageReader.from_file(pkg_file) → Unmarshaller.unmarshal(...) — the whole package is loaded before Document() returns.

### Step 7 — `<site-packages>/docx/opc/pkgreader.py:45-84 (python-docx 1.2.0)`

_load_serialized_parts drives _walk_phys_parts, which walks the attacker-authored relationship graph (_rels/.rels, then each part's _rels/*.rels) and for every non-external, not-yet-visited relationship target executes `blob = phys_reader.blob_for(partname)` — eager full decompression of every relationship-reachable member, no reltype filter, no part-count or total-size cap; visited_partnames only deduplicates partnames. All blobs are retained in the returned _SerializedPart tuple.

### Step 8 — `<site-packages>/docx/opc/phys_pkg.py:76-83 (python-docx 1.2.0)`

_ZipPkgReader.blob_for() → `return self._zipf.read(pack_uri.membername)` — whole-member inflate into memory; CPython zipfile has no decompression-bomb guard and transiently costs ~2x the member's uncompressed size during chunk accumulation + b"".join; the maximum inflation is bounded only by the attacker-declared member file_size in the zip central directory.

### Step 9 — `<site-packages>/docx/opc/package.py:195-205 + docx/opc/part.py:85-87,165-205,230-233 + docx/__init__.py:44-51 + docx/oxml/parser.py:19,29-32 (python-docx 1.2.0)`

_unmarshal_parts constructs one Part per blob while the PackageReader still holds every blob: default Part retains the raw bytes (self._blob); WML-XML content types (document/styles/settings/numbering/header/footer/comments) map to XmlPart subclasses whose load() calls parse_xml(blob) → etree.fromstring — a full lxml element tree on top of the retained blob. oxml_parser is configured resolve_entities=False, so the XML-entity amplification class is closed — the vector is purely archive inflation (plus the optional element-tree multiplier), which is unbounded.

### Step 10 — `audiobook_factory/text_extractor.py:280 + deployment topology (AudiobookMaker_Colab.ipynb cells 14/16, AudiobookMaker_Kaggle.ipynb cells 16/18; app.py:1908-1916)`

Impact materialization: with default Linux overcommit the multi-GB allocations succeed and the OOM-killer SIGKILLs the process hosting the handler — in cloud mode the notebook kernel itself (UI launched in-process; the FastAPI backend is only a subprocess), in local mode the app.py process hosting the only user-facing surface. `except Exception` cannot catch SIGKILL; sub-lethal bombs cause long CPU-bound inflation stalls and are retryable unlimited times (no rate limiting, no upload dedup, no session cost).

### Step 11 — `audiobook_factory/text_extractor.py:463-476 (callers: app.py:1035 on_preview, app.py:1216/1638 on_generate)`

Second decompression sink on the click-gated extract path: _extract_docx calls `d = _DocxDoc(path)` (470) again — the same archive is re-inflated on Preview/Generate/Export even if the scan leg were hardened.

### Step 12 — `gradio 6.27.0 blocks.py:2812,2865,3000 + routes.py:2122-2127 (installed dependency)`

No upload-size backstop anywhere: Blocks.launch(max_file_size=None) is documented "If None, no limit is set" and the upload route computes max_file_size if not None else math.inf; neither app.py's demo.launch(...) nor either notebook launch cell passes max_file_size.

## Fix / patch notes

diff --git a/audiobook_factory/text_extractor.py b/audiobook_factory/text_extractor.py
--- a/audiobook_factory/text_extractor.py
+++ b/audiobook_factory/text_extractor.py
@@ -265,6 +265,42 @@ def _scan_pdf(path: str) -> ScanResult:
     except Exception:
         pass
     return ScanResult(file_type="pdf", has_toc=False, page_count=page_count)
 
 
+# Maximum uncompressed payload a DOCX (zip/OPC) archive may declare before
+# python-docx is allowed to open it. python-docx eagerly decompresses every
+# relationship-reachable part into memory (and builds an lxml tree for
+# XML-typed parts), so a small uploaded file can otherwise demand arbitrary
+# gigabytes of RAM (CWE-409 decompression bomb).
+_MAX_DOCX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024  # 512 MB
+
+
+def _assert_docx_safe(path: str) -> None:
+    """Reject zip-based documents whose declared uncompressed size is
+    unreasonable (decompression-bomb guard).
+
+    Reads only the zip central directory - no decompression. CPython's
+    zipfile never inflates beyond a member's declared file_size, so capping
+    the declared sizes bounds the memory python-docx can be made to
+    allocate.
+    """
+    import zipfile
+
+    with zipfile.ZipFile(path, "r") as z:
+        total = 0
+        for info in z.infolist():
+            total += info.file_size
+            if info.file_size > _MAX_DOCX_UNCOMPRESSED_BYTES:
+                raise ValueError(
+                    f"archive member {info.filename!r} declares "
+                    f"{info.file_size} bytes uncompressed "
+                    f"(limit {_MAX_DOCX_UNCOMPRESSED_BYTES})"
+                )
+        if total > _MAX_DOCX_UNCOMPRESSED_BYTES:
+            raise ValueError(
+                f"archive declares {total} bytes uncompressed in total "
+                f"(limit {_MAX_DOCX_UNCOMPRESSED_BYTES})"
+            )
+
+
 def _scan_docx(path: str) -> ScanResult:
@@ -271,6 +307,7 @@ def _scan_docx(path: str) -> ScanResult:
     page_count = 0
     try:
+        _assert_docx_safe(path)
         import docx2txt  # type: ignore
         # docx doesn't expose page count easily; count sections as proxy
         from docx import Document as _DocxDoc  # type: ignore
         d = _DocxDoc(path)
@@ -465,6 +502,7 @@ def _extract_docx(path, page_ranges, normalizer, log):
     try:
         from docx import Document as _DocxDoc  # type: ignore
     except ImportError:
         log("[ERROR] python-docx not installed.")
         return []
+    _assert_docx_safe(path)
     d    = _DocxDoc(path)

## References

- https://cwe.mitre.org/data/definitions/409.html
- https://cwe.mitre.org/data/definitions/400.html
- https://owasp.org/www-community/attacks/Zip_bomb
- https://docs.python.org/3/library/zipfile.html
- https://python-docx.readthedocs.io/en/latest/

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] ODT book upload zero-interaction scan path (Gradio book_file.upload → odfpy load() eager manifest-entry decompression): attacker-crafted .odt decompression bomb → pre-auth remote memory-exhaustion DoS of the whole shared service process
- key: `BUG-R2-C3-A1-H3`
- disclosure: owner_only
- cwe: CWE-409
- file: `app.py`

# ODT book upload zero-interaction scan path (Gradio book_file.upload → odfpy load() eager manifest-entry decompression): attacker-crafted .odt decompression bomb → pre-auth remote memory-exhaustion DoS of the whole shared service process

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C3-A1-H3
- **CWE:** CWE-409
- **CVSS:** 7.5 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H`)
- **EV priority:** P0
- **EV score:** 8
- **PoC status:** reproduced
- **EXP status:** pending
- **Affected versions:** Audited snapshot: v1.3.0 line (commit dc9aed2cac064f436310a35fdd069f34205fa687); the ODT scan/extract path (text_extractor.py _scan_odt/_extract_odt via odfpy) is present in current main; exact introduction version not established. Library scope: every published odfpy release including the latest 1.4.1 (requirements.txt leaves odfpy unpinned; 1.4.1 is what a fresh install resolves; the project has been dormant since 2019, no fixed release exists). Gradio scope: every version with the default max_file_size=None ('no limit'), which no shipped launch mode overrides (app.py local launch and the Colab/Kaggle notebook launch cells); the snapshot's launch kwargs (theme=/css=) require gradio >= 6.0.0 to launch unmodified, but the vulnerable upload→scan chain itself is version-independent.

## Exploitability rationale

R:N — one anonymous HTTP upload against the advertised public notebook tunnel (Colab/Kaggle notebooks expose the identical no-auth Gradio app; no authentication exists anywhere) is enough: the .upload event fires automatically, no further interaction, no credentials. E:D — the Book upload control is the application's step-1 primary input and the ODT scan runs on every upload in every deployment mode (local run.sh and the cloud notebooks); no option disables it. C:D — purely deterministic archive-inflation logic: deflate on zero bytes yields ~1000:1, zipfile imposes no ratio/total-size cap, and the attacker fully controls the declared member sizes and CRCs; no race or memory layout involved. I:D — denial of service: a ~10-16 MB upload reliably demands 10-16 GB of RSS in the single shared Python process (the root-owned notebook kernel in cloud mode), ending in an uncatchable OOM-killer SIGKILL of the whole service (every co-user's session dies with it), repeatable at will with cheap requests; alternatively a long memory-thrash stall followed by a caught MemoryError. No confidentiality or integrity impact. EV = 3+3+2+0 = 8 → P0.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 168 | `build_app` |
| `app.py` | 635 | `on_book_upload` |
| `app.py` | 656 | `on_book_upload` |
| `app.py` | 756 | `build_app` |
| `app.py` | 1908 | `build_app` |
| `audiobook_factory/text_extractor.py` | 79 | `_detect_type` |
| `audiobook_factory/text_extractor.py` | 197 | `scan` |
| `audiobook_factory/text_extractor.py` | 285 | `_scan_odt` |
| `audiobook_factory/text_extractor.py` | 290 | `_scan_odt` |
| `audiobook_factory/text_extractor.py` | 477 | `_extract_odt` |
| `audiobook_factory/text_extractor.py` | 485 | `_extract_odt` |
| `requirements.txt` | 12 | `—` |

## Background

AudiobookMaker is a self-hosted text-to-speech audiobook generator. Its primary interface is an unauthenticated Gradio web app (app.py) that the README and the bundled Colab/Kaggle notebooks deliberately expose to anonymous visitors through a public SSH tunnel pointing at localhost:7860; on those notebooks the Python process runs as root and hosts everything — the UI, the TTS models and every visitor's session — in one process. Converting a book starts with uploading it: the "Book" tab's gr.File control (app.py:168-171) accepts .epub/.mobi/.pdf/.docx/ .odt/.txt by extension only. An ODT file is an OpenDocument package, i.e. a ZIP archive whose META-INF/manifest.xml lists the package members. To show the book's metadata immediately after upload (zero further interaction), the upload event handler on_book_upload synchronously calls scan(path), which dispatches by extension to _scan_odt and loads the document with odfpy's odf.opendocument.load() (requirements.txt installs odfpy unpinned; 1.4.1 is the latest release and what a fresh install resolves). The same loader is used again on the extraction path when generation starts. odfpy's load() is an eager whole-package reader: it trusts the (attacker-authored) manifest and decompresses every listed member fully into memory, retaining the bytes on the returned document. Because nothing between the network upload and that reader bounds the uncompressed volume, the classic ZIP "decompression bomb" recipe applies unchanged: highly compressible members (deflate on zeros achieves ~1000:1) convert a small upload into a multi-gigabyte in-memory allocation.

## Description

The reachable chain: (1) book_file.upload(on_book_upload, ...) is wired at app.py:756-762 and fires automatically when a file lands; sync Gradio handlers execute in the server process via anyio.to_thread.run_sync (gradio queueing.py), so there is no subprocess isolation. (2) on_book_upload (app.py:635) calls scan(path) (app.py:656). (3) _detect_type (text_extractor.py:79-92) maps the .odt extension to the "odt" reader — content is never inspected. (4) scan() dispatches to _scan_odt (text_extractor.py:197,211,285), which calls odf.opendocument.load(path) (text_extractor.py:290). (5) Inside odfpy 1.4.1 load() (odf/opendocument.py:975-1012): z.read('META-INF/ manifest.xml') parses the manifest with no validation whatsoever (odf/odfmanifest.py:83-86 stores every manifest:file-entry verbatim, keyed by full-path; no count/size/path checks); __loadxmlparts (opendocument.py:872-908) then fully decompresses each listed settings/meta/content/styles.xml member (z.read + .decode, line 898) and SAX-parses it into a retained pure-Python element tree; finally the manifest-entry loop (opendocument.py:990-1009) eagerly decompresses and retains every remaining entry — Pictures/* via doc.addPicture(..., z.read(mentry)) (line 992; bytes stored in doc.Pictures, line 453), Thumbnails/thumbnail.png via doc.addThumbnail(z.read(mentry)) (line 994), and any other non-directory entry via OpaqueObject(..., z.read(mentry)) appended to doc._extra (line 1008; content kept, opendocument.py:95-112). All decompressed bombs therefore live simultaneously on the returned document. These reads happen before the trailing getElementsByType(Body) access (opendocument.py:1012), so even a structurally degenerate package detonates the full allocation loop; a minimal valid package (mimetype + manifest + tiny content.xml with an office:body/office:text) completes the clean path. (6) CPython's zipfile offers no decompression-bomb protection: ZipFile.read() decompresses the whole member into one bytes object; the only caps are the attacker-declared file_size (ZipExtFile._left) and the EOF CRC check — an attacker crafting the archive declares true sizes and correct CRCs, so a ~1000:1 deflated-zeros member set is unconstrained. (7) Impact in the single shared process: with Linux's default heuristic overcommit the genuinely-written decompressed bytes grow RSS until the OOM killer SIGKILLs the process — a signal _scan_odt's except Exception (text_extractor.py:294) cannot catch — killing the whole service (on the advertised notebooks: the root-owned kernel with every session); with strict overcommit the allocation fails and the raised MemoryError is caught by that same handler, but only after a long inflation stall that starves all concurrent requests. Each repeat upload re-arms the attack. A second detonation exists on the extraction path: _extract_odt (text_extractor.py:477-489) calls odf_load(path) again (line 485) plus per-paragraph str(p) re-serialization (line 486), and scan(path) is also re-invoked during generation for cover extraction (app.py:1724). No layer of the chain checks uncompressed size, compression ratio or member count: the app performs no size validation at all (no max_file_size in either launch mode — gradio's default is 'no limit', routes.py /upload receives math.inf; no getsize/setrlimit anywhere in the scan chain), and odfpy contains no such guard (1.4.1 is the latest release; the project is dormant, so no fixed version exists to inherit). odfpy does parse XML with defusedxml, which blocks DTD/entity amplification, but that protection does not extend to zip-member inflation, which is the vector here.

## Attack

An anonymous visitor of the public notebook tunnel URL (the sharing deployment the project advertises) — or anyone who can reach the Gradio port in any other exposure — uploads the crafted file through the Book tab's upload control (or posts it to the equally unauthenticated upload endpoint and triggers the book_file.upload event via the queue/REST API). No click or other interaction is needed: the upload event itself runs the scan synchronously in the service process. Within seconds the process's memory is exhausted; on the default Linux memory configuration the kernel OOM-kills the whole service process (the notebook kernel, terminating every user's session and all in-flight work), and the attacker can repeat the upload to keep the service dead. On strict-overcommit hosts the effect is a long memory-thrashing stall per request that starves all other users, with the eventual MemoryError swallowed by the scan's exception handler. A second trigger is available by letting the same file proceed to generation (extract path), which re-detonates the bomb and additionally re-serializes every paragraph.

### Payload

A ZIP archive named *.odt containing: a stored mimetype member with "application/vnd.oasis.opendocument.text"; a small META-INF/manifest.xml listing manifest:file-entry elements for "/", "content.xml" and N entries "Pictures/p<i>.png" (or any other non-directory names — the OpaqueObject branch accepts them all); a tiny valid content.xml with an office:body/office:text element so the load completes the clean path; and N deflated members of zero bytes (~1 MB compressed each, declared file_size ~1 GB each, correct CRCs). With deflate's ~1000:1 ratio on zeros, a ~10-16 MB file declares and produces 10-16 GB of decompressed bytes; N and member sizes scale the demand freely (a ~100 MB upload can demand 100+ GB).

## Data flow

### Step 1 — `app.py:168-171 (build_app)`

book_file = gr.File(file_types=[".epub", ".mobi", ".pdf", ".docx", ".odt", ".txt"]) — extension-only acceptance filter; no content or size inspection.

### Step 2 — `gradio 6.27.0 routes.py:1932-1952 (/upload) + blocks.py:2812,2865,3000`

Unauthenticated multipart upload endpoint; blocks.max_file_size is None in every shipped launch mode (app.py:1908-1915 and the Colab/Kaggle launch cells pass no max_file_size), so the effective cap is math.inf — the crafted ~10-16 MB .odt is accepted.

### Step 3 — `app.py:756-762 (build_app)`

book_file.upload(on_book_upload, ...) — the event fires automatically when the upload lands (zero further interaction).

### Step 4 — `app.py:635,656 (on_book_upload) + gradio queueing.py:17,912`

on_book_upload resolves the uploaded file's server-side path and synchronously calls scan(path); sync handlers run in-process via anyio.to_thread.run_sync — no subprocess isolation between the scan and the shared service process (the notebook kernel in cloud mode).

### Step 5 — `audiobook_factory/text_extractor.py:79-92,197,211 (_detect_type / scan)`

Extension dispatch: Path(path).suffix.lower() == ".odt" → "odt" → _scan_odt(path); content never inspected.

### Step 6 — `audiobook_factory/text_extractor.py:285-297 (_scan_odt)`

doc = odf_load(path) (line 290) — odf.opendocument.load() is called with no size/ratio/member-count guard; the trailing except Exception (line 294) catches MemoryError but not the OOM killer's SIGKILL.

### Step 7 — `odfpy 1.4.1 odf/opendocument.py:975-1012 (load)`

z.read('META-INF/manifest.xml') (987) + manifestlist (988) parse the fully attacker-authored manifest (odf/odfmanifest.py:83-86 stores every file-entry verbatim); __loadxmlparts (989; body 872-908) fully decompresses settings/meta/content/styles.xml (z.read + decode at 898) into retained pure-Python element trees.

### Step 8 — `odfpy 1.4.1 odf/opendocument.py:990-1009 (load manifest-entry loop)`

Every remaining manifest entry is eagerly decompressed and retained: Pictures/* → doc.addPicture(..., z.read(mentry)) (992; stored in doc.Pictures at 453); Thumbnails/thumbnail.png → doc.addThumbnail(z.read(mentry)) (994); any other non-directory entry → OpaqueObject(..., z.read(mentry)) appended to doc._extra (1008; content kept, 95-112). Reads occur before the trailing getElementsByType(Body) (1012), so the allocation loop runs even for degenerate packages.

### Step 9 — `CPython zipfile (ZipFile.read / ZipExtFile)`

ZipFile.read(name) = whole-member decompression into one bytes object; the only caps are the attacker-declared file_size (ZipExtFile._left, truncation at declared size) and the EOF CRC check — the attacker declares true uncompressed sizes (~1 GB per ~1 MB deflated-zeros member) and correct CRCs, so ~1000:1 amplification is unconstrained. Peak in-memory demand = sum of all manifest-listed members' uncompressed sizes (all retained simultaneously on the returned document).

### Step 10 — `process memory / Linux OOM killer`

A ~10-16 MB .odt with N×~1 GB deflated-zeros members demands 10-16 GB RSS in the shared process → OOM-killer SIGKILL (uncatchable; whole-service loss on the advertised notebooks, repeatable at will) or, under strict overcommit, a caught MemoryError after a long inflation stall that starves all concurrent requests.

### Step 11 — `audiobook_factory/text_extractor.py:303,327,421,477-489 (extract / _extract_paged / _extract_odt) + app.py:1035,1216,1638,1724`

Second detonation on the extraction path: the Generate flow calls extract() → _extract_paged → _extract_odt, which calls odf_load(path) again (485) plus per-paragraph str(p) re-serialization (486); the cover-embedding step re-invokes scan(path) (app.py:1724).

## Fix / patch notes

diff --git a/audiobook_factory/text_extractor.py b/audiobook_factory/text_extractor.py
--- a/audiobook_factory/text_extractor.py
+++ b/audiobook_factory/text_extractor.py
@@ -282,6 +282,29 @@ def _scan_docx(path: str) -> ScanResult:
     return ScanResult(file_type="docx", has_toc=False, page_count=page_count)


+_ZIP_MAX_UNCOMPRESSED = 1 << 30  # 1 GiB total uncompressed payload cap
+_ZIP_MAX_MEMBERS = 2000
+
+
+def _assert_zip_safe(path: str) -> None:
+    """Reject archive-based documents whose declared uncompressed payload
+    exceeds sane bounds BEFORE any eager whole-archive reader (odfpy
+    load(), and the ebooklib/python-docx readers that share this pattern)
+    decompresses it into memory. zipfile truncates member reads at the
+    declared file_size and verifies the CRC at EOF, so the declared total
+    is an upper bound on what a zipfile-based reader can allocate."""
+    import zipfile
+    with zipfile.ZipFile(path) as z:
+        infos = z.infolist()
+        if len(infos) > _ZIP_MAX_MEMBERS:
+            raise ValueError(f"too many archive members: {len(infos)}")
+        total = sum(i.file_size for i in infos)
+        if total > _ZIP_MAX_UNCOMPRESSED:
+            raise ValueError(
+                f"uncompressed payload too large: {total} bytes")
+
+
 def _scan_odt(path: str) -> ScanResult:
     page_count = 0
     try:
+        _assert_zip_safe(path)
         from odf.opendocument import load as odf_load  # type: ignore
         from odf.text import P  # type: ignore
         doc = odf_load(path)
@@ -479,6 +502,10 @@ def _extract_odt(path, page_ranges, normalizer, log):
     except ImportError:
         log("[ERROR] odfpy not installed.")
         return []
+    try:
+        _assert_zip_safe(path)
+    except Exception as e:
+        log(f"[ERROR] Rejected unsafe ODT archive: {e}")
+        return []
     doc  = odf_load(path)
     raw  = "\n\n".join(str(p) for p in doc.text.getElementsByType(P))

## References

- https://cwe.mitre.org/data/definitions/409.html
- https://cwe.mitre.org/data/definitions/400.html
- https://owasp.org/www-community/attacks/Zip_bomb
- https://docs.python.org/3/library/zipfile.html
- https://github.com/eea/odfpy

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] PDF chapter extractor (_extract_pdf_ranges, text_extractor.py) processes anonymous uploads with no bound on extracted-text volume, page-range count or page count — pre-auth memory-exhaustion (OOM-kill of the shared web process) and CPU/queue-wedging denial of service on the public tunnel deployment
- key: `BUG-R2-C3-A3-H1`
- disclosure: owner_only
- cwe: CWE-400
- file: `audiobook_factory/text_extractor.py`

# PDF chapter extractor (_extract_pdf_ranges, text_extractor.py) processes anonymous uploads with no bound on extracted-text volume, page-range count or page count — pre-auth memory-exhaustion (OOM-kill of the shared web process) and CPU/queue-wedging denial of service on the public tunnel deployment

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C3-A3-H1
- **CWE:** CWE-400
- **CVSS:** 7.5 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H`)
- **EV priority:** P1
- **EV score:** 7
- **PoC status:** pending
- **EXP status:** pending
- **Affected versions:** audited snapshot (commit dc9aed2cac064f436310a35fdd069f34205fa687, v1.3.0 line); exact introduction version not established. The missing bounds are in the project's own code and are independent of library versions (vector 2/3); vector 1's amplification ratio depends on PyMuPDF/mupdf text extraction (requirements.txt pins no pymupdf version; all currently shipping release lines behave as described). Gradio upload-size unlimiting applies to the unpinned gradio resolution (verified wheel-level on 4.44.1, 5.0.0, 6.27.0).

## Exploitability rationale

R:N (remote, anonymous) — the sink is reached from the primary anonymous surface: the Colab/Kaggle notebooks (the advertised deployment) expose the exact same no-auth Gradio app through a public Pinggy TCP tunnel (ssh -R 0:localhost:7860 free.pinggy.io); the trigger is one file upload plus one button click (Preview / Generate / Export config) with no authentication anywhere. E:D (default) — the Book-tab upload + page-range flow is the application's core UI path, present and enabled in every deployment mode; every PDF necessarily takes the page-range branch (_scan_pdf always returns has_toc=False), and an empty range string already selects the whole-document extraction branch. C:D (deterministic) — vector 2 (the range-count multiplier) is pure project logic: an unbounded comma list of ranges, each content-clamped to the whole document, with every produced chapter retained in memory; no race, no memory layout, no version dependence — the attacker picks the multiplier k exactly. Vectors 1 (form-XObject text amplification, ratio ~10^4-10^5:1 mechanism-certain, exact ratio pending POC) and 3 (declared page count → per-page interpretation loop) amplify further from KB-MB-scale files. I:D (disruption) — maximal impact is complete availability loss of the single shared service process: peak RSS of several times the attacker-amplified text kills the whole notebook session (kernel OOM-kill, SIGKILL, uncatchable by the app's exception handling) for every user of the instance, repeatable at will and unthrottled; below the kill threshold the extraction wedges the Gradio event queue for the duration. No confidentiality or integrity impact.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `audiobook_factory/text_extractor.py` | 425 | `_extract_pdf_ranges` |
| `audiobook_factory/text_extractor.py` | 440 | `_extract_pdf_ranges` |
| `audiobook_factory/text_extractor.py` | 447 | `_extract_pdf_ranges` |
| `audiobook_factory/text_extractor.py` | 448 | `_extract_pdf_ranges` |
| `audiobook_factory/text_extractor.py` | 452 | `_extract_pdf_ranges` |
| `audiobook_factory/text_extractor.py` | 258 | `_scan_pdf` |
| `app.py` | 168 | `build_app` |
| `app.py` | 202 | `build_app` |
| `app.py` | 1026 | `on_preview` |
| `app.py` | 1090 | `on_generate` |
| `app.py` | 1095 | `on_generate` |
| `app.py` | 1627 | `on_export_config` |
| `app.py` | 1232 | `on_generate` |
| `app.py` | 1306 | `on_generate` |
| `audiobook_factory/pipeline.py` | 384 | `preview_chapters` |
| `audiobook_factory/extractor_engine.py` | 468 | `TextNormalizer.normalize` |
| `audiobook_factory/text_processing.py` | 54 | `smart_sentence_splitter` |

## Background

AudiobookMaker is a self-hosted text-to-speech audiobook generator whose primary interface is an unauthenticated Gradio web app (app.py). The README and the bundled Colab/Kaggle notebooks deliberately expose this app to anonymous visitors through a public SSH tunnel (pinggy.io) forwarding to localhost:7860; on those notebooks the web UI, the FastAPI orchestrator and the torch/TTS stack all live in one Python process (the notebook kernel), which is the machine's main memory consumer. The Book tab accepts ebook uploads (.epub/.mobi/.pdf/.docx/.odt/.txt). PDFs have no chapter structure in this app, so the fast pre-scan (scan → _scan_pdf) only reads the page count and the UI switches to a free-form "page ranges" textbox: each comma-separated range like "1-50" becomes one chapter, and an empty box means "extract the whole document". Extraction runs PyMuPDF (fitz) — a C library (mupdf) interpreting the PDF content streams — then normalizes the text with a chain of full-string regex passes and splits it into a sentence list, all inside the shared web process. This is the trust boundary: fully attacker-controlled file bytes and form strings enter a native parser and unbounded Python string processing with no authentication, no upload-size cap and no resource budgets.

## Description

The PDF extraction sink _extract_pdf_ranges (audiobook_factory/ text_extractor.py:425-460) enforces no resource bound of any kind. (1) Whole-document branch (line 440): all_text = join(doc[i].get_text() for i in range(doc.page_count)) — every declared page is interpreted and the full text accumulated, with no cap on pages or characters; this branch is selected automatically whenever the range box is empty or contains no valid entry (app.py:1095-1096 sets page_ranges = None; the sink's "if not page_ranges" covers both None and []). (2) Range branch (lines 447-457): the loop iterates the caller-supplied list of (start, end) tuples. Each range is only content-clamped — range(max(0, start-1), min(end, doc.page_count)) — so "1-99999" means "the whole document"; the number of ranges, their duplication and the total extracted volume are unchecked, and every produced chapter (text + sentence list) is retained in results and returned. The range list itself is parsed in the UI layer from an attacker-authored free-form string with an unbounded comma-split loop in three handlers (app.py:1026, 1090, 1627); the textbox has no max_length and gradio performs no server-side string validation. (3) Amplification inside the PDF: per PDF 32000-1 §8.8, a form XObject invoked via /Do executes its content stream at each point of use. mupdf's interpreter re-executes the XObject on every invocation and its structured-text device accumulates every shown character — no deduplication, no output-size cap, only nested-depth limits. A ~100 KB structurally valid PDF (one XObject with a long Tj and a base-14 font, referenced 10^5 times via /Do; the repetition flate-compresses to kilobytes) therefore extracts to gigabytes of text — a decompression-bomb analogue executed by the legitimate parser. Independently, a PDF page tree of millions of minimal page objects (stored in compressed object streams) makes the whole-document loop burn hours of CPU. (4) Downstream multipliers turn T bytes of extracted text into several×T of live memory: TextNormalizer.normalize (extractor_engine.py:468-508) runs _remove_duplicate_title (a full text.split("\n") line-list copy), six-plus full-string re.sub passes and two callback-regex passes — peak 2-3× T simultaneously alive; smart_sentence_splitter (text_processing.py:54-80, default max_len=399) adds a paragraph-split copy plus a list-of-strings second copy (~1.1-1.5× T with ~56 bytes object overhead per sentence) plus pure-Python NLTK punkt CPU (~1 MB/s on GB text); the sentence list is retained per chapter; on Preview the pipeline re-splits every chapter's full text a second time just to count sentences (pipeline.py:384-421); on Generate the chapters are re-serialized into JSON for the FastAPI dispatch (app.py:1232-1245) and persisted (text + sentences) into generation_progress.json on disk (app.py:1806-1824). With k duplicate whole-document ranges the retained state is k × (2-2.5) × document text — attacker-chosen. On the advertised single-process Colab/Kaggle deployment (12-16 GB RAM with torch models resident), a few GB of amplified text pushes the kernel past available RAM: the Linux OOM killer SIGKILLs the whole notebook session — an outcome no exception handler can catch — or, below the kill threshold, the extraction wedges the Gradio queue (default queue serializes same-event handlers; on_generate's extraction runs in a detached daemon thread unaffected by client disconnect). Uploads are unlimited in size (gradio max_file_size defaults to None on 4.44.1/5.0.0/6.27.0; neither launch mode overrides it), there is no rate limiting, no timeout and no worker cap, so the attack is repeatable at will by any anonymous visitor of the public tunnel URL, denying service to every user of the instance.

## Attack

Attacker: any anonymous visitor of the public pinggy.io URL of a Colab/Kaggle-deployed AudiobookMaker instance (the deployment mode the README advertises and the notebooks automate; no authentication exists). Step 1: in the Book tab, upload the payload PDF (accepted by the file-type filter; the automatic pre-scan only reads the page count and imposes no gate). Step 2: optionally type the range-multiplier string into the "Page ranges" textbox (or leave it empty for the whole-document branch). Step 3: click "Preview" (or Generate / Export config). The handler parses the ranges and calls extract() → _extract_pdf_ranges in the shared web process; extraction, normalization and sentence-splitting run with no budget. Observable effect within seconds to minutes: the process RSS climbs to several times the extracted text; on a 12-16 GB notebook VM with torch resident the kernel is OOM-killed — the whole session (web UI, API, in-flight generations of other users) dies and must be restarted; on a larger host the extraction instead pins a CPU core for hours (re-extraction per range + regex/punkt passes) while the Gradio queue serializes behind it, blocking every other visitor's interactions. The attacker repeats at will (unthrottled uploads and events; on_generate even spawns a new daemon thread per click). Premises: only knowledge of the public URL; no credentials, no user interaction, no special library versions (the range-multiplier and page-count vectors are pure project code).

### Payload

Three payload shapes, all anonymous and pre-auth. (a) Text bomb: a structurally valid ~100 KB PDF containing one form XObject whose content stream shows a long text string (e.g. 'A'×10000 via Tj with the base-14 Helvetica font, no embedding) and a page content stream invoking that XObject 10^5 times via /Fm0 Do — flate compression reduces the repeated references to kilobytes; leave the page-range box empty (whole-document branch) or set "1-1". Expected extraction: ~10^9 characters (~1 GB) per page from a 100 KB file; multiple pages or higher invocation counts scale linearly. (b) Range multiplier, no PDF crafting: any ordinary text-rich PDF (tens to hundreds of MB — uploads are unlimited) plus the range string "1-999999," repeated k times (e.g. k=100, a 900-byte string): every entry clamps to the whole document, producing k retained full-document chapters (each also normalized and sentence-split) — memory and CPU multiply by exactly k. (c) Page-count CPU bomb: a PDF whose page tree declares millions of minimal pages (page dictionaries inside compressed object streams; a file of a few hundred MB declares ~10^7 pages) with an empty range box — the whole-document loop interprets every page. All three are delivered through the normal UI (upload → optional range string → click Preview / Generate / Export config) or the unauthenticated gradio REST queue API.

## Data flow

### Step 1 — `app.py:168-171 (build_app)`

Anonymous upload source: gr.File(file_types=[..., ".pdf", ...]) accepts arbitrary PDF bytes; gradio upload size is unlimited (max_file_size defaults to None on 4.44.1/5.0.0/6.27.0 and neither launch mode overrides it).

### Step 2 — `app.py:635-652 (on_book_upload) → audiobook_factory/text_extractor.py:258-266 (_scan_pdf)`

Automatic post-upload scan does only fitz.open + doc.page_count — no page-count ceiling or content gate — and sets scan_state with has_toc=False, switching the UI to page-range mode for every PDF.

### Step 3 — `app.py:202-207 (build_app)`

page_ranges_box = gr.Textbox(...) — attacker-authored free-form string; lines=2 is display-only, no max_length; no server-side string validation on any gradio version (settled matrix), including via the unauthenticated REST queue API.

### Step 4 — `app.py:1026-1034 (on_preview), app.py:1089-1096 (on_generate), app.py:1626-1634 (on_export_config)`

Three handlers share the identical unbounded parse loop `for part in page_ranges_str.split(",")` appending (int(s), int(e)) per entry — range count and duplicates unchecked; empty/invalid string yields page_ranges=None → whole-document branch (app.py:1095-1096).

### Step 5 — `audiobook_factory/text_extractor.py:435-440 (_extract_pdf_ranges)`

Sink, whole-document branch: doc = fitz.open(path); all_text = "\n\n".join(doc[i].get_text("text") for i in range(doc.page_count)) — every declared page interpreted, full text accumulated, no cap on pages or chars. A form XObject invoked N times via /Do re-executes its content each time (PDF 32000-1 §8.8; mupdf re-runs the stream, stext device accumulates with no dedup/output cap) → N× text from one XObject; a page tree of millions of minimal pages → one page-load + interpretation per iteration.

### Step 6 — `audiobook_factory/text_extractor.py:447-457 (_extract_pdf_ranges)`

Sink, range branch: for idx, (start, end) in enumerate(page_ranges, 1) — each range only content-clamped (range(max(0, start-1), min(end, doc.page_count))), so "1-99999" = whole document; k duplicate ranges re-extract the document k times and every chapter is appended to results and retained — count/total-volume multiplier fully attacker-chosen.

### Step 7 — `audiobook_factory/text_extractor.py:441/450 → audiobook_factory/extractor_engine.py:451-508 (TextNormalizer.normalize)`

Full-string normalization passes over the accumulated text: _remove_duplicate_title does text.split("\n") (full line-list copy), then ~6 re.sub passes (_strip_noise), 2 callback-regex passes (_fix_isolated_capitals), 2 more in normalize_text — peak keeps 2-3× the text alive simultaneously.

### Step 8 — `audiobook_factory/text_extractor.py:443/455 → audiobook_factory/text_processing.py:54-80 (smart_sentence_splitter)`

Second full-size copy: paragraph split + NLTK punkt sent_tokenize (pure Python, ~1 MB/s — hours on GB text) + list-of-strings ≈ 1.1-1.5× text with ~56 bytes per-object overhead; the sentence list is retained inside every ExtractedChapter (results list grows per range).

### Step 9 — `app.py:1042 → audiobook_factory/pipeline.py:384-421 (preview_chapters)`

Preview path re-splits every chapter's full text a second time (smart_sentence_splitter(ch.text, 9999)) just to count sentences — a third full-text CPU/memory pass per click.

### Step 10 — `app.py:1208-1245 (on_generate/_runner)`

Generate path: extraction runs in a detached daemon thread (app.py:1306, one per click, unbounded, unaffected by client disconnect); the accumulated chapters are re-serialized into JSON for requests.post(json=payload) (another full copy) or passed to run_pipeline; generation_progress.json persists every chapter's text + sentences to disk (app.py:1806-1824).

### Step 11 — `deployment impact (AudiobookMaker_Colab.ipynb / AudiobookMaker_Kaggle.ipynb launch cells)`

All of the above runs in the single shared process (notebook kernel) hosting torch + the TTS stack behind the public Pinggy tunnel: peak RSS of several × amplified text exceeds available RAM on the advertised 12-16 GB VMs → kernel OOM-kill (SIGKILL, uncatchable) of the whole session for every user; below the threshold the extraction wedges the Gradio queue, blocking other visitors; attack is anonymous, pre-auth, and repeatable at will (no rate limit / timeout / worker cap).

## Fix / patch notes

diff --git a/audiobook_factory/text_extractor.py b/audiobook_factory/text_extractor.py
--- a/audiobook_factory/text_extractor.py
+++ b/audiobook_factory/text_extractor.py
@@ -426,6 +426,14 @@
     from audiobook_factory.extractor_engine import TextNormalizer  # type: ignore
     from audiobook_factory.text_processing import smart_sentence_splitter

+    # ── Resource-safety caps (pre-auth anonymous surface) ──────────────────
+    # Bounds total interpreted pages, retained chapters and extracted
+    # characters so that neither a text-amplifying PDF (form-XObject reuse)
+    # nor a duplicated/oversized page-range list can exhaust the process.
+    MAX_RANGES        = 64
+    MAX_PAGES         = 5_000
+    MAX_EXTRACT_CHARS = 20_000_000
+
     try:
         import fitz
     except ImportError:
@@ -434,20 +442,49 @@

     doc = fitz.open(path)
     results = []
+    pages_left = MAX_PAGES
+    chars_left = MAX_EXTRACT_CHARS
+
+    def _page_text(i):
+        nonlocal chars_left
+        t = doc[i].get_text("text")
+        chars_left -= len(t)
+        return t

     if not page_ranges:
         # Whole document
-        all_text = "\n\n".join(doc[i].get_text("text") for i in range(doc.page_count))
+        parts = []
+        for i in range(min(doc.page_count, pages_left)):
+            parts.append(_page_text(i))
+            if chars_left <= 0:
+                log("[WARN] Extracted-text limit reached — truncating extraction.")
+                break
+        all_text = "\n\n".join(parts)
+        del parts
         text = normalizer.normalize(all_text, title="", ocr_block_texts=[])
+        del all_text
         results.append(ExtractedChapter(
             num=1, title="Full Book", text=text,
             sentences=smart_sentence_splitter(text)
         ))
     else:
-        for idx, (start, end) in enumerate(page_ranges, 1):
+        seen: set[tuple[int, int]] = set()
+        idx = 0
+        for start, end in page_ranges:
+            if idx >= MAX_RANGES or pages_left <= 0 or chars_left <= 0:
+                log("[WARN] Page-range limits reached — stopping extraction.")
+                break
+            if (start, end) in seen:   # identical ranges are pure duplicates
+                continue
+            seen.add((start, end))
             pages = range(max(0, start - 1), min(end, doc.page_count))
-            raw   = "\n\n".join(doc[i].get_text("text") for i in pages)
+            if len(pages) > pages_left:
+                pages = pages[:pages_left]
+            pages_left -= len(pages)
+            raw   = "\n\n".join(_page_text(i) for i in pages)
+            idx  += 1
             text  = normalizer.normalize(raw, title="", ocr_block_texts=[])
+            del raw
             results.append(ExtractedChapter(
                 num=idx,
                 title=f"Chapter {idx} (pp. {start}–{end})",
@@ -455,6 +492,9 @@
                 sentences=smart_sentence_splitter(text),
             ))
             log(f"  ✓ Extracted pages {start}–{end}")
+            if pages_left <= 0 or chars_left <= 0:
+                log("[WARN] Page/extracted-text limits reached — stopping extraction.")
+                break

     doc.close()
     return results

## References

- https://cwe.mitre.org/data/definitions/400.html
- https://owasp.org/www-community/attacks/Resource_consumption
- https://iso.org/standard/75839.html (ISO 32000-1, §8.8 Form XObjects / §8.10.2 Do operator — each invocation paints the form again)
- https://mupdf.com/docs (structured text extraction via fz_new_stext_device — accumulates every shown character, no deduplication or output cap)
- https://www.gradio.app/guides/blocks-and-functions (launch(max_file_size=...) defaults — unlimited uploads when unset)

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] EPUB chapter-extraction docling HTML table path (extract → _docling_html → DocumentConverter.convert → HTMLDocumentBackend.parse_table_data): unbounded colspan/rowspan-driven table-grid allocation — anonymous ~2 KB crafted-EPUB upload plus one click OOM-kills the shared web process (pre-auth remote DoS)
- key: `BUG-R2-C3-A4-H1`
- disclosure: owner_only
- cwe: CWE-400
- file: `app.py`

# EPUB chapter-extraction docling HTML table path (extract → _docling_html → DocumentConverter.convert → HTMLDocumentBackend.parse_table_data): unbounded colspan/rowspan-driven table-grid allocation — anonymous ~2 KB crafted-EPUB upload plus one click OOM-kills the shared web process (pre-auth remote DoS)

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R2-C3-A4-H1
- **CWE:** CWE-400
- **CVSS:** 7.5 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H`)
- **EV priority:** P0
- **EV score:** 8
- **PoC status:** reproduced
- **EXP status:** pending
- **Affected versions:** any release shipping the Gradio book-upload extract path feeding docling (app.py on_preview/on_generate/on_export_config → extract() → DocumentIngestor._docling_html → DocumentConverter.convert); requirements.txt declares docling unpinned — verified live in the POC sandbox: pip resolves docling 2.127.0 today, and the PyPI docling 2.127.0 package is a metapackage depending on docling-slim[standard]==2.127.0 (installed stack: docling-slim 2.127.0 + docling-core 2.96.1); the unbounded span→grid arithmetic is present in every docling release examined — 2.0.0 through 2.127.0 (current latest) — with no patched version in existence; current source snapshot dc9aed2 confirmed affected (dynamically: unmodified app killed 3× by the same 1,599-byte upload)

## Exploitability rationale

R:N (remote): the README-advertised primary deployment is the Colab/Kaggle notebook, whose launch cell runs the Gradio UI in-process in the notebook kernel and forwards port 7860 through a public anonymous Pinggy tunnel — the anonymous internet visitor is threat-model attacker A-Remote-Web ("can upload any 'book'"); the trigger click is performed by the attacker on their own upload, so no victim interaction exists. Local-mode deployments (server_name="localhost") degrade reachability to adjacent/local only. E:D (default): book extraction is the core advertised feature and docling is a hard dependency of the EPUB leg (requirements.txt), so the sink is installed and reachable in every shipped launch mode (local app.py, Colab, Kaggle, CLI); requirements.txt declares docling unpinned, resolving to docling/docling-slim 2.127.0 + docling-core 2.96.1 today. C:D (deterministic): the kill is pure arithmetic — attribute values sum directly into num_rows/num_cols and then into the grid allocation, with no race, timing, or memory-layout dependence; the only environmental variable is available RAM, which the attacker scales to at will (each ~30-byte table row buys ~400 MB of allocation) on publicly documented host sizes; the multi-row allocation pattern stays below Linux overcommit refusal thresholds by construction, so the uncatchable OOM-kill regime is reliably entered. I:D (disruption): maximal impact class is whole-process availability loss — kernel OOM-kill (SIGKILL, uncatchable by the app's except Exception) of the single process hosting the UI, torch/TTS stack, and all user sessions; in cloud mode that process is the notebook kernel itself (operator's session dies); unlimited anonymous retries; sub-lethal sizes still wedge the request thread for minutes via span-scaled fill loops. No code execution or information disclosure on this path.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 168 | `build_app (book_file gr.File component, extension filter only)` |
| `app.py` | 1035 | `on_preview → extract() (attack trigger; on_generate:1216 and on_export_config:1638 equivalent)` |
| `audiobook_factory/text_extractor.py` | 349 | `_extract_epub → ingestor.ingest_epub()` |
| `audiobook_factory/extractor_engine.py` | 525 | `DocumentIngestor.__init__ (default DocumentConverter(), no limits)` |
| `audiobook_factory/extractor_engine.py` | 699 | `_docling_html → self._converter.convert(tmp_path)` |
| `audiobook_factory/extractor_engine.py` | 701 | `_docling_html → doc.export_to_markdown() (secondary grid materialization)` |
| `audiobook_factory/extractor_engine.py` | 709 | `_docling_html → doc.export_to_dict() (secondary grid serialization)` |
| `audiobook_factory/extractor_engine.py` | 724 | `_docling_html exception fallback (catches MemoryError; cannot catch OOM SIGKILL)` |
| `audiobook_factory/extractor_engine.py` | 772 | `_group_spine_by_chapter ("name in h" chapter-group reachability condition)` |
| `audiobook_factory/extractor_engine.py` | 868 | `ingest_epub per-file loop → _docling_html (runs before Phase-3 classification)` |
| `audiobook_factory/extractor_engine.py` | 882 | `ingest_epub (merged_ir = ir_dict, IR retention for whole run)` |
| `requirements.txt` | 9 | `unpinned docling dependency` |

## Background

AudiobookMaker is an end-to-end AI audiobook generator: a user uploads a book (EPUB/DOCX/PDF/...), the app extracts and classifies chapters, normalizes the text, and synthesizes audio with a local TTS stack. It ships a Gradio web UI plus a FastAPI backend, and its README advertises Google-Colab/Kaggle notebook deployments, where the UI runs inside the notebook kernel and is forwarded to the public internet through an anonymous Pinggy SSH reverse tunnel (ssh -p 443 -R 0:localhost:7860 free.pinggy.io) — so the "user" of the UI can be any anonymous internet visitor, and the UI process is also the process hosting the torch/TTS stack and all session state.

For EPUB inputs, chapter extraction goes through docling: each spine document item's HTML body is preprocessed with BeautifulSoup, written to a temp .html file, and converted in-process by docling's HTML backend (DocumentConverter.convert) inside the Gradio request thread. docling's HTML table support derives a table's grid dimensions (num_rows, num_cols) directly from the colspan/rowspan attribute values found in the markup, and materializes a num_rows × num_cols Python list-of-lists grid before processing any cell content. Those attributes are ordinary attacker-controlled integers in crafted input: there is no bound on them anywhere in docling (all releases to date) or in the app, and the app configures no resource limits, timeouts, or process isolation around the conversion. The relevant security property is therefore an amplification from a ~2 KB uploaded file to an exactly attacker-sized multi-GB in-process allocation, terminated by the kernel OOM-killer rather than by any Python-level exception the app could catch.

## Description

Root cause (third-party, unpatched in every release examined): docling's HTML backend computes table grid dimensions from raw HTML span attributes. HTMLDocumentBackend._get_cell_spans (docling/backend/html_backend.py:5039-5063, docling 2.127.0) converts the colspan/rowspan attribute to an integer with int(re.search(r"\d+", attr)) and no upper bound (docling 2.0.0 used int(cell.get("colspan", 1)) directly). get_html_table_row_col (html_backend.py:2943-2965) then sums colspans per row (col_count += col_span) into num_cols = max(...) and counts one num_rows per non-header <tr>. parse_table_data (html_backend.py:1820) allocates grid = [[None for _ in range(num_cols)] for _ in range(num_rows)] — 8 bytes per cell — and the fill loops (html_backend.py:1904-1907) iterate range(row_span) × range(col_span) with per-iteration bounds checks, i.e. the raw attribute values again. docling-core's TableData (types/doc/items/table/table_data.py:101-107) declares num_rows/num_cols as plain unbounded ints, and its grid @computed_field (table_data.py:109-141) materializes one 11-field pydantic TableCell object per grid cell on every access (markdown serializer iterates it at transforms/serializer/markdown.py:731/124/549; export_to_dict → model_dump serializes it, types/doc/document.py:3662-3680).

Exposure created by the app: DocumentIngestor is constructed with a plain DocumentConverter() (extractor_engine.py:524-527) — no format caps, no DocumentLimits (docling defaults are sys.maxsize no-ops), no span caps. ingest_epub (extractor_engine.py:798) groups spine items by TOC (_group_spine_by_chapter, "name in h" substring match at :772) and runs _docling_html on every file of every "chapter" group (:868) — before the Phase-3 classifier (:897-916) or any word-count check could discard the item. _docling_html (:684) round-trips the chapter HTML through BeautifulSoup (which preserves span attributes verbatim), writes it to a temp .html file (:693-696), and converts it (:699); it then calls export_to_markdown() (:701) and export_to_dict() (:709), both of which re-materialize the full TableCell grid. The lead file's IR dict is retained for the whole run (:882, :940) and later fully json.dumps'd (:1047; the 5 MB MAX_IR_SIZE cap at :1029 truncates only the on-disk string after full in-memory serialization).

Attack mechanics: an attacker uploads a valid ~1-2 KB EPUB whose single TOC-referenced chapter contains a table with 40 rows, each <tr><td colspan="50000000">x</td></tr>. docling computes num_cols=50M, num_rows=40 and allocates 40 independent ~400 MB row lists (16 GB total) inside the shared web process. Because the grid is built as 40 separate list comprehensions growing in ≤ ~50-100 MB realloc steps, every individual allocation passes Linux overcommit heuristics on any host — the single-huge-allocation pattern that would raise a catchable MemoryError (e.g. the 80-byte colspan="2000000000" variant) is deliberately avoided. Every comprehension append touches the page, so RSS climbs monotonically past physical RAM and the kernel OOM-killer terminates the process with SIGKILL. SIGKILL is not an exception: the app's except Exception fallback (extractor_engine.py:724-726) and its BeautifulSoup fallback only cover the MemoryError regime. On hosts with more RAM than the crafted demand, the attacker adds rows (each ~30 payload bytes buys ~400 MB of allocation), so the kill is targetable at any host size; on swap-equipped hosts the same input instead produces long-lived thrashing plus a minutes-long CPU spin in the span-scaled fill loops (2×10⁹ bounded iterations for the 40×50M payload; a rowspan="1000000000" cell yields ~10⁹ always-failing bounds checks with near-zero memory use). Additionally, tables with moderate spans that survive parsing are amplified ~25-100× per cell at export time when export_to_markdown()/export_to_dict() materialize one pydantic TableCell object per grid cell per access, with the IR dict retained in-memory per chapter group.

## Attack

The attacker is an anonymous visitor of the deployment's public UI URL (the advertised Colab/Kaggle notebooks forward the unauthenticated Gradio UI through a public Pinggy tunnel; a locally exposed instance is equivalent for a local/adjacent attacker). They upload the ~2 KB crafted .epub through the book-file widget (extension filter only, no auth, no size limit), then click Preview (Generate and Export-config are equivalent triggers). The conversion runs synchronously in the request worker thread of the single shared process that also hosts the torch/TTS stack and all user sessions; within seconds the process's RSS passes physical RAM and the kernel OOM-killer SIGKILLs it — in cloud mode this is the notebook kernel itself, killing the operator's whole session and any in-flight generations. The kill is repeatable at will (re-upload and re-click after restart), and no victim interaction, authentication, or special timing is required at any point. Dynamically verified end-to-end (see poc/poc.md and poc/evidence/): anonymous REST upload of the 1,599-byte bomb + on_book_upload auto-scan ("✅ 1 chapters found") + on_preview click → SSE stream died 5.2-6.1 s after the click, app exit code -9 (SIGKILL), cgroup memory.events oom_kill +1, RSS ramp 421 → 2,109-2,122 MB at ~335 MB/s (one ~400 MB row list per ~1.2 s) inside the 2 GiB cgroup-limited sandbox container, app stdout ending mid-ingest with no exception line (uncatchable); reproduced 3× on the unmodified app with a healthy benign control through the identical entry immediately before each kill; with an artificial RLIMIT_AS bound applied to the same app the identical upload instead produced the caught `[Docling HTML] failed: Pipeline SimplePipeline failed` line and the event completed via the BeautifulSoup fallback with the service surviving (oom_kill Δ0) — the shipped deployment sets no such bound.

### Payload

A valid, well-formed EPUB (zip container with mimetype, META-INF/ container.xml, OPF manifest declaring the chapter as an XHTML document item, and a nav/NCX TOC whose entry "Chapter 1" references the payload file so it lands in a "chapter" spine group). The chapter body is pure HTML, no images or fonts: a <table> with 40 rows, each containing a single <td colspan="50000000">x</td> (~1.4 KB of markup, ~1-2 KB zipped). The colspan value is the only attacker-tuned parameter: rows × colspan × 8 bytes = demanded allocation (40 × 50M × 8 B = 16 GB). Dynamically built and used in the POC (poc/evidence/bombs/): bomb_16gib.epub = 1,599 bytes (measured grid demand 16.0 GiB, 10,000,000:1 amplification), bomb_singlecell.epub = 1,583 bytes, bomb_cpu.epub (rowspan wedge) = 1,579 bytes, benign control book = 1,769 bytes. A minimal 80-byte single-cell colspan="2000000000" variant demonstrates the single-request regime (16 GB in one allocation, typically refused with MemoryError on smaller hosts), and rowspan="1000000000" variants provide a pure CPU-wedge payload with negligible memory use.

## Data flow

### Step 1 — `app.py:168-170`

Anonymous book upload: gr.File(file_types=[".epub", ...]) applies an extension filter only; demo.launch (app.py:1908-1916) sets no auth and no max_file_size; the advertised Colab/Kaggle launch cells forward the unauthenticated UI through a public Pinggy reverse tunnel.

### Step 2 — `app.py:1035`

Attacker clicks Preview (on_generate:1216 and on_export_config:1638 are equivalent), calling extract(path, ...) on their own upload — no victim interaction.

### Step 3 — `audiobook_factory/text_extractor.py:303-349`

extract() → _extract_epub() constructs DocumentIngestor (extractor_engine.py:524-527, plain DocumentConverter() with default options — no format/size/span limits) and calls ingest_epub().

### Step 4 — `audiobook_factory/extractor_engine.py:772`

_group_spine_by_chapter matches item basenames to TOC chapter hrefs by substring ("name in h"); the crafted EPUB's TOC entry "Chapter 1" → chapter1.xhtml puts the payload file in a "chapter" group (skip/front groups are the only ones that bypass Phase 2, at :847-857).

### Step 5 — `audiobook_factory/extractor_engine.py:861-868`

Per-file loop over the chapter group: file_item.get_body_content() (EbookLib lxml round-trip, epub.py:378-403 — colspan preserved verbatim) → _docling_html(html_raw, ...). This runs before the Phase-3 classifier (:897-916) could discard the item.

### Step 6 — `audiobook_factory/extractor_engine.py:689-699`

_preprocess_html (BS4 html.parser round-trip touching only spans/images — colspan preserved) → temp .html file → self._converter.convert(tmp_path); the .html extension routes to docling's HTMLDocumentBackend (BeautifulSoup(raw, "html.parser"), html_backend.py:499 — third lenient round-trip, attributes intact).

### Step 7 — `site-packages/docling/backend/html_backend.py:3106-3118 (docling 2.127.0)`

_walk dispatches table ∈ _BLOCK_TAGS (:99, :2106-2118) to _handle_block's table branch: num_rows, num_cols = get_html_table_row_col(tag); TableData(num_rows, num_cols); parse_table_data(...).

### Step 8 — `site-packages/docling/backend/html_backend.py:2943-2965 and :5039-5063`

Grid dimensions are computed from raw attacker attributes: _get_cell_spans does int(re.search(r"\d+", attr)) with no upper bound (2.0.0: int(cell.get("colspan", 1))); get_html_table_row_col sums colspans per row into num_cols and counts num_rows per <td> row. docling-core TableData (table_data.py:101-107) accepts the unbounded ints without validation.

### Step 9 — `site-packages/docling/backend/html_backend.py:1820`

parse_table_data allocates grid = [[None for _ in range(num_cols)] for _ in range(num_rows)] — 8 bytes/cell, touched on every append. 40 × colspan="50000000" → 40 independent ~400 MB row lists = 16 GB of RSS inside the shared web process; each list grows in ≤ ~50-100 MB realloc steps, passing overcommit heuristics on any host (the multi-row pattern structurally avoids the single-allocation refusal that would raise a catchable MemoryError).

### Step 10 — `site-packages/docling/backend/html_backend.py:1904-1907`

Fill loops iterate range(row_span) × range(col_span) with per-iteration bounds checks — iteration counts are the raw attribute values (2×10⁹ iterations for the 40×50M payload; ~10⁹ always-failing checks for a rowspan wedge), adding a minutes-long in-request CPU spin that survives even when memory is sufficient.

### Step 11 — `kernel OOM-killer vs audiobook_factory/extractor_engine.py:724-726`

When RSS passes physical RAM the kernel OOM-killer sends SIGKILL to the (largest-RSS) process hosting the UI, torch/TTS stack, and all sessions — in cloud mode the notebook kernel itself. The app's except Exception fallback catches MemoryError only and cannot intercept SIGKILL; the attack is repeatable after restart.

### Step 12 — `audiobook_factory/extractor_engine.py:701-709, :882, :940, :1047`

Secondary attacker-scaled passes: export_to_markdown() and export_to_dict() each materialize one pydantic TableCell per grid cell (TableData.grid computed_field, docling-core table_data.py:109-141; markdown.py:731/124/549 — ~25-100× per-cell multiplier, re-computed on every access); the lead file's IR dict is retained for the whole run and fully json.dumps'd later (MAX_IR_SIZE truncates only the on-disk string), so moderate-span tables are amplified at export time and across multi-chapter EPUBs.

## Fix / patch notes

diff --git a/audiobook_factory/extractor_engine.py b/audiobook_factory/extractor_engine.py
--- a/audiobook_factory/extractor_engine.py
+++ b/audiobook_factory/extractor_engine.py
@@ -607,6 +607,21 @@ class DocumentIngestor:
         soup = BeautifulSoup(html_content, "html.parser")
         extracted_images = []

+        # Clamp table span attributes before Docling sees them: Docling
+        # scales its table-grid allocation directly by colspan/rowspan
+        # values without any internal bound, so a crafted ~2 KB chapter
+        # can demand a multi-GB grid and OOM-kill the whole service
+        # process (uncatchable SIGKILL).
+        _MAX_TABLE_SPAN = 1000
+        for cell in soup.find_all(["td", "th"]):
+            for attr in ("colspan", "rowspan"):
+                raw = cell.get(attr)
+                if raw is None:
+                    continue
+                digits = "".join(ch for ch in str(raw) if ch.isdigit())
+                if digits and int(digits) > _MAX_TABLE_SPAN:
+                    cell[attr] = str(_MAX_TABLE_SPAN)
+
         for span in soup.find_all("span"):
             classes = " ".join(span.get("class", []))
             style   = span.get("style", "")

## References

- https://cwe.mitre.org/data/definitions/400.html
- https://cwe.mitre.org/data/definitions/770.html
- https://docling-project.github.io/docling/

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] Cross-user settings poisoning of the title-keyed generation_progress.json (on_generate upload-copy) re-consumed verbatim by the documented headless cli.py workflow loads an attacker-chosen HuggingFace repo with trust_remote_code=True — RCE on the victim's host with one documented CLI action, no Gradio-version dependency
- key: `BUG-R4-C1-A4-H1`
- disclosure: owner_only
- cwe: CWE-94
- file: `app.py`

# Cross-user settings poisoning of the title-keyed generation_progress.json (on_generate upload-copy) re-consumed verbatim by the documented headless cli.py workflow loads an attacker-chosen HuggingFace repo with trust_remote_code=True — RCE on the victim's host with one documented CLI action, no Gradio-version dependency

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R4-C1-A4-H1
- **CWE:** CWE-94
- **CVSS:** 8.8 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H`)
- **EV priority:** P0
- **EV score:** 10
- **PoC status:** pending
- **EXP status:** pending
- **Affected versions:** v1.3.0 confirmed (audited snapshot, commit dc9aed2cac064f436310a35fdd069f34205fa687). Per CHANGELOG the consuming chain has existed since the initial v0.5.0 release (progress-JSON upload for session restoration, headless cli.py, Qwen from_pretrained sink); the trust_remote_code VibeVoice sink since v1.3.0; the copy-on-generate write was examined in the v1.3.0 snapshot only. Earlier releases not examined in detail.

## Exploitability rationale

R:N — the planting leg is executed remotely over HTTP by an anonymous visitor of the README-advertised public notebook tunnel URL (Pinggy in Colab/Kaggle; no auth in either launch mode): upload a .json progress file + a dummy book, pick a valid model variant, set the victim's title, click Generate. No privileges and no victim interaction are needed for the planting, and the planted file persists indefinitely (only-if-missing settings logic preserves it across the victim's own UI runs; no cleanup path exists). The final sink execution additionally depends on the victim performing the product's documented headless workflow — one CLI run on the at-rest progress file — a behavior dependency the EV model does not dimension separately and which is disclosed here explicitly. E:D — the title-keyed progress cache, the progress-JSON upload, and the headless CLI are core default features present in every deployment; the shared-instance configuration is the product's advertised cloud usage (settled in the threat model and already rated default exposure in BUG-R2-C1-A2-H5); the victim's CLI consumption is the notebook-recommended workflow ("Prefer the CLI for faster, more reliable generation — especially on long sessions where the Gradio tunnel may expire", AudiobookMaker_Colab.ipynb headless cell). C:D — pure application logic with no framework dependency: the file content never passes through a Gradio component on the delivery path (shutil.copy2 is byte-for-byte), so the server-side Dropdown choices validation that gates the UI-restore delivery of BUG-R2-C1-A1-H1 on Gradio >= 5.0.0 never sees the payload; once the poisoned file is at rest, the CLI settings rebuild, provider dispatch and trust_remote_code model load are deterministic on every install (all anchors verified). I:X — arbitrary code execution in the victim's process: repo-supplied Python executes via AutoProcessor/AutoTokenizer/AutoModelForCausalLM.from_pretrained (model_name, trust_remote_code=True) on every transformers line (VibeVoice branch); the process runs as root on the Colab/Kaggle notebook VM (session data, book texts, environment tokens, network pivot) or as the local user on a workstation where the victim runs the CLI on the downloaded file. settings.output_dir additionally delivers arbitrary-directory writes on the victim's own run. The planting primitive itself (wholesale replacement of a title-keyed progress file by an anonymous visitor) was already dynamically confirmed end-to-end on gradio 6.27.0 in the BUG-R2-C1-A2-H5 EXP.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 1129 | `on_generate` |
| `app.py` | 1069 | `on_generate` |
| `app.py` | 1114 | `on_generate` |
| `app.py` | 151 | `build_app (progress_file_upload component)` |
| `app.py` | 81 | `check_existing_progress` |
| `app.py` | 1374 | `build_app (book_title_box.change oracle wiring)` |
| `app.py` | 1386 | `on_progress_upload` |
| `app.py` | 1339 | `on_generate (final download list offers the progress file)` |
| `app.py` | 1797 | `on_export_config` |
| `app.py` | 1826 | `on_export_config (status message recommends cli.py on prog_path)` |
| `audiobook_factory/progress_io.py` | 63 | `read_progress_file` |
| `audiobook_factory/pipeline.py` | 324 | `_validate_config` |
| `audiobook_factory/pipeline.py` | 443 | `run_pipeline` |
| `audiobook_factory/pipeline.py` | 539 | `run_pipeline (only-if-missing settings persistence)` |
| `audiobook_factory/pipeline.py` | 565 | `run_pipeline (GPU pool provider factory)` |
| `audiobook_factory/pipeline.py` | 1101 | `_process_chapter_with_retry / _synth_single` |
| `audiobook_factory/tts_providers/base_tts_provider.py` | 166 | `get_tts_provider` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 62 | `VibeVoiceTTSProvider._ensure_initialised (substring gate)` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 74 | `VibeVoiceTTSProvider._ensure_initialised (AutoProcessor trust_remote_code)` |
| `audiobook_factory/tts_providers/vibevoice_provider.py` | 84 | `VibeVoiceTTSProvider._ensure_initialised (AutoModelForCausalLM trust_remote_code)` |
| `audiobook_factory/tts_providers/qwen_provider.py` | 593 | `QwenTTSProvider._load_model` |
| `cli.py` | 250 | `_load_config` |
| `cli.py` | 292 | `_build_audiobook_config` |
| `cli.py` | 372 | `_build_audiobook_config` |
| `cli.py` | 383 | `_build_audiobook_config` |
| `cli.py` | 445 | `_load_chapters` |
| `cli.py` | 694 | `main (copies the progress JSON into cfg.output_dir)` |
| `cli.py` | 770 | `main / _runner (local run_pipeline)` |
| `api/server.py` | 167 | `api_voice_test` |
| `README.md` | 191 | `Quick CLI Usage (documented at-rest consumption)` |
| `README.md` | 238 | `CLI Workflow` |
| `AudiobookMaker_Colab.ipynb` | 400 | `headless CLI cell (CONFIG_JSON + os.system)` |

## Background

AudiobookMaker is an end-to-end AI audiobook generator (Gradio web UI on port 7860, an unauthenticated FastAPI orchestrator on 127.0.0.1:8000, and a headless CLI) intended to run on a local workstation or — as advertised in the README — in Google Colab / Kaggle notebooks whose Gradio UI is published through a public Pinggy SSH tunnel URL meant to be shared; there is no authentication and no multi-user model anywhere. A central artifact is the progress file audiobook_output/<sanitized book title>/generation_progress.json: it caches the fully extracted chapter text (so a returning user or the CLI can resume without re-parsing the book) and embeds a complete settings block (TTS provider, model repository, output directory, all generation parameters). The file is written by the UI ("Export Config JSON" or any Generate run), can be re-uploaded through the "Resume from Progress JSON" accordion to restore a session, and is the designated input of the documented headless workflow: README "Quick CLI Usage" (python cli.py audiobook_output/MyBook/generation_progress.json), the app's own export status message ("To generate without Gradio, run: python cli.py \"<path>\""), and the Colab/Kaggle headless cells that run exactly that command via os.system(). The security-relevant design facts are that (1) the output directory name is derived solely from the user-supplied title, making the progress file a resource in a global, guessable namespace with no ownership binding between a file and the session that created it (CWE-639), and (2) every consumer of the file — the CLI config rebuild and the FastAPI config import — treats its settings block as trusted configuration without any authenticity check or field allowlist (CWE-345), ultimately feeding tts_provider_name/tts_model_name into HuggingFace model loaders that execute repository-supplied Python (trust_remote_code=True, CWE-94).

## Description

The chain has three stages. (1) Planting: on_generate copies an uploaded progress JSON byte-for-byte over the title-keyed progress file before any pipeline work starts (shutil.copy2, app.py:1125-1130); the only guards are a book upload and a non-"Base" model choice (app.py:1069-1074), both trivially satisfied by an anonymous co-visitor with a dummy file and any valid dropdown variant. The file content never passes through a Gradio component on this path, so the server-side Dropdown choices validation that blocks the UI-restore delivery vector on Gradio >= 5.0.0 does not apply — the delivery is version-independent. The restore event that fires on upload (on_progress_upload) only applies UI updates (output-side gr.update, warning-only on all versions) and does not modify the file; the attacker sets book_path/voice_file empty in the JSON so no server paths are preset. Victim targeting needs only the sanitized title: the title box is a free oracle (check_existing_progress fires on every change, app.py:81-104/1374-1378), and an empty title lands every no-title user in the shared default directory audiobook_output/audiobook. (2) Persistence: the planted settings survive the victim's own UI Generate runs because run_pipeline only fills settings/book_path/ voice_file when missing (pipeline.py:539-553), chapter status updates are read-modify-write (progress_io.py:151-196), the whole poisoned record is synced to temp/generation_progress.json (pipeline.py:557-562), the app offers the very file as a download after generation (app.py:1339-1342), and no cleanup path ever removes it (only the non-default force_reprocess flag does). The single mechanism that overwrites the settings is the victim's own "Export Config JSON" (app.py:1797), which is not part of the documented at-rest consumption routes and cannot undo a planting that happens after the export (the app keeps running in notebook mode). (3) Consumption and sink: the documented headless workflow runs python cli.py audiobook_output/<BookTitle>/generation_progress.json (README.md:191, README.md:238, the app's own export status message app.py:1826-1830, and the Colab/Kaggle headless cells). cli.py:_load_config takes settings = data.get("settings", {}) verbatim (cli.py:250; CLI flag overrides only on explicit flags) and _build_audiobook_config rebuilds AudiobookConfig with no allowlist — output_dir (cli.py:292), tts_provider_name (cli.py:372), tts_model_name (cli.py:383) pass straight through. Chapters with non-empty cached text are rebuilt from the JSON without any book file (cli.py:445-462), so a "pending" chapter list guarantees synthesis is attempted. run_pipeline then performs the only config validation (quantization, pipeline.py:324-331), a preflight whose voice-reference check is keyed to the attacker-controlled top-level voice_file (empty disables it, pipeline.py:443-447; remaining checks are environmental and pass with CPU fallback), and instantiates the provider selected by the poisoned settings (pipeline.py:565-580 or 1101-1106 -> get_tts_provider, base_tts_provider.py:166-200). For tts_provider_name "vibevoice" (a legitimate, UI-offered value) the model name passes a single substring gate (if "VibeVoice" not in model_name, vibevoice_provider.py:62-64) — any attacker HF repo id containing "VibeVoice" passes — and is loaded with AutoProcessor/AutoTokenizer/AutoModelForCausalLM .from_pretrained(model_name, trust_remote_code=True) (vibevoice_provider.py:74/78/ 84-90), executing repository-supplied Python in the victim's process. ensure_ready() runs before _validate_voice_ref (vibevoice_provider.py:109-110), so no voice prerequisite can preempt the load. On the Qwen branch the same settings field feeds Qwen3TTSModel.from_pretrained(config.tts_model_name) (qwen_provider.py:593), which executes attacker-supplied model code on legacy stacks (transformers < 4.42 / torch < 2.6). The poisoned settings.output_dir additionally redirects all run outputs — including a copy of the poisoned JSON itself (cli.py:694-699) — to an arbitrary absolute directory on the victim's host. The CLI run summary prints Book/Chapters/Voice/Output/Format/Workers/CoverImg/TextCache but not the provider or model (cli.py:596-606), and the attacker can preserve the victim's real chapter titles/text/statuses (readable via the confirmed export-disclosure issue BUG-R2-C1-A2-H5) so the file looks normal.

## Attack

Attacker: an anonymous co-visitor of the shared AudiobookMaker instance (the README-advertised public notebook tunnel URL; no authentication). Steps: (1) optionally enumerate live victim titles via the title-box oracle or read a victim's current progress file via the export disclosure to copy its chapter data; (2) upload the crafted .json through the "Resume from Progress JSON" accordion (the restore event harmlessly applies UI updates); (3) upload any dummy book file and select any non-"Base" model variant to pass the two guards; (4) set the victim's book title (or leave it empty for the shared default directory) and click Generate — on_generate copies the crafted file byte-for-byte over audiobook_output/<victim title>/generation_progress.json before any pipeline work; the run can be cancelled immediately afterwards; (5) host the payload repo on HuggingFace with "VibeVoice" in its id. Victim: runs the documented headless workflow on the at-rest file — python cli.py audiobook_output/<BookTitle>/generation_progress.json in a terminal, or the Colab/Kaggle headless cell that resolves CONFIG_JSON from exactly that path and os.system()s the command (the notebook even recommends the CLI over the UI for cloud sessions). One CLI run rebuilds the configuration from the poisoned settings and loads the attacker's repository with trust_remote_code=True: arbitrary Python executes in the victim's process (root on the Colab/Kaggle VM; the local user on a workstation where the victim ran the CLI on the downloaded file, which the app itself offers as a download after generation). Observable impact: full host compromise of the victim's environment; secondary arbitrary-directory writes via settings.output_dir on the same run.

### Payload

A generation_progress.json whose top-level fields are book_path "", voice_file "" (disables the preflight voice check), book_title set to the victim's title (or empty for the shared default directory), and whose settings block contains tts_provider_name "vibevoice", tts_model_name "<attacker HF repo id containing the substring VibeVoice, e.g. attacker/VibeVoice-demo>" hosting custom modeling/processing Python, output_dir "<arbitrary absolute directory>", quantization "none", selected_chapters [], device "cpu" (avoids CUDA-only device_map failures); plus a chapters[] array with the victim's real chapter titles/text (readable first via the export disclosure) and status "pending" so synthesis — and therefore the model load — is always attempted. The file is a normal .json (the only upload-side check is the extension).

## Data flow

### Step 1 — `app.py:151-155`

Attacker uploads the crafted progress JSON via progress_file_upload (gr.File, file_types=[".json"]) — the extension is the only upload-side check; the .upload restore event (app.py:1386-1554) applies UI updates only (output-side gr.update applies with warning on all Gradio versions) and does not touch the file.

### Step 2 — `app.py:1069-1074`

on_generate guards: a book upload and a non-"Base" model choice — satisfied with any dummy file and any valid dropdown variant; no content check on the uploaded progress file.

### Step 3 — `app.py:1114-1118`

book_out = audiobook_output/<sanitized UI book_title> — the attacker enters the victim's title (enumerable via the check_existing_progress oracle, app.py:81-104/1374-1378; empty title = shared default dir "audiobook").

### Step 4 — `app.py:1125-1130`

shutil.copy2(uploaded_progress_path, dest_progress_path) replaces the victim's generation_progress.json byte-for-byte before cfg construction and any pipeline work — the payload never passes through a Gradio component, so the >= 5.0.0 Dropdown input validation never sees it (version-independent delivery).

### Step 5 — `audiobook_factory/pipeline.py:515-562`

Persistence: any later run that reads the file succeeds (valid JSON) and only fills settings/book_path/voice_file when missing — the planted settings survive the victim's own UI Generate runs and are synced to temp/generation_progress.json; chapter updates are read-modify-write (progress_io.py:151-196); no cleanup path removes the file.

### Step 6 — `README.md:190-191, README.md:233-238, app.py:1826-1830, app.py:1339-1342, AudiobookMaker_Colab.ipynb cells 17-18, AudiobookMaker_Kaggle.ipynb cell 20`

The victim's documented workflow consumes the at-rest file: "python cli.py audiobook_output/MyBook/generation_progress.json" (README basic run), the app's own export status message pointing at the same path, the post-generation download offer of the file, and the notebook headless cells that os.system() the CLI with CONFIG_JSON resolved from that path.

### Step 7 — `cli.py:250`

_load_config: settings = data.get("settings", {}) — verbatim; CLI flag overrides apply only for explicitly passed flags.

### Step 8 — `cli.py:292,372,383`

_build_audiobook_config rebuilds AudiobookConfig with settings.get("output_dir"), settings.get("tts_provider_name"), settings.get("tts_model_name") verbatim — no allowlist (AudiobookConfig.from_dict on the API path, api/server.py:167, filters unknown keys only).

### Step 9 — `cli.py:445-462`

_load_chapters: all chapters have non-empty cached text, so the chapter list is built directly from the JSON (no book file needed); "pending" statuses guarantee synthesis is attempted (pipeline.py:638-660 also regenerates completed-but-missing chapters by default).

### Step 10 — `audiobook_factory/pipeline.py:324-331,443-447`

run_pipeline: _validate_config checks quantization only (attacker sets "none"); preflight voice check keyed to the attacker-controlled top-level voice_file (empty disables it); remaining checks are environmental and pass with CPU fallback.

### Step 11 — `audiobook_factory/pipeline.py:565-580,1101-1106`

Provider instantiation: GPU pool warmup or _synth_single call get_tts_provider(config.tts_provider_name, config).

### Step 12 — `audiobook_factory/tts_providers/base_tts_provider.py:166-200`

get_tts_provider dispatches the name "vibevoice" (a supported, UI-offered value) to VibeVoiceTTSProvider.

### Step 13 — `audiobook_factory/tts_providers/vibevoice_provider.py:62-64`

The only model-name gate: if "VibeVoice" not in model_name — any attacker repo id containing the substring passes unchanged.

### Step 14 — `audiobook_factory/tts_providers/vibevoice_provider.py:74,78,84-90`

AutoProcessor/AutoTokenizer/AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True) — repository-supplied Python executes in the victim's process (ensure_ready() runs before _validate_voice_ref, vibevoice_provider.py:109-110, so nothing preempts the load). RCE on every transformers line; the qwen branch (qwen_provider.py:593, Qwen3TTSModel.from_pretrained(config.tts_model_name)) reaches the same effect on legacy stacks.

### Step 15 — `cli.py:694-699`

Secondary effect on the same run: the CLI copies the poisoned JSON into the attacker-chosen settings.output_dir, and all generation outputs are written there — arbitrary-directory writes on the victim's host.

## Fix / patch notes

diff --git a/cli.py b/cli.py
--- a/cli.py
+++ b/cli.py
@@ -248,6 +248,26 @@ def _load_config(args) -> tuple[dict, dict, list[dict], str]:
     settings = data.get("settings", {})
     chapters_raw = data.get("chapters", [])

+    # Security: generation_progress.json lives in a shared, title-keyed output
+    # directory with no ownership binding, so its settings block is untrusted
+    # input. Never let it select the TTS provider or the model repository;
+    # explicit CLI flags (e.g. --tts-model-name) are applied further below
+    # and remain authoritative for the local user.
+    _ALLOWED_PROVIDERS = {"qwen", "vibevoice", "f5tts"}
+    _ALLOWED_MODELS = {
+        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
+        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
+        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
+        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
+        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
+        "bezzam/VibeVoice-1.5B-hf",
+    }
+    _prov = settings.get("tts_provider_name")
+    if _prov is not None and _prov not in _ALLOWED_PROVIDERS:
+        print(_warn(f"  Untrusted tts_provider_name {_prov!r} in progress file - resetting to 'qwen'."))
+        settings["tts_provider_name"] = "qwen"
+    _model = settings.get("tts_model_name")
+    if _model is not None and _model not in _ALLOWED_MODELS:
+        print(_warn(f"  Untrusted tts_model_name {_model!r} in progress file - resetting to default."))
+        settings["tts_model_name"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
+
     # Apply CLI overrides
     if args.book_path:
         meta["book_path"] = args.book_path
diff --git a/audiobook_factory/tts_providers/vibevoice_provider.py b/audiobook_factory/tts_providers/vibevoice_provider.py
--- a/audiobook_factory/tts_providers/vibevoice_provider.py
+++ b/audiobook_factory/tts_providers/vibevoice_provider.py
@@ -61,7 +61,9 @@ class VibeVoiceTTSProvider(BaseTTSProvider):
                 import torch
                 from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
                 model_name = getattr(self.config, "tts_model_name", "bezzam/VibeVoice-1.5B-hf")
-                if "VibeVoice" not in model_name:
+                # Only the curated repository may be loaded with
+                # trust_remote_code=True; a substring match is not an allowlist.
+                if model_name != "bezzam/VibeVoice-1.5B-hf":
                     model_name = "bezzam/VibeVoice-1.5B-hf"

## References

- https://cwe.mitre.org/data/definitions/94.html
- https://cwe.mitre.org/data/definitions/345.html
- https://cwe.mitre.org/data/definitions/639.html
- https://huggingface.co/docs/transformers/security#trust-remote-code

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._


## [high] Gradio resume-state read-back (title-keyed audiobook_output/<title>/generation_progress.json consumed by on_generate/_load_cached_chapters_if_available and run_pipeline) trusts cross-user poisoned progress files — silent audiobook content forgery, completed-chapter skip with attacker-audio adoption, foreign chunk splicing, and persistent poisoned-settings re-offer into the documented resume/CLI workflows
- key: `BUG-R4-C1-A4-H2`
- disclosure: owner_only
- cwe: CWE-345 / CWE-639
- file: `app.py`

# Gradio resume-state read-back (title-keyed audiobook_output/<title>/generation_progress.json consumed by on_generate/_load_cached_chapters_if_available and run_pipeline) trusts cross-user poisoned progress files — silent audiobook content forgery, completed-chapter skip with attacker-audio adoption, foreign chunk splicing, and persistent poisoned-settings re-offer into the documented resume/CLI workflows

- **Project:** MSpider3/AudiobookMaker
- **Finding key:** BUG-R4-C1-A4-H2
- **CWE:** CWE-345 / CWE-639
- **CVSS:** 8.2 (`CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:L`)
- **EV priority:** P0
- **EV score:** 9
- **PoC status:** pending
- **EXP status:** pending
- **Affected versions:** v1.3.0 (audited snapshot, commit dc9aed2cac064f436310a35fdd069f34205fa687); the consuming features (title-keyed progress cache with cached-chapter loading, chunk-level resume, only-if-missing settings persistence) exist since v1.0.0 per CHANGELOG, but earlier versions were not examined

## Exploitability rationale

R:N — the poisoning leg (wholesale progress-file replacement via progress-JSON upload + victim-title Generate) is EXP-confirmed in BUG-R2-C1-A2-H5 as remotely triggerable by an anonymous visitor of the README-advertised public tunnel URL with no authentication and no victim interaction; the audio-file plant is the same EXP-confirmed same-title re-render. The integrity impact materializes on the victim's next ordinary Generate/Export click — the app's core advertised workflow (auto-resume banner on every title change, chunk-level resume, progress re-upload accordion), not a security decision by the victim, so it is scored like H5's next-run-materializing impacts (the stricter UI:R reading scores CVSS 7.1, still High). E:D — the resume-from-progress cache is the product's core feature (README "Cached Book Extraction", "Chunk-Level Resume"), present in every deployment; the vulnerable shared-instance configuration is the advertised usage ("public shareable Gradio links"). C:D — pure application logic with no framework dependency on the consumption path: once the poisoned file sits at the title-derived path, the cached-text load, chapter-status skip, output-file adoption, chunk adoption and settings persistence are deterministic on both dispatch modes (API relay and local fallback); the attacker's information needs (victim title, chapter titles/nums, output format, max_len) are supplied by the confirmed H5/H2 disclosure. The only probabilistic element is the chunk-WAV plant (graceful-cancel race; deterministic on hard termination). I:S — silent integrity compromise of the victim's deliverable: the generated audiobook/text/subtitles contain attacker-chosen text and spliced attacker audio with zero UI signals, the victim's download list offers attacker-planted files as their finished audiobook, and the attacker's settings persist in the victim's progress file feeding the documented resume-reimport and CLI workflows (the delivery substrate of the cross-user RCE chain HYP-R4-C1-A4-H1 / LEAD-R4-C1-A4-L2). Not code execution by itself.

## Code anchors

| File | Line | Function |
|---|---:|---|
| `app.py` | 1208 | `on_generate / _runner` |
| `app.py` | 975 | `_load_cached_chapters_if_available` |
| `app.py` | 983 | `_load_cached_chapters_if_available (all-text gate)` |
| `app.py` | 996 | `_load_cached_chapters_if_available (title filter)` |
| `app.py` | 1015 | `on_preview` |
| `app.py` | 1123 | `on_generate (uploaded progress copy — the H5 write primitive)` |
| `app.py` | 1231 | `on_generate / _runner (API payload carries cached chapters)` |
| `app.py` | 1294 | `on_generate / _runner (local-fallback run_pipeline)` |
| `app.py` | 1338 | `on_generate (final download list appends progress JSON)` |
| `app.py` | 82 | `check_existing_progress` |
| `app.py` | 148 | `build_app (Resume from Progress JSON accordion)` |
| `app.py` | 701 | `on_book_upload (selection defaults / fuzzy restore)` |
| `app.py` | 1396 | `on_progress_upload` |
| `app.py` | 1493 | `on_progress_upload (pronunciation_map restore)` |
| `app.py` | 1522 | `on_progress_upload (tts_model_name Dropdown restore)` |
| `app.py` | 1601 | `on_export_config` |
| `app.py` | 1790 | `on_export_config (merge preserves cached text)` |
| `audiobook_factory/pipeline.py` | 514 | `run_pipeline (reads back prog_path_out)` |
| `audiobook_factory/pipeline.py` | 539 | `run_pipeline (only-if-missing settings/book_path/voice_file)` |
| `audiobook_factory/pipeline.py` | 557 | `run_pipeline (sync to shared temp/generation_progress.json)` |
| `audiobook_factory/pipeline.py` | 635 | `_process (completed-status skip)` |
| `audiobook_factory/pipeline.py` | 639 | `_process (existing-file adoption into output_files)` |
| `audiobook_factory/pipeline.py` | 1012 | `_process_chapter (completed_chunks validation)` |
| `audiobook_factory/pipeline.py` | 1033 | `_process_chapter (pronunciation + text export sinks)` |
| `audiobook_factory/chapter_pipeline.py` | 42 | `_validate_chunk_file` |
| `audiobook_factory/chapter_pipeline.py` | 409 | `run_chapter_pipeline (cached-chunk adoption)` |
| `audiobook_factory/chapter_pipeline.py` | 232 | `_flush_accumulated_batch (chunk_ch_{idx}_{n}.wav naming)` |
| `audiobook_factory/progress_io.py` | 151 | `update_chapter_status (read-modify-write preserves settings)` |
| `api/worker.py` | 131 | `_process_single_task` |
| `api/server.py` | 276 | `websocket_task (WS early-break regression source)` |

## Background

AudiobookMaker is an end-to-end AI audiobook generator (Gradio web UI on port 7860, an unauthenticated FastAPI orchestrator on 127.0.0.1:8000, and a headless CLI) intended to run on a local workstation or — as advertised in the README — in Google Colab / Kaggle notebooks whose Gradio UI is published through a public Pinggy SSH tunnel URL meant to be shared; there is no authentication and no multi-user model. Its central resume feature is the progress file written at audiobook_output/<sanitized book title>/generation_progress.json: it caches the fully extracted chapter text ("Cached Book Extraction"), per-chapter statuses and completed_chunks (chunk-level resume of interrupted GPU synthesis), and the whole generation settings. Returning users resume by simply re-running their book — the UI advertises on every title change "Generation will automatically resume from the last completed chapter" — or by re-uploading the progress JSON, which the top-of-page accordion instructs ("Start here if you are resuming. Upload your generation_progress.json to restore all settings and chapter selections automatically"); the Colab/Kaggle notebook additionally tells users to feed audiobook_output/<BookTitle>/generation_progress.json to the headless CLI cell. The already-confirmed BUG-R2-C1-A2-H5 established that the output directory is a global, guessable namespace with no ownership binding, so an anonymous co-visitor can (EXP-confirmed) replace another session's progress file wholesale (progress-JSON upload + same-title Generate copies the file byte-for-byte over the victim's) and overwrite the victim's finished chapter audio files with their own same-title run. This finding covers what happens next: how the victim's own documented workflows consume the replaced file as trusted state.

## Description

The resume consumers perform an untrusted read-back: whatever JSON sits at the title-derived path is parsed by read_progress_file (format/encoding/HTML checks only, progress_io.py:64-134 — no signature, schema, or ownership binding) and its values drive synthesis, chapter skipping, chunk adoption, and settings persistence. (1) Cached-text substitution: on_generate's _runner calls _load_cached_chapters_if_available(prog_json_path, selected_chapters, ...) BEFORE parsing the victim's book (app.py:1208-1212); on a hit the book is never re-parsed (app.py:1214-1220). The only gates — every cached chapter has non-empty text (app.py:983) and the title filter st == title or st in title (app.py:996-1004) — are satisfied by preserving the victim's chapter titles (readable via the H5 disclosure); TXT/no-TOC books and fresh sessions bypass the filter entirely, and the victim's own resume flows keep the selection consistent (re-import restores the poisoned selected_chapters, app.py:1506-1507; book re-upload selects all chapters by default or fuzzy-matches by chapter-number prefix, app.py:686-708). The poisoned chapters[].text then becomes the synthesis/export text on both dispatch modes — the API relay ships the cached chapters in the task payload (app.py:1231-1244; api/worker.py:131-152) and the local fallback calls run_pipeline directly (app.py:1294) — producing TTS audio, export_text .txt files (pipeline.py:1038-1046) and subtitles with attacker-chosen content. The only log line ("📦 Using cached chapter text from progress JSON (skipping book re-parsing)", app.py:1008) is identical to a legitimate resume, and the chapter preview table shows the REAL book's stats because on_preview re-extracts from the book and never consults the cache (app.py:1015-1023) — the UI actively reinforces the victim's trust. The victim's own "Export Config JSON" preserves the poisoned text in the merge ("text": cd.get("text") or ec.get("text", ""), app.py:1790-1796) and returns the file with the on-screen CLI instruction (app.py:1826-1832), propagating the poison into the headless CLI workflow. (2) Status-skip + output-file spoofing: chapters whose poisoned entry matches by normalized title or by num (pipeline.py:612-633) with status:"completed" are skipped when force_reprocess is off (default, app.py:504), and any file at the deterministic "Chapter {idx} - <title>.<fmt>" path in the shared output dir is appended to output_files as that chapter's result (pipeline.py:635-648) — the attacker plants those files with their own same-title Generate run (EXP-confirmed in H5 as the re-render overwrite), knowing the victim's output format from the stolen settings. regen_missing (default on) only triggers when the file is MISSING, so nothing is re-synthesized and the "⏩ Already completed. Skipping." log is normal resume behavior; the auto-resume banner even shows completion counts read from the poisoned file (app.py:93-100). (3) Foreign chunk splicing: poisoned completed_chunks are adopted after validation that checks only existence and size ≥ 1000 bytes (pipeline.py:1012-1024 → chapter_pipeline.py:42-61) and spliced into the mastered chapter audio (chapter_pipeline.py:409-420); both preconditions are defaults (resume_incomplete_chunks on, app.py:549-554; force_reprocess off), the chunk dir is inside the shared output namespace (pipeline.py:1000-1001) with attacker-reproducible names chunk_ch_{idx}_{n}.wav (chapter_pipeline.py:232), and nothing binds a chunk file to the run that created it. Chunk-WAV planting requires the attacker's same-title run to terminate without cleanup (process/session death, or the graceful-cancel race — chunk files are deleted on chapter success and by Stage C's finally for received results, chapter_pipeline.py:617-619, 651-655), so this leg is consumption-deterministic but plant-probabilistic through the UI alone. (4) Poisoned-settings persistence and re-offer: the victim's own run_pipeline replaces book_path/voice_file/settings only when missing or empty (pipeline.py:539-549), so the attacker's non-empty values survive the victim's runs; per-chapter updates are read-modify-write cycles that never touch settings (progress_io.py:151-224); the poisoned data is synced to the fixed shared temp/generation_progress.json (pipeline.py:557-561); the Generate download list appends the poisoned progress file itself (app.py:1338-1342, delivered when out_files is non-empty — on the audited snapshot the API-relay path suppresses the whole list through an unrelated WS early-break regression, api/server.py:276-279 + app.py:1185-1206, while the local-fallback path and the always-working Export Config download deliver). On re-import, on_progress_upload restores pronunciation_map unconditionally (written to audiobook_output/restored_pronunciation_fixes.txt and preset into the pronunciation component, app.py:1493-1502), which the victim's next Generate applies as re.sub search-replace to the text before TTS/export (pipeline.py:1033-1035, 369-377) — a version-independent second text-substitution channel; selected_chapters is restored unconditionally (app.py:1506-1507); book_path/voice_file whenever they exist (app.py:1511-1513); settings.tts_model_name is restored into the model Dropdown (app.py:1447, 1522), which on Gradio ≥ 5.0.0 is rejected at the next event submit by the framework's server-side choices check (identical to BUG-R2-C1-A1-H1's gating; on ≤ 4.44.1 it round-trips into the provider loaders — the cross-user RCE delivery tracked by HYP-R4-C1-A4-H1). Accuracy bounds: the victim's own on_export_config overwrites the poisoned settings with the victim's current values (app.py:1797-1800), so the settings persistence holds for the Generate path; and no documented workflow references temp/generation_progress.json (the notebook/README point at audiobook_output/<BookTitle>/generation_progress.json). The only mechanism that cleanses a poisoned cache is force_reprocess (deletes the progress file and chunk files, pipeline.py:487-494, 1002-1010), which is off by default and contrary to the advertised auto-resume. This finding corrects H5's impact bound: the write damage is not "recoverable by re-running" — re-running reproduces the forgery with zero UI signals; recovery requires out-of-band suspicion (comparing the produced audio/text against the source book) plus manual deletion or force-reprocessing. A safe minimal patch cannot be given as a diff: any authenticity marker stored inside the poisoned artifact is copyable by the attacker (the whole file is disclosed by H5/H2), and an instance-wide HMAC only proves "written by this instance", not by the owning session — the shared unauthenticated namespace cannot authenticate its own state. The real fix is a design change: per-session or randomly-named output directories (title kept as metadata only), or server-side ownership state outside the shared namespace.

## Attack

Attacker: any anonymous co-visitor of a shared AudiobookMaker instance — the README-advertised Colab/Kaggle public-link deployment (no authentication; Pinggy tunnel). Preconditions: (1) another user has processed a book under a title T (normal workflow; the progress file persists indefinitely), (2) the attacker can reproduce T (title oracle on every title-box change, public low-entropy titles, or the fixed default directory for empty titles), and (3) optionally the victim's chapter titles/settings — obtained by downloading the victim's progress file through the confirmed H5/H2 disclosure. Attack: (a) the attacker runs Generate with title T and their own book/attacker text to plant finished chapter audio at the predictable shared paths (and, for the splice leg, lets that run die mid-chapter or cancels it to leave chunk WAVs); (b) the attacker uploads the crafted progress JSON with title T and clicks Generate — the file is copied byte-for-byte over audiobook_output/T/generation_progress.json (EXP-confirmed primitive) — then cancels; (c) the victim later returns: their title-box change shows "Existing Progress Found! N/M chapters … Generation will automatically resume", they click Generate (or Export Config), and their run consumes the poison as trusted resume state. Observable result: the victim's audiobook, text exports and subtitles contain attacker-chosen text; the download list / output directory contains attacker-planted audio presented as the victim's finished chapters; the victim's own runs keep the attacker's settings in the progress file and re-offer the poisoned file for download and re-upload; if the victim follows the notebook's headless CLI instruction, the poisoned settings reach the CLI config rebuild (cross-user RCE delivery per HYP-R4-C1-A4-H1). No error or warning appears anywhere in the victim's UI at any point.

### Payload

The payload is the poisoned generation_progress.json uploaded by the attacker with the victim's book title: chapters[] entries that preserve the victim's titles/nums (for the selection filter and status matching) but carry attacker-chosen "text"/"sentences"; selected chapters flagged "status":"completed" for the skip/spoof legs (optionally with attacker-planted "Chapter {n} - <title>.<fmt>" audio files from the attacker's own same-title run); "completed_chunks" lists pointing at pre-planted .temp_chunks/abm_ch{idx}/chunk_ch_{idx}_{n}.wav files; and a non-empty "settings" block (attacker-chosen tts_provider_name/tts_model_name for the RCE-delivery chain, pronunciation_map regex pairs for the unconditional text-substitution channel). The only structural requirements: valid JSON ≥ 10 bytes, non-empty text on every chapter, and the victim's sanitized title as the directory key.

## Data flow

### Step 1 — `app.py:1123-1132`

Attacker (source, EXP-confirmed in BUG-R2-C1-A2-H5): progress-JSON upload + the victim's title in the title box; on_generate's setup copies the uploaded file byte-for-byte over audiobook_output/<title>/generation_progress.json (shutil.copy2) before any pipeline work; the upload content passes through no Gradio component, so no choices/bounds validation applies.

### Step 2 — `app.py:82-101 / app.py:1374-1378`

Victim returns and types/gets their title (auto-filled from book metadata on upload, app.py:663, 717); book_title_box.change fires check_existing_progress, which reads the poisoned file and advertises "Existing Progress Found!" with the attacker-controlled completion counts and the promise "Generation will automatically resume from the last completed chapter".

### Step 3 — `app.py:1208-1212 → app.py:975-1013`

The victim clicks Generate; _runner derives prog_json_path from the same title and calls _load_cached_chapters_if_available BEFORE book parsing. Gates: all chapters non-empty text (983) — attacker-satisfied; title filter (996-1004) — passed because the attacker preserved the victim's titles, or bypassed (selected_titles = None) for TXT/no-TOC books and fresh sessions.

### Step 4 — `app.py:1231-1244 / app.py:1294`

Propagation on both dispatch modes: the API-relay path ships the cached (poisoned) chapters as the task payload (api/worker.py:131-152 rebuilds ExtractedChapter and calls run_pipeline with cfg.output_dir = the same title-derived dir); the local-fallback path calls run_pipeline directly. The victim's own book is never parsed.

### Step 5 — `pipeline.py:1033-1046 / audiobook_factory/pipeline.py:1074-1081`

Sink A (content forgery): the poisoned chapters[].text is applied pronunciation fixes, written to export_text .txt files, split into TTS chunks and synthesized, and rendered into LRC/SRT/VTT subtitles — the victim's entire deliverable carries attacker-chosen content. The only log line (app.py:1008) is identical to a legitimate resume; on_preview (app.py:1015-1023) still shows the real book's stats.

### Step 6 — `pipeline.py:612-648`

Sink B (status-skip + output spoofing): run_pipeline reads back the poisoned file (514-515), matches chapters by normalized title or num (612-633); status "completed" + force_reprocess off (default) → chapter skipped (635) and any file at the deterministic "Chapter {idx} - <title>.<fmt>" path in the shared output dir is appended to output_files and returned as the chapter's result (639-648). The attacker's same-title Generate run planted those files (H5 EXP re-render overwrite); regen_missing never triggers because the file exists.

### Step 7 — `pipeline.py:1012-1024 → chapter_pipeline.py:409-420`

Sink C (chunk splicing): poisoned completed_chunks are validated for existence and size ≥ 1000 bytes only (_validate_chunk_file, chapter_pipeline.py:42-61) and the pre-existing .temp_chunks/abm_ch{idx}/chunk_ch_{idx}_{n}.wav files are adopted as cached synthesis results and spliced into the mastered chapter audio; preconditions are defaults (resume_incomplete_chunks on, app.py:549-554; force_reprocess off). Chunk-WAV plant requires the attacker's same-title run to terminate without cleanup (hard termination, or the graceful-cancel race — chapter_pipeline.py:617-619, 651-655 delete chunks on success/receipt).

### Step 8 — `pipeline.py:539-561 / progress_io.py:151-224`

Sink D (settings persistence): the victim's own run replaces book_path, voice_file and settings only when missing or empty (539-549), so the attacker's non-empty values survive; per-chapter status/chunk updates are read-modify-write cycles that never touch settings; the poisoned data is synced to the fixed shared temp/generation_progress.json (557-561). The poisoned file persists at audiobook_output/<title>/generation_progress.json — the path the Colab notebook tells the victim to feed to the headless CLI and the Resume accordion tells them to re-upload.

### Step 9 — `app.py:1338-1342`

Re-offer: on completion the Generate handler appends the (poisoned) progress JSON to the final download list (delivered when out_files is non-empty; on the audited snapshot the API-relay path suppresses the list via the WS early-break regression — api/server.py:276-279 breaks after the status:completed broadcast and listen_ws swallows the close, app.py:1185-1206 — while the local-fallback path and the Export Config download deliver it).

### Step 10 — `app.py:1396-1557`

Re-import: on_progress_upload restores the poisoned settings into the UI — pronunciation_map unconditionally (written to audiobook_output/restored_pronunciation_fixes.txt and preset into the pronunciation component, 1493-1502; applied as re.sub to the text on the next Generate, pipeline.py:1033-1035), selected_chapters unconditionally (1506-1507), book_path/voice_file when they exist (1511-1513), and tts_model_name into the model Dropdown (1447, 1522 — rejected on Gradio ≥ 5.0.0 by the server-side choices check at the next event submit, round-trips on ≤ 4.44.1: the cross-user RCE delivery tracked by HYP-R4-C1-A4-H1).

### Step 11 — `app.py:1601-1612 / app.py:1790-1832`

Export Config twin: the victim's own on_export_config loads the cached (poisoned) chapters the same way (1604 → 975-1013), preserves the poisoned text in the merge (1790-1796), and returns the file as a download with the on-screen instruction "python cli.py <prog_path>" (1826-1832) — the poisoned text and (until the victim's own values overwrite them at 1797-1800) settings propagate into the documented headless CLI workflow.

## References

- https://cwe.mitre.org/data/definitions/345.html
- https://cwe.mitre.org/data/definitions/639.html
- https://owasp.org/www-community/attacks/Insecure_Direct_Object_References

---

_Rendered from original VulnHunter / VulnForge `report.yaml` by OpenVuln._

