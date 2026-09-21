# Evidence Context & Inventory for Security Hardening

## Target System Information
- **Repository**: `AudiobookMaker` (`MSpider3/AudiobookMaker`)
- **Assessed Commit/Revision**: Working tree (`qa/full-audit`) incorporating fixes for all 13 vulnerability disclosures in `docs/openvuln-MSpider3-AudiobookMaker-full.md`
- **Python Version**: Python 3.11+
- **Primary Frameworks**: FastAPI (REST/WebSocket), Gradio (Web UI), HuggingFace Transformers / PyTorch (TTS)

## Evidence Inventory

| Evidence ID | Category | Title / Summary | Affected Components |
|---|---|---|---|
| `VULN-01` (`BUG-R2-C1-A2-H8`) | RCE / Deserialization | VibeVoice TTS provider arbitrary model loading via `trust_remote_code=True` | `audiobook_factory/tts_providers/vibevoice_provider.py` |
| `VULN-02` (`BUG-R2-C1-A2-H9`) | Path Traversal / Arbitrary Write | FastAPI `/api/v1/generate` task worker `output_dir` path traversal | `api/worker.py` |
| `VULN-03` (`BUG-R2-C2-A2-H3`) | Path Traversal / Arbitrary Write | Unsanitized `output_format` writes audio files outside `output_dir` | `audiobook_factory/filename_sanitizer.py`, `audiobook_factory/pipeline.py` |
| `VULN-04` (`BUG-R2-C3-A1-H1`) | Resource Exhaustion (Zip Bomb) | EPUB decompression bomb in Docling ingestion | `audiobook_factory/extractor_engine.py` |
| `VULN-05` (`BUG-R2-C3-A1-H2`) | Resource Exhaustion (Zip Bomb) | DOCX decompression bomb in docx ingestion | `audiobook_factory/text_extractor.py` |
| `VULN-06` (`BUG-R2-C3-A1-H3`) | Resource Exhaustion (Zip Bomb) | ODT decompression bomb in ODF ingestion | `audiobook_factory/text_extractor.py` |
| `VULN-07` (`BUG-R2-C3-A3-H1`) | Resource Exhaustion (DoS) | PDF page extraction resource exhaustion via unbounded range expansion | `audiobook_factory/text_extractor.py` |
| `VULN-08` (`BUG-R2-C3-A4-H1`) | Memory Exhaustion (DoS) | EPUB table colspan/rowspan memory explosion in Docling engine | `audiobook_factory/extractor_engine.py` |
| `VULN-09` (`BUG-R4-C1-A4-H1`) | RCE / Path Traversal | CLI progress JSON settings poisoning leading to malicious model load or arbitrary directory write | `cli.py` |
| `VULN-10` (`BUG-R2-C1-A2-H2`) | Information Disclosure | Gradio `on_progress_upload` preset reveals server file paths | `app.py` |
| `VULN-11` (`BUG-R2-C1-A2-H3`) | Information Disclosure / Oracle | Gradio `on_progress_upload` allows probing server file existence via upload handler | `app.py` |
| `VULN-12` (`BUG-R2-C1-A2-H5`) | Authorization / Cross-Session | Cross-session title-keyed progress file overwrite in Gradio UI | `app.py` |
| `VULN-13` (`BUG-R4-C1-A4-H2`) | Information Disclosure | Cross-session cached book chapter text disclosure via book title collision | `app.py` |

## Structural Clusters Identified
1. **Multi-tenant State & Filesystem Sandboxing**: Gradio and FastAPI previously operated under ambient authority where client-supplied book titles, output directories, and progress uploads interacted directly with the host filesystem without strong tenant boundaries.
2. **Pre-flight Ingestion & Archive Containment**: Text extraction from archive-based documents (EPUB, DOCX, ODT) and PDFs was distributed across separate engine methods with inconsistent resource quotas and decompression limits.
3. **Execution Model & Model Provenance**: The TTS provider runtime executes Hugging Face models directly inside the application process; models enabling `trust_remote_code=True` require strict provenance boundaries.
