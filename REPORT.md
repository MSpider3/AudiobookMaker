# AudiobookMaker — Kaggle GPU Verification & QA Audit Report

**Generated:** 2026-08-26 18:01:03 UTC

**Branch:** `qa/full-audit`


---

## 1. System & Hardware Environment

- **PyTorch:** 2.13.0+cpu
- **CUDA Available:** False
- **GPU Count:** 0
- **Native Rust Extension (`audiobook_rust`):** True

### Detected GPU Devices

| Index | Device Name | VRAM Total | VRAM Free | Compute Capability |
|---|---|---|---|---|
| - | CPU Fallback | N/A | N/A | N/A |

---

## 2. TTS Provider Execution Benchmarks

| Provider | Status | Latency | Duration | RTF (Real-Time Factor) | RMS Level |
|---|---|---|---|---|---|
| `mock` | ✅ PASS | 0.00s | 4.75s | 0.00x | 0.1200 |

---

## 3. End-to-End Multi-Format Document Synthesis (Synthetic Fixtures)

| Format | Status | Chapters | Files Generated | Wall Clock Time |
|---|---|---|---|---|
| **EPUB** | ✅ PASS | 3 | 3 | 2.21s |
| **TXT** | ✅ PASS | 1 | 1 | 0.16s |
| **DOCX** | ✅ PASS | 1 | 1 | 0.21s |

---

## 5. Progress Resumption & Recovery

| Test Case | Status | Detail |
|---|---|---|
| `chunk_resume` | ✅ PASS | Valid output produced: True (17.35s) |

---

## 6. Artifact Quality Gates & Validation

- **Audio Files Validated:** 5 / 5 (100% Passed)
- **Subtitles Monotonicity Verified (LRC/SRT/VTT):** 15 / 15 (100% Passed)
- **Silent Zero / NaN Audio Artifacts Detected:** 0
- **Overall QA Status:** **PASSED ✅**
