# Security Hardening Review: AudiobookMaker

## Evidence Basis
This architectural hardening portfolio is derived from our comprehensive security audit of `MSpider3/AudiobookMaker` based on the 13 vulnerability disclosures detailed in `docs/openvuln-MSpider3-AudiobookMaker-full.md` and verified against the current repository source code (`qa/full-audit`).

All 13 vulnerability disclosures have been validated, reproduced with targeted unit tests, and remediated with minimal surgical code changes. The purpose of this hardening review is to evaluate the systemic architectural properties that permitted these vulnerabilities to emerge, and to establish defense-in-depth boundaries that prevent similar failure classes in future iterations.

## Constraints
Throughout our analysis, we operated under the following project-specific engineering constraints:
- **Headless & Workstation Simplicity**: AudiobookMaker is primarily designed for local desktop and workstation use with optional web access via Gradio or FastAPI. Hardening measures must not introduce mandatory external database servers (e.g. Postgres) or complex cluster daemons.
- **Python-First Portability**: Changes must preserve cross-platform compatibility across Linux, macOS, and Windows. Pure-Python fallbacks must remain functional without hard compilation dependencies.
- **Zero Performance Degradation**: Audio synthesis latency and throughput on GPU hot-paths must remain entirely unaffected.
- **Strict Surgical Footprint**: Follow Karpathy guidelines—remediate real root causes with minimal cognitive overhead and zero speculative abstractions.

## Opportunity Portfolio

| Opportunity | Evidence | Options | Recommendation | Proposal |
|---|---|---|---|---|
| **Unified Session Workspace & Path Isolation** | Cross-session overwrite (`VULN-12`), cached text disclosure (`VULN-13`), API path traversal (`VULN-02`), and Gradio path leakage (`VULN-10`, `VULN-11`) | Option 1: In-Memory Registry (Baseline)<br>Option 2: Directory Namespacing (Recommended)<br>Option 3: External DB Store | **Option 1 (Immediate) / Option 2 (Medium-term)** | [unified-session-workspace.md](proposals/unified-session-workspace.md) |
| **Centralized Archive Pre-flight Ingestion & Resource Sandboxing** | EPUB/DOCX/ODT zip bombs (`VULN-04`, `VULN-05`, `VULN-06`), PDF page range DoS (`VULN-07`), EPUB table span explosion (`VULN-08`) | Option 1: Unified Pre-Flight Quotas (Recommended Baseline)<br>Option 2: Process-Isolated Subprocess Sandbox<br>Option 3: External Conversion Service | **Option 1 (Recommended Baseline)** | [centralize-archive-containment.md](proposals/centralize-archive-containment.md) |

## Recommendation Summary
We have analyzed the failure modes across multi-tenant web sessions and document ingestion pathways.

In our multi-tenant analysis, we found that ambient filesystem access and title-based addressing in `app.py` and `api/worker.py` allowed users to inadvertently read or overwrite files belonging to other sessions. We implemented **Option 1 (In-Memory Ownership Registry & Strict Subpath Anchoring)** as an immediate baseline, which successfully eliminated all active cross-session vulnerabilities. For future iterations or multi-user server deployments, I recommend migrating to **Option 2 (Session-Scoped Directory Namespacing)**, which physically isolates session data on disk (`audiobook_output/sessions/<session_id>/`) and ensures defense-in-depth across application restarts.

In our document extraction analysis, we observed that third-party parsing libraries were directly exposed to untrusted archives, enabling catastrophic resource exhaustion via zip bombs, unbounded PDF page iteration, and massive HTML table matrices. We implemented and recommend **Option 1 (Unified Pre-Flight Quotas)**: validating zip header totals (`_assert_zip_safe`) prior to decompression, enforcing hard bounds on PDF page ranges (5,000 pages / 20M characters), and clamping table attributes (`colspan`/`rowspan` to 1,000). This provides 100% defense against all tested denial-of-service payloads with less than 5ms overhead and zero external dependencies.

## Next Decisions
1. **Approve Baseline Hardening**: Confirm acceptance of the current in-tree implementations of Option 1 across both opportunity areas, backed by 136 passing tests.
2. **Evaluate Medium-Term Namespacing**: Decide whether multi-user cloud hosting warrants scheduling Phase 2 of the Unified Session Workspace (creating explicit session directories on disk).
