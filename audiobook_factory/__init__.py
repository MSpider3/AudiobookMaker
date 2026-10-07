"""
audiobook_factory
=================
Core library of AudiobookMaker: extraction, text preparation, TTS providers,
the synthesis pipeline and mastering.

Copyright (C) 2025-2026 Mehul Golecha (MSpider3)
Licensed under the GNU Affero General Public License v3.0 or later; see the
LICENSE and NOTICE files in the repository root.
"""
from __future__ import annotations

__license__: str = "AGPL-3.0-or-later"

SOURCE_URL: str = "https://github.com/MSpider3/AudiobookMaker"
# Shown in the web UI and returned by the API so that anyone using a running
# instance over a network can find its source (AGPL section 13). If you deploy
# a modified version, point this at the source of *your* version.
