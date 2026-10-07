"""
tests/kaggle/sync_assets.py
============================
Copies the fixture books into ``tests/kaggle/assets/books`` so the Kaggle test
notebook reads every input (narrator voice and books) from one folder.

The fixtures under ``tests/fixtures/source_documents`` stay the source of
truth; run this after regenerating them:

    python tests/kaggle/sync_assets.py

``tests/unit/test_kaggle_assets.py`` fails if the two folders drift apart.
"""
from __future__ import annotations

import os
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES_DIR: str = os.path.join(_ROOT, "tests", "fixtures", "source_documents")
ASSETS_DIR: str = os.path.join(_ROOT, "tests", "kaggle", "assets")
BOOKS_DIR: str = os.path.join(ASSETS_DIR, "books")

_BOOK_SUFFIXES: tuple[str, ...] = (".epub", ".pdf", ".docx", ".odt", ".txt", ".mobi", ".json")


def book_files() -> list[str]:
    """Names of the fixture files the notebook needs (books and their expectations)."""
    return sorted(
        name for name in os.listdir(FIXTURES_DIR)
        if name.lower().endswith(_BOOK_SUFFIXES)
    )


def sync() -> list[str]:
    """Copies the fixture books into the assets folder and removes stale copies.

    Returns
    -------
    list[str]
        The file names now present in ``assets/books``.
    """
    os.makedirs(BOOKS_DIR, exist_ok=True)
    wanted = book_files()
    for name in os.listdir(BOOKS_DIR):
        if name not in wanted:
            os.remove(os.path.join(BOOKS_DIR, name))
    for name in wanted:
        shutil.copyfile(os.path.join(FIXTURES_DIR, name), os.path.join(BOOKS_DIR, name))
    return wanted


if __name__ == "__main__":
    for copied in sync():
        print(f"assets/books/{copied}")
