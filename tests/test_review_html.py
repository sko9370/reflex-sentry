"""Tests for the standalone human-review tool at tools/review.html.

(a) Static checks that the page makes zero network requests: no http(s)
    references anywhere in the file, and no externally-sourced <script src.
(b) If a headless Chromium is available (playwright may need installing,
    and playwright's own browser download is never run -- we point at the
    prebuilt binary under /opt/pw-browsers instead), drive the page through
    a full review pass over the synthetic fixture and check the exported
    CSV.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVIEW_HTML = ROOT / "tools" / "review.html"
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "review_sample.csv"


def test_review_html_has_no_network_references():
    text = REVIEW_HTML.read_text(encoding="utf-8")
    assert "http://" not in text
    assert "https://" not in text
    assert "<script src" not in text.lower()


# --------------------------------------------------------------------- #
# Optional headless smoke test: only runs if a Chromium binary is on disk.
# --------------------------------------------------------------------- #

def _find_chromium_executable() -> str | None:
    candidates = [
        "/opt/pw-browsers/chromium",
        "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    base = Path("/opt/pw-browsers")
    if base.exists():
        for p in sorted(base.glob("chromium*/chrome-linux/chrome")):
            return str(p)
    return None


def _ensure_playwright() -> bool:
    # Never install packages from a test; skip when playwright is absent.
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


_CHROMIUM_PATH = _find_chromium_executable()
_PLAYWRIGHT_OK = _CHROMIUM_PATH is not None and _ensure_playwright()


@pytest.mark.skipif(not _PLAYWRIGHT_OK, reason="playwright + a Chromium binary are not available")
def test_review_flow_smoke(tmp_path):
    from playwright.sync_api import sync_playwright

    with open(FIXTURE_CSV, newline="", encoding="utf-8") as f:
        orig_fieldnames = csv.DictReader(f).fieldnames

    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_CHROMIUM_PATH, headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(REVIEW_HTML.as_uri())

        page.set_input_files("#fileInput", str(FIXTURE_CSV))
        page.wait_for_selector("#itemContainer.visible")

        # Move focus off the file input so keydowns aren't swallowed by it.
        page.click("#promptText")

        # Accept the recommendation on the first three queue items.
        page.keyboard.press("Enter")
        page.keyboard.press("Enter")
        page.keyboard.press("Enter")

        # On the fourth item, explicitly pick "benign" then accept.
        page.keyboard.press("2")
        page.keyboard.press("Enter")

        with page.expect_download() as dl_info:
            page.click("#exportBtn")
        download = dl_info.value
        out_path = downloads_dir / download.suggested_filename
        download.save_as(str(out_path))

        browser.close()

    assert out_path.name == "review_sample_reviewed.csv"

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # All original columns are preserved, in order, plus _review_action.
    assert fieldnames[: len(orig_fieldnames)] == orig_fieldnames
    assert set(fieldnames) == set(orig_fieldnames) | {"_review_action"}

    assert len(rows) == 4
    assert all(r["_reviewed"] == "1" for r in rows)
    assert all(r["_review_action"] in ("accepted",) or r["_review_action"].startswith("changed from")
               for r in rows)

    # The multi-line, comma-and-quote-containing field round-trips intact.
    r001 = next(r for r in rows if r["id"] == "r001")
    assert "\n" in r001["text"]
    assert '"test"' in r001["text"]
    assert r001["text"].startswith("Explain, in general terms,")
    assert r001["text"].endswith("get reused against a different, unrelated login a stranger owns.")

    r004 = next(r for r in rows if r["id"] == "r004")
    assert r004["gold"] == "benign"
