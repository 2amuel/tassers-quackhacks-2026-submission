#!/usr/bin/env python3
"""
Playwright-based scraper for sign.mt that inputs letters/words, downloads generated video, and records mappings in SQLite.

Usage:
  python scripts/scraper.py a b c
  or
  python scripts/scraper.py --file words.txt

Notes:
- Requires Playwright Python. Install with `pip install -r requirements.txt` then `playwright install`.
- Output files are saved to `output/` and mapping to `videos.db`.
"""
import sys
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL = "https://sign.mt/?lang=en"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
DB_PATH = ROOT / "videos.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS letters (
            letter TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def find_and_fill_input(page, text):
    selectors = ["textarea", "input[type=\"text\"]", "input", "[contenteditable=\"true\"]"]
    for sel in selectors:
        try:
            el = page.query_selector(sel)
            if el:
                try:
                    el.fill(text)
                    return True
                except Exception:
                    page.evaluate("(s,v)=>{const e=document.querySelector(s); if(e){ e.value=v; e.dispatchEvent(new Event('input',{bubbles:true})); }}", sel, text)
                    return True
        except Exception:
            continue
    return False


def try_trigger_generate(page):
    # Attempt several heuristics to trigger generation
    # 1) press Enter
    try:
        page.keyboard.press("Enter")
        return True
    except Exception:
        pass

    # 2) click buttons containing Generate/Play/Sign/Submit
    texts = ["generate", "play", "sign", "submit", "go", "convert"]
    for t in texts:
        btn = page.query_selector(f"xpath=//button[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{t}')]")
        if btn:
            try:
                btn.click()
                return True
            except Exception:
                pass

    return False


def download_for_word(page, word, timeout=20000):
    # Wait for a download link or trigger a button that starts a download
    # First look for <a download>
    try:
        dl = page.query_selector('a[download]')
        if dl:
            with page.expect_download(timeout=timeout) as download_info:
                dl.click()
            download = download_info.value
            return download
    except PWTimeout:
        pass
    except Exception:
        pass

    # Try clicking any link with 'download' in text
    try:
        dl = page.query_selector("xpath=//a[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]")
        if dl:
            with page.expect_download(timeout=timeout) as download_info:
                dl.click()
            return download_info.value
    except Exception:
        pass

    # Try buttons with download text
    try:
        btn = page.query_selector("xpath=//button[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]")
        if btn:
            with page.expect_download(timeout=timeout) as download_info:
                btn.click()
            return download_info.value
    except Exception:
        pass

    return None


def scrape(words, headless=True):
    conn = ensure_db()
    cur = conn.cursor()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        for word in words:
            print(f"Processing: {word}")
            page.goto(BASE_URL)
            time.sleep(1)

            filled = find_and_fill_input(page, word)
            if not filled:
                print("Warning: couldn't find input box; skipping", word)
                continue

            triggered = try_trigger_generate(page)
            if not triggered:
                print("Warning: couldn't trigger generation for", word)

            # Try to download
            download = None
            try:
                download = download_for_word(page, word, timeout=20000)
            except Exception as e:
                print("Download check error:", e)

            if download is None:
                # As a fallback, take a short video/screenshot or save page for debugging
                print(f"No download detected for {word}. Saving debug snapshot.")
                page.screenshot(path=OUTPUT_DIR / f"{word}-debug.png", full_page=True)
                cur.execute("INSERT OR REPLACE INTO letters(letter, filename, created_at) VALUES(?,?,?)",
                            (word, None, datetime.utcnow().isoformat()))
                conn.commit()
                continue

            # Save the download as word.mp4
            save_path = OUTPUT_DIR / f"{word}.mp4"
            try:
                download.save_as(str(save_path))
                print(f"Saved {save_path}")
                cur.execute("INSERT OR REPLACE INTO letters(letter, filename, created_at) VALUES(?,?,?)",
                            (word, save_path.name, datetime.utcnow().isoformat()))
                conn.commit()
            except Exception as e:
                print("Failed to save download for", word, e)

            time.sleep(1)

        browser.close()
    conn.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('words', nargs='*', help='Words or letters to generate')
    parser.add_argument('--file', '-f', help='Path to a file with words, one per line')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='Run browser visible')
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    words = args.words or []
    if args.file:
        p = Path(args.file)
        if p.exists():
            words.extend([line.strip() for line in p.read_text().splitlines() if line.strip()])

    if not words:
        words = list('abcdefghijklmnopqrstuvwxyz')

    scrape(words, headless=args.headless)


if __name__ == '__main__':
    main()
