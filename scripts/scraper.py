#!/usr/bin/env python3
"""
Selenium-based scraper for sign.mt that inputs letters/words, downloads generated videos, and records mappings in SQLite.

Usage:
  python scripts/scraper.py a b c
  or
  python scripts/scraper.py --file words.txt

Notes:
- Requires Selenium Python and webdriver-manager. Install with `pip install -r requirements.txt`.
- Output files are saved to `output/` and mapping to `videos.db`.
"""
import argparse
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

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
            filename TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def create_driver(headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,1000")
    preferences = {
        "download.default_directory": str(OUTPUT_DIR.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    options.add_experimental_option("prefs", preferences)

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver


def fill_text_input(driver, text):
    selectors = [
        (By.CSS_SELECTOR, 'textarea'),
        (By.CSS_SELECTOR, 'input[type="text"]'),
        (By.CSS_SELECTOR, 'input'),
        (By.CSS_SELECTOR, '[contenteditable="true"]'),
    ]
    for by, selector in selectors:
        try:
            element = driver.find_element(by, selector)
            if element:
                try:
                    element.clear()
                    element.send_keys(text)
                    return True
                except WebDriverException:
                    driver.execute_script(
                        "const el = document.querySelector(arguments[0]); if (el) { el.textContent = arguments[1]; el.dispatchEvent(new Event('input', { bubbles: true })); }",
                        selector,
                        text,
                    )
                    return True
        except NoSuchElementException:
            continue
    return False


def click_generate(driver):
    try:
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ENTER)
        return True
    except WebDriverException:
        pass

    text_options = ["generate", "play", "sign", "submit", "go", "convert"]
    for text in text_options:
        xpath = (
            f"//button[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text}')]"
        )
        try:
            button = driver.find_element(By.XPATH, xpath)
            button.click()
            return True
        except NoSuchElementException:
            continue
        except WebDriverException:
            continue
    return False


def find_download_control(driver):
    try:
        return driver.find_element(By.CSS_SELECTOR, 'a[download]')
    except NoSuchElementException:
        pass

    try:
        return driver.find_element(
            By.XPATH,
            "//a[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download') or contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'save')]",
        )
    except NoSuchElementException:
        pass

    try:
        return driver.find_element(
            By.XPATH,
            "//button[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download') or contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'save')]")
    except NoSuchElementException:
        pass

    return None


def wait_for_new_download(existing_names, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        current_files = {p.name for p in OUTPUT_DIR.iterdir() if p.is_file()}
        new_files = current_files - existing_names
        if new_files:
            valid_files = [name for name in new_files if not name.endswith(".crdownload")]
            if valid_files:
                return sorted(valid_files)[0]
        time.sleep(1)
    return None


def scrape(words, headless=True):
    conn = ensure_db()
    cursor = conn.cursor()
    driver = create_driver(headless=headless)

    for word in words:
        print(f"Processing: {word}")
        try:
            driver.get(BASE_URL)
        except WebDriverException as exc:
            print("Failed to load page:", exc)
            break

        time.sleep(2)

        if not fill_text_input(driver, word):
            print("Warning: could not find input field for", word)
            continue

        if not click_generate(driver):
            print("Warning: could not trigger generation for", word)

        time.sleep(2)

        existing_names = {p.name for p in OUTPUT_DIR.iterdir() if p.is_file()}
        control = find_download_control(driver)
        downloaded_name = None

        if control:
            try:
                control.click()
                downloaded_name = wait_for_new_download(existing_names, timeout=45)
            except WebDriverException as exc:
                print("Failed to click download control for", word, exc)
        else:
            print("No download control found for", word)

        if downloaded_name is None:
            screenshot_path = OUTPUT_DIR / f"{word}-debug.png"
            driver.save_screenshot(str(screenshot_path))
            print(f"No video downloaded for {word}. Saved debug screenshot to {screenshot_path}")
            cursor.execute(
                "INSERT OR REPLACE INTO letters(letter, filename, created_at) VALUES(?,?,?)",
                (word, None, datetime.utcnow().isoformat()),
            )
            conn.commit()
            continue

        saved_name = downloaded_name
        print(f"Downloaded {saved_name} for {word}")
        cursor.execute(
            "INSERT OR REPLACE INTO letters(letter, filename, created_at) VALUES(?,?,?)",
            (word, saved_name, datetime.utcnow().isoformat()),
        )
        conn.commit()

        time.sleep(1)

    driver.quit()
    conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Sign.mt video scraper using Selenium.')
    parser.add_argument('words', nargs='*', help='Words or letters to generate')
    parser.add_argument('--file', '-f', help='Path to a file with words, one per line')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='Run browser visible')
    parser.set_defaults(headless=True)
    return parser.parse_args()


def main():
    args = parse_args()
    words = list(args.words)
    if args.file:
        path = Path(args.file)
        if path.exists():
            words.extend([line.strip() for line in path.read_text().splitlines() if line.strip()])

    if not words:
        words = list('abcdefghijklmnopqrstuvwxyz')

    scrape(words, headless=args.headless)


if __name__ == '__main__':
    main()
