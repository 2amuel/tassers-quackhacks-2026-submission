"""Scrape generated sign.mt videos for letters a-z.

Requires:
    pip install playwright
    python -m playwright install chromium

Example:
    python video_scraper.py --concurrency 6 --min-wait 10
"""

from __future__ import annotations

import argparse
import asyncio
import string
from pathlib import Path
from urllib.parse import urlparse


SIGN_MT_URL = "https://sign.mt/?lang=en"
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".m4v")
VIDEO_CONTENT_TYPES = ("video/",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download generated sign.mt videos for letters.")
    parser.add_argument("--url", default=SIGN_MT_URL, help="sign.mt translation page URL.")
    parser.add_argument(
        "--letters",
        default=string.ascii_lowercase,
        help="Letters to scrape, for example 'abc' or 'abcdefghijklmnopqrstuvwxyz'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sign_mt_videos"),
        help="Base output directory. Each letter gets its own subfolder.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=26,
        help="Number of letters to process in parallel.",
    )
    parser.add_argument(
        "--min-wait",
        type=float,
        default=10.0,
        help="Minimum seconds to wait after entering each letter.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum seconds to wait for one generated video.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Show browser windows instead of running headless.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Skip letters that already have a downloaded video file.",
    )
    return parser.parse_args()


def safe_letters(raw_letters: str) -> list[str]:
    letters = []
    for char in raw_letters.lower():
        if char in string.ascii_lowercase and char not in letters:
            letters.append(char)
    if not letters:
        raise ValueError("No letters found. Pass letters like --letters abc.")
    return letters


def video_suffix_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return suffix
    return ".mp4"


async def install_hint() -> None:
    raise SystemExit(
        "Playwright is not installed. Run:\n"
        "  .venv/bin/python -m pip install playwright\n"
        "  .venv/bin/python -m playwright install chromium"
    )


async def fill_translation_box(page, letter: str) -> None:
    candidates = [
        "textarea",
        "[contenteditable='true']",
        "input[type='text']",
        "input:not([type])",
    ]

    for selector in candidates:
        locators = await page.locator(selector).all()
        for locator in locators:
            try:
                if await locator.is_visible() and await locator.is_enabled():
                    await locator.click()
                    modifier = "Meta" if await page.evaluate("navigator.platform.includes('Mac')") else "Control"
                    await page.keyboard.press(f"{modifier}+A")
                    await page.keyboard.type(letter)
                    await page.keyboard.press("Enter")
                    return
            except Exception:
                continue

    role_box = page.get_by_role("textbox").first
    await role_box.click()
    modifier = "Meta" if await page.evaluate("navigator.platform.includes('Mac')") else "Control"
    await page.keyboard.press(f"{modifier}+A")
    await page.keyboard.type(letter)
    await page.keyboard.press("Enter")


async def dom_video_candidate(page) -> tuple[str, bytes | None] | None:
    video_info = await page.evaluate(
        """async () => {
            const video = [...document.querySelectorAll('video')]
                .find(v => v.currentSrc || v.src || v.querySelector('source')?.src);
            if (!video) return null;

            const src = video.currentSrc || video.src || video.querySelector('source')?.src;
            const ready = video.readyState >= 2;
            const duration = Number.isFinite(video.duration) ? video.duration : null;
            return { src, ready, duration };
        }"""
    )
    if not video_info or not video_info.get("src"):
        return None

    src = video_info["src"]
    if src.startswith("blob:"):
        data = await page.evaluate(
            """async (blobUrl) => {
                const response = await fetch(blobUrl);
                const buffer = await response.arrayBuffer();
                return Array.from(new Uint8Array(buffer));
            }""",
            src,
        )
        return ".mp4", bytes(data)

    return src, None


async def download_from_url(page, url: str) -> bytes:
    response = await page.request.get(url)
    if not response.ok:
        raise RuntimeError(f"Video request failed with HTTP {response.status}: {url}")
    return await response.body()


async def scrape_letter(browser, args: argparse.Namespace, letter: str, semaphore: asyncio.Semaphore) -> Path:
    async with semaphore:
        letter_dir = args.output_dir / letter
        letter_dir.mkdir(parents=True, exist_ok=True)

        existing = list(letter_dir.glob(f"{letter}.*"))
        if args.keep_existing and existing:
            print(f"[{letter}] keeping existing {existing[0]}")
            return existing[0]

        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        video_response: dict[str, bytes | str] = {}

        async def on_response(response) -> None:
            url = response.url
            content_type = response.headers.get("content-type", "").lower()
            path = urlparse(url).path.lower()
            looks_like_video = content_type.startswith(VIDEO_CONTENT_TYPES) or path.endswith(VIDEO_EXTENSIONS)
            if not looks_like_video or video_response:
                return
            try:
                body = await response.body()
            except Exception:
                return
            if body:
                video_response["url"] = url
                video_response["body"] = body

        page.on("response", lambda response: asyncio.create_task(on_response(response)))

        try:
            await page.goto(args.url, wait_until="domcontentloaded", timeout=45_000)
            await fill_translation_box(page, letter)

            started = asyncio.get_running_loop().time()
            min_done_at = started + args.min_wait
            timeout_at = started + args.timeout
            candidate: tuple[str, bytes | None] | None = None

            while asyncio.get_running_loop().time() < timeout_at:
                await page.wait_for_timeout(500)
                if video_response:
                    candidate = (str(video_response["url"]), bytes(video_response["body"]))
                else:
                    candidate = await dom_video_candidate(page)

                if candidate and asyncio.get_running_loop().time() >= min_done_at:
                    break

            if not candidate:
                screenshot_path = letter_dir / f"{letter}_debug.png"
                html_path = letter_dir / f"{letter}_debug.html"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                html_path.write_text(await page.content(), encoding="utf-8")
                raise RuntimeError(
                    f"[{letter}] no video found after {args.timeout}s. "
                    f"Saved {screenshot_path} and {html_path}."
                )

            candidate_url_or_suffix, candidate_body = candidate
            if candidate_body is None:
                suffix = video_suffix_from_url(candidate_url_or_suffix)
                video_body = await download_from_url(page, candidate_url_or_suffix)
            else:
                suffix = candidate_url_or_suffix if candidate_url_or_suffix.startswith(".") else video_suffix_from_url(candidate_url_or_suffix)
                video_body = candidate_body

            output_path = letter_dir / f"{letter}{suffix}"
            output_path.write_bytes(video_body)
            print(f"[{letter}] saved {output_path}")
            return output_path
        finally:
            await context.close()


async def main() -> None:
    args = parse_args()

    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError:
        await install_hint()

    letters = safe_letters(args.letters)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not args.headful)
        try:
            semaphore = asyncio.Semaphore(args.concurrency)
            tasks = [scrape_letter(browser, args, letter, semaphore) for letter in letters]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await browser.close()

    failures = []
    for letter, result in zip(letters, results):
        if isinstance(result, Exception):
            failures.append((letter, result))

    if failures:
        for letter, error in failures:
            print(f"[{letter}] failed: {error}")
        raise SystemExit(1)

    print(f"Downloaded {len(results)} videos into {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
