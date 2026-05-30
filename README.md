# ASL Video Creator (Python scraper)

This repository now uses a Python Selenium scraper as the primary method to generate ASL videos from https://sign.mt.

The old JavaScript frontend has been removed from this branch; the focus is on programmatically creating videos for vocabulary (letters, words) and storing them locally with a SQLite mapping.

## Files of interest

- `scripts/scraper.py` — main Python scraper that automates the site, downloads videos into `output/`, and records mappings in `videos.db`.
- `requirements.txt` — Python dependencies (Selenium and webdriver-manager).
- `videos.db` — SQLite database created after running the scraper (not checked into repo).

## Setup

Install dependencies and Chrome webdriver:

```bash
python -m pip install -r requirements.txt
```

## Run

Default (letters a-z):

```bash
python scripts/scraper.py
```

Run with a file containing words (one per line):

```bash
python scripts/scraper.py --file words.txt
```

Run with visible browser for debugging:

```bash
python scripts/scraper.py --no-headless
```

## Output

- `output/` directory with downloaded videos (e.g., `a.mp4`)
- `videos.db` SQLite database mapping letters/words to filenames

## Notes

- The scraper uses heuristics to find input boxes and download links on the site. If the site UI changes, update selectors in `scripts/scraper.py`.
- For more robustness, consider adding retries, explicit waits, or using any available official API from the site.


