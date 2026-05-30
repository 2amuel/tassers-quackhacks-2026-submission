const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();

const DOWNLOAD_DIR = path.resolve(__dirname, '..', 'videos');
const DB_PATH = path.resolve(__dirname, '..', 'videos.db');
const BASE_URL = 'https://sign.mt/?lang=en';

if (!fs.existsSync(DOWNLOAD_DIR)) fs.mkdirSync(DOWNLOAD_DIR, { recursive: true });

function openDb() {
  return new sqlite3.Database(DB_PATH);
}

function ensureTable(db) {
  db.run(`CREATE TABLE IF NOT EXISTS letters (
    letter TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL
  )`);
}

async function main() {
  const db = openDb();
  ensureTable(db);

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();

  const letters = 'abcdefghijklmnopqrstuvwxyz'.split('');

  for (const letter of letters) {
    console.log(`Processing letter: ${letter}`);
    await page.goto(BASE_URL, { waitUntil: 'networkidle' });

    // Heuristic: find a text input / textarea or contenteditable and set the letter
    const candidateSelectors = ['textarea', 'input[type="text"]', 'input', '[contenteditable="true"]'];
    let filled = false;
    for (const sel of candidateSelectors) {
      const exists = await page.$(sel);
      if (exists) {
        try {
          // Prefer fill when possible
          await page.fill(sel, letter);
          filled = true;
          break;
        } catch (e) {
          try {
            await page.evaluate((s, v) => { const el = document.querySelector(s); if (el) { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); } }, sel, letter);
            filled = true;
            break;
          } catch (err) {
            // continue trying other selectors
          }
        }
      }
    }

    if (!filled) {
      console.warn('Could not find an input to type the letter into. Please update the selector heuristics.');
      break;
    }

    // Attempt to click a "Generate" or "Download" control. These labels may change on the site.
    // First try to trigger submit by pressing Enter
    try {
      await page.keyboard.press('Enter');
    } catch (e) {}

    // Wait for a download link or button to appear
    let downloadHandled = false;
    try {
      // Wait for an anchor with a downloadable link
      const dlLink = await page.waitForSelector('a[download]', { timeout: 12000 }).catch(() => null);

      if (dlLink) {
        const [download] = await Promise.all([
          page.waitForEvent('download'),
          dlLink.click()
        ]);

        const suggested = await download.suggestedFilename();
        const saveAs = path.join(DOWNLOAD_DIR, `${letter}.mp4`);
        await download.saveAs(saveAs);
        console.log(`Saved ${saveAs} (suggested name: ${suggested})`);

        db.run('INSERT OR REPLACE INTO letters(letter, filename, created_at) VALUES(?,?,datetime("now"))', [letter, `${letter}.mp4`]);
        downloadHandled = true;
      } else {
        // Try to click any button that contains 'Download' text
        const btn = await page.$('//button[contains(translate(text(), ' + "'ABCDEFGHIJKLMNOPQRSTUVWXYZ'" + ', ' + "'abcdefghijklmnopqrstuvwxyz'" + '), "download")]');
        if (btn) {
          const [download] = await Promise.all([
            page.waitForEvent('download'),
            btn.click()
          ]);
          const saveAs = path.join(DOWNLOAD_DIR, `${letter}.mp4`);
          await download.saveAs(saveAs);
          db.run('INSERT OR REPLACE INTO letters(letter, filename, created_at) VALUES(?,?,datetime("now"))', [letter, `${letter}.mp4`]);
          downloadHandled = true;
        }
      }
    } catch (err) {
      console.warn('Download attempt failed for', letter, err.message);
    }

    if (!downloadHandled) {
      console.log('Could not automatically download for letter:', letter, '- please check site UI and adjust selectors.');
    }

    // Small delay between letters
    await page.waitForTimeout(1200);
  }

  await browser.close();
  db.close();
  console.log('Done. Videos saved into', DOWNLOAD_DIR, 'and mapping stored in', DB_PATH);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
