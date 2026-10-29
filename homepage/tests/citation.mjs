import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const buildRoot = path.join(projectRoot, 'dist', 'RoboAtom');
const artifactRoot = path.join(projectRoot, '.cache', 'citation-tests');
const paperUrl = 'https://arxiv.org/abs/2610.40207';
const readme = await readFile(path.join(projectRoot, '..', 'README.md'), 'utf8');
const expectedCitation = readme.match(/```bibtex\n([\s\S]*?)\n```/)?.[1];
assert.ok(expectedCitation, 'The code README must include BibTeX.');
assert.ok(expectedCitation.startsWith('@misc{wei2026roboatomlargescaleatomicactionsupervision,'));
assert.ok(expectedCitation.includes(`url={${paperUrl}}`));

process.env.PLAYWRIGHT_BROWSERS_PATH ||= path.join(projectRoot, '.cache', 'ms-playwright');
const { chromium } = await import('playwright');
const browser = await chromium.launch({ headless: true });
await mkdir(artifactRoot, { recursive: true });

async function createFixtureContext(basePath, options = {}) {
  const context = await browser.newContext(options);
  const contentTypes = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.svg': 'image/svg+xml',
  };
  await context.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== 'https://roboatom.test') {
      // External media is outside this focused test; never access the network.
      await route.fulfill({ status: 204, body: '' });
      return;
    }
    assert.ok(url.pathname.startsWith(basePath), 'Assets must respect the site subpath.');
    const relative = url.pathname.slice(basePath.length) || 'index.html';
    assert.ok(['index.html', 'styles.css', 'script.js', 'assets/icons.svg'].includes(relative), relative);
    await route.fulfill({
      body: await readFile(path.join(buildRoot, relative)),
      contentType: contentTypes[path.extname(relative)],
    });
  });
  return context;
}

async function checkContentAndLayout(page) {
  assert.equal(await page.locator('#citation-bibtex').textContent(), expectedCitation);
  const links = await page.locator('[data-resource="paper"]').evaluateAll((elements) =>
    elements.map((element) => ({ href: element.getAttribute('href'), tag: element.tagName })));
  assert.equal(links.length, 2);
  assert.ok(links.every((link) => link.tag === 'A' && link.href === paperUrl));
  assert.equal(await page.locator('a[href=""], a[href="#"]').count(), 0);
  const state = await page.evaluate(() => {
    const citation = document.getElementById('citation');
    const code = citation.querySelector('pre');
    const footer = document.querySelector('footer');
    return {
      isLastSection: document.querySelector('main > section:last-child') === citation,
      isBeforeFooter: Boolean(citation.compareDocumentPosition(footer) & Node.DOCUMENT_POSITION_FOLLOWING),
      overflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - innerWidth,
      codeContained: code.getBoundingClientRect().right <= innerWidth,
      codeScrollable: getComputedStyle(code).overflowX === 'auto',
    };
  });
  assert.ok(state.isLastSection && state.isBeforeFooter, 'Citation belongs before the footer.');
  assert.ok(state.overflow <= 0 && state.codeContained && state.codeScrollable,
    `Citation must not overflow the page: ${JSON.stringify(state)}`);
}

let scenarios = 0;
try {
  for (const basePath of ['/', '/RoboAtom/']) {
    for (const viewport of [
      { width: 1440, height: 900 },
      { width: 1280, height: 800 },
      { width: 390, height: 844 },
      { width: 360, height: 800 },
    ]) {
      const context = await createFixtureContext(basePath, {
        viewport,
        permissions: ['clipboard-read', 'clipboard-write'],
      });
      try {
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        await page.goto(`https://roboatom.test${basePath}`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => document.documentElement.classList.contains('js'));
        await checkContentAndLayout(page);
        const button = page.locator('#copy-citation');
        await button.focus();
        await page.keyboard.press('Enter');
        await page.getByRole('button', { name: 'Copied', exact: true }).waitFor();
        assert.equal(await page.evaluate(() => navigator.clipboard.readText()), expectedCitation);
        assert.equal(await page.locator('#citation-status').textContent(), 'BibTeX copied to clipboard.');
        assert.equal(await button.isDisabled(), false);
        await page.locator('.citation-code').focus();
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction(() => document.querySelector('.citation-code').scrollLeft > 0);
        await page.locator('#citation').screenshot({ path: path.join(artifactRoot,
          `${basePath === '/' ? 'root' : 'subpath'}-${viewport.width}.png`) });
        assert.deepEqual(errors, [], 'No uncaught JavaScript errors.');
        scenarios += 1;
      } finally {
        await context.close();
      }
    }

    const noScript = await createFixtureContext(basePath, {
      javaScriptEnabled: false,
      viewport: { width: 390, height: 844 },
    });
    try {
      const page = await noScript.newPage();
      await page.goto(`https://roboatom.test${basePath}`, { waitUntil: 'domcontentloaded' });
      await checkContentAndLayout(page);
      assert.equal(await page.locator('#copy-citation').isVisible(), false);
      assert.equal(await page.locator('#citation-bibtex').isVisible(), true);
      scenarios += 1;
    } finally {
      await noScript.close();
    }

    for (const mode of ['denied', 'unavailable']) {
      const context = await createFixtureContext(basePath);
      try {
        await context.addInitScript((clipboardMode) => {
          Object.defineProperty(navigator, 'clipboard', {
            configurable: true,
            value: clipboardMode === 'unavailable' ? undefined : {
              writeText: () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError')),
            },
          });
        }, mode);
        const page = await context.newPage();
        await page.goto(`https://roboatom.test${basePath}`, { waitUntil: 'domcontentloaded' });
        const button = page.getByRole('button', { name: 'Copy BibTeX', exact: true });
        await button.click();
        await page.waitForFunction(() => document.getElementById('citation-status')
          .textContent.includes('Please select and copy'));
        assert.equal(await button.isDisabled(), false, 'Allow retry after clipboard failure.');
        assert.equal(await page.locator('#citation-bibtex').textContent(), expectedCitation);
        assert.equal(await page.getByRole('button', { name: 'Copied', exact: true }).count(), 0);
        scenarios += 1;
      } finally {
        await context.close();
      }
    }
  }
  console.info(`Citation checks passed: ${scenarios} browser scenarios; screenshots in .cache/citation-tests/.`);
} finally {
  await browser.close();
}
