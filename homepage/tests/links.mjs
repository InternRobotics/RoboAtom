import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const buildRoot = path.join(projectRoot, 'dist', 'RoboAtom');
const artifactRoot = path.join(projectRoot, '.cache', 'link-tests');
const destinations = {
  website: 'https://internrobotics.github.io/RoboAtom/',
  code: 'https://github.com/InternRobotics/RoboAtom',
  paper: 'https://arxiv.org/abs/2610.40207',
  data: 'https://huggingface.co/datasets/InternRobotics/RoboAtom',
  models: 'https://huggingface.co/InternRobotics/RoboAtom',
};
const badgePrefix = 'https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/badges/';
const codeReadme = await readFile(path.join(projectRoot, '..', 'README.md'), 'utf8');
const badgeRow = codeReadme.match(/<p align="center">[\s\S]*?<\/p>/g)
  ?.find((row) => row.includes(badgePrefix));
assert.ok(badgeRow, 'The README must include its resource row.');
const overviewRow = codeReadme.match(/<p align="center">[\s\S]*?<\/p>/g)
  ?.find((row) => row.includes('alt="RoboAtom overview"'));
assert.ok(overviewRow, 'The overview must have a centered responsive wrapper.');
const overviewUrl = 'https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/figures/roboatom-overview.png';

process.env.PLAYWRIGHT_BROWSERS_PATH ||= path.join(projectRoot, '.cache', 'ms-playwright');
const { chromium } = await import('playwright');
const browser = await chromium.launch({ headless: true });
await mkdir(artifactRoot, { recursive: true });

async function createContext(basePath, options = {}) {
  const context = await browser.newContext({ reducedMotion: 'reduce', ...options });
  await context.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.href.startsWith(badgePrefix)) {
      const name = url.pathname.split('/').at(-1);
      assert.ok(Object.keys(destinations).some((key) => name === `${key}.svg`));
      await route.fulfill({ body: await readFile(path.join(buildRoot, 'assets', 'badges', name)), contentType: 'image/svg+xml' });
    } else if (url.pathname.endsWith('/roboatom-overview.png')) {
      // Match the original dimensions; real-asset screenshots are checked separately.
      await route.fulfill({ body: '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1080"><rect x="1" y="1" width="2398" height="1078" fill="white" stroke="black"/></svg>', contentType: 'image/svg+xml' });
    } else if (route.request().isNavigationRequest() && Object.values(destinations).includes(url.href)) {
      await route.fulfill({ body: '<!doctype html><title>Destination fixture</title>', contentType: 'text/html' });
    } else if (url.origin === 'https://roboatom.test' && url.pathname === '/README.html') {
      await route.fulfill({
        body: `<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{margin:16px}a{display:block}img{display:block;margin:32px auto;max-width:100%;height:auto}</style>${badgeRow}`,
        contentType: 'text/html',
      });
    } else if (url.origin === 'https://roboatom.test' && url.pathname === '/README-overview.html') {
      const html = url.searchParams.has('strip-styles') ? overviewRow.replace(/\sstyle="[^"]*"/g, '') : overviewRow;
      await route.fulfill({
        body: `<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{margin:16px}img{max-width:100%;height:auto}</style>${html}`,
        contentType: 'text/html',
      });
    } else if (url.origin === 'https://roboatom.test') {
      assert.ok(url.pathname.startsWith(basePath));
      const name = url.pathname.slice(basePath.length) || 'index.html';
      assert.ok(['index.html', 'styles.css', 'script.js', 'assets/icons.svg'].includes(name), name);
      const contentType = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.svg': 'image/svg+xml' }[path.extname(name)];
      await route.fulfill({ body: await readFile(path.join(buildRoot, name)), contentType });
    } else {
      // Public availability is audited separately; this test never uses the network.
      await route.fulfill({ status: 204, body: '' });
    }
  });
  return context;
}

async function checkPageLinks(page) {
  await page.waitForFunction(() => [...document.querySelectorAll('.title-link svg, .site-footer .icon-link svg')]
    .every((icon) => icon.getBBox().width > 0));
  const links = await page.locator('a').evaluateAll((elements) => elements
    .filter((link) => link.textContent.trim() && !link.matches('.skip-link, .brand, .figure-image, video a'))
    .map((link) => ({
      label: link.textContent.trim(),
      icon: Boolean(link.querySelector('svg use')),
      decoration: getComputedStyle(link).textDecorationLine,
      href: link.getAttribute('href'),
    })));
  assert.ok(links.length > 10);
  assert.ok(links.every((link) => link.icon && link.decoration === 'none' && link.href), JSON.stringify(links));
  const anchors = await page.locator('a[href^="#"]').evaluateAll((elements) =>
    elements.map((link) => Boolean(document.getElementById(link.getAttribute('href').slice(1)))));
  assert.ok(anchors.every(Boolean), 'All fragment links must have a target.');
  assert.ok(await page.evaluate(() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) <= innerWidth),
    'Icon links must not cause horizontal overflow.');
}

let scenarios = 0;
try {
  for (const basePath of ['/', '/RoboAtom/']) {
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }, { width: 360, height: 800 }]) {
      const context = await createContext(basePath, { viewport });
      try {
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        await page.goto(`https://roboatom.test${basePath}`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => document.documentElement.classList.contains('js'));
        await checkPageLinks(page);
        const title = page.locator('#benchmark-title a');
        const normalBackground = await title.evaluate((link) => getComputedStyle(link).backgroundColor);
        await title.hover();
        const hovered = await title.evaluate((link) => ({
          decoration: getComputedStyle(link).textDecorationLine,
          background: getComputedStyle(link).backgroundColor,
        }));
        assert.equal(hovered.decoration, 'none');
        assert.notEqual(hovered.background, normalBackground, 'Hover must provide a visible cue.');
        await page.keyboard.press('Tab');
        await title.focus();
        assert.equal(await title.evaluate((link) => getComputedStyle(link).outlineStyle), 'solid');
        await page.locator('#benchmark-title').screenshot({ path: path.join(artifactRoot,
          `${basePath === '/' ? 'root' : 'subpath'}-title-${viewport.width}.png`) });
        for (const key of ['paper', 'code', 'data', 'models']) {
          const selector = key === 'paper' ? '[data-resource="paper"]' : `[data-release="${key}"]`;
          const [popup] = await Promise.all([
            page.waitForEvent('popup'),
            page.locator(`.project-resources ${selector}`).click(),
          ]);
          await popup.waitForLoadState('domcontentloaded');
          assert.equal(popup.url(), destinations[key]);
          await popup.close();
        }
        if (viewport.width < 900) {
          await page.locator('.menu-toggle').click();
          await page.locator('#mobile-nav a[href="#video"]').click();
          assert.equal(await page.locator('.menu-toggle').getAttribute('aria-expanded'), 'false');
          assert.equal(new URL(page.url()).hash, '#video');
        }
        await page.locator('.site-footer .icon-link').click();
        await page.waitForFunction(() => scrollY < 2);
        assert.equal(new URL(page.url()).hash, '#home');
        assert.deepEqual(errors, []);
        scenarios += 1;
      } finally {
        await context.close();
      }
    }
    const context = await createContext(basePath, { javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
    try {
      const page = await context.newPage();
      await page.goto(`https://roboatom.test${basePath}`, { waitUntil: 'domcontentloaded' });
      await checkPageLinks(page);
      assert.equal(await page.locator('#copy-citation').isVisible(), false);
      scenarios += 1;
    } finally {
      await context.close();
    }
  }

  for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
    const context = await createContext('/', { viewport });
    try {
      const page = await context.newPage();
      await page.goto('https://roboatom.test/README.html');
      const badges = await page.locator('a').evaluateAll(async (links) => Promise.all(links.map(async (link) => {
        const image = link.querySelector('img');
        await image.decode();
        return { href: link.href, src: image.src, label: image.alt, width: image.naturalWidth, height: image.naturalHeight };
      })));
      assert.equal(badges.length, 5);
      const badgeLayout = await page.locator('p').evaluate((row) => ({
        height: row.getBoundingClientRect().height,
        inline: [...row.querySelectorAll('a, img')].every((element) => getComputedStyle(element).display === 'inline-block'),
      }));
      assert.ok(badgeLayout.inline && badgeLayout.height < 140,
        'Badge attributes must override prose styles that otherwise stack images vertically.');
      for (const [key, href] of Object.entries(destinations)) {
        const badge = badges.find((item) => item.href === href);
        assert.ok(badge && badge.label && badge.width > 90 && badge.height === 36, key);
        assert.equal(badge.src, `${badgePrefix}${key}.svg`);
      }
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({ path: path.join(artifactRoot, `readme-badges-${viewport.width}.png`) });
      scenarios += 1;
    } finally {
      await context.close();
    }
  }
  for (const width of [1920, 1440, 390, 360]) {
    for (const stripStyles of [false, true]) {
      const context = await createContext('/', { viewport: { width, height: 1080 } });
      try {
        const page = await context.newPage();
        await page.goto(`https://roboatom.test/README-overview.html${stripStyles ? '?strip-styles' : ''}`);
        const image = page.getByAltText('RoboAtom overview');
        const geometry = await image.evaluate(async (element) => {
          await element.decode();
          const bounds = element.getBoundingClientRect();
          return {
            width: bounds.width,
            ratioError: Math.abs(bounds.width / bounds.height / (element.naturalWidth / element.naturalHeight) - 1),
            centerOffset: Math.abs(bounds.left + bounds.width / 2 - innerWidth / 2),
          };
        });
        assert.ok(Math.abs(geometry.width - Math.min(1040, width - 32)) < 1, JSON.stringify(geometry));
        assert.ok(geometry.ratioError < 0.002 && geometry.centerOffset < 1, JSON.stringify(geometry));
        assert.equal(await page.locator('a').getAttribute('href'), overviewUrl);
        assert.equal(await page.locator('a').getAttribute('target'), '_blank');
        const [popup] = await Promise.all([page.waitForEvent('popup'), image.click()]);
        await popup.waitForLoadState('domcontentloaded');
        assert.equal(popup.url(), overviewUrl);
        await popup.close();
        scenarios += 1;
      } finally {
        await context.close();
      }
    }
  }
  console.info(`Link and badge checks passed: ${scenarios} browser scenarios; screenshots in .cache/link-tests/.`);
} finally {
  await browser.close();
}
