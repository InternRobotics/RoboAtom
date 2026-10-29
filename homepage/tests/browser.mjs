import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const artifactRoot = path.join(projectRoot, 'artifacts');
const screenshotRoot = path.join(artifactRoot, 'screenshots');
const baseUrl = process.env.BASE_URL;
process.env.PLAYWRIGHT_BROWSERS_PATH ||= path.join(projectRoot, '.cache', 'ms-playwright');
const { chromium } = await import('playwright');

if (!baseUrl) {
  throw new Error('Set BASE_URL to the local preview URL.');
}

const viewports = [
  { width: 1440, height: 900 },
  { width: 1280, height: 800 },
  { width: 390, height: 844 },
  { width: 360, height: 800 },
];

const expectedTables = {
  annotation: [
    ['GPT-4o', '31.5', '13.9', '7.2', '47.7', '0.971', '46.1', '14.6', '0.800'],
    ['GPT-5.6-sol', '44.0', '28.0', '21.4', '63.4', '4.330', '41.9', '31.3', '3.375'],
    ['Gemini 3.1 Pro', '42.5', '24.4', '16.1', '61.8', '4.516', '52.1', '29.6', '4.389'],
    ['Qwen3.8-27B', '38.4', '20.1', '10.9', '56.6', '0.103', '51.3', '24.3', '0.096'],
    ['Qwen3.5-9B', '9.6', '6.2', '3.5', '28.7', '0.008', '12.8', '1.7', '0.007'],
    ['RoboAtom-Anno-9B', '55.1', '41.0', '30.1', '76.7', '0.006', '61.7', '48.3', '0.004'],
    ['RoboAtom-Anno-27B', '56.4', '43.2', '33.8', '76.8', '0.078', '66.0', '49.2', '0.054'],
  ],
  control: [
    ['\u03c00.5', '95.00', '93.01', '85.23', '79.93', '78.56', '84.05', '80.70', '48.04', '64.37'],
    ['Ours (dual system)', '98.20', '92.46', '91.86', '86.34', '82.29', '88.18', '82.02', '50.32', '66.17'],
  ],
  extension: [
    ['GPT-4o', '48.2', '20.7', '11.1', '10', '0.888'],
    ['Gemini 3.1 Pro', '61.1', '44.9', '39.7', '39', '2.877'],
    ['GPT-5.6-sol', '58.9', '37.8', '35.0', '46', '3.510'],
    ['Qwen3.8-27B', '46.1', '25.4', '19.1', '25', '0.051'],
    ['Cortex', '66.5', '49.8', '48.5', '37', '4.8 \u00d7 10-4'],
    ['PhysExtend', '76.8', '71.6', '71.0', '69', '2.8 \u00d7 10-5'],
  ],
};

await mkdir(screenshotRoot, { recursive: true });
const browser = await chromium.launch({ headless: true });
const report = {
  baseUrl,
  browser: await browser.version(),
  viewports: [],
  noJavaScript: null,
  configuredResources: null,
  backgroundPlayback: null,
  passed: true,
};

function record(checks, label, passed, details = undefined) {
  checks.push({ label, passed: Boolean(passed), ...(details === undefined ? {} : { details }) });
}

async function pageDiagnostics(page) {
  return page.evaluate(() => {
    const images = Array.from(document.images).map((image) => ({
      src: image.getAttribute('src'),
      loaded: image.complete && image.naturalWidth > 0,
      width: image.naturalWidth,
      height: image.naturalHeight,
    }));
    const resources = Array.from(document.querySelectorAll('[data-resource]')).map((element) => ({
      key: element.dataset.resource,
      tag: element.tagName,
      tabIndex: element.tabIndex,
      href: element.getAttribute('href'),
    }));
    const invalidLinks = Array.from(document.querySelectorAll('a')).filter((link) => {
      const href = link.getAttribute('href');
      return href === '' || href === '#';
    }).length;
    const rootRelativeAssets = Array.from(document.querySelectorAll('img[src], script[src], link[href], video[poster], video source[src]'))
      .map((element) => element.getAttribute('src') || element.getAttribute('href') || element.getAttribute('poster'))
      .filter((value) => value?.startsWith('/'));
    const videos = Array.from(document.querySelectorAll('video')).map((video) => ({
      src: video.querySelector('source')?.getAttribute('src'),
      poster: video.getAttribute('poster'),
      readyState: video.readyState,
      duration: video.duration,
      loaded: video.readyState >= 2 && Number.isFinite(video.duration) && video.duration > 0,
    }));
    const heroMedia = document.querySelector('.hero video').getBoundingClientRect();
    return {
      title: document.title,
      images,
      videos,
      resources,
      invalidLinks,
      rootRelativeAssets,
      horizontalOverflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - window.innerWidth,
      heroVisible: heroMedia.top < window.innerHeight && heroMedia.bottom > 0,
    };
  });
}

async function loadAllImages(page) {
  await page.evaluate(async () => {
    const images = Array.from(document.images);
    for (const image of images) {
      image.loading = 'eager';
      image.scrollIntoView({ block: 'center' });
      if (!image.complete) {
        await new Promise((resolve) => {
          image.addEventListener('load', resolve, { once: true });
          image.addEventListener('error', resolve, { once: true });
        });
      }
      try {
        await image.decode();
      } catch {
        // The diagnostics below record broken images; decoding must not hide them.
      }
    }
    const videos = Array.from(document.querySelectorAll('video'));
    for (const video of videos) {
      video.scrollIntoView({ block: 'center' });
      video.preload = 'auto';
      if (video.readyState < 2) {
        await Promise.race([
          new Promise((resolve) => {
            video.addEventListener('loadeddata', resolve, { once: true });
            video.addEventListener('error', resolve, { once: true });
          }),
          new Promise((resolve) => setTimeout(resolve, 10000)),
        ]);
      }
    }
  });
  await page.evaluate(() => window.scrollTo(0, 0));
}

async function checkPaperContent(page, checks) {
  const actual = await page.evaluate(() => {
    const rows = (panel) => Array.from(document.querySelectorAll(`#panel-${panel} tbody tr`)).map((row) =>
      Array.from(row.children).map((cell) => cell.textContent.trim().replace(/\s+/g, ' ')));
    return {
      annotation: rows('annotation'),
      control: rows('control'),
      extension: rows('extension'),
      text: document.body.innerText,
    };
  });
  for (const [panel, expected] of Object.entries(expectedTables)) {
    const passed = JSON.stringify(actual[panel]) === JSON.stringify(expected);
    record(checks, `${panel} table matches paper values`, passed, passed ? undefined : { expected, actual: actual[panel] });
  }
  record(checks, 'repeated extension notes are removed',
    !actual.text.includes('100 trajectories (1 per task)') && !actual.text.includes('71.0 with one reference'));
  record(checks, 'standalone number strips are removed', await page.locator('.stats-band, .benchmark-facts').count() === 0);
}

async function checkReleaseLinks(page, checks) {
  const expected = {
    code: 'https://github.com/InternRobotics/RoboAtom',
    data: 'https://huggingface.co/datasets/InternRobotics/RoboAtom',
    models: 'https://huggingface.co/InternRobotics/RoboAtom',
  };
  const links = await page.locator('[data-release]').evaluateAll((elements) => elements.map((element) => ({
    key: element.dataset.release,
    tag: element.tagName,
    href: element.getAttribute('href'),
    target: element.target,
    rel: element.rel,
  })));
  record(checks, 'resource links match the release README destinations', links.length === 12
    && links.every((link) => link.tag === 'A' && link.href.split('#')[0] === expected[link.key]
      && link.target === '_blank' && link.rel.includes('noopener') && link.rel.includes('noreferrer')), links);
  const mediaSources = await page.locator('video source').evaluateAll((elements) => elements.map((element) => element.getAttribute('src')));
  record(checks, 'all website videos use public Hugging Face file URLs', mediaSources.length === 20
    && mediaSources.every((url) => url.startsWith('https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/')), mediaSources);
  record(checks, 'benchmark and model titles link to Hugging Face',
    await page.locator('#benchmark-title a').getAttribute('href') === expected.data
    && await page.locator('#anno-title a').getAttribute('href') === expected.models);
}

async function checkIdentityAndResources(page, checks) {
  const state = await page.evaluate(() => {
    const resources = document.querySelector('.project-resources');
    const video = document.getElementById('overview-video');
    const paper = resources.querySelector('[data-resource="paper"]');
    const artwork = document.querySelector('#hero-title img');
    return {
      favicon: document.querySelector('link[rel="icon"]').getAttribute('href'),
      heroArtwork: artwork.getAttribute('src'),
      heroArtworkReady: artwork.complete && artwork.naturalWidth === 2060,
      brandIcon: document.querySelector('.brand-icon').getAttribute('src'),
      resourcesAboveVideo: resources.getBoundingClientRect().bottom <= video.getBoundingClientRect().top,
      noVideoHeading: Array.from(document.querySelectorAll('#video h2')).every((heading) => heading.textContent.trim() !== 'Video'),
      paperLinked: paper.tagName === 'A' && paper.getAttribute('href') === 'https://arxiv.org/abs/2610.40207'
        && paper.getClientRects().length > 0,
      iconsReady: Array.from(resources.querySelectorAll('.resource-link')).every((element) => {
        const icon = element.querySelector('svg');
        return icon && icon.getBBox().width > 0 && icon.getBBox().height > 0;
      }),
      subtitleSize: parseFloat(getComputedStyle(document.querySelector('.hero-paper-title')).fontSize),
      resourceSize: Math.min(...Array.from(resources.querySelectorAll('.resource-link')).map((element) => parseFloat(getComputedStyle(element).fontSize))),
      minimumTableSize: Math.min(...Array.from(document.querySelectorAll('tbody td')).map((element) =>
        parseFloat(getComputedStyle(element).fontSize))),
      minimumHeaderSize: Math.min(...Array.from(document.querySelectorAll('thead th')).map((element) =>
        parseFloat(getComputedStyle(element).fontSize))),
    };
  });
  record(checks, 'RoboAtom uses the original artwork for the title, brand and favicon',
    state.favicon === 'https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/brand/roboatom-icon.png' && state.brandIcon === state.favicon
    && state.heroArtwork === 'https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/brand/roboatom-wordmark.png' && state.heroArtworkReady, state);
  record(checks, 'icon resource links replace the standalone Video heading',
    state.resourcesAboveVideo && state.noVideoHeading && state.iconsReady);
  record(checks, 'paper link is present with and without JavaScript', state.paperLinked);
  record(checks, 'hero subtitle and resource entries are enlarged', state.subtitleSize >= 28 && state.resourceSize >= 22);
  record(checks, 'tables retain readable body and header sizes', state.minimumTableSize >= 22 && state.minimumHeaderSize >= 19,
    { body: state.minimumTableSize, header: state.minimumHeaderSize });
}

async function checkNumericAlignment(page, checks) {
  const auxiliaryNumbers = await page.locator('tbody td').evaluateAll((elements) =>
    elements.map((element) => getComputedStyle(element).fontVariantNumeric));
  record(checks, 'table values use lining tabular figures', auxiliaryNumbers.every((value) =>
    value.includes('lining-nums') && value.includes('tabular-nums')));
}

async function checkConciseContent(page, checks) {
  const state = await page.evaluate(() => ({
    terminalPeriods: Array.from(document.querySelectorAll('p, span')).filter((element) =>
      /[.。]$/.test(element.textContent.trim())).map((element) => element.textContent.trim()),
    anchorTargets: ['home', 'video', 'data', 'overview', 'examples', 'method', 'physextend', 'anno', 'benchmark', 'results', 'control']
      .every((id) => document.getElementById(id)),
    sectionOrder: Array.from(document.querySelectorAll('main > section[id]')).map((element) => element.id),
    noFigureCaptions: document.querySelectorAll('figcaption, [data-caption], #dialog-caption, .table-note, .caption-detail').length === 0,
    noRemovedNotes: !document.body.textContent.includes('(one reference)') && !document.body.textContent.includes('†'),
  }));
  record(checks, 'paragraphs and spans have no terminal periods', state.terminalPeriods.length === 0, state.terminalPeriods);
  record(checks, 'existing deep links remain available', state.anchorTargets);
  record(checks, 'sections follow the simplified reading order', JSON.stringify(state.sectionOrder)
    === JSON.stringify(['home', 'video', 'examples', 'data', 'benchmark', 'anno', 'control']), state.sectionOrder);
  record(checks, 'figure and table explanatory fine print is removed', state.noFigureCaptions && state.noRemovedNotes);
}

async function checkResearchLayout(page, checks, viewport) {
  const state = await page.evaluate(() => ({
    researchSections: Array.from(document.querySelectorAll('.research-section')).map((element) => element.id),
    whiteSections: Array.from(document.querySelectorAll('.research-section')).every((element) =>
      getComputedStyle(element).backgroundColor === 'rgb(255, 255, 255)'),
    bareFigures: Array.from(document.querySelectorAll('.figure-image')).every((element) => {
      const style = getComputedStyle(element);
      return style.borderTopWidth === '0px' && style.borderRadius === '0px' && style.paddingTop === '0px';
    }),
    visibleTiles: Array.from(document.querySelectorAll('[data-wall-video]')).filter((video) => video.getClientRects().length).length,
    uniqueSources: new Set(Array.from(document.querySelectorAll('[data-wall-video] source')).map((source) => source.getAttribute('src'))).size,
    removedTrainingText: ['Robot-domain mid-training', 'Supervised fine-tuning', 'GRPO reward optimization'].every((text) => !document.body.textContent.includes(text)),
    datasetRows: Array.from(document.querySelectorAll('#dataset-comparison tbody tr')).map((row) =>
      Array.from(row.children).map((cell) => cell.textContent.trim().replaceAll('†', ''))),
    panels: ['data-composition', 'benchmark-radar', 'benchmark-distribution', 'annotation-progress'].every((name) =>
      document.querySelector(`img[src^="https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/figures/${name}.svg"]`)),
  }));
  const datasetRows = [
    ['AgiBot World Beta', 'Real', '2,976.4', '141,563', '1,001,552', '77', '217', '1'],
    ['AgiBot World 2026', 'Real', '591.6', '28,331', '327,355', '43', '37', '1'],
    ['Galaxea Open-World', 'Real', '500.0', '19,183', '108,775', '58', '150', '1'],
    ['RoboInter-Data', 'Real', '2,417.1', '235,880', '752,388', '15', '232', '6'],
    ['BEHAVIOR-1K 2026', 'Sim', '1,952.9', '20,000', '406,341', '35', '100', '1'],
    ['RoboAtom-Data', 'Mixed', '16,882.1', '1,446,974', '6,876,282', '29', '3,334', '32'],
  ];
  record(checks, 'four research sections follow the requested order', JSON.stringify(state.researchSections)
    === JSON.stringify(['data', 'benchmark', 'anno', 'control']), state.researchSections);
  record(checks, 'video wall uses sixteen distinct source videos', state.uniqueSources === 16);
  record(checks, 'video wall adapts to desktop and mobile', state.visibleTiles === (viewport.width <= 600 ? 8 : 16), state.visibleTiles);
  record(checks, 'three-step training list is removed', state.removedTrainingText);
  record(checks, 'dataset comparison matches the paper', JSON.stringify(state.datasetRows) === JSON.stringify(datasetRows), state.datasetRows);
  record(checks, 'research figures include both benchmark charts', state.panels);
  const benchmarkLayout = await page.evaluate(() => {
    const figures = Array.from(document.querySelectorAll('.benchmark-layout figure')).map((figure) => figure.getBoundingClientRect().toJSON());
    return { figures, tables: document.querySelectorAll('#benchmark table').length };
  });
  const [radar, ring] = benchmarkLayout.figures;
  record(checks, 'benchmark charts adapt to the viewport without a duplicate subset table', radar && ring
    && (viewport.width > 900 ? ring.left >= radar.right : ring.top >= radar.bottom)
    && benchmarkLayout.tables === 0
    && (viewport.width <= 900 || Math.abs(ring.height - radar.height * 0.92) < 3)
    && (viewport.width <= 900 || Math.abs(ring.top - radar.top) < 2), benchmarkLayout);
  for (const name of ['benchmark-radar', 'benchmark-distribution']) {
    await page.locator(`a[href^="https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/figures/${name}.svg"]`).click();
    const dialog = page.locator('#figure-dialog');
    record(checks, `${name} opens in the figure viewer`, await dialog.isVisible()
      && (await dialog.locator('img').getAttribute('src')).includes(`${name}.svg`));
    await page.keyboard.press('Escape');
    record(checks, `${name} viewer supports Escape`, !(await dialog.isVisible()));
  }
  record(checks, 'dual system keeps its architecture and results table without the duplicate chart',
    await page.locator('#control img[src="https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/figures/dual-system.png"]').count() === 1
    && await page.locator('#control img[src="https://huggingface.co/datasets/InternRobotics/RoboAtom/resolve/main/project_media/figures/control-results.svg"]').count() === 0
    && await page.locator('#panel-control table').count() === 1);
  record(checks, 'research sections use a continuous white background', state.whiteSections);
  record(checks, 'figures have no surrounding cards or padding', state.bareFigures);
}

async function navigateToOverview(page) {
  const desktopLink = page.locator('.desktop-nav a[href="#video"]');
  if (await desktopLink.isVisible()) await desktopLink.click();
  else {
    await page.locator('.menu-toggle').click();
    await page.locator('#mobile-nav a[href="#video"]').click();
  }
}

async function exerciseOverview(page, checks, viewport) {
  await page.locator('#home').evaluate((element) => element.scrollIntoView());
  await page.waitForFunction(() => Array.from(document.querySelectorAll('[data-wall-video]')).every((video) => video.getClientRects().length ? !video.paused && video.currentTime > 0 : video.paused));
  record(checks, 'video wall loops automatically without a pause button', await page.locator('#hero-playback').count() === 0);
  await navigateToOverview(page);
  const player = page.locator('#overview-video');
  await page.waitForFunction(() => document.getElementById('overview-video').readyState >= 2);
  const state = await player.evaluate((video) => ({
    width: video.videoWidth, height: video.videoHeight, duration: video.duration, controls: video.controls, muted: video.muted,
    fit: getComputedStyle(video).objectFit, paused: video.paused,
  }));
  record(checks, 'Video link reaches the inline player', new URL(page.url()).hash === '#video'
    && await page.locator('#video-dialog').count() === 0);
  record(checks, 'inline overview preserves full aspect ratio and native controls', state.width === 1920
    && state.height === 1080 && Math.abs(state.duration - 88.5) < 0.1 && !state.muted && state.controls && state.fit === 'contain', state);
  record(checks, 'overview waits for user playback', state.paused);
  await player.focus();
  await page.keyboard.press('Space');
  await page.waitForFunction(() => !document.getElementById('overview-video').paused
    && document.getElementById('overview-video').currentTime > 0);
  record(checks, 'inline overview supports keyboard playback', true);
  await page.keyboard.press('Space');
  await page.waitForFunction(() => document.getElementById('overview-video').paused);
  await player.evaluate((video) => { video.currentTime = 47.5; });
  await page.waitForFunction(() => {
    const video = document.getElementById('overview-video');
    return !video.seeking && Math.abs(video.currentTime - 47.5) < 0.1;
  });
  record(checks, 'inline overview supports seeking', true);
  await page.screenshot({ path: path.join(screenshotRoot, `${viewport.width}x${viewport.height}-overview.png`) });
  await player.evaluate((video) => video.play());
  await page.locator('#data').evaluate((element) => element.scrollIntoView());
  await page.waitForFunction(() => document.getElementById('overview-video').paused);
  record(checks, 'inline overview pauses outside the viewport', true);
}

async function exerciseInteractions(page, viewport, checks) {
  await exerciseOverview(page, checks, viewport);
  const menu = page.locator('.menu-toggle');
  const name = `${viewport.width}x${viewport.height}`;
  if (viewport.width <= 900) {
    await menu.evaluate((element) => { element.click(); element.click(); element.click(); });
    record(checks, 'mobile menu opens', await menu.getAttribute('aria-expanded') === 'true'
      && await page.locator('#mobile-nav').isVisible());
    record(checks, 'rapid menu clicks leave one consistent navigation', await page.locator('#mobile-nav').count() === 1
      && await menu.getAttribute('aria-expanded') === 'true');
    await page.screenshot({ path: path.join(screenshotRoot, `${name}-menu.png`) });
    await page.keyboard.press('Escape');
    record(checks, 'Escape closes menu and restores focus', await menu.getAttribute('aria-expanded') === 'false'
      && await menu.evaluate((element) => element === document.activeElement));
    await menu.click();
    await page.locator('#mobile-nav a[href="#video"]').click();
    record(checks, 'mobile navigation closes after selection', await menu.getAttribute('aria-expanded') === 'false'
      && await page.locator('#mobile-nav').isHidden());
  } else {
    await page.locator('.desktop-nav a[href="#video"]').click();
    record(checks, 'desktop navigation reaches video', new URL(page.url()).hash === '#video');
  }

  record(checks, 'results are visible in their respective sections',
    await page.locator('#data #panel-extension').isVisible()
    && await page.locator('#anno #panel-annotation').isVisible()
    && await page.locator('#control #panel-control').isVisible());
  await page.locator('#control').evaluate((element) => element.scrollIntoView());
  await page.screenshot({ path: path.join(screenshotRoot, `${name}-control.png`) });

  const figureTrigger = page.locator('#overview [data-figure]');
  await figureTrigger.evaluate((element) => { element.click(); element.click(); });
  const dialog = page.locator('#figure-dialog');
  record(checks, 'figure dialog opens with loaded image', await dialog.evaluate((element) => element.open)
    && await dialog.locator('img').evaluate((image) => image.complete && image.naturalWidth > 0));
  record(checks, 'rapid figure clicks leave one modal', await page.locator('dialog[open]').count() === 1);
  record(checks, 'figure dialog focuses close button', await page.locator('.dialog-close').evaluate((element) => element === document.activeElement));
  await page.screenshot({ path: path.join(screenshotRoot, `${name}-dialog.png`) });
  await page.locator('#figure-zoom').click();
  record(checks, 'figure zoom maintains state', await page.locator('#figure-zoom').getAttribute('aria-pressed') === 'true'
    && await page.locator('#figure-viewport').evaluate((element) => element.classList.contains('is-zoomed')));
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.getElementById('figure-dialog').open
    && document.activeElement === document.querySelector('#overview [data-figure]'));
  record(checks, 'Escape closes dialog and restores trigger focus', await dialog.evaluate((element) => !element.open)
    && await figureTrigger.evaluate((element) => element === document.activeElement));

  const heroCard = page.locator('[data-video="nesting"]');
  const timelineButton = heroCard.locator('.timeline-segment').nth(2);
  await page.waitForFunction(() => {
    const video = document.querySelector('[data-video="nesting"] video');
    return video.seekable.length > 0 && video.seekable.end(0) > 3.6;
  });
  await timelineButton.click();
  await page.waitForFunction(() => document.querySelector('[data-video="nesting"] video').currentTime >= 3.5);
  const videoState = await heroCard.evaluate((card) => ({
    currentTime: card.querySelector('video').currentTime,
    activeSkill: card.querySelector('[data-active-skill]').textContent,
    pressed: card.querySelectorAll('.timeline-segment[aria-pressed="true"]').length,
  }));
  record(checks, 'annotation timeline seeks video and updates state', videoState.currentTime >= 3.5
    && videoState.activeSkill === 'Pick'
    && videoState.pressed === 1, videoState);

  if (viewport.width <= 600) {
    const table = page.locator('#panel-annotation .table-scroll');
    const tableScroll = await table.evaluate((element) => {
      element.scrollLeft = 160;
      return {
        scrolled: element.scrollLeft > 0,
        pageOverflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - window.innerWidth,
      };
    });
    record(checks, 'table scrolls without page overflow', tableScroll.scrolled && tableScroll.pageOverflow <= 0, tableScroll);
    await table.evaluate((element) => { element.scrollLeft = 0; });
  }

  if (viewport.width <= 900) {
    await menu.click();
    await page.setViewportSize({ width: 1000, height: viewport.height });
    await page.waitForFunction(() => document.querySelector('.menu-toggle').getAttribute('aria-expanded') === 'false');
    record(checks, 'desktop resize closes mobile navigation', await page.locator('#mobile-nav').isHidden());
    await page.setViewportSize(viewport);
  }
}

for (const viewport of viewports) {
  const name = `${viewport.width}x${viewport.height}`;
  const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const result = { name, viewport, checks: [], consoleErrors: [], pageErrors: [], failedRequests: [], cancelledMediaRequests: [], errorResponses: [] };
  page.on('console', (message) => {
    if (message.type() === 'error') result.consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => result.pageErrors.push(error.message));
  page.on('requestfailed', (request) => {
    const failure = { url: request.url(), error: request.failure()?.errorText };
    // Chromium cancels metadata range requests when it has enough media buffered.
    // Video readiness is checked separately, so actual load failures still fail QA.
    if (request.resourceType() === 'media' && failure.error === 'net::ERR_ABORTED') {
      result.cancelledMediaRequests.push(failure);
    } else {
      result.failedRequests.push(failure);
    }
  });
  page.on('response', (response) => {
    if (response.status() >= 400) result.errorResponses.push({ url: response.url(), status: response.status() });
  });

  try {
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await loadAllImages(page);
    const diagnostics = await pageDiagnostics(page);
    result.diagnostics = diagnostics;
    record(result.checks, 'RoboAtom title is present', diagnostics.title.includes('RoboAtom'));
    record(result.checks, 'all figures render', diagnostics.images.every((image) => image.loaded), diagnostics.images);
    record(result.checks, 'all videos load metadata', diagnostics.videos.length === 20
      && diagnostics.videos.every((video) => video.loaded), diagnostics.videos);
    record(result.checks, 'no page-level horizontal overflow', diagnostics.horizontalOverflow <= 0, diagnostics.horizontalOverflow);
    record(result.checks, 'hero video is visible in first viewport', diagnostics.heroVisible);
    record(result.checks, 'hero has no dataset labels or playback controls', await page.locator('#hero-playback, .wall-label, .hero-bottom').count() === 0);
    record(result.checks, 'paper links point to arXiv', diagnostics.resources.length === 2
      && diagnostics.resources.every((resource) => resource.tag === 'A'
        && resource.href === 'https://arxiv.org/abs/2610.40207'), diagnostics.resources);
    record(result.checks, 'no empty or placeholder links', diagnostics.invalidLinks === 0);
    record(result.checks, 'all site assets use relative paths', diagnostics.rootRelativeAssets.length === 0, diagnostics.rootRelativeAssets);
    await checkPaperContent(page, result.checks);
    await checkConciseContent(page, result.checks);
    await checkReleaseLinks(page, result.checks);
    await checkIdentityAndResources(page, result.checks);
    await checkResearchLayout(page, result.checks, viewport);
    await checkNumericAlignment(page, result.checks);
    const pdfLinks = await page.locator('a[href$=".pdf"]').evaluateAll((links) => [...new Set(links.map((link) => link.getAttribute('href')))]);
    for (const href of pdfLinks) {
      const response = await page.request.get(new URL(href, page.url()).href);
      record(result.checks, `PDF loads: ${href}`, response.ok(), response.status());
    }
    const unnamedControls = await page.locator('button, a').evaluateAll((controls) => controls.filter((control) =>
      !(control.getAttribute('aria-label') || control.textContent.trim())).length);
    record(result.checks, 'all interactive controls have accessible names', unnamedControls === 0);
    await page.screenshot({ path: path.join(screenshotRoot, `${name}-hero.png`) });
    await page.screenshot({ path: path.join(screenshotRoot, `${name}.png`), fullPage: true });
    await exerciseInteractions(page, viewport, result.checks);
  } catch (error) {
    result.fatalError = error.stack || error.message;
  } finally {
    result.passed = result.checks.every((check) => check.passed)
      && !result.fatalError
      && !result.consoleErrors.length
      && !result.pageErrors.length
      && !result.failedRequests.length
      && !result.errorResponses.length;
    report.viewports.push(result);
    await page.close();
  }
}

{
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, javaScriptEnabled: false });
  const page = await context.newPage();
  const checks = [];
  const errorResponses = [];
  page.on('response', (response) => {
    if (response.status() >= 400) errorResponses.push({ url: response.url(), status: response.status() });
  });
  try {
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await loadAllImages(page);
    const state = await page.evaluate(() => ({
      heading: document.querySelector('h1').textContent,
      visiblePanels: Array.from(document.querySelectorAll('.result-panel')).filter((panel) => getComputedStyle(panel).display !== 'none').length,
      loadedImages: Array.from(document.images).every((image) => image.complete && image.naturalWidth > 0),
      videosReady: Array.from(document.querySelectorAll('video')).every((video) => video.readyState >= 2 && video.duration > 0),
      exampleCount: document.querySelectorAll('#examples [data-video-card]').length,
      watchHref: document.querySelector('.desktop-nav a[href="#video"]').getAttribute('href'),
      wallHasControls: Boolean(document.getElementById('hero-playback')),
      timelinesPopulated: document.querySelectorAll('.timeline-segment').length,
      overflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - window.innerWidth,
      tabsVisible: Boolean(document.querySelector('.results-tabs')),
    }));
    record(checks, 'core heading remains readable', state.heading === 'RoboAtom');
    await checkReleaseLinks(page, checks);
    await checkIdentityAndResources(page, checks);
    record(checks, 'all result tables remain visible', state.visiblePanels === 3);
    record(checks, 'figures render without JavaScript', state.loadedImages);
    record(checks, 'videos remain available without JavaScript', state.videosReady && state.timelinesPopulated === 0);
    record(checks, 'three annotated examples remain visible', state.exampleCount === 3);
    record(checks, 'inline video remains reachable without JavaScript', state.watchHref === '#video' && !state.wallHasControls);
    record(checks, 'no horizontal overflow without JavaScript', state.overflow <= 0);
    record(checks, 'nonfunctional tabs are not shown', !state.tabsVisible);
    await page.screenshot({ path: path.join(screenshotRoot, '390x844-no-js.png'), fullPage: true });
    report.noJavaScript = { checks, state, errorResponses, passed: checks.every((check) => check.passed) && !errorResponses.length };
  } catch (error) {
    report.noJavaScript = { checks, errorResponses, passed: false, fatalError: error.stack || error.message };
  } finally {
    await context.close();
  }
}

{
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  const checks = [];
  const configuredUrl = 'https://example.com/roboatom-paper';
  const configuredRequests = [];
  page.on('request', (request) => {
    if (request.url() === configuredUrl) configuredRequests.push(request.url());
  });
  await page.route('**/script.js', async (route) => {
    const response = await route.fetch();
    const source = await response.text();
    const body = source.replace(/paper:\s*'[^']*',/, `paper: '${configuredUrl}',`);
    await route.fulfill({ response, body });
  });
  try {
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    const state = await page.evaluate(() => ({
      paperLinks: Array.from(document.querySelectorAll('[data-resource="paper"]')).map((element) => ({
        tag: element.tagName,
        href: element.getAttribute('href'),
        available: element.classList.contains('is-available'),
        pending: element.classList.contains('is-pending'),
        target: element.getAttribute('target'),
        rel: element.getAttribute('rel'),
      })),
      labels: Array.from(document.querySelectorAll('[data-resource="paper"]')).map((element) => element.textContent.trim()),
      releaseLinksRemainAvailable: Array.from(document.querySelectorAll('[data-release]'))
        .every((element) => element.tagName === 'A' && element.href.startsWith('https://')),
    }));
    record(checks, 'configured URL creates two real paper links', state.paperLinks.length === 2
      && state.paperLinks.every((link) => link.tag === 'A' && link.href === configuredUrl
        && link.target === '_blank' && link.rel.includes('noopener')));
    record(checks, 'configured links use available style', state.paperLinks.every((link) => link.available && !link.pending));
    record(checks, 'resource labels remain concise', state.labels.every((label) => label === 'Paper'));
    record(checks, 'release links remain available when the paper is configured', state.releaseLinksRemainAvailable);
    record(checks, 'link generation does not navigate to configured URL', configuredRequests.length === 0);
    report.configuredResources = { checks, state, configuredRequests, passed: checks.every((check) => check.passed) };
  } catch (error) {
    report.configuredResources = { checks, configuredRequests, passed: false, fatalError: error.stack || error.message };
  } finally {
    await context.close();
  }
}

{
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const checks = [];
  try {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => Array.from(document.querySelectorAll('[data-wall-video]')).every((video) => video.getClientRects().length ? !video.paused && video.currentTime > 0 : video.paused));
    record(checks, 'muted background autoplays with normal motion preferences', await page.locator('#hero-video')
      .evaluate((video) => video.muted && video.loop && video.playsInline));
    await page.locator('#data').evaluate((element) => element.scrollIntoView());
    await page.waitForFunction(() => Array.from(document.querySelectorAll('[data-wall-video]')).every((video) => video.paused));
    record(checks, 'background pauses outside the viewport', true);
    await page.locator('#home').evaluate((element) => element.scrollIntoView());
    await page.waitForFunction(() => Array.from(document.querySelectorAll('[data-wall-video]')).every((video) => video.getClientRects().length ? !video.paused && video.currentTime > 0 : video.paused));
    await navigateToOverview(page);
    await page.waitForFunction(() => Array.from(document.querySelectorAll('[data-wall-video]')).every((video) => video.paused));
    record(checks, 'scrolling to the full video pauses the wall', true);
    await page.locator('#overview-video').scrollIntoViewIfNeeded();
    await page.locator('#overview-video').evaluate((video) => video.play());
    await page.locator('#home').evaluate((element) => element.scrollIntoView());
    await page.waitForFunction(() => document.getElementById('overview-video').paused
      && Array.from(document.querySelectorAll('[data-wall-video]')).every((video) => video.getClientRects().length ? !video.paused : video.paused));
    record(checks, 'returning to the hero pauses the film and resumes the wall', true);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForFunction(() => {
      const videos = Array.from(document.querySelectorAll('[data-wall-video]'));
      return videos.filter((video) => video.getClientRects().length).length === 8
        && videos.every((video) => video.getClientRects().length ? !video.paused : video.paused);
    });
    record(checks, 'mobile resize pauses hidden tiles and keeps eight playing', true);

    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.waitForFunction(() => Array.from(document.querySelectorAll('[data-wall-video]')).every((video) => video.getClientRects().length ? !video.paused : video.paused));
    record(checks, 'the requested video wall keeps looping without manual controls', true);
    report.backgroundPlayback = { checks, passed: checks.every((check) => check.passed) };
  } catch (error) {
    report.backgroundPlayback = { checks, passed: false, fatalError: error.stack || error.message };
  } finally {
    await page.close();
  }
}

{
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const checks = [];
  try {
    await page.addInitScript(() => {
      HTMLMediaElement.prototype.play = function () {
        return Promise.reject(new DOMException('Playback requires a gesture', 'NotAllowedError'));
      };
    });
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    record(checks, 'blocked autoplay leaves posters visible without adding controls', await page.locator('#hero-playback').count() === 0
      && await page.locator('#hero-video').evaluate((video) => video.paused && Boolean(video.poster)));
    await navigateToOverview(page);
    record(checks, 'blocked autoplay still allows the native player', await page.locator('#overview-video')
      .evaluate((video) => video.controls && video.paused));
    report.blockedAutoplay = { checks, passed: checks.every((check) => check.passed) };
  } catch (error) {
    report.blockedAutoplay = { checks, passed: false, fatalError: error.stack || error.message };
  } finally {
    await page.close();
  }
}

await browser.close();
report.passed = report.viewports.every((result) => result.passed)
  && Boolean(report.noJavaScript?.passed)
  && Boolean(report.configuredResources?.passed)
  && Boolean(report.backgroundPlayback?.passed)
  && Boolean(report.blockedAutoplay?.passed);
await writeFile(path.join(artifactRoot, 'browser-results.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
  passed: report.passed,
  viewports: report.viewports.map((result) => ({ name: result.name, passed: result.passed, fatalError: result.fatalError })),
  noJavaScript: report.noJavaScript?.passed,
  configuredResources: report.configuredResources?.passed,
  backgroundPlayback: report.backgroundPlayback?.passed,
  blockedAutoplay: report.blockedAutoplay?.passed,
  report: path.relative(projectRoot, path.join(artifactRoot, 'browser-results.json')),
}, null, 2));
if (!report.passed) process.exitCode = 1;
