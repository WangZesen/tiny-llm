import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';

test('homepage follows publication priorities with clean navigation and readable figures', async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('./');
  await expect(
    page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link'),
  ).toHaveText(['Results', 'Usage', 'Performance']);
  expect(
    await page.locator('main > section').evaluateAll((nodes) => nodes.map((n) => n.id)),
  ).toEqual(['schedules', 'workers', 'protocol', 'usage', 'performance']);
  await expect(
    page.getByRole('table', { name: 'Current winning configurations' }).locator('tbody tr'),
  ).toHaveCount(15);
  await expect(page.locator('main')).not.toContainText(/lr-expanded|Earlier cosine-study winners/);
  await page.screenshot({
    path: info.outputPath('publication-home.png'),
    fullPage: info.project.name === 'desktop',
    scale: 'css',
  });
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await page.screenshot({
    path: info.outputPath('publication-home-dark.png'),
    fullPage: info.project.name === 'desktop',
    scale: 'css',
  });
  if (info.project.name === 'mobile') {
    const figures = page.locator('#schedules .figure-narrow');
    await expect(figures).toBeVisible();
    await figures.scrollIntoViewIfNeeded();
    await expect
      .poll(() =>
        figures
          .locator('img')
          .evaluateAll((images) =>
            images.every((image) => (image as HTMLImageElement).naturalWidth > 0),
          ),
      )
      .toBeTruthy();
    await figures.screenshot({
      path: info.outputPath('publication-mobile-figures.png'),
      scale: 'css',
    });
  }
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

test('current explorer filters, restores URLs, downloads logs and exposes missing coverage', async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('results/explorer/');
  await expect(page.getByLabel('Schedule', { exact: true })).toHaveValue('cosine');
  await expect(page.getByLabel('Training mode', { exact: true })).toHaveValue('sync');
  const table = page.getByRole('table', { name: 'Current configuration rankings' });
  await expect(table.locator('tbody tr')).toHaveCount(24);
  await expect(table.locator('tbody tr').first()).toContainText('3.549818');
  await page.getByLabel('Learning rate', { exact: true }).selectOption('0.01');
  await page.getByLabel('Beta 2', { exact: true }).selectOption('0.99');
  await expect(table.locator('tbody tr')).toHaveCount(1);
  await page.getByRole('button', { name: 'Inspect current rank 1', exact: true }).click();
  await expect(
    page.getByRole('table', { name: 'Current matched comparisons' }).locator('tbody tr'),
  ).toHaveCount(2);
  await page.getByLabel('Loss view', { exact: true }).selectOption('train');
  await page.getByLabel('Show individual seeds').check();
  await expect(
    page
      .getByRole('img', { name: 'Current training trajectories', exact: true })
      .locator('.main-svg')
      .first(),
  ).toBeVisible();
  await page.getByLabel('Gradient-norm y-axis scale', { exact: true }).selectOption('log');
  await expect(
    page
      .getByRole('img', { name: 'Current gradient-norm trajectories', exact: true })
      .locator('.main-svg')
      .first(),
  ).toBeVisible();
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download filtered CSV', exact: true }).click();
  const download = await pending;
  expect((await readFile((await download.path())!, 'utf8')).trim().split('\n')).toHaveLength(2);
  await page.reload();
  await expect(page.getByLabel('Loss view', { exact: true })).toHaveValue('train');
  await expect(page.getByLabel('Show individual seeds')).toBeChecked();
  await expect(page.getByLabel('Gradient-norm y-axis scale', { exact: true })).toHaveValue('log');
  const logs = page
    .getByRole('table', { name: 'Current seed results' })
    .getByRole('link', { name: 'Metrics log' });
  const href = (await logs.first().getAttribute('href'))!;
  const [container, entryPath] = href.split('#');
  const response = await page.request.get(container);
  expect(response.ok()).toBeTruthy();
  const body = await response.body();
  // Preview servers may advertise Content-Encoding: gzip; HTTP clients decode it.
  const decoded = body[0] === 0x1f && body[1] === 0x8b ? gunzipSync(body) : body;
  const entry = decoded
    .toString()
    .trim()
    .split('\n')
    .map((s) => JSON.parse(s))
    .find((e) => e.path === decodeURIComponent(entryPath));
  expect(entry).toBeTruthy();
  const events = entry.text
    .trim()
    .split('\n')
    .map((s: string) => JSON.parse(s));
  expect(events.at(-1).event).toBe('final_validation');
  // Retention policy: the per-worker diagnostics are deliberately not published.
  expect(events[0]).not.toHaveProperty('local_losses');
  await page.getByLabel('Training mode', { exact: true }).selectOption('awc4');
  await page.getByLabel('Schedule', { exact: true }).selectOption('wsd');
  await expect(page.getByRole('status')).toHaveCount(1);
  await expect(page.getByRole('status')).toContainText('Four-worker WSD is unavailable');
  await page.getByLabel('Training mode', { exact: true }).selectOption('sync');
  await page.getByLabel('Horizon', { exact: true }).selectOption('160');
  await expect(table.locator('tbody tr')).toHaveCount(12);
  await page.getByText('Grid coverage', { exact: true }).click();
  await expect(
    page
      .getByRole('table', { name: 'Current grid coverage' })
      .getByRole('cell', { name: 'Pruned', exact: true }),
  ).toHaveCount(3);
  await page.getByText('Trajectory sources', { exact: true }).click();
  await expect(
    page
      .getByRole('region', { name: 'Selected configuration' })
      .getByText('Earlier terminal decays are excluded.', { exact: false }),
  ).toBeVisible();
  await page.screenshot({
    path: info.outputPath('publication-explorer.png'),
    fullPage: info.project.name === 'desktop',
    scale: 'css',
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

test('unavailable curves preserve final results and downloads', async ({ page }) => {
  await page.route('**/assets/data/current-training/curves/**', (route) =>
    route.fulfill({ status: 404, body: '' }),
  );
  await page.goto('results/explorer/?schedule=cosine&method=awc8&horizon=80');
  await expect(page.getByRole('alert')).toHaveCount(1);
  await expect(page.getByRole('alert')).toContainText('Recorded curves could not load');
  await expect(
    page.getByRole('table', { name: 'Current seed results' }).locator('tbody tr'),
  ).toHaveCount(3);
  await expect(
    page
      .getByRole('table', { name: 'Current seed results' })
      .getByRole('link', { name: 'Metrics log' }),
  ).toHaveCount(3);
});

test('comparison survives scope changes and round-trips through the URL', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('results/explorer/');
  const tray = page.getByRole('region', { name: 'Trajectory comparison' });
  const pinned = tray.getByRole('button', { name: /^Remove .* from comparison$/ });
  await page.getByRole('checkbox', { name: 'Compare current rank 1', exact: true }).check();
  await page.getByLabel('Training mode', { exact: true }).selectOption('awc8');
  // Pinning is cross-scope: the first selection must survive changing the training mode.
  await page.getByRole('checkbox', { name: 'Compare current rank 1', exact: true }).check();
  await expect(pinned).toHaveCount(2);
  // The chip names the scope its curve came from, which the filters no longer imply.
  await expect(
    tray.getByRole('button', { name: /^Inspect Cosine-to-zero · 1 worker · h20 / }),
  ).toHaveCount(1);
  expect(page.url()).toContain('compare=');
  await page.reload();
  await expect(pinned).toHaveCount(2);
  await expect(
    tray
      .getByRole('img', { name: 'Current validation trajectories', exact: true })
      .locator('.main-svg')
      .first(),
  ).toBeVisible();
  await expect(page.getByRole('status')).toHaveCount(0);
  await pinned.first().click();
  await expect(pinned).toHaveCount(1);
  await page.getByRole('button', { name: 'Pin best at this horizon', exact: true }).click();
  await expect(pinned).toHaveCount(5);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

// A visitor can hold a cached curve payload older than the code that reads it, which must
// cost the gradient-norm figure only, never the rest of the explorer.
test('curves cached without a gradient-norm series still render the loss comparison', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.route('**/curves/*.json', async (route) => {
    const response = await route.fetch();
    const curve = await response.json();
    for (const run of curve.runs) delete run.gradientNorm;
    await route.fulfill({ response, json: curve });
  });
  await page.goto('results/explorer/');
  await page.getByRole('checkbox', { name: 'Compare current rank 1', exact: true }).check();
  await expect(
    page
      .getByRole('img', { name: 'Current validation trajectories', exact: true })
      .locator('.main-svg')
      .first(),
  ).toBeVisible();
  await expect(
    page.getByRole('img', { name: 'Current gradient-norm trajectories', exact: true }),
  ).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('GH200 measurements expose all horizons, dates and per-run downloads', async ({ page }) => {
  await page.goto('performance/training/');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Training on one GH200');
  await expect(page.locator('article table tbody tr')).toHaveCount(9);
  await expect(page.getByText('Measurements from jobs started', { exact: false })).toContainText(
    /\d{4}-\d{2}-\d{2}–\d{4}-\d{2}-\d{2} \(UTC\)/,
  );
  const response = await page.request.get(
    (await page.getByRole('link', { name: 'Per-run measurements CSV' }).getAttribute('href'))!,
  );
  expect(response.ok()).toBeTruthy();
  const lines = (await response.text()).trim().split('\n');
  expect(lines).toHaveLength(1189);
  expect(lines[0]).toContain('measurementDate');
});

test('usage navigation excludes archived references; search excludes archive and maintenance', async ({
  page,
}) => {
  await page.goto('guides/overview/');
  const sidebar = page.getByRole('complementary', { name: 'Section navigation' });
  await expect(sidebar).not.toContainText('benchmarking');
  await expect(sidebar).not.toContainText('maintenance');
  await page.getByRole('button', { name: 'Search documentation' }).click();
  await page.locator('#search input').fill('training');
  await expect(page.locator('.pagefind-ui__result').first()).toBeVisible();
  await expect(
    page.locator(
      '.pagefind-ui__result-link[href*="/archive/"], .pagefind-ui__result-link[href*="/explorer/wsd/"], .pagefind-ui__result-link[href*="/guides/website/"]',
    ),
  ).toHaveCount(0);
});
