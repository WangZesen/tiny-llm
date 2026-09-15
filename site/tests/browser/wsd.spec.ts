import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('WSD filters, ranking, seeds, URL restoration, downloads and themes', async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('explorer/wsd/');
  await expect(page.getByLabel('Method', { exact: true })).toHaveValue('awc8');
  await expect(page.getByLabel('Horizon', { exact: true })).toHaveValue('20');
  const rows = page.getByRole('table', { name: 'WSD configuration rankings' }).locator('tbody tr');
  await expect(rows).toHaveCount(36);
  await expect(rows.first()).toContainText('3.593128');
  await expect(
    page.getByRole('table', { name: 'WSD seed results' }).locator('tbody tr'),
  ).toHaveCount(3);
  await expect(page.locator('.js-plotly-plot')).toHaveCount(5);
  await page.getByLabel('Beta 1', { exact: true }).selectOption('0.95');
  await expect(rows).toHaveCount(18);
  await page.getByLabel('Learning rate', { exact: true }).selectOption('0.006');
  await page.getByLabel('Beta 2', { exact: true }).selectOption('0.999');
  await expect(rows).toHaveCount(1);
  await page.getByRole('button', { name: 'Inspect WSD rank 1', exact: true }).click();
  await page.getByLabel('Loss view', { exact: true }).selectOption('train');
  await page.getByLabel('Show individual seeds').check();
  await expect(page.getByRole('img', { name: 'WSD training trajectories' })).toBeVisible();
  for (const ext of ['CSV', 'JSON']) {
    const pending = page.waitForEvent('download');
    await page.getByRole('button', { name: `Download filtered ${ext}`, exact: true }).click();
    const file = await pending;
    expect(file.suggestedFilename()).toBe(`awc8-wsd-h20-filtered.${ext.toLowerCase()}`);
    const content = await readFile((await file.path())!, 'utf8');
    if (ext === 'JSON') {
      const r = JSON.parse(content);
      expect(r).toHaveLength(1);
      expect(r[0].runs).toHaveLength(3);
      expect(r[0].id).toBe('awc8-wsd-h20--0.006--0.95--0.999');
    } else {
      expect(content.trim().split('\n')).toHaveLength(2);
      expect(content).toContain('lossSeed44');
    }
  }
  await page.reload();
  for (const [name, value] of [
    ['Learning rate', '0.006'],
    ['Beta 1', '0.95'],
    ['Beta 2', '0.999'],
    ['Loss view', 'train'],
  ])
    await expect(page.getByLabel(name, { exact: true })).toHaveValue(value);
  await expect(page.getByLabel('Show individual seeds')).toBeChecked();
  await expect(page).toHaveURL(/selected=awc8-wsd-h20--0.006--0.95--0.999/);
  await page.getByRole('button', { name: 'Reset filters', exact: true }).click();
  await page.getByRole('button', { name: 'Mean loss', exact: true }).click();
  await expect(rows.last()).toContainText('3.593128');
  await page.getByLabel('Horizon', { exact: true }).selectOption('160');
  await expect(rows).toHaveCount(12);
  await page.getByLabel('Method', { exact: true }).selectOption('sync');
  await expect(rows).toHaveCount(12);
  await page.getByText('Grid coverage', { exact: true }).click();
  const grid = page.getByRole('table', { name: 'WSD grid coverage' });
  await expect(grid.getByRole('cell', { name: 'Pruned', exact: true })).toHaveCount(18);
  await expect(grid.getByRole('cell', { name: 'Outside original grid', exact: true })).toHaveCount(
    6,
  );
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page
    .getByRole('region', { name: 'WSD configuration explorer' })
    .screenshot({ path: info.outputPath('wsd-rankings.png') });
  await page
    .getByRole('img', { name: 'WSD horizon heatmap' })
    .screenshot({ path: info.outputPath('wsd-heatmap-dark.png') });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});
test('WSD heatmap, chart export, continuation sources and history', async ({ page }, info) => {
  await page.goto('explorer/wsd/?method=awc8&horizon=80');
  const plot = page.getByRole('img', { name: 'WSD horizon heatmap' }),
    heat = plot.locator('.heatmaplayer image');
  await expect(heat).toBeVisible();
  await heat.scrollIntoViewIfNeeded();
  const box = (await heat.boundingBox())!;
  await page.mouse.click(box.x + box.width / 12, box.y + box.height / 12);
  await expect(page).toHaveURL(/selected=awc8-wsd-h80--0.002--0.95--0.999/);
  const selected = page.getByRole('region', { name: 'Selected WSD configuration' });
  await expect(selected).toContainText('LR 0.002 · β₁ 0.95 · β₂ 0.999');
  await page.getByText('Continuation sources and curve data', { exact: true }).click();
  for (const h of [20, 40, 80]) await expect(selected).toContainText(`H${h}:`);
  await page.evaluate(() => {
    history.pushState(null, '', '?method=sync&horizon=40&lr=0.004');
    dispatchEvent(new PopStateEvent('popstate'));
  });
  await expect(page.getByLabel('Method', { exact: true })).toHaveValue('sync');
  await expect(page.getByLabel('Horizon', { exact: true })).toHaveValue('40');
  await expect(
    page.getByRole('table', { name: 'WSD configuration rankings' }).locator('tbody tr'),
  ).toHaveCount(6);
  await page
    .getByRole('img', { name: 'WSD subset-validation trajectories' })
    .scrollIntoViewIfNeeded();
  await page.screenshot({ path: info.outputPath('wsd-curves.png') });
  const pending = page.waitForEvent('download');
  await plot.locator('.modebar-btn[data-title*="Download"]').click({ force: true });
  expect((await pending).suggestedFilename()).toMatch(/\.svg$/);
});
test('WSD invalid selections and unavailable curves stay explicit', async ({ page }) => {
  await page.goto(
    'explorer/wsd/?method=sync&horizon=160&lr=0.01&selected=awc8-wsd-h20--0.006--0.95--0.999',
  );
  await expect(page.getByLabel('Learning rate', { exact: true })).toHaveValue('all');
  await expect(page.getByRole('region', { name: 'Selected WSD configuration' })).toContainText(
    'Synchronous · horizon 160',
  );
  await expect(page).not.toHaveURL(/selected=awc8/);
  await page.route('**/curves/awc8-wsd-h20--0.006--0.95--0.999.json', (route) =>
    route.fulfill({ status: 404, body: 'missing' }),
  );
  await page.goto('explorer/wsd/');
  await expect(page.getByRole('alert')).toContainText('Recorded curves are unavailable');
  await expect(
    page.getByRole('table', { name: 'WSD seed results' }).locator('tbody tr'),
  ).toHaveCount(3);
});
test('WSD static article, winners, figures and downloads work without JavaScript', async ({
  browser,
  request,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false }),
    page = await context.newPage();
  await page.goto('http://127.0.0.1:4321/tiny-llm/explorer/wsd/');
  await expect(
    page.getByRole('table', { name: 'WSD winning configurations' }).locator('tbody tr'),
  ).toHaveCount(10);
  const response = await request.get(
    (await page.getByRole('link', { name: 'All seed results CSV' }).getAttribute('href'))!,
  );
  expect(response.ok()).toBeTruthy();
  expect((await response.text()).trim().split('\n')).toHaveLength(595);
  const pdf = await request.get(
    (await page.getByRole('link', { name: 'Comparison figure PDF' }).getAttribute('href'))!,
  );
  expect((await pdf.body()).subarray(0, 4).toString()).toBe('%PDF');
  await page.goto('http://127.0.0.1:4321/tiny-llm/results/wsd-horizon-tuning/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('WSD');
  await expect(page.locator('article')).toContainText('0.000015');
  await expect(page.locator('article img').first()).toBeVisible();
  await context.close();
});
