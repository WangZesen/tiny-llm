import { test, expect } from '@playwright/test';
test('home, math, theme, archive and local search', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('How small models learn');
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.goto('methods/algorithms/');
  await expect(page.locator('.katex').first()).toBeVisible();
  await expect(page.locator('.katex-error')).toHaveCount(0);
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.getByRole('button', { name: 'Search documentation' }).click();
  await page.locator('#search input').fill('gradient');
  await expect(page.locator('.pagefind-ui__result').first()).toBeVisible();
  await expect(page.locator('.pagefind-ui__result-link[href*="/archive/"]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Close search' }).click();
  await page.getByRole('link', { name: 'Historical archive', exact: true }).click();
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Keep the record');
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});
test('filters, sorting, inspection, curves, downloads, URL restoration', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('explorer/');
  await page.getByRole('button', { name: 'Synchronous', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Configuration rankings' })).toBeVisible();
  await expect(page.locator('.js-plotly-plot').first()).toBeVisible();
  await page.getByLabel('Learning rate', { exact: true }).selectOption('0.0056');
  await expect(
    page.getByRole('table', { name: 'Configuration rankings' }).locator('tbody tr'),
  ).toHaveCount(3);
  await page.getByRole('button', { name: 'Inspect rank 1', exact: true }).click();
  await expect(
    page.getByRole('table', { name: 'Selected seed results' }).locator('tbody tr'),
  ).toHaveCount(3);
  await page.getByRole('button', { name: 'Add to comparison', exact: true }).click();
  await page.getByLabel('Loss view', { exact: true }).selectOption('train');
  await page.getByLabel('Show individual seeds').check();
  await expect(page.getByRole('img', { name: 'Training-loss trajectories' })).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download filtered CSV' }).click();
  expect((await download).suggestedFilename()).toBe('sync-filtered.csv');
  await page.reload();
  await expect(page.getByLabel('Loss view', { exact: true })).toHaveValue('train');
  await expect(page.getByLabel('Show individual seeds')).toBeChecked();
  await expect(page.getByLabel('Learning rate', { exact: true })).toHaveValue('0.0056');
  await page.getByRole('button', { name: 'Mean loss', exact: false }).click();
  await expect(
    page.getByRole('table', { name: 'Configuration rankings' }).locator('tbody tr').first(),
  ).toContainText('0.999');
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});
test('missing curves remain explicit and selection is capped at four', async ({ page }) => {
  await page.goto('explorer/?study=awc4&selected=awc4--0.008--0.95--0.98');
  await expect(page.getByText('Curves unavailable for', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Reset filters' }).click();
  const boxes = page.getByRole('table', { name: 'Configuration rankings' }).getByRole('checkbox');
  for (let i = 0; i < 4; i++) await boxes.nth(i).check();
  await expect(boxes.nth(4)).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Clear comparison' })).toBeVisible();
});
test('heatmap clicks select the matching recipe and chart export works', async ({ page }) => {
  await page.goto('explorer/?study=awc4&heatLR=0.008');
  const heatmap = page.locator('.heatmaplayer image');
  await expect(heatmap).toBeVisible();
  await heatmap.scrollIntoViewIfNeeded();
  const box = (await heatmap.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.125, box.y + box.height * 0.125);
  await expect(page.getByRole('region', { name: 'Selected configuration' })).toContainText(
    'β₁ 0.975 · β₂ 0.95',
  );
  await expect(
    page.getByRole('table', { name: 'Configuration rankings' }).locator('tr.selected'),
  ).toContainText('3.693064');
  await expect(page).toHaveURL(/selected=awc4--0.008--0.975--0.95/);
  const download = page.waitForEvent('download');
  await page.locator('.modebar-btn[data-title*="Download"]').first().click({ force: true });
  expect((await download).suggestedFilename()).toMatch(/\.svg$/);
});
test('articles and summary downloads work without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4321/tiny-llm/');
  await expect(page.getByRole('table', { name: 'Current winning configurations' })).toBeVisible();
  await page.goto('http://127.0.0.1:4321/tiny-llm/methods/algorithms/');
  await expect(page.locator('.katex').first()).toBeVisible();
  await context.close();
});
