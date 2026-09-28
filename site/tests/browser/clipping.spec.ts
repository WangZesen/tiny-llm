import { test, expect, type Locator } from '@playwright/test';

const tokens = [300000000, 408027136];
const file = (name: string, counts: number[][] | number[]) => ({
  name,
  mimeType: 'application/jsonl',
  buffer: Buffer.from(
    counts
      .map((count, index) =>
        JSON.stringify({
          event: 'validation',
          tokens: tokens[index],
          epoch: index + 1,
          loss: 4 - index / 2,
          grad_clip_count: Array.isArray(count) ? count.reduce((a, b) => a + b, 0) : count,
          ...(Array.isArray(count) ? { local_grad_clip_counts: count } : {}),
        }),
      )
      .join('\n'),
  ),
});
type Trace = {
  name: string;
  x: number[];
  y: number[];
  customdata: number[][];
  mode: string;
  hovertemplate: string;
  line: { color: string };
  marker: { symbol: string };
};
const traces = (plot: Locator) =>
  plot.evaluate((node) =>
    ((node as HTMLElement & { data?: Trace[] }).data ?? []).filter(
      (t) => t.mode === 'lines+markers',
    ),
  );

test('local counts support totals, workers, hover, focus, themes, and worker removal', async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('results/explorer/');
  await expect(
    page.getByText('No clipping-count measurements available', { exact: false }),
  ).toBeVisible();
  await page.getByLabel('Local metrics files', { exact: true }).setInputFiles([
    file('sync.jsonl', [0, 4]),
    file('packed.jsonl', [
      [1, 2, 0],
      [0, 0, 0],
    ]),
  ]);
  const plot = page.getByRole('img', { name: 'Current gradient-clipping counts', exact: true });
  const view = page.getByLabel('Clipping count view', { exact: true });
  await expect
    .poll(() => traces(plot))
    .toMatchObject([
      {
        name: 'Local · sync.jsonl',
        x: tokens,
        y: [0, 4],
        customdata: [
          [1, 1],
          [2, 1],
        ],
      },
      { name: 'Local · packed.jsonl', x: tokens, y: [3, 0] },
    ]);
  const loss = page.getByRole('img', { name: 'Current validation trajectories', exact: true });
  const lossLocal = await loss.evaluate((node) =>
    (node as any).data.find((t: Trace) => t.name === 'Local · sync.jsonl'),
  );
  const original = (await traces(plot))[0];
  expect(original.line.color).toBe(lossLocal.line.color);
  expect(original.marker.symbol).toBe(lossLocal.marker.symbol);
  // Plotly receives hover events on a transparent drag layer above its SVG markers.
  await plot
    .locator('.scatterlayer .trace')
    .first()
    .locator('.point')
    .nth(1)
    .hover({ force: true });
  await expect(plot.locator('.hoverlayer')).toContainText('Epoch 2');
  await expect(plot.locator('.hoverlayer')).toContainText('Clips 4');
  await expect(plot.locator('.hoverlayer')).toContainText('Total across workers');
  await view.selectOption('1');
  await expect
    .poll(() => traces(plot))
    .toMatchObject([{ name: 'Local · packed.jsonl', y: [2, 0] }]);
  expect(await traces(plot)).toHaveLength(1);
  await expect(page.getByText('No Worker 1 clipping counts for:', { exact: false })).toContainText(
    'sync.jsonl',
  );
  await expect.poll(() => new URL(page.url()).searchParams.get('clipWorker')).toBe('1');
  await view.selectOption('2');
  await expect.poll(() => traces(plot)).toMatchObject([{ y: [0, 0] }]);
  await expect
    .poll(() => plot.evaluate((node) => (node as any).layout.yaxis.range))
    .toEqual([0, 1]);
  await page.getByLabel('Show individual seeds').check();
  await page.getByLabel('Loss view', { exact: true }).selectOption('train');
  await expect.poll(() => traces(plot)).toMatchObject([{ y: [0, 0] }]);
  await page.getByLabel('Focus on the final losses').check();
  await expect
    .poll(() => plot.evaluate((node) => (node as any).layout.xaxis.range[0]))
    .toBe(tokens[1] / 2);
  await view.selectOption('total');
  await page
    .getByRole('textbox', { name: 'Local run name: sync.jsonl', exact: true })
    .fill('Synchronous run');
  await expect.poll(() => traces(plot)).toMatchObject([{ name: 'Local · Synchronous run' }, {}]);
  await plot.screenshot({ path: info.outputPath('clipping-light.png') });
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect.poll(async () => (await traces(plot))[0].line.color).not.toBe(original.line.color);
  await plot.screenshot({ path: info.outputPath('clipping-dark.png') });
  await view.selectOption('2');
  await page
    .getByRole('button', { name: 'Remove local run packed.jsonl from comparison', exact: true })
    .click();
  await expect(view).toHaveValue('total');
  await expect
    .poll(() => traces(plot))
    .toMatchObject([{ name: 'Local · Synchronous run', y: [0, 4] }]);
  await expect.poll(() => new URL(page.url()).searchParams.get('clipWorker')).toBeNull();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

test('published counts aggregate seeds and restore worker selection from a comparison URL', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/curves/*.json', async (route) => {
    const response = await route.fetch();
    const curve = await response.json();
    for (const [index, run] of curve.runs.entries()) {
      run.gradientClipping = run.validation.slice(-2).map((point: number[], i: number) => ({
        tokens: point[0],
        epoch: run.validation.length - 1 + i,
        total: i ? 0 : 2 * index + 3,
        workers: i ? [0, 0, 0, 0] : [index + 1, index + 2, 0, 0],
      }));
    }
    await route.fulfill({ response, json: curve });
  });
  await page.goto('results/explorer/?method=awc4&clipWorker=1');
  const plot = page.getByRole('img', { name: 'Current gradient-clipping counts', exact: true });
  const view = page.getByLabel('Clipping count view', { exact: true });
  await expect(view).toHaveValue('1');
  await expect
    .poll(() => traces(plot))
    .toMatchObject([
      {
        y: [3, 0],
        customdata: [
          [39, 3],
          [40, 3],
        ],
      },
    ]);
  await expect
    .poll(() => plot.evaluate((node) => (node as any).data.slice(0, 2).map((t: Trace) => t.y)))
    .toEqual([
      [2, 0],
      [4, 0],
    ]);
  await view.selectOption('total');
  await expect.poll(() => traces(plot)).toMatchObject([{ y: [5, 0] }]);
  await view.selectOption('1');
  await page.reload();
  await expect(view).toHaveValue('1');
  await expect.poll(() => traces(plot)).toMatchObject([{ y: [3, 0] }]);
  await page.getByLabel('Show individual seeds').check();
  await expect
    .poll(() => traces(plot))
    .toMatchObject([
      { name: expect.stringContaining('Seed 42'), y: [2, 0] },
      { name: expect.stringContaining('Seed 43'), y: [3, 0] },
      { name: expect.stringContaining('Seed 44'), y: [4, 0] },
    ]);
  expect(errors).toEqual([]);
});

test('cached legacy payloads and missing seeds never turn into zero counts', async ({ page }) => {
  await page.route('**/curves/*.json', async (route) => {
    const response = await route.fetch();
    const curve = await response.json();
    for (const run of curve.runs) delete run.gradientClipping;
    const run = curve.runs[0];
    run.gradientClipping = [{ tokens: run.validation[0][0], epoch: 1, total: 2, workers: [2] }];
    await route.fulfill({ response, json: curve });
  });
  await page.goto('results/explorer/?clipWorker=9');
  const view = page.getByLabel('Clipping count view', { exact: true });
  const plot = page.getByRole('img', { name: 'Current gradient-clipping counts', exact: true });
  await expect(view).toHaveValue('total');
  await expect.poll(() => traces(plot)).toMatchObject([{ y: [2], customdata: [[1, 1]] }]);
  await expect(page.getByText('No clipping counts for:', { exact: false })).toContainText(
    'seeds 43, 44',
  );
  await expect(
    page.getByRole('img', { name: 'Current validation trajectories', exact: true }),
  ).toBeVisible();
});
