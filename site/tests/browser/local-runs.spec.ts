import { test, expect, type Locator } from '@playwright/test';

const events = [
  { event: 'train', tokens: 300000000, loss: 4.2, grad_norm: 1.5 },
  { event: 'validation', tokens: 300000000, loss: 4.1 },
  { event: 'train', tokens: 408027136, loss: 3.7, grad_norm: 0.5 },
  { event: 'validation', tokens: 408027136, loss: 3.6 },
  { event: 'final_validation', tokens: 408027136, loss: 3.55 },
];
const file = (name: string, content: object[] | string = events) => ({
  name,
  mimeType: 'application/jsonl',
  buffer: Buffer.from(
    typeof content === 'string' ? content : content.map((row) => JSON.stringify(row)).join('\n'),
  ),
});
type Trace = {
  name: string;
  legendgroup: string;
  x: number[];
  y: number[];
  line: { color: string; dash: string };
  marker: { symbol: string };
  mode: string;
};
const localTraces = (plot: Locator) =>
  plot.evaluate((node) =>
    ((node as HTMLElement & { data?: Trace[] }).data ?? []).filter((trace) =>
      trace.legendgroup?.startsWith('local-'),
    ),
  );

test('local files join both plots, keep their names and controls, and stay out of URLs', async ({
  page,
  context,
}, info) => {
  const errors: string[] = [];
  const requests: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    if (request.method() !== 'GET' || /\/curves\/local-/.test(request.url()))
      requests.push(request.url());
  });
  await page.goto('results/explorer/');
  await page.getByRole('checkbox', { name: 'Compare current rank 1', exact: true }).check();
  const publishedComparison = new URL(page.url()).searchParams.get('compare');
  const picker = page.getByLabel('Local metrics files', { exact: true });
  await picker.setInputFiles([
    file('metrics.jsonl'),
    file(
      'metrics.jsonl',
      events.map((event) => ({ ...event, loss: event.loss + 0.2 })),
    ),
  ]);
  const names = page.getByRole('textbox', { name: 'Local run name: metrics.jsonl', exact: true });
  await expect(names).toHaveCount(2);
  await expect(names.nth(0)).toHaveValue('metrics.jsonl');
  await expect(names.nth(1)).toHaveValue('metrics.jsonl (2)');
  await names.nth(0).fill('My experiment');
  await names.nth(0).blur();
  const validation = page.getByRole('img', {
    name: 'Current validation trajectories',
    exact: true,
  });
  const gradient = page.getByRole('img', {
    name: 'Current gradient-norm trajectories',
    exact: true,
  });
  await expect
    .poll(() => localTraces(validation))
    .toMatchObject([
      {
        name: 'Local · My experiment',
        x: [300000000, 408027136],
        y: [4.1, 3.6],
        line: { dash: 'dot' },
      },
      { name: 'Local · metrics.jsonl (2)', y: [4.3, 3.8000000000000003] },
    ]);
  await expect
    .poll(() => localTraces(gradient))
    .toMatchObject([
      {
        name: 'Local · My experiment',
        x: [300000000, 408027136],
        y: [1.5, 0.5],
        line: { dash: 'dot' },
      },
      { name: 'Local · metrics.jsonl (2)', y: [1.5, 0.5] },
    ]);
  const lossTrace = (await localTraces(validation))[0];
  const gradientTrace = (await localTraces(gradient))[0];
  expect(gradientTrace.line.color).toBe(lossTrace.line.color);
  expect(gradientTrace.marker.symbol).toBe(lossTrace.marker.symbol);

  await page.getByLabel('Loss view', { exact: true }).selectOption('train');
  const training = page.getByRole('img', { name: 'Current training trajectories', exact: true });
  await expect
    .poll(() => localTraces(training))
    .toMatchObject([{ y: [4.2, 3.7] }, { y: [4.4, 3.9000000000000004] }]);
  await page.getByLabel('Show individual seeds').check();
  await expect
    .poll(() => localTraces(training))
    .toMatchObject([
      { name: 'Local · My experiment', y: [4.2, 3.7] },
      { name: 'Local · metrics.jsonl (2)' },
    ]);
  await page.getByLabel('Y-axis scale', { exact: true }).selectOption('log');
  await page.getByLabel('Gradient-norm y-axis scale', { exact: true }).selectOption('log');
  for (const plot of [training, gradient]) {
    await expect.poll(() => plot.evaluate((node) => (node as any).layout?.yaxis?.type)).toBe('log');
  }
  await page.getByLabel('Focus on the final losses').check();
  await expect
    .poll(() => gradient.evaluate((node) => (node as any).layout?.xaxis?.range?.[0]))
    .toBe(204013568);

  await page.getByLabel('Training mode', { exact: true }).selectOption('awc8');
  await expect(names.nth(0)).toHaveValue('My experiment');
  await page.getByRole('button', { name: 'Reset filters', exact: true }).click();
  await expect(names).toHaveCount(2);
  await expect
    .poll(() => localTraces(validation))
    .toMatchObject([
      { line: { color: lossTrace.line.color } },
      { name: 'Local · metrics.jsonl (2)' },
    ]);
  await expect
    .poll(() => new URL(page.url()).searchParams.get('compare'))
    .toBe(publishedComparison);
  expect(page.url()).not.toContain('local-');
  expect(page.url()).not.toContain('metrics.jsonl');
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.getByRole('button', { name: 'Copy comparison link', exact: true }).click();
  await expect(page.getByText('Comparison link copied.', { exact: false })).toContainText(
    'Local runs are not included',
  );
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(page.url());
  const comparison = page.getByRole('region', { name: 'Trajectory comparison' });
  await comparison.screenshot({ path: info.outputPath('local-comparison.png'), scale: 'css' });
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect
    .poll(async () => (await localTraces(validation))[0]?.line.color)
    .not.toBe(lossTrace.line.color);
  await comparison.screenshot({ path: info.outputPath('local-comparison-dark.png'), scale: 'css' });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2),
  ).toBeTruthy();
  expect(requests).toEqual([]);
  await page.reload();
  await expect(page.getByLabel('Schedule', { exact: true })).toHaveValue('cosine');
  await expect(names).toHaveCount(0);
  await expect
    .poll(() => new URL(page.url()).searchParams.get('compare'))
    .toBe(publishedComparison);
  expect(errors).toEqual([]);
});

test('partial and invalid logs report per-file feedback even without published results', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('results/explorer/?schedule=wsd&method=awc4');
  await expect(page.getByRole('button', { name: 'Add local runs', exact: true })).toBeVisible();
  await page
    .getByLabel('Local metrics files')
    .setInputFiles([
      file('broken.jsonl', '{"event": nope}'),
      file('partial.jsonl', JSON.stringify(events[0]) + '\n{"event":'),
      file('loss-only.jsonl', [{ event: 'train', tokens: 408027136, loss: 4 }]),
    ]);
  await expect(page.getByRole('alert')).toContainText('broken.jsonl, line 1: invalid JSON record');
  await expect(
    page.getByText('partial.jsonl, line 2: incomplete final record ignored.', { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText('loss-only.jsonl: no subset-validation loss or gradient norm measurements.', {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText('No subset-validation loss measurements available in this comparison.', {
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByText('Loading recorded curves…', { exact: true })).toHaveCount(0);
  const gradient = page.getByRole('img', {
    name: 'Current gradient-norm trajectories',
    exact: true,
  });
  await expect
    .poll(() => localTraces(gradient))
    .toMatchObject([{ y: [1.5], mode: 'lines+markers' }]);
  await page.getByLabel('Loss view', { exact: true }).selectOption('train');
  const training = page.getByRole('img', { name: 'Current training trajectories', exact: true });
  await expect.poll(() => localTraces(training)).toMatchObject([{ y: [4.2] }, { y: [4] }]);
  await page.getByLabel('Focus on the final losses').check();
  await expect
    .poll(() => training.evaluate((node) => (node as any).layout?.xaxis?.range?.[0]))
    .toBe(204013568);
  await page
    .getByRole('button', { name: 'Remove local run partial.jsonl from comparison', exact: true })
    .click();
  await expect(
    page.getByRole('textbox', { name: 'Local run name: partial.jsonl', exact: true }),
  ).toHaveCount(0);
  await expect(gradient).toHaveCount(0);
  await expect(
    page.getByText('partial.jsonl, line 2: incomplete final record ignored.', { exact: true }),
  ).toHaveCount(0);
  await page.getByRole('button', { name: 'Clear comparison', exact: true }).click();
  await expect(page.getByRole('textbox', { name: /^Local run name:/ })).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Add local runs', exact: true })).toBeVisible();
  // Selecting the same file again must trigger a fresh read.
  await page
    .getByLabel('Local metrics files')
    .setInputFiles(file('loss-only.jsonl', [{ event: 'train', tokens: 408027136, loss: 3 }]));
  await expect.poll(() => localTraces(training)).toMatchObject([{ y: [3] }]);
  expect(errors).toEqual([]);
});

test('imports share the eight-curve capacity and preserve existing comparisons', async ({
  page,
}) => {
  await page.goto('results/explorer/');
  await page.getByRole('checkbox', { name: 'Compare current rank 1', exact: true }).check();
  const picker = page.getByLabel('Local metrics files');
  await picker.setInputFiles(Array.from({ length: 8 }, (_, i) => file(`run-${i + 1}.jsonl`)));
  const names = page.getByRole('textbox', { name: /^Local run name:/ });
  await expect(names).toHaveCount(7);
  await expect(page.getByRole('alert')).toContainText('run-8.jsonl: not added');
  await expect(
    page.getByRole('checkbox', { name: 'Compare current rank 1', exact: true }),
  ).toBeChecked();
  await expect(
    page.getByRole('checkbox', { name: 'Compare current rank 2', exact: true }),
  ).toBeDisabled();
  await page.getByRole('button', { name: 'Pin best at this horizon', exact: true }).click();
  await expect(names).toHaveCount(7);
  await page
    .getByRole('button', { name: 'Remove local run run-1.jsonl from comparison', exact: true })
    .click();
  await picker.setInputFiles(file('run-8.jsonl'));
  await expect(names).toHaveCount(7);
  await expect(
    page.getByRole('textbox', { name: 'Local run name: run-8.jsonl', exact: true }),
  ).toHaveValue('run-8.jsonl');
  await expect(page.getByRole('alert')).toHaveCount(0);
  const comparison = page.getByRole('region', { name: 'Trajectory comparison' });
  await expect(comparison.locator('.chip')).toHaveCount(8);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2),
  ).toBeTruthy();
  await page.getByRole('button', { name: 'Clear comparison', exact: true }).click();
  await expect(names).toHaveCount(0);
  await expect(comparison.locator('.chip')).toHaveCount(1);
});
