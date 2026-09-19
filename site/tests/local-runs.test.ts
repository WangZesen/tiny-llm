import test from 'node:test';
import assert from 'node:assert/strict';
import { localRunLabel, parseLocalMetrics } from '../src/lib/local-runs.ts';

const log = (...events: object[]) => events.map((event) => JSON.stringify(event)).join('\n');
const train = { event: 'train', tokens: 131072, loss: 4.2, grad_norm: 1.5 };

test('trainer logs preserve global tokens and separate subset from full validation', () => {
  const metrics = parseLocalMetrics(
    '\uFEFF' +
      log(
        { event: 'startup', message: 'ready' },
        { ...train, tokens_per_model: 32768, local_losses: [1, 2], local_grad_norms: [1, 1.5] },
        { event: 'validation', tokens: 131072, loss: 4.1, evaluation_tokens: 1000 },
        { event: 'final_validation', tokens: 131072, loss: 4.05 },
      ).replaceAll('\n', '\r\n') +
      '\r\n\r\n',
    'metrics.jsonl',
  );
  assert.deepEqual(metrics, {
    train: [[131072, 4.2]],
    validation: [[131072, 4.1]],
    gradientNorm: [[131072, 1.5]],
    warnings: [],
  });
});

test('unfinished runs and independently missing measurements are usable', () => {
  const metrics = parseLocalMetrics(
    log({ event: 'train', tokens: 0, grad_norm: 0 }, { event: 'train', tokens: 10, loss: 4 }),
    'partial.jsonl',
  );
  assert.deepEqual(metrics.train, [[10, 4]]);
  assert.deepEqual(metrics.gradientNorm, [[0, 0]]);
  assert.deepEqual(metrics.validation, []);
  assert.deepEqual(
    parseLocalMetrics(log({ event: 'validation', tokens: 4, loss: 2 }), 'validation.jsonl').train,
    [],
  );
});

test('repeated positions use the last measurement in each series and sort by tokens', () => {
  const metrics = parseLocalMetrics(
    log(
      { ...train, tokens: 20 },
      { ...train, tokens: 10 },
      { event: 'train', tokens: 20, loss: 3 },
      { event: 'validation', tokens: 20, loss: 3.2 },
      { event: 'validation', tokens: 20, loss: 3.1 },
    ),
    'resumed.jsonl',
  );
  assert.deepEqual(metrics.train, [
    [10, 4.2],
    [20, 3],
  ]);
  assert.deepEqual(metrics.gradientNorm, [
    [10, 1.5],
    [20, 1.5],
  ]);
  assert.deepEqual(metrics.validation, [[20, 3.1]]);
});

test('incomplete final records warn, while valid records without a final newline are kept', () => {
  for (const tail of ['{', '{"event"', '{"event":', '{"event":"tr', '{"event":"train","tokens":']) {
    const metrics = parseLocalMetrics(log(train) + '\n' + tail, 'live.jsonl');
    assert.deepEqual(metrics.train, [[131072, 4.2]]);
    assert.deepEqual(metrics.warnings, ['live.jsonl, line 2: incomplete final record ignored.']);
  }
  assert.equal(parseLocalMetrics(log(train), 'complete.jsonl').warnings.length, 0);
});

test('malformed records identify their file and line, including invalid final records', () => {
  for (const bad of ['not json', '{"event": nope}', '[]', 'null', '42', '{"event": truX']) {
    assert.throws(
      () => parseLocalMetrics(log(train) + '\n' + bad, 'bad.jsonl'),
      /bad.jsonl, line 2:/,
    );
  }
  assert.throws(() => parseLocalMetrics('{\n' + log(train), 'bad.jsonl'), /bad.jsonl, line 1:/);
  assert.throws(() => parseLocalMetrics(log(train) + '\n{\n', 'bad.jsonl'), /bad.jsonl, line 2:/);
});

test('tokens and present measurements must be valid numbers', () => {
  for (const tokens of [-1, 1.1, '100', null, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(
      () => parseLocalMetrics(log({ ...train, tokens }), 'bad.jsonl'),
      /line 1: tokens/,
    );
  }
  for (const field of ['loss', 'grad_norm']) {
    for (const value of ['4', null, true]) {
      assert.throws(
        () => parseLocalMetrics(log({ ...train, [field]: value }), 'bad.jsonl'),
        new RegExp(`line 1: ${field}`),
      );
    }
    assert.throws(
      () => parseLocalMetrics(`{"event":"train","tokens":10,"${field}":1e999}`, 'bad.jsonl'),
      /finite number/,
    );
  }
});

test('empty logs and full-validation-only logs cannot become empty curves', () => {
  for (const content of [
    '',
    '\n \n',
    log({ event: 'startup' }),
    log({ event: 'final_validation', tokens: 10, loss: 4 }),
    '{',
  ]) {
    assert.throws(() => parseLocalMetrics(content, 'empty.jsonl'), /empty.jsonl: no usable/);
  }
});

test('default run labels distinguish identically named files', () => {
  assert.equal(localRunLabel('metrics.jsonl', []), 'metrics.jsonl');
  assert.equal(
    localRunLabel('metrics.jsonl', ['metrics.jsonl', 'metrics.jsonl (2)']),
    'metrics.jsonl (3)',
  );
});
