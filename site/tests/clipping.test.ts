import test from 'node:test';
import assert from 'node:assert/strict';
import { aggregateClipping, clippingValues, validateClippingSeries } from '../src/lib/clipping.ts';

const sync = [{ tokens: 10, epoch: 1, total: 0, workers: [0] }];
const packed = [{ tokens: 10, epoch: 1, total: 6, workers: [2, 4] }];

test('worker selection distinguishes zero, total, and an absent worker', () => {
  assert.deepEqual(clippingValues(sync, null), [{ tokens: 10, epoch: 1, count: 0 }]);
  assert.deepEqual(clippingValues(sync, 0), clippingValues(sync, null));
  assert.deepEqual(clippingValues(sync, 1), []);
  assert.deepEqual(clippingValues(packed, null), [{ tokens: 10, epoch: 1, count: 6 }]);
  assert.deepEqual(clippingValues(packed, 1), [{ tokens: 10, epoch: 1, count: 4 }]);
});

test('seed aggregation counts only observations present at each epoch', () => {
  assert.deepEqual(aggregateClipping([sync, packed, []], null), [
    { tokens: 10, epoch: 1, mean: 3, sd: Math.sqrt(18), n: 2 },
  ]);
  assert.deepEqual(aggregateClipping([sync, packed, []], 1), [
    { tokens: 10, epoch: 1, mean: 4, sd: 0, n: 1 },
  ]);
  assert.deepEqual(aggregateClipping([[], []], null), []);
});

test('published clipping series allow missing epochs but reject invalid and nonmonotonic data', () => {
  const later = { tokens: 30, epoch: 3, total: 0, workers: [0] };
  validateClippingSeries([]);
  validateClippingSeries([...sync, later]);
  for (const invalid of [
    null,
    {},
    [later, ...sync],
    [...sync, ...sync],
    [...sync, { ...later, epoch: 1 }],
    [{ ...later, total: 1 }],
    [{ ...later, workers: [] }],
    [{ ...later, total: NaN }],
  ])
    assert.throws(() => validateClippingSeries(invalid));
});
