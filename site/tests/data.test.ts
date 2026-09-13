import test from 'node:test';
import assert from 'node:assert/strict';
import { aggregate, matchedGroups, validatePoints } from '../src/lib/statistics.ts';
import { loadStudies, dataRoot, json } from '../tools/data.ts';
import type { CurveRun, Configuration } from '../src/lib/types.ts';
import path from 'node:path';

test('committed studies preserve audited counts, seed statistics and winners', () => {
  const studies = loadStudies();
  assert.equal(
    studies.reduce((s, d) => s + d.runs, 0),
    753,
  );
  assert.equal(
    studies.reduce((s, d) => s + d.groups.length, 0),
    251,
  );
  const available = studies.flatMap((s) => s.groups).reduce((s, g) => s + g.curveSeeds.length, 0);
  const coverage = json(path.join(dataRoot, 'curves/coverage.json'));
  assert.equal(available, coverage.available);
  assert.equal(available + coverage.missing.length, 753);
  assert.ok(available >= 726, 'Published curve coverage must not regress');
  assert.equal(studies.find((s) => s.id === 'atc4')!.matches.length, 21);
  assert.equal(studies.find((s) => s.id === 'awc8')!.matches.length, 28);
  assert.equal(
    studies.find((s) => s.id === 'noclip4')!.matches.filter((m) => m.difference > 0).length,
    27,
  );
});
test('curve statistics use matching positions and sample standard deviation', () => {
  const runs = [
    {
      validation: [
        [10, 1],
        [20, 4],
      ],
    },
    {
      validation: [
        [10, 3],
        [30, 9],
      ],
    },
  ] as CurveRun[];
  const result = aggregate(runs, 'validation');
  assert.deepEqual(
    result.map((r) => [r.tokens, r.mean, r.n]),
    [
      [10, 2, 2],
      [20, 4, 1],
      [30, 9, 1],
    ],
  );
  assert.equal(result[0].sd, Math.SQRT2);
  assert.throws(() =>
    validatePoints([
      [20, 1],
      [10, 2],
    ]),
  );
  assert.throws(() => validatePoints([[10, Number.NaN]]));
});
test('matching does not join measurements with different beta1', () => {
  const a = [{ lr: 0.008, beta1: 0.9, beta2: 0.99, mean: 3.6 }] as Configuration[];
  const b = [{ lr: 0.008, beta1: 0.95, beta2: 0.99, mean: 3.5 }] as Configuration[];
  assert.deepEqual(matchedGroups(a, b), []);
  assert.equal(matchedGroups(a, [{ ...a[0], mean: 3.4 }])[0].difference, 3.4 - 3.6);
});
