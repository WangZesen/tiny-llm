import test from 'node:test';
import assert from 'node:assert/strict';
import { loadPublication } from '../tools/publication.ts';
import {
  winners,
  containerFor,
  curveLabel,
  seriesColors,
  MAX_COMPARE,
} from '../src/lib/current.ts';
import { publicationRoot } from '../tools/publication.ts';
import { readFileSync } from 'node:fs';
import path from 'node:path';
const data = loadPublication();

test('current publication preserves counts, coverage, selected losses and worker availability', () => {
  assert.equal(data.groups.filter((g) => g.schedule === 'cosine').length, 396);
  assert.deepEqual(
    ['sync', 'awc4', 'awc8'].map(
      (m) => data.groups.filter((g) => g.schedule === 'cosine' && g.method === m).length * 3,
    ),
    [216, 432, 540],
  );
  assert.equal(data.groups.filter((g) => g.schedule === 'wsd').length, 198);
  assert.equal(data.groups.filter((g) => g.schedule === 'wsd' && g.method === 'awc4').length, 0);
  assert.equal(data.groups.filter((g) => g.schedule === 'cosine' && g.horizon > 80).length, 0);
  assert.equal(winners(data).length, 15);
  assert.equal(winners(data, false).length, 19);
  assert.equal(
    data.groups
      .find(
        (g) => g.schedule === 'cosine' && g.method === 'sync' && g.horizon === 20 && g.rank === 1,
      )!
      .mean.toFixed(6),
    '3.549818',
  );
  assert.equal(
    data.groups.find(
      (g) => g.schedule === 'cosine' && g.method === 'awc8' && g.horizon === 20 && g.rank === 1,
    )!.lr,
    0.014,
  );
  assert.equal(data.performance.filter((p) => p.horizon === 80).length, 3);
  assert.equal(data.matches.length, 338);
});

test('retained artifacts resolve to checksummed containers', () => {
  const checksums: Record<string, string> = JSON.parse(
    readFileSync(path.join(publicationRoot, 'checksums.json'), 'utf8'),
  );
  const containers = new Set<string>();
  for (const group of data.groups)
    for (const run of group.runs)
      for (const artifact of Object.values(run.artifacts)) {
        const container = containerFor(artifact);
        assert.ok(checksums[container], artifact);
        containers.add(container);
      }
  // 19 campaign stages, each with a metrics and a records container, plus the shared
  // environments: the grouping must stay coarse rather than drift back towards per run.
  assert.equal(containers.size, 39);
  assert.ok(Object.keys(checksums).length < 100, 'the bundle must stay a handful of files');
});

test('every curve is distinguishable in a comparison legend', () => {
  assert.equal(new Set(seriesColors.light).size, MAX_COMPARE);
  assert.equal(new Set(seriesColors.dark).size, MAX_COMPARE);
  assert.equal(new Set(data.groups.map(curveLabel)).size, data.groups.length);
});
