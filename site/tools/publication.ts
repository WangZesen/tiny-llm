import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { containerFor, type Publication, type RecordedCurve } from '../src/lib/current.ts';
import { mean, sd, validatePoints } from '../src/lib/statistics.ts';

export const publicationRoot = fileURLToPath(
  new URL('../../doc/data/current-training/', import.meta.url),
);
const read = (root: string, name: string) =>
  JSON.parse(readFileSync(path.join(root, name), 'utf8'));
const near = (a: number, b: number) =>
  assert.ok(Number.isFinite(a) && Math.abs(a - b) < 1e-10, String(a));

/** Recorded curves are stored one campaign stage per container; explode on first use. */
function curveLoader(root: string) {
  const loaded = new Map<string, Map<string, RecordedCurve>>();
  return (id: string): RecordedCurve => {
    const group = id.split('--')[0];
    let curves = loaded.get(group);
    if (!curves) {
      curves = new Map();
      const body = gunzipSync(readFileSync(path.join(root, 'curves', group + '.jsonl.gz')));
      for (const line of body.toString().trimEnd().split('\n')) {
        const parsed: RecordedCurve = JSON.parse(line);
        curves.set(parsed.configuration, parsed);
      }
      loaded.set(group, curves);
    }
    const curve = curves.get(id);
    assert.ok(curve, 'Missing recorded curve: ' + id);
    return curve;
  };
}

export function loadPublication(root = publicationRoot): Publication {
  const checksums: Record<string, string> = read(root, 'checksums.json');
  for (const [name, expected] of Object.entries(checksums)) {
    assert.ok(!path.isAbsolute(name) && !name.split('/').includes('..'));
    assert.equal(
      createHash('sha256')
        .update(readFileSync(path.join(root, name)))
        .digest('hex'),
      expected,
      'Publication checksum: ' + name,
    );
  }
  const data: Publication = read(root, 'dataset.json');
  assert.equal(data.version, 1);
  assert.deepEqual(data.protocol.seeds, [42, 43, 44]);
  assert.equal(data.protocol.warmupSteps, 312);
  assert.equal(data.groups.length, 594);
  assert.equal(new Set(data.groups.map((g) => g.id)).size, data.groups.length);
  assert.equal(data.groups.flatMap((g) => g.runs).length, 1782);
  const lookup = new Map(data.groups.map((g) => [g.id, g]));
  const curveOf = curveLoader(root);
  for (const group of data.groups) {
    assert.deepEqual(
      group.runs.map((r) => r.seed),
      [42, 43, 44],
    );
    near(mean(group.runs.map((r) => r.loss)), group.mean);
    near(sd(group.runs.map((r) => r.loss)), group.sd);
    for (const run of group.runs)
      for (const source of Object.values(run.artifacts))
        assert.ok(checksums[containerFor(source)], source);
    const curve: RecordedCurve = curveOf(group.id);
    assert.equal(curve.configuration, group.id);
    assert.deepEqual(
      curve.runs.map((r) => r.seed),
      [42, 43, 44],
    );
    const stage = data.campaigns
      .find((c) => c.schedule === group.schedule && c.method === group.method)!
      .stages.find((s) => s.horizon === group.horizon)!;
    for (const r of curve.runs) {
      near(r.final, group.runs.find((s) => s.seed === r.seed)!.loss);
      assert.equal(r.validation.length, stage.epochs);
      for (const points of [r.train, r.validation]) {
        validatePoints(points);
        assert.equal(points.at(-1)![0], stage.tokens);
      }
      assert.equal(r.sources[0].startTokens, 0);
      assert.equal(r.sources.at(-1)!.endTokens, stage.tokens);
      for (const [i, s] of r.sources.entries()) {
        assert.equal(s.container, containerFor(s.path));
        assert.ok(checksums[s.container]);
        assert.ok(s.startTokens < s.endTokens);
        if (i) assert.equal(s.startTokens, r.sources[i - 1].endTokens);
      }
    }
  }
  for (const campaign of data.campaigns) {
    for (const stage of campaign.stages) {
      const groups = data.groups.filter(
        (g) =>
          g.schedule === campaign.schedule &&
          g.method === campaign.method &&
          g.horizon === stage.horizon,
      );
      assert.equal(groups.length, stage.groups);
      assert.equal(groups.length * 3, stage.runs);
      const ordered = [...groups].sort(
        (a, b) => a.mean - b.mean || a.lr - b.lr || a.beta1 - b.beta1 || a.beta2 - b.beta2,
      );
      ordered.forEach((g, i) => assert.equal(g.rank, i + 1));
    }
  }
  for (const m of data.matches) {
    const a = lookup.get(m.left)!,
      b = lookup.get(m.right)!;
    for (const k of ['horizon', 'lr', 'beta1', 'beta2'] as const) assert.equal(a[k], b[k]);
    if (m.kind === 'schedule') assert.equal(a.method, b.method);
    else assert.equal(a.schedule, b.schedule);
    const differences = b.runs.map((r, i) => r.loss - a.runs[i].loss);
    near(mean(differences), m.difference);
    near(sd(differences), m.pairedSD);
    differences.forEach((d, i) => near(d, m.seedDifferences[i]));
  }
  assert.equal(
    data.performance.reduce((n, r) => n + r.n, 0),
    1188,
  );
  for (const p of data.performance)
    for (const d of Object.values(p.metrics))
      assert.ok(Number.isFinite(d.median) && d.q1 > 0 && d.q1 <= d.median && d.median <= d.q3);
  return data;
}
