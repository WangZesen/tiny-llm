import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { mean, sd, validatePoints } from '../src/lib/statistics.ts';
import { wsdId, wsdOrder, type WsdData, type WsdCurve } from '../src/lib/wsd.ts';

export const wsdRoot = fileURLToPath(
  new URL('../../doc/data/wsd-horizon-tuning/', import.meta.url),
);
const json = (p: string) => JSON.parse(readFileSync(p, 'utf8'));
const near = (a: number, b: number) =>
  assert.ok(
    Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-11,
    `${a} differs from ${b}`,
  );
export function verifyChecksums(root = wsdRoot) {
  const checksums: Record<string, string> = json(path.join(root, 'checksums.json'));
  assert.ok(Object.keys(checksums).length >= 210);
  for (const [name, expected] of Object.entries(checksums)) {
    assert.ok(!path.isAbsolute(name) && !name.split('/').includes('..'));
    assert.match(expected, /^[a-f0-9]{64}$/);
    assert.equal(
      createHash('sha256')
        .update(readFileSync(path.join(root, name)))
        .digest('hex'),
      expected,
      `Checksum mismatch: ${name}`,
    );
  }
}
export function loadWsd(root = wsdRoot): WsdData {
  verifyChecksums(root);
  const data: WsdData = json(path.join(root, 'dataset.json'));
  assert.equal(data.version, 1);
  assert.deepEqual(data.protocol.horizons, [20, 40, 80, 120, 160]);
  assert.deepEqual(data.protocol.seeds, [42, 43, 44]);
  assert.equal(data.protocol.validationTokens, 197411295);
  assert.equal(data.protocol.schedule, 'wsd');
  assert.equal(data.protocol.warmupSteps, 312);
  assert.equal(data.protocol.decayFraction, 0.1);
  assert.deepEqual(
    data.campaigns.map((c) => c.id),
    ['sync', 'awc8'],
  );
  const ids = new Set<string>();
  let runs = 0;
  for (const c of data.campaigns) {
    assert.deepEqual(c.grid.beta1, [0.9, 0.95]);
    assert.deepEqual(c.grid.beta2, [0.9, 0.98, 0.999]);
    assert.deepEqual(
      c.grid.lr,
      c.id === 'sync'
        ? [0.008, 0.006, 0.004, 0.003, 0.002]
        : [0.01, 0.008, 0.006, 0.004, 0.003, 0.002],
    );
    let ceiling = Math.max(...c.grid.lr);
    for (const [index, s] of c.stages.entries()) {
      assert.equal(s.horizon, data.protocol.horizons[index]);
      assert.equal(s.lrCeiling, ceiling);
      assert.deepEqual(
        s.eligibleLRs,
        c.grid.lr.filter((lr) => lr <= ceiling),
      );
      assert.equal(s.groups.length, s.eligibleLRs.length * 6);
      assert.equal(s.runs, s.groups.length * 3);
      assert.deepEqual(s.groups, [...s.groups].sort(wsdOrder));
      for (const [i, g] of s.groups.entries()) {
        assert.equal(g.id, wsdId(c.id, s.horizon, g.lr, g.beta1, g.beta2));
        assert.ok(!ids.has(g.id));
        ids.add(g.id);
        assert.equal(g.method, c.id);
        assert.equal(g.horizon, s.horizon);
        assert.equal(g.rank, i + 1);
        assert.ok(
          s.eligibleLRs.includes(g.lr) &&
            c.grid.beta1.includes(g.beta1) &&
            c.grid.beta2.includes(g.beta2),
        );
        assert.deepEqual(
          g.runs.map((r) => r.seed),
          [42, 43, 44],
        );
        near(mean(g.runs.map((r) => r.loss)), g.mean);
        near(sd(g.runs.map((r) => r.loss)), g.sd);
        runs += 3;
        const file = path.join(root, 'curves', g.id + '.json');
        assert.ok(existsSync(file));
        const curve: WsdCurve = json(file);
        assert.equal(curve.configuration, g.id);
        assert.deepEqual(
          curve.runs.map((r) => r.seed),
          [42, 43, 44],
        );
        for (const r of curve.runs) {
          near(r.final, g.runs.find((x) => x.seed === r.seed)!.loss);
          assert.equal(r.validation.length, s.epochs);
          assert.equal(r.sources.length, index + 1);
          assert.equal(r.sources[0].startTokens, 0);
          assert.equal(r.sources.at(-1)!.endTokens, s.tokens);
          r.sources.forEach((source, j) => {
            assert.ok(source.startTokens < source.endTokens);
            if (j) assert.equal(source.startTokens, r.sources[j - 1].endTokens);
            assert.match(source.sha256, /^[a-f0-9]{64}$/);
            assert.match(source.resultSha256, /^[a-f0-9]{64}$/);
          });
          for (const kind of ['train', 'validation'] as const) {
            validatePoints(r[kind]);
            assert.equal(r[kind].at(-1)![0], s.tokens);
            if (index) {
              const parentId = wsdId(c.id, c.stages[index - 1].horizon, g.lr, g.beta1, g.beta2);
              const parent: WsdCurve = json(path.join(root, 'curves', parentId + '.json'));
              const cut = r.sources.at(-1)!.startTokens;
              assert.deepEqual(
                r[kind].filter((p) => p[0] <= cut),
                parent.runs.find((x) => x.seed === r.seed)![kind].filter((p) => p[0] <= cut),
              );
              assert.equal(
                r.validation.filter((p) => p[0] <= cut).length,
                c.stages[index - 1].checkpointEpoch,
              );
            }
          }
        }
      }
      ceiling = s.groups[0].lr;
      assert.deepEqual(
        s.prunedLRs,
        c.grid.lr.filter((lr) => lr > ceiling),
      );
    }
    assert.deepEqual(
      c.stages.map((s) => s.runs),
      c.id === 'sync' ? [90, 54, 54, 36, 36] : [108, 72, 72, 36, 36],
    );
  }
  assert.equal(runs, 594);
  assert.equal(ids.size, 198);
  assert.equal(data.matches.length, 90);
  const matchedIds = new Set<string>();
  for (const m of data.matches) {
    const key = wsdId('match', m.horizon, m.lr, m.beta1, m.beta2);
    assert.ok(!matchedIds.has(key));
    matchedIds.add(key);
    const groups = data.campaigns.map((c) =>
      c.stages
        .find((s) => s.horizon === m.horizon)!
        .groups.find((g) => g.lr === m.lr && g.beta1 === m.beta1 && g.beta2 === m.beta2),
    );
    assert.ok(groups[0] && groups[1]);
    near(groups[0].mean, m.sync);
    near(groups[1].mean, m.awc8);
    near(m.awc8 - m.sync, m.difference);
    const differences = groups[0].runs.map((r, i) => groups[1]!.runs[i].loss - r.loss);
    differences.forEach((d, i) =>
      near(d, [m.differenceSeed42, m.differenceSeed43, m.differenceSeed44][i]),
    );
    near(sd(differences), m.pairedSD);
  }
  const flat = json(path.join(root, 'runs.json'));
  assert.equal(flat.length, 594);
  const runIds = new Set<string>();
  for (const r of flat) {
    const g = data.campaigns
      .find((c) => c.id === r.method)!
      .stages.find((s) => s.horizon === r.horizon)!
      .groups.find((g) => g.id === r.configuration)!;
    near(r.loss, g.runs.find((x) => x.seed === r.seed)!.loss);
    assert.equal(r.id, `${g.id}--s${r.seed}`);
    assert.ok(!runIds.has(r.id));
    runIds.add(r.id);
    assert.equal(g.runs.find((x) => x.seed === r.seed)!.id, r.id);
  }
  return data;
}
