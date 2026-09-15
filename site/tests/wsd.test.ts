import test from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { loadWsd, verifyChecksums, wsdRoot } from '../tools/wsd.ts';
import { cellStatus, wsdId, wsdOrder, type WsdData } from '../src/lib/wsd.ts';
const data = loadWsd();
const read = (name: string) => JSON.parse(readFileSync(path.join(wsdRoot, name), 'utf8'));

test('WSD audit preserves counts, complete seeds, winner settings and matched comparisons', () => {
  assert.deepEqual(
    data.campaigns.map((c) => c.stages.map((s) => s.runs)),
    [
      [90, 54, 54, 36, 36],
      [108, 72, 72, 36, 36],
    ],
  );
  assert.deepEqual(
    data.campaigns.map((c) =>
      c.stages.map((s) => [s.groups[0].beta1, s.groups[0].beta2, s.groups[0].lr]),
    ),
    [
      [
        [0.95, 0.98, 0.004],
        [0.95, 0.98, 0.004],
        [0.95, 0.999, 0.003],
        [0.95, 0.999, 0.003],
        [0.95, 0.999, 0.003],
      ],
      [
        [0.95, 0.999, 0.006],
        [0.95, 0.999, 0.006],
        [0.95, 0.999, 0.003],
        [0.95, 0.999, 0.003],
        [0.95, 0.999, 0.003],
      ],
    ],
  );
  assert.deepEqual(
    data.protocol.horizons.map((h) => data.matches.filter((m) => m.horizon === h).length),
    [30, 18, 18, 12, 12],
  );
  assert.equal(data.matches.filter((m) => m.difference > 0).length, 88);
  const stage = data.campaigns[1].stages[2];
  assert.ok(Math.abs(stage.groups[1].mean - stage.groups[0].mean - 0.000015214682539) < 1e-12);
  const p = read('provenance.json');
  assert.equal(p.audit.status, 'passed');
  assert.equal(p.audit.artifacts, 4404);
  assert.equal(p.audit.continuations, 396);
  assert.deepEqual(
    p.campaigns.sync.definition.cache_identity,
    p.campaigns.awc8.definition.cache_identity,
  );
  assert.equal(p.campaigns.sync.definition.code.package, p.campaigns.awc8.definition.code.package);
});
test('exact ties use LR, then beta1, then beta2, and IDs separate campaign and horizon', () => {
  const base = data.campaigns[0].stages[0].groups[0];
  const groups = [
    { ...base, lr: 0.004, beta1: 0.9, beta2: 0.9 },
    { ...base, lr: 0.003, beta1: 0.95, beta2: 0.9 },
    { ...base, lr: 0.003, beta1: 0.9, beta2: 0.999 },
    { ...base, lr: 0.003, beta1: 0.9, beta2: 0.98 },
  ];
  assert.deepEqual([...groups].sort(wsdOrder), [groups[3], groups[2], groups[1], groups[0]]);
  assert.notEqual(wsdId('sync', 20, 0.004, 0.95, 0.98), wsdId('awc8', 20, 0.004, 0.95, 0.98));
  assert.notEqual(wsdId('sync', 20, 0.004, 0.95, 0.98), wsdId('sync', 40, 0.004, 0.95, 0.98));
});
test('pruned, outside grid and missing observations have distinct meanings', () => {
  const c = data.campaigns[0],
    s = c.stages[1];
  assert.equal(cellStatus(c, s, 0.01, 0.95, 0.98), 'Outside original grid');
  assert.equal(cellStatus(c, s, 0.004, 0.925, 0.98), 'Outside original grid');
  assert.equal(cellStatus(c, s, 0.008, 0.95, 0.98), 'Pruned');
  assert.equal(cellStatus(c, s, 0.004, 0.95, 0.98), 'Measured');
  assert.equal(cellStatus(c, { ...s, groups: [] }, 0.004, 0.95, 0.98), 'Missing observations');
});
test('every continuation boundary preserves only the parent stable prefix', () => {
  let continuations = 0;
  for (const c of data.campaigns)
    for (const [i, s] of c.stages.entries())
      if (i)
        for (const g of s.groups) {
          const child = read('curves/' + g.id + '.json');
          const parent = read(
            'curves/' + wsdId(c.id, c.stages[i - 1].horizon, g.lr, g.beta1, g.beta2) + '.json',
          );
          for (const [j, r] of child.runs.entries()) {
            continuations++;
            const cut = r.sources.at(-1).startTokens;
            assert.equal(
              r.validation.filter(([t]: number[]) => t <= cut).length,
              c.stages[i - 1].checkpointEpoch,
            );
            assert.ok(
              parent.runs[j].validation.some(([t]: number[]) => t > cut),
              'parent terminal decay exists',
            );
            for (const kind of ['train', 'validation']) {
              assert.deepEqual(
                r[kind].filter(([t]: number[]) => t <= cut),
                parent.runs[j][kind].filter(([t]: number[]) => t <= cut),
              );
              assert.equal(new Set(r[kind].map(([t]: number[]) => t)).size, r[kind].length);
            }
            assert.equal(r.validation.length, s.epochs);
            assert.equal(r.validation.at(-1)[0], s.tokens);
          }
        }
  assert.equal(continuations, 396);
});
function fixture(fn: (root: string) => void) {
  const root = mkdtempSync(path.join(tmpdir(), 'tiny-llm-wsd-publication-'));
  try {
    cpSync(wsdRoot, root, { recursive: true });
    fn(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}
function mutate(root: string, name: string, fn: (value: any) => void) {
  const file = path.join(root, name),
    value = JSON.parse(readFileSync(file, 'utf8'));
  fn(value);
  writeFileSync(file, JSON.stringify(value));
  const checksums = path.join(root, 'checksums.json'),
    manifest = JSON.parse(readFileSync(checksums, 'utf8'));
  manifest[name] = createHash('sha256').update(readFileSync(file)).digest('hex');
  writeFileSync(checksums, JSON.stringify(manifest));
}
test('publication checksum failures stop loading', () =>
  fixture((root) => {
    writeFileSync(path.join(root, 'runs.csv'), 'corrupt');
    assert.throws(() => verifyChecksums(root), /Checksum mismatch: runs.csv/);
  }));
for (const [name, change] of [
  [
    'incomplete seeds',
    (d: WsdData) => {
      d.campaigns[0].stages[0].groups[0].runs.pop();
    },
  ],
  [
    'wrong winner',
    (d: WsdData) => {
      d.campaigns[0].stages[0].groups.reverse();
    },
  ],
  [
    'pruning re-entry',
    (d: WsdData) => {
      d.campaigns[1].stages[3].lrCeiling = 0.004;
    },
  ],
  [
    'cross-horizon matching',
    (d: WsdData) => {
      d.matches[0].horizon = 160;
    },
  ],
  [
    'duplicate configuration IDs',
    (d: WsdData) => {
      d.campaigns[0].stages[0].groups[1].id = d.campaigns[0].stages[0].groups[0].id;
    },
  ],
] as const)
  test(`data validation rejects ${name}`, () =>
    fixture((root) => {
      mutate(root, 'dataset.json', change);
      assert.throws(() => loadWsd(root));
    }));
test('a duplicate per-run export is rejected', () =>
  fixture((root) => {
    mutate(root, 'runs.json', (r) => {
      r[1] = r[0];
    });
    assert.throws(() => loadWsd(root));
  }));
test('corrupt continuation prefix is rejected even with an updated file checksum', () =>
  fixture((root) => {
    const g = data.campaigns[1].stages[4].groups[0];
    mutate(root, 'curves/' + g.id + '.json', (c) => {
      c.runs[0].validation[0][1] += 0.01;
    });
    assert.throws(() => loadWsd(root));
  }));
