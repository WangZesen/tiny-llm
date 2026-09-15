import { readFileSync, writeFileSync, mkdirSync, cpSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import { parse } from 'yaml';
import type { RegistryEntry, Study, Curves, Configuration } from '../src/lib/types.ts';
import { configurationId, mean, sd, validatePoints } from '../src/lib/statistics.ts';
import { loadWsd } from './wsd.ts';

export const root = fileURLToPath(new URL('../../', import.meta.url));
export const dataRoot = path.join(root, 'doc/data');
export const json = (p: string) => JSON.parse(readFileSync(p, 'utf8'));
export const registry: RegistryEntry[] = json(path.join(dataRoot, 'studies.json'));
const near = (a: number, b: number) =>
  assert.ok(Number.isFinite(a) && Math.abs(a - b) < 1e-11, `${a} differs from ${b}`);

export function loadStudies(requireCurves = true): Study[] {
  assert.equal(new Set(registry.map((s) => s.id)).size, registry.length);
  return registry.map((entry) => {
    const raw = json(path.join(dataRoot, entry.directory, 'results.json'));
    const protocol = raw.protocol ?? raw;
    const unique = new Set<string>();
    const groups: Configuration[] = raw.groups
      .map((g: any) => {
        const beta1 = g.beta1 ?? raw.fixed_beta1 ?? 0.9;
        const id = configurationId(entry.id, g.lr, beta1, g.beta2);
        const runs = raw.runs
          .filter(
            (r: any) =>
              r.lr === g.lr && (r.beta1 ?? raw.fixed_beta1 ?? 0.9) === beta1 && r.beta2 === g.beta2,
          )
          .map((r: any) => {
            const key = `${id}--${r.seed}`;
            assert.ok(!unique.has(key), `Duplicate ${key}`);
            unique.add(key);
            assert.ok(Number.isFinite(r.loss));
            return {
              id: r.run_id,
              seed: r.seed,
              loss: r.loss,
              directory: r.directory,
              archive: r.archive ?? raw.archive,
            };
          })
          .sort((a: any, b: any) => a.seed - b.seed);
        assert.deepEqual(
          runs.map((r: any) => r.seed),
          [42, 43, 44],
          id,
        );
        near(mean(runs.map((r: any) => r.loss)), g.mean_loss);
        near(sd(runs.map((r: any) => r.loss)), g.std_loss);
        const curvePath = path.join(dataRoot, 'curves', `${id}.json`);
        const curveSeeds: number[] = [];
        if (existsSync(curvePath)) {
          const curves: Curves = json(curvePath);
          assert.equal(curves.configuration, id);
          for (const curve of curves.runs) {
            assert.ok(!curveSeeds.includes(curve.seed));
            curveSeeds.push(curve.seed);
            const run = runs.find((r: any) => r.seed === curve.seed);
            assert.ok(run);
            near(run.loss, curve.final);
            validatePoints(curve.train);
            validatePoints(curve.validation);
            assert.equal(curve.validation.length, 40);
            assert.equal(curve.train.at(-1)?.[0], protocol.training_tokens);
            assert.equal(curve.validation.at(-1)?.[0], protocol.training_tokens);
            assert.match(curve.sha256, /^[a-f0-9]{64}$/);
            assert.match(curve.resultSha256, /^[a-f0-9]{64}$/);
          }
        } else if (requireCurves) throw new Error(`Missing curve snapshot: ${curvePath}`);
        return {
          id,
          study: entry.id,
          lr: g.lr,
          beta1,
          beta2: g.beta2,
          mean: g.mean_loss,
          sd: g.std_loss,
          runs,
          curveSeeds,
          rank: 0,
        };
      })
      .sort(
        (a: Configuration, b: Configuration) =>
          a.mean - b.mean || a.lr - b.lr || a.beta1 - b.beta1 || a.beta2 - b.beta2,
      );
    groups.forEach((g, i) => (g.rank = i + 1));
    assert.equal(groups.length, entry.configurations);
    assert.equal(unique.size, entry.runs);
    assert.equal(raw.runs.length, entry.runs);
    const winner = groups[0];
    near(winner.mean, raw.winner.mean_loss);
    for (const k of ['lr', 'beta2'] as const) assert.equal(winner[k], raw.winner[k]);
    if (entry.preset) {
      const preset = parse(readFileSync(path.join(root, 'configs', entry.preset), 'utf8'));
      for (const k of ['lr', 'beta1', 'beta2'] as const)
        near(preset.optimizer[k] ?? 0.9, winner[k]);
      assert.equal(preset.training.batch_tokens, protocol.batch_tokens);
      assert.equal(preset.decentralized?.num_models ?? 1, entry.workers);
      assert.equal(preset.decentralized?.scheme ?? 'synchronous', entry.scheme);
      assert.equal(preset.optimizer.grad_clip, entry.clip);
    }
    const matches = (
      raw.matched_awc_comparisons ??
      raw.matched_packed4 ??
      raw.matched_clipped ??
      []
    ).map((m: any) => {
      const left = m.awc_mean ?? m.four_mean ?? m.clipped_mean;
      const right = m.atc_mean ?? m.eight_mean ?? m.unclipped_mean;
      const difference =
        m.difference_atc_minus_awc ??
        m.difference_eight_minus_four ??
        m.difference_unclipped_minus_clipped;
      near(right - left, difference);
      const reference = json(path.join(dataRoot, 'recipe_sweep_packed4_20m_128k/results.json'));
      const matchedReference = reference.groups.find(
        (g: any) => g.lr === m.lr && g.beta1 === (m.beta1 ?? 0.9) && g.beta2 === m.beta2,
      );
      assert.ok(matchedReference, 'Missing matched AWC reference');
      near(left, matchedReference.mean_loss);
      const matchedTarget = groups.find(
        (g) => g.lr === m.lr && g.beta1 === (m.beta1 ?? 0.9) && g.beta2 === m.beta2,
      );
      assert.ok(matchedTarget, 'Missing matched target');
      near(right, matchedTarget.mean);
      if (raw.matched_runs) {
        const pairs = raw.matched_runs.filter(
          (r: any) => r.lr === m.lr && r.beta1 === m.beta1 && r.beta2 === m.beta2,
        );
        assert.deepEqual(pairs.map((r: any) => r.seed).sort(), [42, 43, 44]);
        near(mean(pairs.map((r: any) => r.clipped_loss)), left);
        near(mean(pairs.map((r: any) => r.unclipped_loss)), right);
        for (const r of pairs) near(r.unclipped_loss - r.clipped_loss, r.difference);
        near(sd(pairs.map((r: any) => r.difference)), m.paired_std);
      }
      return {
        lr: m.lr,
        beta1: m.beta1 ?? 0.9,
        beta2: m.beta2,
        left,
        right,
        difference,
        leftLabel: 'AWC · 4 workers (clipped)',
        rightLabel: entry.title,
        pairedSD: m.paired_std,
      };
    });
    return {
      ...entry,
      groups,
      matches,
      date: raw.measured_date ?? raw.measured_dates?.join(' – ') ?? '2026-09-11 – 2026-09-12',
      batch: protocol.batch_tokens,
      tokens: protocol.training_tokens,
      parameters: protocol.parameters,
      steps: protocol.optimizer_steps,
      validationTokens: protocol.validation_tokens,
    };
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const studies = loadStudies();
  const manifest = json(path.join(dataRoot, 'curves/coverage.json'));
  const available = studies.flatMap((s) => s.groups).reduce((n, g) => n + g.curveSeeds.length, 0);
  const total = studies.reduce((n, s) => n + s.runs, 0);
  assert.equal(available, manifest.available);
  assert.equal(total - available, manifest.missing.length);
  const actualMissing = studies
    .flatMap((s) =>
      s.groups.flatMap((g) =>
        g.runs.filter((r) => !g.curveSeeds.includes(r.seed)).map((r) => `${g.id}--${r.seed}`),
      ),
    )
    .sort();
  assert.deepEqual(actualMissing, manifest.missing.map((m: any) => m.id).sort());
  const generated = path.join(root, 'site/src/generated');
  mkdirSync(generated, { recursive: true });
  writeFileSync(path.join(generated, 'studies.json'), JSON.stringify(studies));
  writeFileSync(path.join(generated, 'wsd.json'), JSON.stringify(loadWsd()));
  const assets = path.join(root, 'site/public/assets');
  mkdirSync(assets, { recursive: true });
  cpSync(dataRoot, path.join(assets, 'data'), { recursive: true });
  cpSync(
    path.join(root, 'doc/adaptive-consensus-investigation'),
    path.join(assets, 'adaptive-consensus-investigation'),
    { recursive: true },
  );
  cpSync(path.join(root, 'configs'), path.join(assets, 'configs'), { recursive: true });
  console.log(
    `Validated ${total} runs, ${studies.reduce((n, s) => n + s.groups.length, 0)} configurations; ${available} curves, ${total - available} unavailable.`,
  );
}
