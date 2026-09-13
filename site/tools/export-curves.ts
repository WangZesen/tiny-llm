import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parse } from 'yaml';
import { loadStudies, root, dataRoot } from './data.ts';
import { validatePoints } from '../src/lib/statistics.ts';
import type { Curves, Point } from '../src/lib/types.ts';

const args = process.argv.slice(2);
const offset = args.indexOf('--runs-root');
assert.ok(
  args.length === 0 || (offset === 0 && args.length === 2),
  'Usage: npm run data:refresh -- [--runs-root /path/to/runs]',
);
const runsRoot = offset < 0 ? path.join(root, 'runs') : path.resolve(args[offset + 1]);
const sha = (s: Buffer) => createHash('sha256').update(s).digest('hex');
const missing: { id: string; source: string; reason: string }[] = [];
let available = 0;
const output = path.join(dataRoot, 'curves');
mkdirSync(output, { recursive: true });
for (const study of loadStudies(false))
  for (const group of study.groups) {
    const snapshot: Curves = { configuration: group.id, runs: [] };
    for (const run of group.runs) {
      const source = run.directory.startsWith('experiments/')
        ? `${run.archive}/${run.directory}`
        : run.directory;
      assert.ok(source.startsWith('runs/'), `Unrecognized run path ${source}`);
      const directory = path.join(runsRoot, source.slice(5));
      const file = path.join(directory, 'metrics.jsonl');
      if (!existsSync(file)) {
        missing.push({
          id: `${group.id}--${run.seed}`,
          source: `${source}/metrics.jsonl`,
          reason: 'Source log unavailable',
        });
        continue;
      }
      const bytes = readFileSync(file);
      const resultBytes = readFileSync(path.join(directory, 'result.json'));
      const result = JSON.parse(resultBytes.toString());
      const recipe = parse(readFileSync(path.join(directory, 'resolved.yaml'), 'utf8'));
      assert.equal(recipe.runtime.seed, run.seed);
      for (const k of ['lr', 'beta1', 'beta2'] as const)
        assert.equal(recipe.optimizer[k], group[k]);
      assert.equal(recipe.training.batch_tokens, study.batch);
      assert.equal(recipe.optimizer.grad_clip, study.clip);
      assert.equal(result.status, 'complete');
      assert.equal(result.tokens, study.tokens);
      assert.equal(result.final_validation.validation_complete, true);
      assert.equal(result.final_validation.tokens, study.validationTokens);
      assert.ok(Math.abs(result.final_validation.loss - run.loss) < 1e-12);
      const events = bytes
        .toString()
        .trim()
        .split('\n')
        .map((s) => JSON.parse(s));
      const final = events.filter((e) => e.event === 'final_validation');
      assert.equal(final.length, 1);
      assert.ok(Math.abs(final[0].loss - run.loss) < 1e-12);
      const points = (kind: string): Point[] =>
        events.filter((e) => e.event === kind).map((e) => [e.tokens, e.loss]);
      const train = points('train'),
        validation = points('validation');
      validatePoints(train);
      validatePoints(validation);
      assert.equal(validation.length, 40);
      assert.equal(train.at(-1)?.[0], study.tokens);
      snapshot.runs.push({
        seed: run.seed,
        train,
        validation,
        final: final[0].loss,
        source: `${source}/metrics.jsonl`,
        sha256: sha(bytes),
        resultSha256: sha(resultBytes),
        recipeIdentity: result.recipe_identity,
      });
      available++;
    }
    writeFileSync(path.join(output, `${group.id}.json`), JSON.stringify(snapshot) + '\n');
  }
writeFileSync(
  path.join(output, 'coverage.json'),
  JSON.stringify(
    {
      schemaVersion: 1,
      available,
      missing,
      note: 'Train loss is the logged token-weighted window loss. Validation curves use the fixed epoch subset. Final full-validation loss is separate. Source paths are historical provenance, not website URLs.',
    },
    null,
    2,
  ) + '\n',
);
console.log(`Exported ${available} runs; ${missing.length} source logs unavailable.`);
