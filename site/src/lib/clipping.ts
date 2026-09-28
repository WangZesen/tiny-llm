import { mean, sd } from './statistics';

/** One completed epoch; counts are never cumulative across epochs. */
export interface GradientClippingPoint {
  tokens: number;
  epoch: number;
  total: number;
  workers: number[];
}

export function parseClippingPoint(value: unknown): GradientClippingPoint {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('gradient clipping measurement must be an object.');
  const point = value as Record<string, unknown>;
  const integer = (v: unknown): v is number =>
    typeof v === 'number' && Number.isSafeInteger(v) && v >= 0;
  if (!integer(point.tokens))
    throw new Error('clipping tokens must be a nonnegative safe integer.');
  if (!integer(point.epoch) || point.epoch === 0)
    throw new Error('clipping epoch must be a positive safe integer.');
  if (!integer(point.total)) throw new Error('grad_clip_count must be a nonnegative safe integer.');
  if (!Array.isArray(point.workers) || !point.workers.length || !point.workers.every(integer))
    throw new Error(
      'local_grad_clip_counts must be a nonempty array of nonnegative safe integers.',
    );
  if (point.workers.reduce((sum, count) => sum + count, 0) !== point.total)
    throw new Error('local_grad_clip_counts must sum to grad_clip_count.');
  return {
    tokens: point.tokens,
    epoch: point.epoch,
    total: point.total,
    workers: point.workers,
  };
}

export function validateClippingSeries(value: unknown): asserts value is GradientClippingPoint[] {
  if (!Array.isArray(value)) throw new Error('gradientClipping must be an array.');
  let tokens = -1;
  let epoch = 0;
  for (const raw of value) {
    const point = parseClippingPoint(raw);
    if (point.tokens <= tokens || point.epoch <= epoch)
      throw new Error('gradientClipping must have increasing tokens and epochs.');
    tokens = point.tokens;
    epoch = point.epoch;
  }
}

export function clippingValues(points: GradientClippingPoint[], worker: number | null) {
  return points.flatMap((point) => {
    const count = worker === null ? point.total : point.workers[worker];
    return count === undefined ? [] : [{ tokens: point.tokens, epoch: point.epoch, count }];
  });
}

export function aggregateClipping(runs: GradientClippingPoint[][], worker: number | null) {
  const positions = new Map<number, { epoch: number; counts: number[] }>();
  for (const run of runs)
    for (const point of clippingValues(run, worker)) {
      const group = positions.get(point.tokens) ?? { epoch: point.epoch, counts: [] };
      group.counts.push(point.count);
      positions.set(point.tokens, group);
    }
  return [...positions]
    .sort(([a], [b]) => a - b)
    .map(([tokens, { epoch, counts }]) => ({
      tokens,
      epoch,
      mean: mean(counts),
      sd: sd(counts),
      n: counts.length,
    }));
}
