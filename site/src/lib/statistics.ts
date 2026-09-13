import type { Configuration, CurveRun, Point } from './types';
export const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
export const sd = (xs: number[]) =>
  xs.length > 1 ? Math.sqrt(xs.reduce((s, x) => s + (x - mean(xs)) ** 2, 0) / (xs.length - 1)) : 0;
export const configurationId = (study: string, lr: number, beta1: number, beta2: number) =>
  `${study}--${lr}--${beta1}--${beta2}`;
export function aggregate(runs: CurveRun[], kind: 'train' | 'validation') {
  const positions = new Map<number, number[]>();
  for (const run of runs)
    for (const [tokens, loss] of run[kind]) {
      const values = positions.get(tokens) ?? [];
      values.push(loss);
      positions.set(tokens, values);
    }
  return [...positions]
    .sort((a, b) => a[0] - b[0])
    .map(([tokens, values]) => ({ tokens, mean: mean(values), sd: sd(values), n: values.length }));
}
export function validatePoints(points: Point[]) {
  let last = -1;
  for (const [tokens, loss] of points) {
    if (!Number.isInteger(tokens) || tokens <= last || !Number.isFinite(loss))
      throw new Error('Invalid or non-monotonic curve point');
    last = tokens;
  }
}
export function matchedGroups(left: Configuration[], right: Configuration[]) {
  return left.flatMap((a) => {
    const b = right.find((b) => a.lr === b.lr && a.beta1 === b.beta1 && a.beta2 === b.beta2);
    return b ? [{ a, b, difference: b.mean - a.mean }] : [];
  });
}
