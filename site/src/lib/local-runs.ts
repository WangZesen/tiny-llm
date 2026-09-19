import type { Point } from './types';

export interface LocalMetrics {
  train: Point[];
  validation: Point[];
  gradientNorm: Point[];
  warnings: string[];
}

export interface LocalRun extends LocalMetrics {
  id: string;
  filename: string;
  label: string;
}

export const localSeriesLabels = {
  train: 'training loss',
  validation: 'subset-validation loss',
  gradientNorm: 'gradient norm',
} as const;

export function localRunLabel(filename: string, existing: string[]): string {
  const labels = new Set(existing);
  let label = filename;
  for (let suffix = 2; labels.has(label); suffix++) label = `${filename} (${suffix})`;
  return label;
}

/** Parse a trainer log without depending on browser APIs or publication metadata. */
export function parseLocalMetrics(text: string, filename: string): LocalMetrics {
  const points = {
    train: new Map<number, number>(),
    validation: new Map<number, number>(),
    gradientNorm: new Map<number, number>(),
  };
  const warnings: string[] = [];
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/);
  const fail = (line: number, reason: string): never => {
    throw new Error(`${filename}, line ${line}: ${reason}`);
  };
  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    let row: unknown;
    try {
      row = JSON.parse(line);
    } catch (error) {
      // A file copied during training can end partway through its last record. Only
      // tolerate unexpected EOF, not arbitrary syntax errors at the end of a file.
      const message = error instanceof Error ? error.message : '';
      const position = /position (\d+)/.exec(message);
      const incomplete =
        /unexpected end|unterminated|end of (?:the )?(?:json |data)/i.test(message) ||
        (position !== null && Number(position[1]) === line.length);
      if (index === lines.length - 1 && incomplete) {
        warnings.push(`${filename}, line ${index + 1}: incomplete final record ignored.`);
        continue;
      }
      fail(index + 1, 'invalid JSON record.');
    }
    if (!row || typeof row !== 'object' || Array.isArray(row))
      fail(index + 1, 'expected a JSON object.');
    const event = row as Record<string, unknown>;
    if (event.event !== 'train' && event.event !== 'validation') continue;
    if (typeof event.tokens !== 'number' || !Number.isSafeInteger(event.tokens) || event.tokens < 0)
      fail(index + 1, 'tokens must be a nonnegative safe integer.');
    const add = (series: keyof typeof points, field: string) => {
      const value = event[field];
      if (value === undefined) return;
      if (typeof value !== 'number' || !Number.isFinite(value))
        fail(index + 1, `${field} must be a finite number.`);
      // Resumed logs may repeat token positions; the last logged value wins.
      points[series].set(event.tokens as number, value as number);
    };
    if (event.event === 'train') {
      add('train', 'loss');
      add('gradientNorm', 'grad_norm');
    } else add('validation', 'loss');
  }
  if (!Object.values(points).some((series) => series.size))
    throw new Error(
      `${filename}: no usable training loss, subset-validation loss, or gradient norm measurements.`,
    );
  const sorted = (series: Map<number, number>): Point[] =>
    [...series].sort(([left], [right]) => left - right);
  return {
    train: sorted(points.train),
    validation: sorted(points.validation),
    gradientNorm: sorted(points.gradientNorm),
    warnings,
  };
}
