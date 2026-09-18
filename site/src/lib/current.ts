export type Schedule = 'cosine' | 'wsd';
export type Method = 'sync' | 'awc4' | 'awc8';
export const methods: Record<Method, string> = {
  sync: 'Synchronous',
  awc4: 'Four workers',
  awc8: 'Eight workers',
};
export const schedules: Record<Schedule, string> = { cosine: 'Cosine-to-zero', wsd: 'WSD' };
export const colors: Record<Method, string> = { sync: '#30658a', awc4: '#b25c3c', awc8: '#766193' };
export interface SeedResult {
  id: string;
  seed: number;
  loss: number;
  jobId: string;
  artifacts: Record<string, string>;
}
export interface Group {
  id: string;
  schedule: Schedule;
  method: Method;
  horizon: number;
  lr: number;
  beta1: number;
  beta2: number;
  mean: number;
  sd: number;
  rank: number;
  runs: SeedResult[];
}
export interface Campaign {
  schedule: Schedule;
  method: Method;
  title: string;
  grid: { lr: number[]; beta1: number[]; beta2: number[] };
  stages: {
    horizon: number;
    tokens: number;
    steps: number;
    epochs: number;
    eligibleLRs: number[];
    groups: number;
    runs: number;
  }[];
  environment: {
    gpu: string;
    python: string;
    torch: string;
    cuda: string;
    sourceHash: string;
    cacheIdentity: string;
  };
}
export interface Match {
  kind: 'schedule' | 'workers';
  left: string;
  right: string;
  horizon: number;
  lr: number;
  beta1: number;
  beta2: number;
  difference: number;
  pairedSD: number;
  seedDifferences: number[];
}
export interface Distribution {
  median: number;
  q1: number;
  q3: number;
}
export interface Performance {
  method: Method;
  horizon: number;
  n: number;
  measurementDates: { first: string; last: string };
  metrics: Record<
    'steady' | 'training' | 'elapsed' | 'validationSeconds' | 'sessionSeconds' | 'peakGiB',
    Distribution
  >;
}
export interface Publication {
  version: number;
  snapshotDate: string;
  protocol: {
    parameters: number;
    batchTokens: number;
    validationTokens: number;
    warmupSteps: number;
    seeds: number[];
  };
  campaigns: Campaign[];
  groups: Group[];
  matches: Match[];
  performance: Performance[];
}
export interface RecordedCurve {
  configuration: string;
  runs: {
    seed: number;
    final: number;
    train: [number, number][];
    validation: [number, number][];
    gradientNorm: [number, number][];
    sources: {
      path: string;
      container: string;
      sha256: string;
      startTokens: number;
      endTokens: number;
    }[];
  }[];
}
export const groupLabel = (g: Group) => schedules[g.schedule] + ' · ' + methods[g.method];
export const publicationAsset = (name: string) =>
  `${import.meta.env.BASE_URL.replace(/\/$/, '')}/assets/data/current-training/${name}`;

/** Retained artifacts live in grouped containers; placement mirrors container_of() in
 *  scripts/export_publication.py, and the site checks every mapping against checksums.json. */
export const containerFor = (path: string) =>
  path.startsWith('shared/')
    ? 'shared/environments.jsonl.gz'
    : path.startsWith('sources/')
      ? 'sources/campaigns.jsonl.gz'
      : `logs/${path.split('/').slice(1, 4).join('-')}.${
          path.endsWith('/metrics.jsonl') ? 'metrics' : 'records'
        }.jsonl.gz`;
export const evidenceAsset = (path: string) => publicationAsset(containerFor(path)) + '#' + path;

export const MAX_COMPARE = 8;
/** Eight categorical slots in fixed order, validated for lightness, chroma, colour-vision
 *  separation and contrast against both surfaces (--bg #faf6ee light, #201d19 dark).
 *  Assign by insertion order and never recycle, so filtering cannot repaint a curve. */
export const seriesColors: Record<'light' | 'dark', string[]> = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};
/** No eight hues separate for every reader, so each slot carries a marker symbol too and
 *  every curve is named in full in the legend, the tray and the hover. */
export const seriesSymbols = [
  'circle',
  'square',
  'diamond',
  'triangle-up',
  'cross',
  'x',
  'star',
  'hexagon',
];
const workers: Record<Method, string> = {
  sync: '1 worker',
  awc4: '4 workers',
  awc8: '8 workers',
};
/** Short enough to stay on one chip line at 412px. */
export const scopeLabel = (g: Group) =>
  `${schedules[g.schedule]} · ${workers[g.method]} · h${g.horizon}`;
/** Full identity: a comparison spans scopes, so the filters no longer imply it. */
export const curveLabel = (g: Group) => `${scopeLabel(g)} · LR ${g.lr} · β ${g.beta1}/${g.beta2}`;

export async function loadCurve(id: string, signal: AbortSignal): Promise<RecordedCurve> {
  const response = await fetch(publicationAsset('curves/' + id + '.json'), { signal });
  if (!response.ok) throw new Error('Missing curve');
  const curve: RecordedCurve = await response.json();
  if (curve.configuration !== id) throw new Error('Wrong curve');
  return curve;
}
export const winners = (data: Publication, shared = true) =>
  data.groups
    .filter((g) => g.rank === 1 && (!shared || g.horizon <= 80))
    .sort(
      (a, b) =>
        a.horizon - b.horizon ||
        a.schedule.localeCompare(b.schedule) ||
        Object.keys(methods).indexOf(a.method) - Object.keys(methods).indexOf(b.method),
    );
