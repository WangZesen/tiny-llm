export type Method = 'sync' | 'awc8';
export type Horizon = 20 | 40 | 80 | 120 | 160;
export interface WsdRun {
  id: string;
  seed: number;
  loss: number;
  jobId: string;
}
export interface WsdGroup {
  id: string;
  method: Method;
  horizon: Horizon;
  beta1: number;
  beta2: number;
  lr: number;
  mean: number;
  sd: number;
  rank: number;
  runs: WsdRun[];
}
export interface WsdStage {
  horizon: Horizon;
  tokens: number;
  epochs: number;
  steps: number;
  lrCeiling: number;
  eligibleLRs: number[];
  prunedLRs: number[];
  checkpointEpoch: number | null;
  groups: WsdGroup[];
  runs: number;
}
export interface WsdCampaign {
  id: Method;
  title: string;
  workers: number;
  color: string;
  identity: string;
  grid: { beta1: number[]; beta2: number[]; lr: number[]; seeds: number[] };
  stages: WsdStage[];
}
export interface WsdMatch {
  horizon: number;
  beta1: number;
  beta2: number;
  lr: number;
  sync: number;
  awc8: number;
  difference: number;
  pairedSD: number;
  differenceSeed42: number;
  differenceSeed43: number;
  differenceSeed44: number;
}
export interface WsdData {
  version: number;
  protocol: {
    schedule: string;
    warmupSteps: number;
    decayFraction: number;
    decayShape: string;
    parameters: number;
    batchTokens: number;
    validationTokens: number;
    seeds: number[];
    horizons: number[];
    evaluation: string;
    budget: string;
  };
  campaigns: WsdCampaign[];
  matches: WsdMatch[];
}
export interface WsdCurve {
  configuration: string;
  runs: {
    seed: number;
    final: number;
    train: [number, number][];
    validation: [number, number][];
    sources: {
      path: string;
      sha256: string;
      resultSha256: string;
      startTokens: number;
      endTokens: number;
      horizon: number;
      recipeIdentity: string;
    }[];
  }[];
}
export const wsdId = (method: string, horizon: number, lr: number, beta1: number, beta2: number) =>
  `${method}-wsd-h${horizon}--${lr}--${beta1}--${beta2}`;
export const wsdOrder = (a: WsdGroup, b: WsdGroup) =>
  a.mean - b.mean || a.lr - b.lr || a.beta1 - b.beta1 || a.beta2 - b.beta2;
export function cellStatus(c: WsdCampaign, s: WsdStage, lr: number, b1: number, b2: number) {
  if (!c.grid.lr.includes(lr) || !c.grid.beta1.includes(b1) || !c.grid.beta2.includes(b2))
    return 'Outside original grid';
  if (lr > s.lrCeiling) return 'Pruned';
  return s.groups.some((g) => g.lr === lr && g.beta1 === b1 && g.beta2 === b2)
    ? 'Measured'
    : 'Missing observations';
}
