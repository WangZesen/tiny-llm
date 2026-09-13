export interface RegistryEntry {
  id: string;
  title: string;
  label: string;
  directory: string;
  workers: number;
  scheme: string;
  clip: number | null;
  color: string;
  preset: string | null;
  runs: number;
  configurations: number;
}
export interface Run {
  id: string;
  seed: number;
  loss: number;
  directory: string;
  archive?: string;
}
export interface Configuration {
  id: string;
  study: string;
  lr: number;
  beta1: number;
  beta2: number;
  mean: number;
  sd: number;
  rank: number;
  runs: Run[];
  curveSeeds: number[];
}
export interface Matched {
  lr: number;
  beta1: number;
  beta2: number;
  left: number;
  right: number;
  difference: number;
  leftLabel: string;
  rightLabel: string;
  pairedSD?: number;
}
export interface Study extends RegistryEntry {
  groups: Configuration[];
  date: string;
  batch: number;
  tokens: number;
  parameters: number;
  steps: number;
  validationTokens: number;
  matches: Matched[];
}
export type Point = [tokens: number, loss: number];
export interface CurveRun {
  seed: number;
  train: Point[];
  validation: Point[];
  final: number;
  source: string;
  sha256: string;
  resultSha256: string;
  recipeIdentity: string;
}
export interface Curves {
  configuration: string;
  runs: CurveRun[];
}
