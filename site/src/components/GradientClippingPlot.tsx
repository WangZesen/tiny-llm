import { useEffect, useMemo } from 'react';
import type { Data } from 'plotly.js';
import Plot from './Plot';
import { aggregateClipping, clippingValues, type GradientClippingPoint } from '../lib/clipping';

export interface ClippingCurve {
  id: string;
  name: string;
  color: string;
  symbol: string;
  dash: 'solid' | 'dash' | 'dot';
  local: boolean;
  runs: { seed?: number; points: GradientClippingPoint[] }[];
}

type ClippingSeries = Omit<ClippingCurve, 'dash'> & {
  dash: ClippingCurve['dash'] | 'dashdot';
  points: ReturnType<typeof aggregateClipping>;
  averaged: boolean;
};

export default function GradientClippingPlot({
  curves,
  worker,
  onWorkerChange,
  seeds,
  focus,
  loading,
}: {
  curves: ClippingCurve[];
  worker: number | null;
  onWorkerChange: (worker: number | null) => void;
  seeds: boolean;
  focus: boolean;
  loading: boolean;
}) {
  const workers = useMemo(
    () =>
      curves.reduce(
        (maximum, curve) =>
          curve.runs.reduce(
            (maximum, run) =>
              run.points.reduce(
                (maximum, point) => Math.max(maximum, point.workers.length),
                maximum,
              ),
            maximum,
          ),
        0,
      ),
    [curves],
  );
  useEffect(() => {
    // Wait for published curves before deciding that a worker restored from the URL is absent.
    if (!loading && worker !== null && worker >= workers) onWorkerChange(null);
  }, [loading, worker, workers, onWorkerChange]);
  const series = useMemo(
    () =>
      curves.flatMap<ClippingSeries>((curve) => {
        if (curve.local || seeds)
          return curve.runs.flatMap((run, index) => {
            const points = clippingValues(run.points, worker).map((p) => ({
              ...p,
              mean: p.count,
              sd: 0,
              n: 1,
            }));
            return points.length
              ? [
                  {
                    ...curve,
                    name: curve.name + (run.seed === undefined ? '' : ` · Seed ${run.seed}`),
                    dash: curve.local
                      ? curve.dash
                      : (['solid', 'dot', 'dashdot'] as const)[index % 3],
                    points,
                    averaged: false,
                  },
                ]
              : [];
          });
        const points = aggregateClipping(
          curve.runs.map((r) => r.points),
          worker,
        );
        return points.length ? [{ ...curve, points, averaged: true }] : [];
      }),
    [curves, worker, seeds],
  );
  const unavailable = useMemo(
    () =>
      curves.flatMap((curve) => {
        const missing = curve.runs.filter((run) => !clippingValues(run.points, worker).length);
        if (!missing.length) return [];
        return [
          curve.name +
            (missing.length === curve.runs.length
              ? ''
              : ` (seeds ${missing.map((run) => run.seed).join(', ')})`),
        ];
      }),
    [curves, worker],
  );
  const view = worker === null ? 'Total across workers' : `Worker ${worker}`;
  const traces = useMemo<Data[]>(
    () =>
      series.flatMap((curve) => {
        // Break lines over unrecorded epochs instead of suggesting counts for missing history.
        const points: ((typeof curve.points)[number] | null)[] = [];
        for (const point of curve.points) {
          const previous = points.at(-1);
          if (previous && point.epoch > previous.epoch + 1) points.push(null);
          points.push(point);
        }
        const x = points.map((p) => p?.tokens ?? null);
        const bands: Data[] =
          curve.averaged && points.some((p) => p && p.n > 1)
            ? [
                {
                  type: 'scatter',
                  mode: 'lines',
                  x,
                  y: points.map((p) => (p ? Math.max(0, p.mean - p.sd) : null)),
                  line: { width: 0 },
                  showlegend: false,
                  hoverinfo: 'skip',
                  legendgroup: curve.id,
                },
                {
                  type: 'scatter',
                  mode: 'lines',
                  x,
                  y: points.map((p) => (p ? p.mean + p.sd : null)),
                  line: { width: 0 },
                  fill: 'tonexty',
                  fillcolor: curve.color + '2e',
                  showlegend: false,
                  hoverinfo: 'skip',
                  legendgroup: curve.id,
                },
              ]
            : [];
        return [
          ...bands,
          {
            type: 'scatter',
            mode: 'lines+markers',
            x,
            y: points.map((p) => p?.mean ?? null),
            customdata: points.map((p) => (p ? [p.epoch, p.n] : [null, null])),
            name: curve.name,
            legendgroup: curve.id,
            line: { color: curve.color, dash: curve.dash, width: 2 },
            marker: { color: curve.color, symbol: curve.symbol, size: 6 },
            hovertemplate: `Epoch %{customdata[0]}<br>%{x:,} tokens<br>${view}<br>${
              curve.averaged ? 'Mean clips %{y:.2f}<br>%{customdata[1]} seeds' : 'Clips %{y:,d}'
            }<extra>%{fullData.name}</extra>`,
          } as Data,
        ];
      }),
    [series, view],
  );
  const layout = useMemo(() => {
    const points = series.flatMap((curve) => curve.points);
    const end = points.reduce((end, point) => Math.max(end, point.tokens), 0);
    const maximum = points.reduce(
      (max, point) =>
        !focus || point.tokens >= end / 2 ? Math.max(max, point.mean + point.sd) : max,
      0,
    );
    return {
      height: 520 + 18 * Math.max(0, series.length - 1),
      margin: { l: 65, r: 15, t: 15, b: 65 + 16 * Math.ceil(series.length / 2) },
      legend: { orientation: 'h' as const, y: -0.2 - 0.035 * series.length, font: { size: 9 } },
      xaxis: {
        title: { text: 'Global training tokens' },
        tickformat: '~s',
        ...(focus && end > 0 ? { range: [end / 2, end * 1.01] } : {}),
      },
      yaxis: {
        title: { text: 'Clipping events per epoch' },
        type: 'linear' as const,
        range: [0, Math.max(1, maximum * 1.1)],
      },
    };
  }, [series, focus]);
  return (
    <>
      <h3 className="plot-heading">Gradient clipping per epoch</h3>
      <div className="curve-controls">
        <label>
          Clipping count view
          <select
            aria-label="Clipping count view"
            value={worker === null ? 'total' : String(worker)}
            onChange={(e) =>
              onWorkerChange(e.target.value === 'total' ? null : Number(e.target.value))
            }
          >
            <option value="total">Total</option>
            {Array.from({ length: workers }, (_, index) => (
              <option key={index} value={index}>
                Worker {index}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="small">
        Each point counts updates above the clipping threshold within one epoch, placed at that
        epoch&rsquo;s ending token count. Total sums all workers, so multiple workers clipping in
        one update count separately. Synchronous runs have only Worker 0. Published curves show seed
        means with sample SD bands, bounded below by zero; the seed and focus controls apply here
        too. Local runs show exact counts.
      </p>
      {traces.length > 0 ? (
        <Plot data={traces} layout={layout} label="Current gradient-clipping counts" />
      ) : loading ? (
        <p className="loading" aria-live="polite">
          Loading clipping counts…
        </p>
      ) : (
        <p className="small muted">
          No clipping-count measurements available in this comparison. Import a metrics log with
          per-epoch clipping counts to display them.
        </p>
      )}
      {!loading && unavailable.length > 0 && (
        <p className="small muted" aria-live="polite">
          No {worker === null ? 'clipping counts' : `Worker ${worker} clipping counts`} for:{' '}
          {unavailable.join('; ')}. Missing measurements are not zero; older sampled norms cannot
          reconstruct exact counts.
        </p>
      )}
    </>
  );
}
