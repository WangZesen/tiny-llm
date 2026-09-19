import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Data } from 'plotly.js';
import Plot from './Plot';
import {
  methods,
  schedules,
  colors,
  groupLabel,
  publicationAsset,
  evidenceAsset,
  winners,
  loadCurve,
  curveLabel,
  scopeLabel,
  seriesColors,
  seriesSymbols,
  MAX_COMPARE,
  type Publication,
  type RecordedCurve,
  type Schedule,
  type Method,
  type Group,
} from '../lib/current';
import { aggregate } from '../lib/statistics';
import { useTheme } from '../lib/theme';
import {
  localRunLabel,
  localSeriesLabels,
  parseLocalMetrics,
  type LocalRun,
} from '../lib/local-runs';

type State = {
  schedule: Schedule;
  method: Method;
  horizon: string;
  lr: string;
  beta1: string;
  beta2: string;
  selected: string;
  /** Pinned configurations and local runs, insertion-ordered. Independent of the scope
   *  filters: the comparisons worth making cross schedules, worker counts and horizons. */
  compare: string[];
  localRuns: Record<string, LocalRun>;
  importMessages: string[];
  curve: 'train' | 'validation';
  seeds: boolean;
  focus: boolean;
  yScale: 'linear' | 'log';
  gradScale: 'linear' | 'log';
};
const defaults: State = {
  schedule: 'cosine',
  method: 'sync',
  horizon: '20',
  lr: 'all',
  beta1: 'all',
  beta2: 'all',
  selected: '',
  compare: [],
  localRuns: {},
  importMessages: [],
  curve: 'validation',
  seeds: false,
  focus: false,
  yScale: 'linear',
  gradScale: 'linear',
};
const horizons = ['20', '40', '80', '120', '160'];
const fmt = (v: number) => v.toFixed(6);
const signed = (v: number) => (v > 0 ? '+' : '') + fmt(v);
function download(name: string, body: string, type: string) {
  const object = URL.createObjectURL(new Blob([body], { type }));
  const link = document.createElement('a');
  link.href = object;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(object), 1000);
}
function csv(groups: Group[]) {
  const keys = [
    'schedule',
    'method',
    'horizon',
    'rank',
    'lr',
    'beta1',
    'beta2',
    'mean',
    'sd',
  ] as const;
  return [
    ...[keys.join(',') + ',lossSeed42,lossSeed43,lossSeed44'],
    ...groups.map(
      (g) => keys.map((k) => g[k]).join(',') + ',' + g.runs.map((r) => r.loss).join(','),
    ),
  ].join('\n');
}
export default function CurrentExplorer({ data }: { data: Publication }) {
  const [state, setState] = useState<State>(defaults);
  const [ready, setReady] = useState(false);
  const [descending, setDescending] = useState(false);
  const [curves, setCurves] = useState<Record<string, RecordedCurve>>({});
  const [failed, setFailed] = useState<string[]>([]);
  const [notice, setNotice] = useState('');
  const [importing, setImporting] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const importGeneration = useRef(0);
  const localSequence = useRef(0);
  const theme = useTheme();
  const lookup = useMemo(() => new Map(data.groups.map((g) => [g.id, g])), [data]);
  useEffect(() => {
    const restore = () => {
      const p = new URLSearchParams(location.search);
      const next = { ...defaults, compare: [] as string[] };
      if (p.get('schedule') === 'wsd') next.schedule = 'wsd';
      if (p.get('method') === 'awc4' || p.get('method') === 'awc8')
        next.method = p.get('method') as Method;
      if (horizons.includes(p.get('horizon') ?? '')) next.horizon = p.get('horizon')!;
      // A configuration carries its own scope, so a bare ?selected= link resolves.
      const target = lookup.get(p.get('selected') ?? '');
      if (target) {
        next.schedule = target.schedule;
        next.method = target.method;
        next.horizon = String(target.horizon);
        next.selected = target.id;
      }
      const scope = data.groups.filter(
        (g) =>
          g.schedule === next.schedule &&
          g.method === next.method &&
          String(g.horizon) === next.horizon,
      );
      for (const k of ['lr', 'beta1', 'beta2'] as const)
        if (scope.some((g) => String(g[k]) === p.get(k))) next[k] = p.get(k)!;
      // Pinned curves are validated against every group, not the current scope.
      next.compare = [...new Set((p.get('compare') ?? '').split(',').filter(Boolean))]
        .filter((id) => lookup.has(id))
        .slice(0, MAX_COMPARE);
      next.curve = p.get('curve') === 'train' ? 'train' : 'validation';
      next.seeds = p.get('seeds') === '1';
      next.focus = p.get('focus') === '1';
      next.yScale = p.get('yScale') === 'log' ? 'log' : 'linear';
      next.gradScale = p.get('gradScale') === 'log' ? 'log' : 'linear';
      setState((previous) => ({
        ...next,
        localRuns: previous.localRuns,
        importMessages: previous.importMessages,
        compare: [
          ...previous.compare.filter((id) => previous.localRuns[id]),
          ...next.compare,
        ].slice(0, MAX_COMPARE),
      }));
      setReady(true);
    };
    restore();
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
  }, [data, lookup]);
  useEffect(() => {
    if (!ready) return;
    const params = new URLSearchParams({
      schedule: state.schedule,
      method: state.method,
      horizon: state.horizon,
    });
    for (const k of ['lr', 'beta1', 'beta2', 'selected'] as const)
      if (state[k] && state[k] !== 'all') params.set(k, state[k]);
    const published = state.compare.filter((id) => lookup.has(id));
    if (published.length) params.set('compare', published.join(','));
    if (state.curve === 'train') params.set('curve', 'train');
    if (state.seeds) params.set('seeds', '1');
    if (state.focus) params.set('focus', '1');
    if (state.yScale === 'log') params.set('yScale', 'log');
    if (state.gradScale === 'log') params.set('gradScale', 'log');
    // A comma is a legal sub-delimiter and keeps eight pinned ids readable in the bar.
    const query = params.toString().replace(/%2C/g, ',');
    history.replaceState(null, '', location.pathname + '?' + query);
  }, [state, ready, lookup]);
  const scoped = useMemo(
    () =>
      data.groups.filter(
        (g) =>
          g.schedule === state.schedule &&
          g.method === state.method &&
          String(g.horizon) === state.horizon,
      ),
    [data, state.schedule, state.method, state.horizon],
  );
  const filtered = useMemo(
    () =>
      scoped.filter((g) =>
        (['lr', 'beta1', 'beta2'] as const).every(
          (k) => state[k] === 'all' || String(g[k]) === state[k],
        ),
      ),
    [scoped, state.lr, state.beta1, state.beta2],
  );
  const ordered = useMemo(
    () => [...filtered].sort((a, b) => (a.mean - b.mean) * (descending ? -1 : 1)),
    [filtered, descending],
  );
  const selected = filtered.find((g) => g.id === state.selected) ?? filtered[0];
  const campaign = data.campaigns.find(
    (c) => c.schedule === state.schedule && c.method === state.method,
  );
  const stage = campaign?.stages.find((s) => String(s.horizon) === state.horizon);
  const matches = useMemo(
    () => data.matches.filter((m) => m.left === selected?.id || m.right === selected?.id),
    [data, selected?.id],
  );
  // Pinned curves plus, when there is room, the configuration being inspected. Browsing
  // the rankings therefore never disturbs a comparison, and each pin keeps its colour.
  const curveIds = useMemo(() => {
    const ids = [...state.compare];
    if (selected && !ids.includes(selected.id) && ids.length < MAX_COMPARE) ids.push(selected.id);
    return ids;
  }, [state.compare, selected?.id]);
  const curveKey = curveIds.filter((id) => lookup.has(id)).join(',');
  const known = useRef<Record<string, RecordedCurve>>({});
  useEffect(() => {
    const controller = new AbortController();
    const wanted = curveIds.filter((id) => lookup.has(id) && !known.current[id]);
    if (wanted.length) {
      setFailed((f) => f.filter((id) => !wanted.includes(id)));
      void Promise.all(
        wanted.map((id) =>
          loadCurve(id, controller.signal)
            .then((curve) => {
              known.current[id] = curve;
              setCurves((prev) => ({ ...prev, [id]: curve }));
            })
            .catch((e) => {
              if (e.name !== 'AbortError') setFailed((f) => (f.includes(id) ? f : [...f, id]));
            }),
        ),
      );
    }
    return () => controller.abort();
    // Depending on `curves` here would abort the siblings of every arriving response.
  }, [curveKey]);
  function scope(key: 'schedule' | 'method' | 'horizon', value: string) {
    // `compare` survives: a comparison is the point of leaving the current scope.
    setState((s) => ({ ...s, [key]: value, selected: '', lr: 'all', beta1: 'all', beta2: 'all' }));
    setDescending(false);
  }
  function pin(id: string) {
    setNotice(
      !state.compare.includes(id) && state.compare.length >= MAX_COMPARE
        ? `Comparison holds ${MAX_COMPARE} curves. Remove one to add another.`
        : '',
    );
    setState((s) =>
      s.compare.includes(id) || s.compare.length >= MAX_COMPARE
        ? s
        : { ...s, compare: [...s.compare, id] },
    );
  }
  function unpin(id: string) {
    setNotice('');
    setState((s) => {
      const localRuns = { ...s.localRuns };
      delete localRuns[id];
      return { ...s, localRuns, compare: s.compare.filter((c) => c !== id) };
    });
  }
  const toggleCompare = (id: string) => (state.compare.includes(id) ? unpin(id) : pin(id));
  async function importFiles(files: File[]) {
    if (!files.length) return;
    const generation = ++importGeneration.current;
    setImporting(true);
    setNotice('');
    const results: ({ run: LocalRun } | { error: string })[] = [];
    for (const file of files) {
      try {
        const metrics = parseLocalMetrics(await file.text(), file.name);
        results.push({
          run: {
            ...metrics,
            id: `local-${++localSequence.current}`,
            filename: file.name,
            label: file.name,
          },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : 'could not read file.';
        results.push({
          error: message.startsWith(file.name) ? message : `${file.name}: ${message}`,
        });
      }
      if (generation !== importGeneration.current) return;
    }
    setState((s) => {
      const compare = [...s.compare];
      const localRuns = { ...s.localRuns };
      const importMessages: string[] = [];
      for (const result of results) {
        if ('error' in result) {
          importMessages.push(result.error);
          continue;
        }
        const run = result.run;
        if (compare.length >= MAX_COMPARE) {
          importMessages.push(
            `${run.filename}: not added. Comparison holds ${MAX_COMPARE} curves; remove one and select the file again.`,
          );
          continue;
        }
        localRuns[run.id] = {
          ...run,
          label: localRunLabel(
            run.filename,
            Object.values(localRuns).map((local) => local.label),
          ),
        };
        compare.push(run.id);
      }
      return { ...s, compare, localRuns, importMessages };
    });
    setImporting(false);
  }
  function pinBest() {
    setNotice('');
    setState((s) => {
      const best = data.groups
        .filter((g) => g.rank === 1 && String(g.horizon) === s.horizon)
        .sort(
          (a, b) =>
            a.schedule.localeCompare(b.schedule) ||
            Object.keys(methods).indexOf(a.method) - Object.keys(methods).indexOf(b.method),
        );
      return {
        ...s,
        compare: [...new Set([...s.compare, ...best.map((g) => g.id)])].slice(0, MAX_COMPARE),
      };
    });
  }
  const winnerTraces = useMemo<Data[]>(
    () =>
      data.campaigns.map((c) => {
        const gs = winners(data, false).filter(
          (g) => g.schedule === c.schedule && g.method === c.method,
        );
        return {
          x: gs.map((g) => g.horizon),
          y: gs.map((g) => g.mean),
          name: c.title,
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: colors[c.method], dash: c.schedule === 'cosine' ? 'solid' : 'dash' },
          marker: { symbol: c.schedule === 'cosine' ? 'circle' : 'diamond' },
          error_y: { type: 'data', array: gs.map((g) => g.sd), visible: true },
          customdata: gs.map((g) => g.id),
          hovertemplate: '%{x} tokens/parameter<br>Loss %{y:.6f}<extra>%{fullData.name}</extra>',
        };
      }),
    [data],
  );
  const responseTraces = useMemo<Data[]>(
    () =>
      [...new Set(filtered.map((g) => g.beta1 + '/' + g.beta2))].map((pair) => {
        const gs = filtered
          .filter((g) => g.beta1 + '/' + g.beta2 === pair)
          .sort((a, b) => a.lr - b.lr);
        return {
          x: gs.map((g) => g.lr),
          y: gs.map((g) => g.mean),
          name: 'β₁ / β₂ ' + pair,
          type: 'scatter',
          mode: 'lines+markers',
          customdata: gs.map((g) => g.id),
          error_y: { type: 'data', array: gs.map((g) => g.sd), visible: true },
        };
      }),
    [filtered],
  );
  const palette = seriesColors[theme === 'dark' ? 'dark' : 'light'];
  /** Loss and gradient norm share one curve identity, so both figures are built the same way
   *  and a colour, dash or marker means the same configuration in each. */
  const buildTraces = useCallback(
    (series: 'train' | 'validation' | 'gradientNorm', quantity: string, format: string): Data[] =>
      curveIds.flatMap((id, i) => {
        const color = palette[i % palette.length];
        const local = state.localRuns[id];
        if (local) {
          const points = local[series];
          if (!points.length) return [];
          return [
            {
              type: 'scatter',
              mode: points.length === 1 || curveIds.length > 3 ? 'lines+markers' : 'lines',
              x: points.map((pt) => pt[0]),
              y: points.map((pt) => pt[1]),
              name: 'Local · ' + (local.label || local.filename),
              legendgroup: id,
              line: { color, width: 2, dash: 'dot' },
              marker: {
                color,
                symbol: seriesSymbols[i % seriesSymbols.length],
                size: 6,
                maxdisplayed: 8,
              },
              hovertemplate: `%{x:,} tokens<br>${quantity} %{y:${format}}<extra>%{fullData.name}</extra>`,
            },
          ] as Data[];
        }
        const recorded = curves[id];
        const g = lookup.get(id);
        if (!recorded || !g) return [];
        // Curve payloads are fetched from a separately cached URL, so a visitor can hold an
        // older one than the code asking for a series it predates. Drop those runs instead
        // of failing the whole explorer; the cache entry expires within minutes.
        const runs = recorded.runs.filter((r) => r[series]?.length);
        if (!runs.length) return [];
        // Redundant with colour, because eight hues cannot separate for every reader.
        const dash = g.schedule === 'wsd' ? 'dash' : 'solid';
        const name = curveLabel(g);
        if (state.seeds)
          return runs.map((r, j): Data => ({
            type: 'scatter',
            mode: 'lines',
            x: r[series].map((pt) => pt[0]),
            y: r[series].map((pt) => pt[1]),
            line: { color, width: 1, dash: (['solid', 'dot', 'dashdot'] as const)[j] },
            opacity: 0.7,
            legendgroup: id,
            showlegend: curveIds.length === 1 || j === 0,
            name: curveIds.length === 1 ? 'Seed ' + r.seed : name,
            hovertemplate: `%{x:,} tokens<br>${quantity} %{y:${format}}<extra>%{fullData.name}</extra>`,
          }));
        const points = aggregate(runs, series);
        const x = points.map((pt) => pt.tokens);
        // The band must sit immediately after its lower bound for fill: 'tonexty'.
        return [
          {
            type: 'scatter',
            x,
            y: points.map((pt) => pt.mean - pt.sd),
            mode: 'lines',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            legendgroup: id,
          },
          {
            type: 'scatter',
            x,
            y: points.map((pt) => pt.mean + pt.sd),
            mode: 'lines',
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: color + (curveIds.length <= 3 ? '2e' : '1a'),
            showlegend: false,
            hoverinfo: 'skip',
            legendgroup: id,
          },
          {
            type: 'scatter',
            x,
            y: points.map((pt) => pt.mean),
            name,
            legendgroup: id,
            mode: curveIds.length > 3 ? 'lines+markers' : 'lines',
            marker: {
              symbol: seriesSymbols[i % seriesSymbols.length],
              size: 6,
              maxdisplayed: 8,
              color,
            },
            line: { color, width: 2, dash },
            customdata: points.map((pt) => pt.n),
            hovertemplate: `%{x:,} tokens<br>${quantity} %{y:${format}}<br>%{customdata} seeds<extra>%{fullData.name}</extra>`,
          },
        ] as Data[];
      }),
    [curveIds, curves, state.seeds, state.localRuns, lookup, palette],
  );
  const curveTraces = useMemo(
    () => buildTraces(state.curve, 'Loss', '.6f'),
    [buildTraces, state.curve],
  );
  const gradientTraces = useMemo(
    () => buildTraces('gradientNorm', 'Gradient norm', '.4f'),
    [buildTraces],
  );
  const rangeFor = useCallback(
    (series: 'train' | 'validation' | 'gradientNorm') => {
      const plotted = curveIds.map((id) => {
        const local = state.localRuns[id];
        return aggregate(
          local ? [local] : (curves[id]?.runs ?? []).filter((r) => r[series]?.length),
          series,
        );
      });
      const end = Math.max(0, ...plotted.map((points) => points.at(-1)?.tokens ?? 0));
      const from = end / 2;
      let low = Infinity;
      let high = -Infinity;
      for (const points of plotted)
        for (const pt of points)
          if (pt.tokens >= from) {
            low = Math.min(low, pt.mean - pt.sd);
            high = Math.max(high, pt.mean + pt.sd);
          }
      if (!Number.isFinite(low)) return null;
      const pad = (high - low || 0.01) * 0.15;
      return { x: [from, end * 1.01], y: [low - pad, high + pad] };
    },
    [curveIds, curves, state.localRuns],
  );
  const focusRange = useMemo(() => rangeFor(state.curve), [rangeFor, state.curve]);
  const gradientRange = useMemo(() => rangeFor('gradientNorm'), [rangeFor]);
  const curveLayout = useMemo(
    () => ({
      height: 520 + 18 * Math.max(0, curveIds.length - 1),
      margin: { l: 55, r: 15, t: 15, b: 65 + 16 * Math.ceil(curveIds.length / 2) },
      legend: { orientation: 'h' as const, y: -0.2 - 0.035 * curveIds.length, font: { size: 9 } },
      // Pinned horizons differ eightfold, so abbreviate rather than print every digit.
      xaxis: {
        title: { text: 'Global training tokens' },
        tickformat: '~s',
        ...(state.focus && focusRange ? { range: focusRange.x } : {}),
      },
      yaxis: {
        title: { text: 'Loss (nats)' },
        type: state.yScale,
        // Plotly log-axis ranges are expressed in log10 space; the focus range is computed
        // linearly, so it only applies cleanly to the linear axis.
        ...(state.focus && focusRange && state.yScale === 'linear' ? { range: focusRange.y } : {}),
      },
    }),
    [curveIds.length, state.focus, focusRange, state.yScale],
  );
  const gradientLayout = useMemo(
    () => ({
      ...curveLayout,
      xaxis: {
        ...curveLayout.xaxis,
        ...(state.focus && gradientRange ? { range: gradientRange.x } : {}),
      },
      yaxis: {
        title: { text: 'Gradient norm' },
        type: state.gradScale,
        ...(state.focus && gradientRange && state.gradScale === 'linear'
          ? { range: gradientRange.y }
          : {}),
      },
    }),
    [curveLayout, state.focus, gradientRange, state.gradScale],
  );
  const missing = curveIds.filter((id) => failed.includes(id));
  const loading = curveIds.some((id) => lookup.has(id) && !curves[id] && !failed.includes(id));
  function inspect(id: string) {
    const g = lookup.get(id);
    if (g)
      setState((s) => ({
        ...s,
        schedule: g.schedule,
        method: g.method,
        horizon: String(g.horizon),
        selected: g.id,
        lr: 'all',
        beta1: 'all',
        beta2: 'all',
      }));
  }
  const fileStem = state.method + '-' + state.schedule + '-h' + state.horizon + '-filtered';
  return (
    <div className="current-explorer">
      <section className="publication-section" aria-label="Winners across horizons">
        <h2>The best tested recipes, across budgets.</h2>
        <p>
          Color identifies the training mode; solid lines are cosine-to-zero and dashed lines are
          WSD. Error bars show sample SD. Click a point to inspect its configuration.
        </p>
        <Plot
          data={winnerTraces}
          label="Selected results across training horizons"
          layout={{
            height: 450,
            xaxis: {
              title: { text: 'Global tokens per parameter' },
              tickvals: [20, 40, 80, 120, 160],
            },
            yaxis: { title: { text: 'Final full-validation loss (nats)' } },
          }}
          onPoint={inspect}
        />
      </section>
      <section className="publication-section" aria-label="Current configuration explorer">
        <div className="section-heading">
          <h2>Explore each experiment.</h2>
          <button
            onClick={() => {
              setState((s) => ({
                ...defaults,
                compare: s.compare,
                localRuns: s.localRuns,
                importMessages: s.importMessages,
              }));
              setNotice('');
              setDescending(false);
            }}
          >
            Reset filters
          </button>
        </div>
        <div className="publication-filters">
          <label>
            Schedule
            <select
              aria-label="Schedule"
              value={state.schedule}
              onChange={(e) => scope('schedule', e.target.value)}
            >
              {Object.entries(schedules).map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Training mode
            <select
              aria-label="Training mode"
              value={state.method}
              onChange={(e) => scope('method', e.target.value)}
            >
              {Object.entries(methods).map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Horizon
            <select
              aria-label="Horizon"
              value={state.horizon}
              onChange={(e) => scope('horizon', e.target.value)}
            >
              {horizons.map((h) => (
                <option key={h} value={h}>
                  {h} tokens/parameter
                </option>
              ))}
            </select>
          </label>
          {(
            [
              ['lr', 'Learning rate'],
              ['beta1', 'Beta 1'],
              ['beta2', 'Beta 2'],
            ] as const
          ).map(([key, label]) => (
            <label key={key}>
              {label}
              <select
                aria-label={label}
                value={state[key]}
                onChange={(e) => setState((s) => ({ ...s, [key]: e.target.value, selected: '' }))}
              >
                <option value="all">All</option>
                {[...new Set(scoped.map((g) => g[key]))]
                  .sort((a, b) => a - b)
                  .map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
              </select>
            </label>
          ))}
        </div>
        {!stage ? (
          <p className="note" role="status">
            {state.method === 'awc4' && state.schedule === 'wsd'
              ? 'Four-worker WSD is unavailable. Select cosine-to-zero for the measured four-worker results.'
              : 'Cosine-to-zero is unavailable at this horizon. Select WSD for the longer training budgets.'}
          </p>
        ) : (
          <p className="muted">
            {schedules[state.schedule]} · {methods[state.method]} · {stage.tokens.toLocaleString()}{' '}
            global tokens. {stage.groups} measured configurations, {stage.runs} seed results.
            {state.schedule === 'wsd' &&
              Number(state.horizon) > 20 &&
              ' Continued horizons share parent training history.'}
          </p>
        )}
        {campaign && stage && (
          <details className="coverage">
            <summary>Grid coverage</summary>
            <p>
              β₁: {campaign.grid.beta1.join(', ')}. β₂: {campaign.grid.beta2.join(', ')}.
            </p>
            <div className="table-wrap">
              <table aria-label="Current grid coverage">
                <thead>
                  <tr>
                    <th>Learning rate</th>
                    <th>Status at this horizon</th>
                  </tr>
                </thead>
                <tbody>
                  {campaign.grid.lr.map((lr) => (
                    <tr key={lr}>
                      <td>{lr}</td>
                      <td>{stage.eligibleLRs.includes(lr) ? 'Measured' : 'Pruned'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
        {scoped.length > 0 && (
          <>
            <div className="download-row">
              <button onClick={() => download(fileStem + '.csv', csv(ordered), 'text/csv')}>
                Download filtered CSV
              </button>
              <button
                onClick={() =>
                  download(fileStem + '.json', JSON.stringify(ordered, null, 2), 'application/json')
                }
              >
                Download filtered JSON
              </button>
              <a href={publicationAsset('configurations.csv')}>All configurations CSV</a>
            </div>
            <p className="small">
              {filtered.length} configurations. Ranking uses final full-validation loss; ± denotes
              sample SD across three seeds.
            </p>
            <div className="table-wrap ranking-scroll">
              <table aria-label="Current configuration rankings">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>LR</th>
                    <th>β₁</th>
                    <th>β₂</th>
                    <th aria-sort={descending ? 'descending' : 'ascending'}>
                      <button onClick={() => setDescending((v) => !v)}>
                        Mean loss {descending ? '↓' : '↑'}
                      </button>
                    </th>
                    <th>Sample SD</th>
                    <th>Compare</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {ordered.map((g) => (
                    <tr key={g.id} className={g.id === selected?.id ? 'selected' : ''}>
                      <td>{g.rank}</td>
                      <td>{g.lr}</td>
                      <td>{g.beta1}</td>
                      <td>{g.beta2}</td>
                      <td>{fmt(g.mean)}</td>
                      <td>{fmt(g.sd)}</td>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={'Compare current rank ' + g.rank}
                          checked={state.compare.includes(g.id)}
                          disabled={
                            !state.compare.includes(g.id) && state.compare.length >= MAX_COMPARE
                          }
                          onChange={() => toggleCompare(g.id)}
                        />
                        {curveIds.includes(g.id) && (
                          <span
                            className="dot"
                            style={{
                              background: palette[curveIds.indexOf(g.id) % palette.length],
                            }}
                          />
                        )}
                      </td>
                      <td>
                        <button
                          aria-label={'Inspect current rank ' + g.rank}
                          onClick={() => setState((s) => ({ ...s, selected: g.id }))}
                        >
                          Inspect ↗
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {filtered.length === 0 && <p role="status">No configurations match these filters.</p>}
            <h3 className="plot-heading">Learning-rate response</h3>
            <Plot
              data={responseTraces}
              label="Current learning-rate response"
              layout={{
                xaxis: { title: { text: 'Peak learning rate' } },
                yaxis: { title: { text: 'Final full-validation loss' } },
              }}
              onPoint={(id) => setState((s) => ({ ...s, selected: id }))}
            />
          </>
        )}
      </section>
      <section className="publication-section" aria-label="Trajectory comparison">
        <h2>Compare trajectories.</h2>
        <div className="comparison-tray" aria-label="Comparison tray">
          {curveIds.map((id, i) => {
            const local = state.localRuns[id];
            if (local)
              return (
                <span key={id} className="chip local-run" title={local.filename}>
                  <span
                    className="dot"
                    style={{ background: palette[i % palette.length] }}
                    aria-hidden="true"
                  />
                  <span className="small muted">Local</span>
                  <input
                    className="local-run-name"
                    aria-label={'Local run name: ' + local.filename}
                    value={local.label}
                    onChange={(event) => {
                      const label = event.target.value;
                      setState((s) => ({
                        ...s,
                        localRuns: { ...s.localRuns, [id]: { ...s.localRuns[id], label } },
                      }));
                    }}
                    onBlur={() =>
                      setState((s) => ({
                        ...s,
                        localRuns: {
                          ...s.localRuns,
                          [id]: {
                            ...s.localRuns[id],
                            label:
                              s.localRuns[id].label.trim() ||
                              localRunLabel(
                                local.filename,
                                Object.values(s.localRuns)
                                  .filter((run) => run.id !== id)
                                  .map((run) => run.label),
                              ),
                          },
                        },
                      }))
                    }
                  />
                  <button
                    className="chip-remove"
                    aria-label={
                      'Remove local run ' + (local.label || local.filename) + ' from comparison'
                    }
                    onClick={() => unpin(id)}
                  >
                    ×
                  </button>
                </span>
              );
            const g = lookup.get(id)!;
            const pinned = state.compare.includes(id);
            return (
              <span
                key={id}
                className={'chip' + (id === selected?.id ? ' inspecting' : '')}
                title={curveLabel(g)}
              >
                <span
                  className="dot"
                  style={{ background: palette[i % palette.length] }}
                  aria-hidden="true"
                />
                <button
                  className="chip-label"
                  aria-label={'Inspect ' + curveLabel(g)}
                  onClick={() => inspect(id)}
                >
                  {scopeLabel(g)} · LR {g.lr}
                </button>
                <button
                  className="chip-remove"
                  aria-label={
                    pinned
                      ? 'Remove ' + curveLabel(g) + ' from comparison'
                      : 'Pin ' + curveLabel(g) + ' to comparison'
                  }
                  onClick={() => toggleCompare(id)}
                >
                  {pinned ? '×' : '+'}
                </button>
              </span>
            );
          })}
        </div>
        <div className="download-row">
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".jsonl"
            aria-label="Local metrics files"
            hidden
            disabled={importing}
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              event.target.value = '';
              void importFiles(files);
            }}
          />
          <button disabled={importing} onClick={() => fileInput.current?.click()}>
            {importing ? 'Reading local runs…' : 'Add local runs'}
          </button>
          <button onClick={pinBest}>Pin best at this horizon</button>
          {(state.compare.length > 0 || importing) && (
            <button
              onClick={() => {
                importGeneration.current++;
                setImporting(false);
                setState((s) => ({ ...s, compare: [], localRuns: {}, importMessages: [] }));
                setNotice('');
              }}
            >
              Clear comparison
            </button>
          )}
          <button
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(location.href);
                setNotice(
                  'Comparison link copied.' +
                    (Object.keys(state.localRuns).length
                      ? ' Local runs are not included; select their files again to compare them.'
                      : ''),
                );
              } catch {
                setNotice(
                  'Could not copy the link. Copy the address from your browser; local runs must be imported again.',
                );
              }
            }}
          >
            Copy comparison link
          </button>
        </div>
        <p className="small muted">
          Select one or more metrics.jsonl files. Files stay in your browser and are available until
          refresh. Comparison links include published runs only; import local files again after
          opening a link. Up to {MAX_COMPARE} curves can be compared.
        </p>
        {state.importMessages.length > 0 && (
          <ul className="local-run-messages small" role="alert">
            {state.importMessages.map((message, index) => (
              <li key={index}>{message}</li>
            ))}
          </ul>
        )}
        <div className="local-run-messages small" aria-live="polite">
          {Object.values(state.localRuns).map((run) => {
            const unavailable = (
              Object.keys(localSeriesLabels) as (keyof typeof localSeriesLabels)[]
            )
              .filter((series) => !run[series].length)
              .map((series) => localSeriesLabels[series]);
            return unavailable.length || run.warnings.length ? (
              <div key={run.id}>
                {unavailable.length > 0 && (
                  <p>
                    {run.label || run.filename}: no {unavailable.join(' or ')} measurements.
                  </p>
                )}
                {run.warnings.map((warning, index) => (
                  <p key={index}>{warning}</p>
                ))}
              </div>
            ) : null;
          })}
        </div>
        <p className="small muted" aria-live="polite">
          {notice}
        </p>
        <div className="curve-controls">
          <label>
            Loss view
            <select
              aria-label="Loss view"
              value={state.curve}
              onChange={(e) => setState((s) => ({ ...s, curve: e.target.value as State['curve'] }))}
            >
              <option value="validation">Subset validation</option>
              <option value="train">Training</option>
            </select>
          </label>
          <label>
            <input
              type="checkbox"
              checked={state.seeds}
              onChange={(e) => setState((s) => ({ ...s, seeds: e.target.checked }))}
            />{' '}
            Show individual seeds
          </label>
          <label>
            <input
              type="checkbox"
              checked={state.focus}
              onChange={(e) => setState((s) => ({ ...s, focus: e.target.checked }))}
            />{' '}
            Focus on the final losses
          </label>
          <label>
            Y-axis scale
            <select
              aria-label="Y-axis scale"
              value={state.yScale}
              onChange={(e) =>
                setState((s) => ({ ...s, yScale: e.target.value as State['yScale'] }))
              }
            >
              <option value="linear">Linear</option>
              <option value="log">Log</option>
            </select>
          </label>
        </div>
        <p className="small">
          Colour and marker identify each curve, matching the tray above. Solid lines are
          cosine-to-zero and dashed lines are WSD; bands show sample SD across published seeds.
          Dotted lines show individual local runs, which are unchanged by the seed control. Curves
          show recorded training-window or subset-validation losses, joined without smoothing. Final
          full-validation results are the separate values below.
        </p>
        {missing.length > 0 && (
          <p role="alert">
            Recorded curves could not load
            {missing.length < curveIds.length
              ? ' for ' + missing.map((id) => curveLabel(lookup.get(id)!)).join('; ')
              : ''}
            . Seed results and retained log downloads remain available.
          </p>
        )}
        {curveTraces.length > 0 ? (
          <Plot
            data={curveTraces}
            label={
              state.curve === 'train'
                ? 'Current training trajectories'
                : 'Current validation trajectories'
            }
            layout={curveLayout}
          />
        ) : loading ? (
          <p className="loading" aria-live="polite">
            Loading recorded curves…
          </p>
        ) : (
          <p className="small muted">
            {curveIds.length
              ? `No ${localSeriesLabels[state.curve]} measurements available in this comparison.`
              : 'Add local runs or select a published configuration to compare trajectories.'}
          </p>
        )}
        {gradientTraces.length > 0 && (
          <>
            <h3 className="plot-heading">Gradient norms</h3>
            <div className="curve-controls">
              <label>
                Gradient-norm y-axis scale
                <select
                  aria-label="Gradient-norm y-axis scale"
                  value={state.gradScale}
                  onChange={(e) =>
                    setState((s) => ({ ...s, gradScale: e.target.value as State['gradScale'] }))
                  }
                >
                  <option value="linear">Linear</option>
                  <option value="log">Log</option>
                </select>
              </label>
            </div>
            <p className="small">
              Gradient norm before clipping, recorded at the same training steps as the loss above,
              so the curve identities match. Synchronous runs record the model&rsquo;s total norm;
              decentralized runs record the largest of their per-worker norms. The seed and focus
              controls apply here too; the loss view does not, because the norm is logged only on
              training steps.
            </p>
            <Plot
              data={gradientTraces}
              label="Current gradient-norm trajectories"
              layout={gradientLayout}
            />
          </>
        )}
      </section>
      {selected && (
        <section className="publication-section" aria-label="Selected configuration">
          <p className="eyebrow">
            {groupLabel(selected)} · {selected.horizon} tokens/parameter
          </p>
          <h2>Inside one configuration.</h2>
          <p>
            LR {selected.lr} · β₁ {selected.beta1} · β₂ {selected.beta2} · Final loss{' '}
            <strong>
              {fmt(selected.mean)} ± {fmt(selected.sd)}
            </strong>
          </p>
          <div className="table-wrap">
            <table aria-label="Current seed results">
              <thead>
                <tr>
                  <th>Seed</th>
                  <th>Final full-validation loss</th>
                  <th>Retained evidence</th>
                </tr>
              </thead>
              <tbody>
                {selected.runs.map((r) => (
                  <tr key={r.id}>
                    <td>{r.seed}</td>
                    <td>{fmt(r.loss)}</td>
                    <td>
                      <a
                        href={evidenceAsset(r.artifacts['metrics.jsonl'])}
                        title={r.artifacts['metrics.jsonl']}
                      >
                        Metrics log
                      </a>{' '}
                      ·{' '}
                      <a
                        href={evidenceAsset(r.artifacts['result.json'])}
                        title={r.artifacts['result.json']}
                      >
                        Run records
                      </a>{' '}
                      ·{' '}
                      <a
                        href={evidenceAsset(r.artifacts['environment.json'])}
                        title={r.artifacts['environment.json']}
                      >
                        Environment
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="download-row">
            <button onClick={() => toggleCompare(selected.id)}>
              {state.compare.includes(selected.id)
                ? 'Remove from comparison'
                : 'Pin this configuration'}
            </button>
          </div>
          {curves[selected.id] && (
            <details>
              <summary>Trajectory sources</summary>
              <p>
                Each retained log is stored once, inside the container for its campaign stage.
                Continued WSD curves use the parent only through the saved cursor, then append the
                child observations. Earlier terminal decays are excluded.
              </p>
              <ul>
                {curves[selected.id].runs.map((r) => (
                  <li key={r.seed}>
                    Seed {r.seed}:{' '}
                    {r.sources.map((src, i) => (
                      <span key={src.path}>
                        {i > 0 && ' → '}
                        <a href={evidenceAsset(src.path)} title={src.path}>
                          {src.startTokens.toLocaleString()}–{src.endTokens.toLocaleString()} tokens
                        </a>
                      </span>
                    ))}
                  </li>
                ))}
              </ul>
            </details>
          )}
          <h3 className="plot-heading">Matched comparisons for this configuration</h3>
          <p className="small">
            Same horizon, LR, betas, and seeds. Differences are right minus left; paired SD
            describes per-seed differences. Schedule comparisons still have different training
            histories and trainer versions.
          </p>
          {matches.length ? (
            <div className="table-wrap">
              <table aria-label="Current matched comparisons">
                <thead>
                  <tr>
                    <th>Left</th>
                    <th>Right</th>
                    <th>Difference</th>
                    <th>Paired SD</th>
                  </tr>
                </thead>
                <tbody>
                  {matches.map((m) => (
                    <tr key={m.left + m.right}>
                      <td>
                        <button onClick={() => inspect(m.left)}>
                          {groupLabel(lookup.get(m.left)!)}
                        </button>
                      </td>
                      <td>
                        <button onClick={() => inspect(m.right)}>
                          {groupLabel(lookup.get(m.right)!)}
                        </button>
                      </td>
                      <td>{signed(m.difference)}</td>
                      <td>{fmt(m.pairedSD)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p>No matching measurements are available for this hyperparameter combination.</p>
          )}
        </section>
      )}
    </div>
  );
}
