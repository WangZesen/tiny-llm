import { useEffect, useMemo, useRef, useState } from 'react';
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

type State = {
  schedule: Schedule;
  method: Method;
  horizon: string;
  lr: string;
  beta1: string;
  beta2: string;
  selected: string;
  /** Pinned configurations, insertion-ordered. Deliberately independent of the scope
   *  filters: the comparisons worth making cross schedules, worker counts and horizons. */
  compare: string[];
  curve: 'train' | 'validation';
  seeds: boolean;
  focus: boolean;
  yScale: 'linear' | 'log';
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
  curve: 'validation',
  seeds: false,
  focus: false,
  yScale: 'linear',
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
      setState(next);
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
    if (state.compare.length) params.set('compare', state.compare.join(','));
    if (state.curve === 'train') params.set('curve', 'train');
    if (state.seeds) params.set('seeds', '1');
    if (state.focus) params.set('focus', '1');
    if (state.yScale === 'log') params.set('yScale', 'log');
    // A comma is a legal sub-delimiter and keeps eight pinned ids readable in the bar.
    const query = params.toString().replace(/%2C/g, ',');
    history.replaceState(null, '', location.pathname + '?' + query);
  }, [state, ready]);
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
  const curveKey = curveIds.join(',');
  const known = useRef<Record<string, RecordedCurve>>({});
  useEffect(() => {
    const controller = new AbortController();
    const wanted = curveIds.filter((id) => !known.current[id]);
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
    setState((s) => ({ ...s, compare: s.compare.filter((c) => c !== id) }));
  }
  const toggleCompare = (id: string) => (state.compare.includes(id) ? unpin(id) : pin(id));
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
  const curveTraces = useMemo<Data[]>(
    () =>
      curveIds.flatMap((id, i) => {
        const recorded = curves[id];
        const g = lookup.get(id);
        if (!recorded || !g) return [];
        const color = palette[i % palette.length];
        // Redundant with colour, because eight hues cannot separate for every reader.
        const dash = g.schedule === 'wsd' ? 'dash' : 'solid';
        const name = curveLabel(g);
        if (state.seeds)
          return recorded.runs.map((r, j): Data => ({
            type: 'scatter',
            mode: 'lines',
            x: r[state.curve].map((pt) => pt[0]),
            y: r[state.curve].map((pt) => pt[1]),
            line: { color, width: 1, dash: (['solid', 'dot', 'dashdot'] as const)[j] },
            opacity: 0.7,
            legendgroup: id,
            showlegend: curveIds.length === 1 || j === 0,
            name: curveIds.length === 1 ? 'Seed ' + r.seed : name,
            hovertemplate: '%{x:,} tokens<br>Loss %{y:.6f}<extra>%{fullData.name}</extra>',
          }));
        const points = aggregate(recorded.runs, state.curve);
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
            hovertemplate:
              '%{x:,} tokens<br>Loss %{y:.6f}<br>%{customdata} seeds<extra>%{fullData.name}</extra>',
          },
        ] as Data[];
      }),
    [curveIds, curves, state.curve, state.seeds, lookup, palette],
  );
  const focusRange = useMemo(() => {
    const plotted = curveIds
      .map((id) => curves[id])
      .filter(Boolean)
      .map((recorded) => aggregate(recorded.runs, state.curve));
    const end = Math.max(0, ...plotted.map((points) => points.at(-1)?.tokens ?? 0));
    const from = end / 2;
    const values = plotted
      .flatMap((points) => points.filter((pt) => pt.tokens >= from))
      .flatMap((pt) => [pt.mean - pt.sd, pt.mean + pt.sd]);
    if (!values.length) return null;
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = (high - low || 0.01) * 0.15;
    return { x: [from, end * 1.01], y: [low - pad, high + pad] };
  }, [curveIds, curves, state.curve]);
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
  const missing = curveIds.filter((id) => failed.includes(id));
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
              setState((s) => ({ ...defaults, compare: s.compare }));
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
      {curveIds.length > 0 && (
        <section className="publication-section" aria-label="Trajectory comparison">
          <h2>Compare trajectories.</h2>
          <div className="comparison-tray" aria-label="Comparison tray">
            {curveIds.map((id, i) => {
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
            <button onClick={pinBest}>Pin best at this horizon</button>
            {state.compare.length > 0 && (
              <button
                onClick={() => {
                  setState((s) => ({ ...s, compare: [] }));
                  setNotice('');
                }}
              >
                Clear comparison
              </button>
            )}
            <button
              onClick={() => {
                void navigator.clipboard?.writeText(location.href);
                setNotice('Comparison link copied.');
              }}
            >
              Copy comparison link
            </button>
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
                onChange={(e) =>
                  setState((s) => ({ ...s, curve: e.target.value as State['curve'] }))
                }
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
            cosine-to-zero and dashed lines are WSD; bands show sample SD across three seeds. Curves
            show recorded training-window or subset-validation losses, joined without smoothing.
            Final full-validation results are the separate values below.
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
          ) : (
            missing.length < curveIds.length && (
              <p className="loading" aria-live="polite">
                Loading recorded curves…
              </p>
            )
          )}
        </section>
      )}
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
