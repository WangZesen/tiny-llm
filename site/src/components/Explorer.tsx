import { useEffect, useMemo, useState } from 'react';
import type { Data } from 'plotly.js';
import type { Configuration, Curves, Study } from '../lib/types';
import { aggregate } from '../lib/statistics';
import Plot from './Plot';

const asset = (p: string) => `${import.meta.env.BASE_URL.replace(/\/$/, '')}/assets/${p}`;
const fmt = (x: number) => x.toFixed(6);
const num = (x: number) => x.toLocaleString('en-US');
const label = (g: Configuration) => `LR ${g.lr} · β₁ ${g.beta1} · β₂ ${g.beta2}`;
const unique = (xs: number[]) => [...new Set(xs)].sort((a, b) => a - b);
type State = {
  study: string;
  lr: string;
  beta1: string;
  beta2: string;
  selected: string;
  compare: string[];
  kind: 'validation' | 'train';
  seeds: boolean;
  heatLR: string;
};
const initial: State = {
  study: 'all',
  lr: 'all',
  beta1: 'all',
  beta2: 'all',
  selected: '',
  compare: [],
  kind: 'validation',
  seeds: false,
  heatLR: '',
};
function saveDownload(name: string, text: string, type: string) {
  const href = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = href;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

export default function Explorer({ studies }: { studies: Study[] }) {
  const [state, setState] = useState<State>(initial);
  const [ready, setReady] = useState(false);
  const [sort, setSort] = useState<{
    key: 'mean' | 'sd' | 'lr' | 'beta1' | 'beta2';
    desc: boolean;
  }>({ key: 'mean', desc: false });
  const [curveData, setCurveData] = useState<Record<string, Curves>>({});
  const [curveError, setCurveError] = useState('');
  const [notice, setNotice] = useState('');
  const all = useMemo(() => studies.flatMap((s) => s.groups), [studies]);
  const lookup = useMemo(() => new Map(all.map((g) => [g.id, g])), [all]);
  useEffect(() => {
    const read = () => {
      const p = new URLSearchParams(location.search);
      const study = studies.find((s) => s.id === p.get('study'));
      const next = { ...initial, study: study?.id ?? 'all' };
      for (const key of ['lr', 'beta1', 'beta2'] as const) {
        const value = p.get(key);
        if (value && study?.groups.some((g) => String(g[key]) === value)) next[key] = value;
      }
      const selected = p.get('selected') ?? '';
      if (lookup.get(selected)?.study === next.study) next.selected = selected;
      next.compare = [...new Set((p.get('compare') ?? '').split(','))]
        .filter((id) => lookup.has(id))
        .slice(0, 4);
      next.kind = p.get('curve') === 'train' ? 'train' : 'validation';
      next.seeds = p.get('seeds') === '1';
      const heat = p.get('heatLR');
      if (heat && study?.groups.some((g) => String(g.lr) === heat)) next.heatLR = heat;
      setState(next);
      setReady(true);
    };
    read();
    window.addEventListener('popstate', read);
    return () => window.removeEventListener('popstate', read);
  }, [studies, lookup]);
  useEffect(() => {
    if (!ready) return;
    const p = new URLSearchParams();
    if (state.study !== 'all') p.set('study', state.study);
    for (const key of ['lr', 'beta1', 'beta2'] as const)
      if (state[key] !== 'all') p.set(key, state[key]);
    if (state.selected) p.set('selected', state.selected);
    if (state.compare.length) p.set('compare', state.compare.join(','));
    if (state.kind === 'train') p.set('curve', 'train');
    if (state.seeds) p.set('seeds', '1');
    if (state.heatLR) p.set('heatLR', state.heatLR);
    history.replaceState(null, '', location.pathname + (p.size ? '?' + p.toString() : ''));
  }, [state, ready]);
  const study = studies.find((s) => s.id === state.study);
  const selected = lookup.get(state.selected);
  const filtered = useMemo(
    () =>
      (study?.groups ?? [])
        .filter((g) =>
          (['lr', 'beta1', 'beta2'] as const).every(
            (k) => state[k] === 'all' || Number(state[k]) === g[k],
          ),
        )
        .sort((a, b) => (a[sort.key] - b[sort.key]) * (sort.desc ? -1 : 1)),
    [study, state.lr, state.beta1, state.beta2, sort],
  );
  const pickStudy = (s?: Study) =>
    setState((prev) => ({
      ...initial,
      compare: prev.compare,
      kind: prev.kind,
      seeds: prev.seeds,
      study: s?.id ?? 'all',
      selected: s?.groups[0].id ?? '',
      beta1: s ? String(s.groups[0].beta1) : 'all',
      heatLR: s ? String(s.groups[0].lr) : '',
    }));
  const pick = (id: string) => {
    const g = lookup.get(id);
    if (!g) return;
    setState((prev) => ({ ...prev, selected: id, study: g.study, heatLR: String(g.lr) }));
  };
  const toggleCompare = (id: string) => {
    setNotice('');
    setState((prev) => {
      if (prev.compare.includes(id))
        return { ...prev, compare: prev.compare.filter((g) => g !== id) };
      if (prev.compare.length >= 4) {
        setNotice('Compare up to four configurations. Remove one to add another.');
        return prev;
      }
      return { ...prev, compare: [...prev.compare, id] };
    });
  };
  const curveIds = state.compare.length ? state.compare : selected ? [selected.id] : [];
  const curveKey = curveIds.join(',');
  useEffect(() => {
    let active = true;
    setCurveError('');
    Promise.all(
      curveIds.map(async (id) => {
        const response = await fetch(asset(`data/curves/${id}.json`));
        if (!response.ok)
          throw new Error('Curve data could not load. Try again or download the source data.');
        return [id, await response.json()] as [string, Curves];
      }),
    )
      .then((entries) => {
        if (active) setCurveData(Object.fromEntries(entries));
      })
      .catch((e) => {
        if (active) setCurveError(String(e.message));
      });
    return () => {
      active = false;
    };
  }, [curveKey]);
  const responseData = useMemo(() => {
    if (!study) return [];
    const groups = study.groups.filter(
      (g) =>
        (state.beta1 === 'all' || g.beta1 === Number(state.beta1)) &&
        (state.beta2 === 'all' || g.beta2 === Number(state.beta2)),
    );
    const slices = [...new Set(groups.map((g) => `${g.beta1}/${g.beta2}`))];
    const colors = ['#2867b2', '#127d72', '#8754ad', '#bb6330', '#b64965', '#70792f'];
    return slices.map((slice, i): Data => {
      const rows = groups
        .filter((g) => `${g.beta1}/${g.beta2}` === slice)
        .sort((a, b) => a.lr - b.lr);
      return {
        type: 'scatter',
        mode: 'lines+markers',
        x: rows.map((g) => g.lr),
        y: rows.map((g) => g.mean),
        customdata: rows.map((g) => g.id),
        error_y: {
          type: 'data',
          array: rows.map((g) => g.sd),
          visible: true,
          thickness: 1,
          width: 3,
        },
        name: `β₁ ${rows[0].beta1} · β₂ ${rows[0].beta2}`,
        line: { color: colors[i % colors.length], width: 1.7 },
        marker: { size: 6 },
        hovertemplate: 'LR %{x}<br>Mean %{y:.6f} nats<extra>%{fullData.name}</extra>',
      };
    });
  }, [study, state.beta1, state.beta2]);
  const heatLR = Number(state.lr !== 'all' ? state.lr : state.heatLR || study?.groups[0].lr);
  const heatData = useMemo((): Data[] => {
    if (!study) return [];
    const b1 = unique(study.groups.map((g) => g.beta1)),
      b2 = unique(study.groups.map((g) => g.beta2));
    const find = (a: number, b: number) =>
      study.groups.find((g) => g.lr === heatLR && g.beta1 === a && g.beta2 === b);
    const gap: { x: string; y: string }[] = [];
    for (const a of b1)
      for (const b of b2) if (!find(a, b)) gap.push({ x: String(b), y: String(a) });
    return [
      {
        type: 'heatmap',
        x: b2.map(String),
        y: b1.map(String),
        z: b1.map((a) => b2.map((b) => find(a, b)?.mean ?? null)),
        customdata: b1.map((a) => b2.map((b) => find(a, b)?.id ?? '')),
        text: b1.map((a) =>
          b2.map((b) => (find(a, b) ? fmt(find(a, b)!.mean) : 'Not measured')),
        ) as any,
        texttemplate: '%{text}',
        textfont: { size: 10 },
        colorscale: [
          [0, '#e7f2c4'],
          [0.5, '#55a888'],
          [1, '#174d4c'],
        ],
        colorbar: { title: { text: 'Nats' }, thickness: 10 },
        xgap: 3,
        ygap: 3,
        hoverongaps: false,
        hovertemplate: 'β₁ %{y} · β₂ %{x}<br>Mean %{z:.6f} nats<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'text',
        x: gap.map((g) => g.x),
        y: gap.map((g) => g.y),
        text: gap.map(() => 'Not measured'),
        textfont: { size: 9, color: '#778678' },
        hoverinfo: 'skip',
        showlegend: false,
      },
    ];
  }, [study, heatLR]);
  const curvePlot = useMemo(
    (): Data[] =>
      curveIds.flatMap((id, index) => {
        const curves = curveData[id];
        if (!curves?.runs.length) return [];
        const g = lookup.get(id)!;
        const s = studies.find((s) => s.id === g.study)!;
        const rows = aggregate(curves.runs, state.kind);
        const color = s.color;
        const name = `${s.title} · LR ${g.lr} · β ${g.beta1}/${g.beta2}`;
        const x = rows.map((r) => r.tokens);
        const traces: Data[] = [
          {
            type: 'scatter',
            x,
            y: rows.map((r) => r.mean - r.sd),
            mode: 'lines',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            legendgroup: id,
          },
          {
            type: 'scatter',
            x,
            y: rows.map((r) => r.mean + r.sd),
            mode: 'lines',
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: color + '22',
            showlegend: false,
            hoverinfo: 'skip',
            legendgroup: id,
          },
          {
            type: 'scatter',
            x,
            y: rows.map((r) => r.mean),
            customdata: rows.map((r) => r.n),
            name,
            legendgroup: id,
            mode: 'lines',
            line: { color, width: 2, dash: (['solid', 'dash', 'dot', 'dashdot'] as const)[index] },
            hovertemplate:
              '%{x:,} global tokens<br>Loss %{y:.6f} nats<br>Seeds: %{customdata}<extra>%{fullData.name}</extra>',
          },
        ];
        if (state.seeds)
          for (const run of curves.runs)
            traces.push({
              type: 'scatter',
              x: run[state.kind].map((p) => p[0]),
              y: run[state.kind].map((p) => p[1]),
              mode: 'lines',
              line: { color, width: 1, dash: 'dot' },
              opacity: 0.55,
              name: `${s.title} seed ${run.seed}`,
              legendgroup: id,
              hovertemplate: '%{x:,} global tokens<br>Loss %{y:.6f}<extra>%{fullData.name}</extra>',
            });
        return traces;
      }),
    [curveKey, curveData, state.kind, state.seeds, lookup, studies],
  );
  const sortBy = (key: typeof sort.key) =>
    setSort((prev) => ({ key, desc: prev.key === key ? !prev.desc : false }));
  const download = (format: 'csv' | 'json') => {
    const rows = filtered.map((g) => ({
      study: g.study,
      rank: g.rank,
      lr: g.lr,
      beta1: g.beta1,
      beta2: g.beta2,
      mean_loss: g.mean,
      sample_sd: g.sd,
      seeds: g.runs.length,
      available_curve_seeds: g.curveSeeds.length,
    }));
    if (format === 'json')
      saveDownload(
        `${state.study}-filtered.json`,
        JSON.stringify(rows, null, 2),
        'application/json',
      );
    else
      saveDownload(
        `${state.study}-filtered.csv`,
        'study,rank,lr,beta1,beta2,mean_loss,sample_sd,seeds,available_curve_seeds\n' +
          rows.map((r) => Object.values(r).join(',')).join('\n') +
          '\n',
        'text/csv',
      );
  };
  if (!ready) return <p className="loading">Loading the results explorer…</p>;
  return (
    <>
      <div className="study-tabs" aria-label="Choose a study">
        <button aria-pressed={!study} onClick={() => pickStudy()}>
          Overview
        </button>
        {studies.map((s) => (
          <button key={s.id} aria-pressed={s.id === state.study} onClick={() => pickStudy(s)}>
            <span className="dot" style={{ background: s.color }} />
            {s.title}
          </button>
        ))}
      </div>
      {!study ? (
        <>
          <div className="cards">
            {studies.map((s) => (
              <button
                key={s.id}
                className="card"
                onClick={() => pickStudy(s)}
                style={{ textAlign: 'left' }}
              >
                <span className="number">
                  {s.runs} RUNS / {s.configurations} CONFIGURATIONS
                </span>
                <h3>{s.title}</h3>
                <p>
                  {s.label}
                  <br />
                  Best tested:{' '}
                  <strong>
                    {fmt(s.groups[0].mean)} ± {fmt(s.groups[0].sd)}
                  </strong>{' '}
                  nats
                </p>
                <span className="arrow">Explore search space →</span>
              </button>
            ))}
          </div>
          <p className="note">
            Each recipe was selected separately by mean final full-validation loss across three
            seeds. These winners compare tuned systems; they do not isolate the effect of a single
            algorithmic choice.
          </p>
        </>
      ) : (
        <>
          <div className="section-heading">
            <div>
              <h2>{study.title}</h2>
              <p className="small muted">
                {study.runs} runs · {study.configurations} configurations · {num(study.batch)}{' '}
                targets/update · {study.date}
              </p>
            </div>
            <a href={asset(`data/${study.directory}/results.json`)}>Source & provenance ↗</a>
          </div>
          <div className="filter-bar">
            {(['lr', 'beta1', 'beta2'] as const).map((key, index) => (
              <label key={key}>
                {['Learning rate', 'Beta 1', 'Beta 2'][index]}
                <select
                  aria-label={['Learning rate', 'Beta 1', 'Beta 2'][index]}
                  value={state[key]}
                  onChange={(e) =>
                    setState((prev) => ({ ...prev, [key]: e.target.value, selected: '' }))
                  }
                >
                  <option value="all">All measured</option>
                  {unique(study.groups.map((g) => g[key])).map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            <button
              onClick={() =>
                setState((prev) => ({ ...prev, lr: 'all', beta1: 'all', beta2: 'all' }))
              }
            >
              Reset filters
            </button>
            <span className="filter-count" aria-live="polite">
              {filtered.length} of {study.configurations} configurations
            </span>
          </div>
          <div className="chart-grid">
            <section className="chart-panel">
              <h3>Learning-rate response</h3>
              <p>
                Beta filters select lines. All measured learning rates remain visible. Error bars:
                sample SD.
              </p>
              <Plot
                data={responseData}
                label="Learning-rate response"
                layout={{
                  xaxis: { title: { text: 'Learning rate' }, type: 'log' },
                  yaxis: { title: { text: 'Final full-validation loss (nats)' } },
                }}
                onPoint={pick}
              />
            </section>
            <section className="chart-panel">
              <h3>Beta response surface</h3>
              <div className="toolbar">
                <label>
                  Heatmap learning rate{' '}
                  <select
                    aria-label="Heatmap learning rate"
                    value={heatLR}
                    onChange={(e) =>
                      setState((prev) => ({ ...prev, heatLR: e.target.value, lr: 'all' }))
                    }
                  >
                    {unique(study.groups.map((g) => g.lr)).map((lr) => (
                      <option value={lr} key={lr}>
                        {lr}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <p>Complete measured beta grid at this LR. Click a cell to inspect its recipe.</p>
              <Plot
                data={heatData}
                label="Beta response heatmap"
                layout={{
                  xaxis: { title: { text: 'Beta 2' }, type: 'category' },
                  yaxis: { title: { text: 'Beta 1' }, type: 'category' },
                }}
                onPoint={(id) => {
                  const g = lookup.get(id)!;
                  setState((prev) => ({
                    ...prev,
                    selected: id,
                    lr: String(g.lr),
                    beta1: String(g.beta1),
                    beta2: 'all',
                  }));
                }}
              />
            </section>
          </div>
          <div className="section-heading">
            <h2>Measured configurations</h2>
            <span className="small muted">Lower loss is better</span>
          </div>
          <div className="toolbar">
            <button onClick={() => download('csv')}>Download filtered CSV</button>
            <button onClick={() => download('json')}>Download filtered JSON</button>
            <a href={asset(`data/${study.directory}/runs.csv`)}>All seed records ↗</a>
            {study.preset && <a href={asset(`configs/${study.preset}`)}>Selected preset ↗</a>}
          </div>
          <div className="table-wrap">
            <table aria-label="Configuration rankings">
              <thead>
                <tr>
                  <th>Compare</th>
                  <th>Rank</th>
                  {(['lr', 'beta1', 'beta2', 'mean', 'sd'] as const).map((k, i) => (
                    <th
                      key={k}
                      aria-sort={sort.key === k ? (sort.desc ? 'descending' : 'ascending') : 'none'}
                    >
                      <button className="sort-button" onClick={() => sortBy(k)}>
                        {['LR', 'Beta 1', 'Beta 2', 'Mean loss', 'Sample SD'][i]}{' '}
                        {sort.key === k ? (sort.desc ? '↓' : '↑') : '↕'}
                      </button>
                    </th>
                  ))}
                  <th>Curves</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((g) => (
                  <tr key={g.id} className={selected?.id === g.id ? 'selected' : ''}>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`Compare ${label(g)}`}
                        checked={state.compare.includes(g.id)}
                        disabled={!state.compare.includes(g.id) && state.compare.length >= 4}
                        onChange={() => toggleCompare(g.id)}
                      />
                    </td>
                    <td>
                      <button
                        className="rank-button"
                        onClick={() => pick(g.id)}
                        aria-label={`Inspect rank ${g.rank}`}
                      >
                        {g.rank} ↗
                      </button>
                    </td>
                    <td>{g.lr}</td>
                    <td>{g.beta1}</td>
                    <td>{g.beta2}</td>
                    <td>{fmt(g.mean)}</td>
                    <td>{fmt(g.sd)}</td>
                    <td>
                      {g.curveSeeds.length ? `${g.curveSeeds.length}/3 seeds` : 'Unavailable'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!filtered.length && (
              <p className="empty">
                No measured configurations match these filters. Reset filters to see the measured
                grid.
              </p>
            )}
          </div>
          {selected && (
            <section className="note" aria-label="Selected configuration">
              <h3>{label(selected)}</h3>
              <div className="detail">
                <div>
                  <span className="number">{fmt(selected.mean)}</span> ± {fmt(selected.sd)} nats
                  <p>
                    Final full-validation mean ± sample SD.
                    <br />
                    Curves: {selected.curveSeeds.length}/3 seeds available.
                  </p>
                  <button onClick={() => toggleCompare(selected.id)}>
                    {state.compare.includes(selected.id)
                      ? 'Remove from comparison'
                      : 'Add to comparison'}
                  </button>
                </div>
                <table aria-label="Selected seed results">
                  <thead>
                    <tr>
                      <th>Runtime seed</th>
                      <th>Full-validation loss</th>
                      <th>Curve</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selected.runs.map((r) => (
                      <tr key={r.seed}>
                        <td>{r.seed}</td>
                        <td>{r.loss.toFixed(9)}</td>
                        <td>
                          {selected.curveSeeds.includes(r.seed) ? 'Available' : 'Unavailable'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
      {curveIds.length > 0 && (
        <section>
          <div className="section-heading">
            <h2>Training trajectories</h2>
            <span className="small muted">Up to four configurations</span>
          </div>
          <div className="comparison-tray">
            {state.compare.map((id) => {
              const g = lookup.get(id)!;
              return (
                <button
                  className="chip"
                  key={id}
                  onClick={() => toggleCompare(id)}
                  aria-label={`Remove ${id} from comparison`}
                >
                  {studies.find((s) => s.id === g.study)!.title} · {label(g)} ×
                </button>
              );
            })}
            {state.compare.length > 0 && (
              <button onClick={() => setState((prev) => ({ ...prev, compare: [] }))}>
                Clear comparison
              </button>
            )}
          </div>
          <p role="status" className="small muted">
            {notice}
          </p>
          <div className="chart-panel">
            <div className="toolbar">
              <label>
                Loss view{' '}
                <select
                  aria-label="Loss view"
                  value={state.kind}
                  onChange={(e) =>
                    setState((prev) => ({ ...prev, kind: e.target.value as State['kind'] }))
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
                  onChange={(e) => setState((prev) => ({ ...prev, seeds: e.target.checked }))}
                />{' '}
                Show individual seeds
              </label>
              <button
                onClick={() =>
                  saveDownload(
                    'selected-curves.json',
                    JSON.stringify(curveIds.map((id) => curveData[id]).filter(Boolean), null, 2),
                    'application/json',
                  )
                }
              >
                Download selected curves
              </button>
              <button
                onClick={() => {
                  navigator.clipboard
                    ?.writeText(location.href)
                    .then(() => setNotice('Comparison link copied.'))
                    .catch(() => setNotice('Copy the URL from your browser to share this view.'));
                }}
              >
                Copy view link
              </button>
            </div>
            <p>
              {state.kind === 'validation'
                ? 'Fixed epoch subset evaluated at averaged weights for packed methods. This is not final full-validation loss.'
                : 'Logged token-weighted training-window loss. Packed training measures local models; validation measures averaged weights.'}{' '}
              Lines show means; bands show sample SD. No smoothing or interpolation.
            </p>
            {curveError ? (
              <p role="alert">{curveError}</p>
            ) : curveIds.some((id) => !curveData[id]) ? (
              <p className="loading">Loading recorded trajectories…</p>
            ) : (
              <>
                {curveIds
                  .filter((id) => !curveData[id]?.runs.length)
                  .map((id) => (
                    <p key={id} className="note">
                      Curves unavailable for{' '}
                      {studies.find((s) => s.id === lookup.get(id)?.study)?.title}:{' '}
                      {label(lookup.get(id)!)}. Final results are retained.
                    </p>
                  ))}
                {curvePlot.length > 0 && (
                  <Plot
                    data={curvePlot}
                    label={
                      state.kind === 'validation'
                        ? 'Subset-validation trajectories'
                        : 'Training-loss trajectories'
                    }
                    layout={{
                      height: 430,
                      xaxis: { title: { text: 'Global training tokens' }, tickformat: '~s' },
                      yaxis: {
                        title: {
                          text: `${state.kind === 'validation' ? 'Subset-validation' : 'Training'} loss (nats)`,
                        },
                      },
                      margin: { b: 100, l: 60, r: 20, t: 20 },
                    }}
                  />
                )}
              </>
            )}
          </div>
        </section>
      )}
      {study && study.matches.length > 0 && (
        <section>
          <div className="section-heading">
            <h2>Matched comparisons</h2>
          </div>
          <p className="small muted">
            Same LR, betas, and runtime seeds. Difference = {study.title} minus clipped four-worker
            AWC; negative favors {study.title}. Matched references are the corresponding
            configurations in the four-worker AWC dataset; provenance retains their original
            sources.
          </p>
          <div className="table-wrap">
            <table aria-label="Matched comparisons">
              <thead>
                <tr>
                  <th>LR</th>
                  <th>Beta 1</th>
                  <th>Beta 2</th>
                  <th>Clipped AWC-4</th>
                  <th>{study.title}</th>
                  <th>Difference</th>
                </tr>
              </thead>
              <tbody>
                {study.matches
                  .filter(
                    (m) =>
                      (state.lr === 'all' || m.lr === +state.lr) &&
                      (state.beta1 === 'all' || m.beta1 === +state.beta1) &&
                      (state.beta2 === 'all' || m.beta2 === +state.beta2),
                  )
                  .map((m) => (
                    <tr key={`${m.lr}/${m.beta1}/${m.beta2}`}>
                      <td>{m.lr}</td>
                      <td>{m.beta1}</td>
                      <td>{m.beta2}</td>
                      <td>{fmt(m.left)}</td>
                      <td>{fmt(m.right)}</td>
                      <td>
                        {m.difference > 0 ? '+' : ''}
                        {fmt(m.difference)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
