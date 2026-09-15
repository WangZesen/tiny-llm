import { useEffect, useMemo, useState } from 'react';
import type { Data } from 'plotly.js';
import Plot from './Plot';
import { aggregate } from '../lib/statistics';
import { cellStatus, type WsdData, type WsdCurve } from '../lib/wsd';

type Controls = {
  method: string;
  horizon: string;
  lr: string;
  beta1: string;
  beta2: string;
  selected: string;
  kind: 'validation' | 'train';
  seeds: boolean;
};
const defaults: Controls = {
  method: 'awc8',
  horizon: '20',
  lr: 'all',
  beta1: 'all',
  beta2: 'all',
  selected: '',
  kind: 'validation',
  seeds: false,
};
const asset = (p: string) =>
  `${import.meta.env.BASE_URL.replace(/\/$/, '')}/assets/data/wsd-horizon-tuning/${p}`;
const fmt = (n: number) => n.toFixed(6);
function download(name: string, value: string, type: string) {
  const href = URL.createObjectURL(new Blob([value], { type }));
  const a = document.createElement('a');
  a.href = href;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}
export default function WsdExplorer({ data }: { data: WsdData }) {
  const [state, setState] = useState<Controls>(defaults);
  const [ready, setReady] = useState(false);
  const [sort, setSort] = useState<{
    key: 'mean' | 'sd' | 'lr' | 'beta1' | 'beta2';
    desc: boolean;
  }>({ key: 'mean', desc: false });
  const [curve, setCurve] = useState<WsdCurve | null>(null);
  const [curveError, setCurveError] = useState('');
  useEffect(() => {
    const read = () => {
      const p = new URLSearchParams(location.search);
      const next = { ...defaults };
      const c = data.campaigns.find((c) => c.id === p.get('method')) ?? data.campaigns[1];
      next.method = c.id;
      const s = c.stages.find((s) => String(s.horizon) === p.get('horizon')) ?? c.stages[0];
      next.horizon = String(s.horizon);
      for (const key of ['lr', 'beta1', 'beta2'] as const) {
        const value = p.get(key);
        if (value && s.groups.some((g) => String(g[key]) === value)) next[key] = value;
      }
      const id = p.get('selected');
      if (id && s.groups.some((g) => g.id === id)) next.selected = id;
      next.kind = p.get('curve') === 'train' ? 'train' : 'validation';
      next.seeds = p.get('seeds') === '1';
      setState(next);
      setReady(true);
    };
    read();
    window.addEventListener('popstate', read);
    return () => window.removeEventListener('popstate', read);
  }, [data]);
  useEffect(() => {
    if (!ready) return;
    const p = new URLSearchParams({ method: state.method, horizon: state.horizon });
    for (const key of ['lr', 'beta1', 'beta2', 'selected'] as const)
      if (state[key] && state[key] !== 'all') p.set(key, state[key]);
    if (state.kind === 'train') p.set('curve', 'train');
    if (state.seeds) p.set('seeds', '1');
    history.replaceState(null, '', location.pathname + '?' + p.toString());
  }, [state, ready]);
  const campaign = data.campaigns.find((c) => c.id === state.method)!;
  const stage = campaign.stages.find((s) => String(s.horizon) === state.horizon)!;
  const filtered = useMemo(
    () =>
      stage.groups.filter((g) =>
        (['lr', 'beta1', 'beta2'] as const).every(
          (k) => state[k] === 'all' || String(g[k]) === state[k],
        ),
      ),
    [stage, state],
  );
  const ordered = [...filtered].sort(
    (a, b) => (a[sort.key] - b[sort.key]) * (sort.desc ? -1 : 1) || a.rank - b.rank,
  );
  const selected = filtered.find((g) => g.id === state.selected) ?? filtered[0];
  useEffect(() => {
    setCurve(null);
    setCurveError('');
    if (!selected) return;
    const controller = new AbortController();
    fetch(asset('curves/' + selected.id + '.json'), { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error('unavailable');
        return r.json();
      })
      .then((value: WsdCurve) => {
        if (value.configuration !== selected.id) throw new Error('mismatch');
        setCurve(value);
      })
      .catch((e) => {
        if (e.name !== 'AbortError')
          setCurveError(
            'Recorded curves are unavailable. Final seed results and downloads remain available.',
          );
      });
    return () => controller.abort();
  }, [selected?.id]);
  const select = (id: string) =>
    setState((s) => ({ ...s, selected: id, lr: 'all', beta1: 'all', beta2: 'all' }));
  const scope = (key: 'method' | 'horizon', value: string) =>
    setState((s) => ({ ...s, [key]: value, selected: '', lr: 'all', beta1: 'all', beta2: 'all' }));
  const winnerTraces: Data[] = data.campaigns.map((c) => ({
    type: 'scatter',
    mode: 'lines+markers',
    name: c.title,
    x: c.stages.map((s) => s.horizon),
    y: c.stages.map((s) => s.groups[0].mean),
    line: { color: c.color },
    error_y: { type: 'data', array: c.stages.map((s) => s.groups[0].sd), visible: true },
    hovertemplate: 'Horizon %{x}<br>Mean %{y:.6f}<extra>%{fullData.name}</extra>',
  }));
  const gaps = data.campaigns[0].stages.map(
    (s, i) => data.campaigns[1].stages[i].groups[0].mean - s.groups[0].mean,
  );
  const pairs = campaign.grid.beta1.flatMap((a) => campaign.grid.beta2.map((b) => [a, b]));
  const colors = ['#2867b2', '#127d72', '#8754ad', '#bb6330', '#b64965', '#70792f'];
  const lrTraces: Data[] = pairs.map(([a, b], i) => {
    const gs = filtered.filter((g) => g.beta1 === a && g.beta2 === b).sort((x, y) => x.lr - y.lr);
    return {
      type: 'scatter',
      mode: 'lines+markers',
      name: `β₁ ${a} · β₂ ${b}`,
      x: gs.map((g) => g.lr),
      y: gs.map((g) => g.mean),
      customdata: gs.map((g) => g.id),
      line: { color: colors[i] },
      error_y: { type: 'data', array: gs.map((g) => g.sd), visible: true },
    };
  });
  const allLRs = [...new Set(data.campaigns.flatMap((c) => c.grid.lr))].sort((a, b) => a - b);
  const ys = pairs.map(([a, b]) => `${a} / ${b}`);
  const grid = pairs.map(([a, b]) =>
    allLRs.map((lr) => stage.groups.find((g) => g.lr === lr && g.beta1 === a && g.beta2 === b)),
  );
  const absent = grid.flatMap((row, i) =>
    row.flatMap((g, j) =>
      g
        ? []
        : [
            {
              x: String(allLRs[j]),
              y: ys[i],
              text: cellStatus(campaign, stage, allLRs[j], ...(pairs[i] as [number, number])),
            },
          ],
    ),
  );
  const heat: Data[] = [
    {
      type: 'heatmap',
      x: allLRs.map(String),
      y: ys,
      z: grid.map((row) => row.map((g) => g?.mean ?? null)),
      customdata: grid.map((row) => row.map((g) => g?.id ?? '')),
      colorscale: 'YlGnBu',
      reversescale: true,
      hoverongaps: false,
      colorbar: { title: { text: 'Mean loss' } },
      hovertemplate: 'LR %{x}<br>β₁ / β₂ %{y}<br>Mean %{z:.6f}<extra></extra>',
    },
    {
      type: 'scatter',
      mode: 'text',
      x: absent.map((p) => p.x),
      y: absent.map((p) => p.y),
      text: absent.map((p) =>
        p.text === 'Pruned' ? 'P' : p.text === 'Outside original grid' ? '—' : '?',
      ),
      hovertext: absent.map((p) => p.text),
      textfont: { size: 12 },
      hoverinfo: 'text',
      showlegend: false,
    },
  ];
  const matches = data.matches.filter(
    (m) =>
      m.horizon === stage.horizon &&
      (['lr', 'beta1', 'beta2'] as const).every(
        (k) => state[k] === 'all' || String(m[k]) === state[k],
      ),
  );
  const curveTraces: Data[] = [];
  if (curve) {
    if (state.seeds)
      for (const r of curve.runs)
        curveTraces.push({
          type: 'scatter',
          mode: 'lines',
          name: `Seed ${r.seed}`,
          x: r[state.kind].map((p) => p[0]),
          y: r[state.kind].map((p) => p[1]),
          line: { width: 1 },
        });
    else {
      const points = aggregate(curve.runs, state.kind);
      curveTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Mean ± sample SD',
        x: points.map((p) => p.tokens),
        y: points.map((p) => p.mean),
        error_y: {
          type: 'data',
          array: points.map((p) => p.sd),
          visible: true,
          thickness: 0.5,
          width: 0,
        },
        customdata: points.map((p) => p.n),
        hovertemplate: '%{x:,} tokens<br>Loss %{y:.6f}<br>%{customdata} seeds<extra></extra>',
        line: { color: campaign.color },
      });
    }
    curveTraces.push({
      type: 'scatter',
      mode: 'markers',
      name: 'Final full-validation (each seed)',
      x: curve.runs.map(() => stage.tokens),
      y: curve.runs.map((r) => r.final),
      marker: { symbol: 'diamond', size: 9, color: '#bd5568' },
    });
  }
  const exportRows = filtered.map(({ runs, ...g }) => ({
    ...g,
    lossSeed42: runs[0].loss,
    lossSeed43: runs[1].loss,
    lossSeed44: runs[2].loss,
  }));
  const exportCSV = () => {
    const keys = Object.keys(exportRows[0] ?? {});
    download(
      `${state.method}-wsd-h${state.horizon}-filtered.csv`,
      [
        keys.join(','),
        ...exportRows.map((r) => keys.map((k) => JSON.stringify(r[k as keyof typeof r])).join(',')),
      ].join('\n'),
      'text/csv',
    );
  };
  const sortBy = (key: typeof sort.key) =>
    setSort((s) => ({ key, desc: s.key === key ? !s.desc : false }));
  return (
    <div className="wsd-explorer">
      <div className="wsd-charts">
        <section>
          <h2>Winners across horizons</h2>
          <p>Separately tuned configurations; bars show sample SD over three seeds.</p>
          <Plot
            data={winnerTraces}
            label="WSD winning loss by horizon"
            layout={{
              xaxis: {
                title: { text: 'Global tokens per parameter' },
                tickvals: data.protocol.horizons,
              },
              yaxis: { title: { text: 'Full-validation loss (nats/token)' } },
            }}
          />
        </section>
        <section>
          <h2>Selected mean gap</h2>
          <p>Positive values favor synchronous training. Winner settings may differ.</p>
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines+markers',
                x: data.protocol.horizons,
                y: gaps,
                name: 'AWC − synchronous',
              },
            ]}
            label="WSD selected loss gap"
            layout={{
              xaxis: {
                title: { text: 'Global tokens per parameter' },
                tickvals: data.protocol.horizons,
              },
              yaxis: { title: { text: 'AWC − synchronous (nats/token)' } },
            }}
          />
        </section>
      </div>
      <section aria-label="WSD configuration explorer">
        <h2>Inspect a horizon</h2>
        <div className="wsd-controls">
          <label>
            Method
            <select
              aria-label="Method"
              value={state.method}
              onChange={(e) => scope('method', e.target.value)}
            >
              {data.campaigns.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.title}
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
              {data.protocol.horizons.map((h) => (
                <option key={h} value={h}>
                  {h} tokens/parameter
                </option>
              ))}
            </select>
          </label>
          {(['lr', 'beta1', 'beta2'] as const).map((k) => (
            <label key={k}>
              {k === 'lr' ? 'Learning rate' : k === 'beta1' ? 'Beta 1' : 'Beta 2'}
              <select
                aria-label={k === 'lr' ? 'Learning rate' : k === 'beta1' ? 'Beta 1' : 'Beta 2'}
                value={state[k]}
                onChange={(e) => setState((s) => ({ ...s, [k]: e.target.value, selected: '' }))}
              >
                <option value="all">All</option>
                {[...new Set(stage.groups.map((g) => g[k]))]
                  .sort((a, b) => a - b)
                  .map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
              </select>
            </label>
          ))}
          <button
            onClick={() =>
              setState((s) => ({ ...s, lr: 'all', beta1: 'all', beta2: 'all', selected: '' }))
            }
          >
            Reset filters
          </button>
        </div>
        <p>
          {campaign.title} · horizon {stage.horizon}: {filtered.length} of {stage.groups.length}{' '}
          configurations, {stage.runs} complete seed results. Entering LR ceiling: {stage.lrCeiling}
          . Global tokens: {stage.tokens.toLocaleString('en-US')}.
        </p>
        <p className="note">
          Ranks apply only within this method and horizon. All eligible triples include seeds 42, 43
          and 44. Pruned LRs never re-enter the search.
        </p>
        <div className="actions">
          <button disabled={!filtered.length} onClick={exportCSV}>
            Download filtered CSV
          </button>
          <button
            onClick={() =>
              download(
                `${state.method}-wsd-h${state.horizon}-filtered.json`,
                JSON.stringify(filtered, null, 2),
                'application/json',
              )
            }
          >
            Download filtered JSON
          </button>
        </div>
        <div className="table-wrap">
          <table aria-label="WSD configuration rankings">
            <thead>
              <tr>
                <th>Rank</th>
                {(['lr', 'beta1', 'beta2', 'mean', 'sd'] as const).map((k) => (
                  <th
                    key={k}
                    aria-sort={sort.key === k ? (sort.desc ? 'descending' : 'ascending') : 'none'}
                  >
                    <button onClick={() => sortBy(k)}>
                      {
                        { lr: 'LR', beta1: 'β₁', beta2: 'β₂', mean: 'Mean loss', sd: 'Sample SD' }[
                          k
                        ]
                      }
                    </button>
                  </th>
                ))}
                <th>Inspect</th>
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
                    <button
                      aria-label={`Inspect WSD rank ${g.rank}`}
                      onClick={() => setState((s) => ({ ...s, selected: g.id }))}
                    >
                      Seeds & curves
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!filtered.length && <p>No configurations match these filters.</p>}
      </section>
      <div className="wsd-charts">
        <section>
          <h2>Learning-rate response</h2>
          <p>Filtered beta pairs; mean ± sample SD. Click a point to inspect its seeds.</p>
          <Plot
            data={lrTraces}
            label="WSD learning-rate response"
            onPoint={select}
            layout={{
              xaxis: { title: { text: 'Learning rate' } },
              yaxis: { title: { text: 'Full-validation loss' } },
            }}
          />
        </section>
        <section>
          <h2>Full horizon grid</h2>
          <p>
            All beta pairs and original LR coverage, independent of filters. Click a measured cell
            to inspect it.
          </p>
          <p className="small">
            P = pruned · — = outside the original grid · ? = missing observations.
          </p>
          <Plot
            data={heat}
            label="WSD horizon heatmap"
            onPoint={select}
            layout={{
              xaxis: { type: 'category', title: { text: 'Learning rate' } },
              yaxis: { type: 'category', title: { text: 'β₁ / β₂' } },
            }}
          />
          <details>
            <summary>Grid coverage</summary>
            <div className="table-wrap">
              <table aria-label="WSD grid coverage">
                <thead>
                  <tr>
                    <th>β₁ / β₂</th>
                    {allLRs.map((lr) => (
                      <th key={lr}>{lr}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pairs.map(([a, b], i) => (
                    <tr key={ys[i]}>
                      <th>{ys[i]}</th>
                      {allLRs.map((lr) => (
                        <td key={lr}>{cellStatus(campaign, stage, lr, a, b)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      </div>
      {selected && (
        <section aria-label="Selected WSD configuration">
          <h2>Selected configuration</h2>
          <p>
            {campaign.title} · horizon {stage.horizon} · LR {selected.lr} · β₁ {selected.beta1} · β₂{' '}
            {selected.beta2}
          </p>
          <div className="table-wrap">
            <table aria-label="WSD seed results">
              <thead>
                <tr>
                  <th>Seed</th>
                  <th>Final full-validation loss</th>
                  <th>Job ID</th>
                </tr>
              </thead>
              <tbody>
                {selected.runs.map((r) => (
                  <tr key={r.seed}>
                    <td>{r.seed}</td>
                    <td>{r.loss.toFixed(9)}</td>
                    <td>{r.jobId}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="wsd-controls">
            <label>
              Loss view
              <select
                aria-label="Loss view"
                value={state.kind}
                onChange={(e) =>
                  setState((s) => ({ ...s, kind: e.target.value as Controls['kind'] }))
                }
              >
                <option value="validation">Epoch subset validation</option>
                <option value="train">Training windows</option>
              </select>
            </label>
            <label>
              <input
                type="checkbox"
                checked={state.seeds}
                onChange={(e) => setState((s) => ({ ...s, seeds: e.target.checked }))}
              />
              Show individual seeds
            </label>
          </div>
          <p>
            Recorded observations are joined without smoothing. Earlier terminal decays are excluded
            at continuation boundaries. Diamonds show final full-validation loss; the subset and
            training losses are different measurements.
          </p>
          {curveError ? (
            <p role="alert">{curveError}</p>
          ) : curve ? (
            <Plot
              data={curveTraces}
              label={
                state.kind === 'train'
                  ? 'WSD training trajectories'
                  : 'WSD subset-validation trajectories'
              }
              layout={{
                xaxis: { title: { text: 'Global training tokens' } },
                yaxis: { title: { text: 'Loss (nats/token)' } },
              }}
            />
          ) : (
            <p role="status">Loading recorded curves…</p>
          )}
          {curve && (
            <details>
              <summary>Continuation sources and curve data</summary>
              <p>
                {curve.runs[0].sources
                  .map(
                    (s) =>
                      `H${s.horizon}: ${s.startTokens.toLocaleString()}–${s.endTokens.toLocaleString()} tokens`,
                  )
                  .join('; ')}
                .
              </p>
              <a href={asset('curves/' + selected.id + '.json')}>
                Download recorded points and source hashes
              </a>
            </details>
          )}
        </section>
      )}
      <section>
        <h2>Matched optimizer comparisons</h2>
        <p>
          Same horizon, LR, beta pair, and runtime seeds. Positive AWC − synchronous values favor
          synchronous training. SD describes paired seed differences, not a confidence interval.
        </p>
        <div className="table-wrap">
          <table aria-label="WSD matched comparisons">
            <thead>
              <tr>
                <th>LR</th>
                <th>β₁</th>
                <th>β₂</th>
                <th>Sync mean</th>
                <th>AWC mean</th>
                <th>AWC − sync</th>
                <th>Paired SD</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((m) => (
                <tr key={`${m.lr}/${m.beta1}/${m.beta2}`}>
                  <td>{m.lr}</td>
                  <td>{m.beta1}</td>
                  <td>{m.beta2}</td>
                  <td>{fmt(m.sync)}</td>
                  <td>{fmt(m.awc8)}</td>
                  <td>{fmt(m.difference)}</td>
                  <td>{fmt(m.pairedSD)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!matches.length && <p>No measurements at matching settings in the other campaign.</p>}
      </section>
    </div>
  );
}
