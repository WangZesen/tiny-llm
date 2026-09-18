import { useEffect, useRef, useState } from 'react';
import type { Data, Layout, PlotMouseEvent } from 'plotly.js';
import { useTheme } from '../lib/theme';

export default function Plot({
  data,
  layout = {},
  label,
  onPoint,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  label: string;
  onPoint?: (id: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState(false);
  const theme = useTheme();
  const handler = useRef(onPoint);
  handler.current = onPoint;
  useEffect(() => {
    let cancelled = false;
    const node = ref.current;
    if (!node) return;
    let observer: ResizeObserver | undefined;
    let cleanup: (() => void) | undefined;
    import('plotly.js-cartesian-dist-min')
      .then(async ({ default: Plotly }) => {
        if (cancelled) return;
        const dark = theme === 'dark';
        const ink = dark ? '#eae1d3' : '#37312b';
        const grid = dark ? '#494037' : '#e1d8cb';
        await Plotly.react(
          node,
          data,
          {
            autosize: true,
            height: 360,
            margin: { l: 55, r: 15, t: 15, b: 65 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'DM Sans, sans-serif', size: 11, color: ink },
            hovermode: 'closest',
            legend: { orientation: 'h', y: -0.23, font: { size: 9 } },
            ...layout,
            xaxis: { gridcolor: grid, zerolinecolor: grid, automargin: true, ...layout.xaxis },
            yaxis: { gridcolor: grid, zerolinecolor: grid, automargin: true, ...layout.yaxis },
          },
          {
            responsive: true,
            displaylogo: false,
            scrollZoom: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
              format: 'svg',
              filename: 'tiny-llm-' + label.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
              width: 1100,
              height: 650,
              scale: 1,
            },
          },
        );
        if (cancelled) {
          Plotly.purge(node);
          return;
        }
        const plot = node as any;
        plot.removeAllListeners?.('plotly_click');
        plot.on('plotly_click', (event: PlotMouseEvent) => {
          const id = event.points[0]?.customdata;
          if (typeof id === 'string') handler.current?.(id);
        });
        observer = new ResizeObserver(() => {
          Plotly.Plots.resize(node);
        });
        observer.observe(node);
        cleanup = () => Plotly.purge(node);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
      observer?.disconnect();
      cleanup?.();
    };
  }, [data, layout, theme]);
  return (
    <div>
      <div className="plot" ref={ref} role="img" aria-label={label} />
      {error && (
        <p role="alert">Chart could not load. The results table and downloads remain available.</p>
      )}
    </div>
  );
}
