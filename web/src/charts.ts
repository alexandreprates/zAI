import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import { listMetrics } from "./api";

const COLORS = ["#4ea8ff", "#f5c451", "#3ecf8e", "#e5484d", "#b58cff", "#ff8f5c", "#5ce1e6", "#9aa5b1"];
/** Chart groups shown first; anything else follows alphabetically. */
const ORDER = [
  "/gate/",
  "/anchor/",
  "/elo/",
  "/xplay/",
  "/eval/",
  "/eval_vs_learned_human/",
  "/episode_A/",
  "/episode/",
  "/league/",
  "/pool/",
  "/tactics/",
  "/tactics_train/",
  "/train/",
  "/perf/",
];
const AXIS = { stroke: "#8b93a1", grid: { stroke: "#2e3440", width: 1 }, ticks: { stroke: "#2e3440" } };

interface Series {
  label: string;
  points: Map<number, number>; // step -> value
}

interface Chart {
  title: string;
  series: Series[];
  plot: uPlot | null;
  el: HTMLElement;
}

/**
 * Groups related metrics on one chart:
 * - `eval/baseline_x` is drawn with `eval/x` (label `baseline`);
 * - `eval/x_lo` / `eval/x_hi` (95% interval bounds) are drawn with `eval/x`;
 * - `eval/x_z3` (per zombie count) go to one `eval/x by zombie count` chart (label `z3`).
 */
export function chartKey(model: string, key: string): { chart: string; label: string } {
  const m = key.match(/^(.*\/)(baseline_)?(.+?)(_lo|_hi|_z\d+)?$/);
  if (!m) return { chart: `${model}/${key}`, label: model };
  const [, dir, baseline, name, suffix] = m;
  const who = baseline ? "baseline" : model;
  if (suffix?.startsWith("_z")) {
    return {
      chart: `${model}/${dir}${name} by zombie count`,
      label: `${baseline ? "baseline " : ""}${suffix.slice(1)}`,
    };
  }
  const bound = suffix === "_lo" ? " 95% lo" : suffix === "_hi" ? " 95% hi" : "";
  return { chart: `${model}/${dir}${name}`, label: `${who}${bound}` };
}

/** Training curves of one run, live-updated through the metrics SSE stream. */
export class Charts {
  private charts = new Map<string, Chart>();
  private source: EventSource | null = null;
  private runId: string | null = null;
  private resize = new ResizeObserver(() => this.layout());

  constructor(private el: HTMLElement) {
    this.resize.observe(el);
  }

  async show(runId: string): Promise<void> {
    this.hide();
    this.runId = runId;
    this.el.hidden = false;
    const points = await listMetrics(runId);
    for (const p of points) this.add(p.model, p.key, p.step, p.value);
    this.render();
    this.source = new EventSource("/api/metrics/stream");
    this.source.onmessage = (ev) => {
      const m = JSON.parse(ev.data) as {
        run_id: string;
        model: string;
        step: number;
        scalars: Record<string, number>;
      };
      if (m.run_id !== this.runId) return;
      for (const [k, v] of Object.entries(m.scalars)) this.add(m.model, k, m.step, v);
      this.render();
    };
  }

  hide(): void {
    this.source?.close();
    this.source = null;
    this.runId = null;
    for (const c of this.charts.values()) c.plot?.destroy();
    this.charts.clear();
    this.el.replaceChildren();
    this.el.hidden = true;
  }

  private add(model: string, key: string, step: number, value: number): void {
    const { chart: title, label } = chartKey(model, key);
    let chart = this.charts.get(title);
    if (!chart) {
      const el = document.createElement("div");
      el.className = "chart";
      chart = { title, series: [], plot: null, el };
      this.charts.set(title, chart);
    }
    let series = chart.series.find((s) => s.label === label);
    if (!series) {
      series = { label, points: new Map() };
      chart.series.push(series);
      chart.plot?.destroy();
      chart.plot = null;
    }
    series.points.set(step, value);
  }

  private width(): number {
    // Container padding 2 x 16, gap 16, card padding + border 2 x 9.
    const cols = this.el.clientWidth > 900 ? 2 : 1;
    const inner = this.el.clientWidth - 32 - 16 * (cols - 1) - 18 * cols;
    return Math.max(260, Math.floor(inner / cols));
  }

  private data(chart: Chart): uPlot.AlignedData {
    const steps = [...new Set(chart.series.flatMap((s) => [...s.points.keys()]))].sort((a, b) => a - b);
    return [steps, ...chart.series.map((s) => steps.map((x) => s.points.get(x) ?? null))];
  }

  private render(): void {
    const rank = (t: string) => ORDER.findIndex((p) => t.includes(p)) >>> 0;
    const sorted = [...this.charts.values()].sort(
      (a, b) => rank(a.title) - rank(b.title) || a.title.localeCompare(b.title),
    );
    if (this.el.childElementCount !== sorted.length) this.el.replaceChildren(...sorted.map((c) => c.el));
    for (const chart of sorted) {
      if (chart.plot) {
        chart.plot.setData(this.data(chart));
        continue;
      }
      chart.plot = new uPlot(
        {
          title: chart.title,
          width: this.width(),
          height: 200,
          scales: { x: { time: false } },
          axes: [AXIS, AXIS],
          series: [
            { label: "step" },
            ...chart.series.map((s, i) => ({
              label: s.label,
              stroke: COLORS[i % COLORS.length],
              width: 1.5,
              spanGaps: true,
            })),
          ],
        },
        this.data(chart),
        chart.el,
      );
    }
  }

  private layout(): void {
    const w = this.width();
    for (const c of this.charts.values()) c.plot?.setSize({ width: w, height: 200 });
  }
}
