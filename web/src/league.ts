import { getLeague } from "./api";
import type { ArenaEntry, CrossplayRow, LeagueReport, PoolEntry } from "./types";

/**
 * League panel of a training run (M6.4): the cross-play matrix of the arena (human snapshots x
 * zombie teams, cell color = human score), and per side the pool entries with their Elo rating,
 * PFSP weight and the other learner's score against them. Hidden for runs that are not leagues.
 */
export class LeaguePanel {
  private runId: string | null = null;
  private timer: number | undefined;

  constructor(private el: HTMLElement) {}

  async show(runId: string): Promise<void> {
    this.runId = runId;
    window.clearInterval(this.timer);
    await this.refresh();
    this.timer = window.setInterval(() => void this.refresh(), 30_000);
  }

  hide(): void {
    this.runId = null;
    window.clearInterval(this.timer);
    this.el.hidden = true;
    this.el.replaceChildren();
  }

  private async refresh(): Promise<void> {
    const runId = this.runId;
    if (!runId) return;
    const report = await getLeague(runId).catch(() => null);
    if (runId !== this.runId) return;
    if (!report) {
      this.el.hidden = true;
      return;
    }
    this.render(report);
    this.el.hidden = false;
  }

  private render(report: LeagueReport): void {
    const elo = new Map<string, ArenaEntry>((report.arena?.entries ?? []).map((e) => [e.id, e]));
    const parts: HTMLElement[] = [];
    const title = document.createElement("h3");
    const updated = report.arena?.updated ?? report.pool?.updated;
    title.textContent = `League ${report.id}` + (updated ? ` · updated ${age(updated)}` : "");
    parts.push(title);
    if (report.arena && report.arena.crossplay.length) parts.push(heatmap(report.arena.crossplay, elo));
    if (report.pool) {
      const tables = document.createElement("div");
      tables.className = "league-tables";
      for (const side of ["human", "zombie"] as const) {
        tables.append(poolTable(side, report.pool.sides[side], elo, report.arena?.latest[side] ?? null));
      }
      parts.push(tables);
    }
    this.el.replaceChildren(...parts);
  }
}

function age(unix: number): string {
  const s = Math.max(0, Date.now() / 1000 - unix);
  return s < 120 ? `${Math.round(s)} s ago` : s < 7200 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
}

const fmt = (x: number, d = 0) => (Number.isFinite(x) ? x.toFixed(d) : "–");

function eloCell(e: ArenaEntry | undefined): string {
  return e ? `${fmt(e.elo)} [${fmt(e.elo_lo)}, ${fmt(e.elo_hi)}]` : "–";
}

/** Human score 0 (red, zombies win) .. 0.5 (neutral) .. 1 (green, the human wins). */
function scoreColor(score: number): string {
  const t = Math.min(1, Math.max(0, score));
  const hue = 120 * t;
  return `hsl(${hue} 55% ${18 + 14 * Math.abs(t - 0.5)}%)`;
}

function heatmap(rows: CrossplayRow[], elo: Map<string, ArenaEntry>): HTMLElement {
  const order = (side: "human" | "zombie") =>
    [...new Set(rows.map((r) => r[side]))].sort(
      (a, b) => (elo.get(a)?.steps ?? 0) - (elo.get(b)?.steps ?? 0) || a.localeCompare(b),
    );
  const humans = order("human");
  const zombies = order("zombie");
  const cell = new Map(rows.map((r) => [`${r.human}|${r.zombie}`, r]));
  const table = document.createElement("table");
  table.className = "heatmap";
  const head = table.createTHead().insertRow();
  head.insertCell().textContent = "human \\ zombies";
  for (const z of zombies) head.insertCell().textContent = z;
  const body = table.createTBody();
  for (const h of humans) {
    const tr = body.insertRow();
    tr.insertCell().textContent = h;
    for (const z of zombies) {
      const td = tr.insertCell();
      const r = cell.get(`${h}|${z}`);
      if (!r) continue;
      td.textContent = fmt(r.score, 2);
      td.style.background = scoreColor(r.score);
      td.title = `${h} vs ${z}: survived ${fmt(100 * r.survived)}%, caught ${fmt(100 * r.caught)}%, starved ${fmt(100 * r.starved)}% (${r.episodes} episodes)`;
    }
  }
  const box = document.createElement("div");
  box.className = "league-block";
  const caption = document.createElement("div");
  caption.className = "label";
  caption.textContent = "Cross-play: human score (survived 1, starved 0.5, caught 0)";
  box.append(caption, table);
  return box;
}

function poolTable(
  side: "human" | "zombie",
  entries: PoolEntry[],
  elo: Map<string, ArenaEntry>,
  latest: string | null,
): HTMLElement {
  const table = document.createElement("table");
  table.className = "pool";
  const head = table.createTHead().insertRow();
  for (const h of ["entry", "source", "steps", "Elo [95%]", "PFSP", "learner score", "games"]) {
    head.insertCell().textContent = h;
  }
  const body = table.createTBody();
  const sorted = [...entries].sort((a, b) => b.steps - a.steps || a.id.localeCompare(b.id));
  for (const e of sorted) {
    const tr = body.insertRow();
    if (e.id === latest) tr.className = "latest";
    const cells = [
      e.id + (e.protected ? " *" : ""),
      e.source,
      `${fmt(e.steps / 1e6, 1)} M`,
      eloCell(elo.get(e.id)),
      e.path ? fmt(e.weight, 3) : "pinned",
      fmt(e.p_learner, 2),
      String(e.games),
    ];
    for (const c of cells) tr.insertCell().textContent = c;
  }
  const box = document.createElement("div");
  box.className = "league-block";
  const caption = document.createElement("div");
  caption.className = "label";
  caption.textContent = `${side === "human" ? "Human" : "Zombie"} pool (* protected)`;
  box.append(caption, table);
  return box;
}
