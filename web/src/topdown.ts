import { Application, Container, Graphics } from "pixi.js";

import type { InitFrame, SnapshotFrame } from "./types";

const COLORS = {
  background: 0x14171c,
  ground: 0x1c2027,
  border: 0x3a414d,
  wall: 0xc9ced6,
  doorOpen: 0x3ecf8e,
  doorClosed: 0xe5484d,
  food: 0xf5c451,
  human: 0x4ea8ff,
  vision: 0x4ea8ff,
  zombie: 0xe5484d,
  noise: [0xa0a7b4, 0xf5c451, 0xe5484d],
};

interface Ring {
  x: number;
  y: number;
  radius: number;
  kind: number;
  born: number;
}

const RING_MS = 900;

/** Top-down renderer with interpolation between decision-step snapshots. */
export class TopDown {
  private app = new Application();
  private world = new Container();
  private mapLayer = new Graphics();
  private foodLayer = new Graphics();
  private ringLayer = new Graphics();
  private entityLayer = new Graphics();
  private init: InitFrame | null = null;
  private prev: SnapshotFrame | null = null;
  private curr: SnapshotFrame | null = null;
  private currAt = 0;
  private lastDoors = -1;
  private rings: Ring[] = [];
  private scale = 1;

  async mount(el: HTMLElement): Promise<void> {
    await this.app.init({ resizeTo: el, background: COLORS.background, antialias: true });
    el.appendChild(this.app.canvas);
    this.world.addChild(this.mapLayer, this.foodLayer, this.ringLayer, this.entityLayer);
    this.app.stage.addChild(this.world);
    this.app.renderer.on("resize", () => this.layout());
    this.app.ticker.add(() => this.draw());
  }

  setInit(frame: InitFrame): void {
    this.init = frame;
    this.prev = this.curr = null;
    this.lastDoors = -1;
    this.rings = [];
    this.layout();
  }

  pushSnapshot(frame: SnapshotFrame): void {
    const now = performance.now();
    // A jump backwards (seek, new episode) must not interpolate across it.
    const continuous = this.curr !== null && frame.tick === this.curr.tick + 1;
    this.prev = continuous ? this.curr : frame;
    this.curr = frame;
    this.currAt = now;
    for (const [kind, x, y, radius] of frame.events) this.rings.push({ kind, x, y, radius, born: now });
    if (frame.doors !== this.lastDoors) {
      this.lastDoors = frame.doors;
      this.drawMap();
    }
    this.drawFood();
  }

  private layout(): void {
    if (!this.init) return;
    const { width, height } = this.app.screen;
    const size = this.init.map_size;
    this.scale = (Math.min(width, height) * 0.94) / size;
    this.world.position.set((width - size * this.scale) / 2, (height - size * this.scale) / 2);
    this.drawMap();
    this.drawFood();
  }

  /** World (y up) to local screen coordinates (y down). */
  private sx(x: number): number {
    return x * this.scale;
  }
  private sy(y: number): number {
    return ((this.init?.map_size ?? 0) - y) * this.scale;
  }

  private drawMap(): void {
    const g = this.mapLayer.clear();
    if (!this.init) return;
    const size = this.init.map_size * this.scale;
    g.rect(0, 0, size, size).fill(COLORS.ground).stroke({ width: 1, color: COLORS.border });
    const line = (s: number[], color: number, width: number) =>
      g.moveTo(this.sx(s[0]), this.sy(s[1])).lineTo(this.sx(s[2]), this.sy(s[3])).stroke({ width, color });
    this.init.houses.forEach((h, i) => {
      for (const w of h.walls) line(w, COLORS.wall, Math.max(1.5, 0.35 * this.scale));
      h.doors.forEach((d, k) => {
        const closed = this.lastDoors >= 0 && (this.lastDoors >> (2 * i + k)) & 1;
        line(d, closed ? COLORS.doorClosed : COLORS.doorOpen, Math.max(1.5, 0.5 * this.scale));
      });
    });
  }

  private drawFood(): void {
    const g = this.foodLayer.clear();
    if (!this.curr) return;
    const r = Math.max(1.5, 0.5 * this.scale);
    for (const [x, y] of this.curr.food) g.circle(this.sx(x), this.sy(y), r);
    g.fill(COLORS.food);
  }

  private draw(): void {
    const g = this.entityLayer.clear();
    const rg = this.ringLayer.clear();
    if (!this.init || !this.curr || !this.prev) return;
    const now = performance.now();
    const interval = this.curr.speed > 0 ? (this.init.decision_dt * 1000) / this.curr.speed : 0;
    const t = interval > 0 && !this.curr.paused ? Math.min(1, (now - this.currAt) / interval) : 1;
    const lerp = (a: number, b: number) => a + (b - a) * t;

    // Noise rings expand to their radius and fade.
    this.rings = this.rings.filter((r) => now - r.born < RING_MS);
    for (const ring of this.rings) {
      const k = (now - ring.born) / RING_MS;
      rg.circle(this.sx(ring.x), this.sy(ring.y), ring.radius * this.scale * Math.min(1, k * 1.5)).stroke({
        width: 1.5,
        color: COLORS.noise[ring.kind] ?? COLORS.noise[0],
        alpha: 1 - k,
      });
    }

    // Human with its vision cone.
    const p = this.prev.human;
    const c = this.curr.human;
    const hx = this.sx(lerp(p.x, c.x));
    const hy = this.sy(lerp(p.y, c.y));
    const vr = this.init.config.human.vision_radius_m * this.scale;
    const fov = (this.init.config.human.vision_fov_deg * Math.PI) / 180;
    const heading = Math.atan2(c.hy, c.hx);
    if (fov >= 2 * Math.PI - 1e-3) {
      g.circle(hx, hy, vr).fill({ color: COLORS.vision, alpha: 0.08 });
    } else {
      // Screen y is flipped, so world angles are negated.
      g.moveTo(hx, hy)
        .arc(hx, hy, vr, -heading - fov / 2, -heading + fov / 2)
        .lineTo(hx, hy)
        .fill({ color: COLORS.vision, alpha: 0.1 });
    }
    const er = Math.max(3, this.init.config.entity_radius_m * this.scale * 2);
    g.circle(hx, hy, er).fill(COLORS.human);

    // Zombies.
    this.curr.zombies.forEach((z, i) => {
      const pz = this.prev!.zombies[i] ?? z;
      g.circle(this.sx(lerp(pz[0], z[0])), this.sy(lerp(pz[1], z[1])), er);
    });
    g.fill(COLORS.zombie);
  }
}
