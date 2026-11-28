import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { headingYaw, lerpYaw, verticalFov, visibility } from "./firstperson";
import type { InitFrame, SnapshotFrame } from "./types";

// Radial fog: three's default fog uses view depth, which at the edges of a 120 degree view
// would show objects up to twice the fog range away; vision is a radius (4.9.6).
THREE.ShaderChunk.fog_vertex = THREE.ShaderChunk.fog_vertex.replace(
  "vFogDepth = - mvPosition.z;",
  "vFogDepth = length( mvPosition.xyz );",
);

/** `orbit`: free camera; `follow`: orbit around the human; `first`: the human's eyes. */
export type CameraMode = "orbit" | "follow" | "first";

// Soft pastel look: light floor, cream walls, rounded characters with eyes.
const COLORS = {
  sky: 0xc9d8e6,
  ground: 0xe7e0d1,
  grid: 0xb9ae98,
  houseFloor: 0xd6c9b0,
  wall: 0xf6f2e9,
  border: 0xa9a294,
  doorClosed: 0xd9534f,
  doorOpen: 0x3ecf8e,
  food: 0xf5c451,
  human: 0x3d8bfd,
  vision: 0x3d8bfd,
  zombie: 0x6cc24a,
  zombieDark: 0x2f4a22,
  zombieEye: 0xf2e7a6,
  zombiePupil: 0xff2a1a,
  mouth: 0x2a0d0d,
  tooth: 0xf3eedb,
  eye: 0xffffff,
  pupil: 0x1b1f26,
  noise: [0x7d8594, 0xe0a82e, 0xd9534f],
};

const WALL_HEIGHT = 2.4;
const WALL_THICKNESS = 0.3;
const RING_MS = 900;
const FALL_MS = 600;
const MAX_FOOD = 512;
// First person: eye height, the caught camera's resting height, and zombies drawn 1.75 m tall
// (visual only; the collision radius stays 0.3 m, the capsule is ~0.99 m).
const EYE_M = 1.6;
const FALLEN_EYE_M = 0.4;
const ZOMBIE_FP_SCALE = 1.75 / 0.99;
const FP_FOG = [70, 100];

interface Ring {
  line: THREE.LineLoop;
  radius: number;
  born: number;
}

interface Character {
  /** Yaw (heading) and position. */
  root: THREE.Group;
  /** Body, tipped over when the human dies; bobs while walking. */
  body: THREE.Group;
  /** Drawn translucent (first-person x-ray of an unseen zombie). */
  ghost?: boolean;
}

const ground = (x: number, y: number, h = 0) => new THREE.Vector3(x, h, -y);

/** 3D renderer over the same frames as the top-down view (world y-up maps to three's -z, so
 * the map reads the same way from the default camera): an orbit or follow camera, or the
 * human's first-person view, which draws only what the human sees. Interpolates between
 * decision-step snapshots like `TopDown`. */
export class Scene3D {
  private renderer = new THREE.WebGLRenderer({ antialias: true });
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(50, 1, 0.1, 3000);
  private controls: OrbitControls;
  private sun = new THREE.DirectionalLight(0xffffff, 1.7);
  private el: HTMLElement | null = null;
  private active = false;
  private frameId = 0;

  private mapGroup = new THREE.Group();
  private doorMeshes: { closed: THREE.Mesh; open: THREE.Mesh }[][] = [];
  private food: THREE.InstancedMesh;
  private human: Character;
  private vision = new THREE.Group();
  private zombies: Character[] = [];
  private zombieLayer = new THREE.Group();
  private rings: Ring[] = [];
  private ringGeometry: THREE.BufferGeometry;

  private init: InitFrame | null = null;
  private prev: SnapshotFrame | null = null;
  private curr: SnapshotFrame | null = null;
  private currAt = 0;
  private lastDoors = -1;
  private endedAt = 0;
  /** Characters are drawn larger than their 0.3 m radius on big maps so they stay visible. */
  private bodyScale = 1;
  private mode: CameraMode = "follow";
  private xray = false;
  private focus = new THREE.Vector3();
  private framedSize = -1;
  private fog = new THREE.Fog(COLORS.sky, FP_FOG[0], FP_FOG[1]);
  private yaw = 0;
  /** Tint over the first-person view: red when caught, grey when starved. */
  private overlay = document.createElement("div");
  private fpsEl: HTMLElement | null = null;
  private frameTimes: number[] = [];

  constructor() {
    const r = this.renderer;
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.background = new THREE.Color(COLORS.sky);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xb3a58a, 1.4));
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const s = this.sun.shadow.camera;
    s.left = s.bottom = -45;
    s.right = s.top = 45;
    s.near = 1;
    s.far = 200;
    this.sun.shadow.bias = -0.0005;
    this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.sun, this.sun.target);

    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.minDistance = 3;

    const points: THREE.Vector3[] = [];
    for (let i = 0; i < 96; i++) {
      const a = (i / 96) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
    }
    this.ringGeometry = new THREE.BufferGeometry().setFromPoints(points);

    this.food = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.45, 0.3, 0.35),
      new THREE.MeshStandardMaterial({ color: COLORS.food, roughness: 0.6 }),
      MAX_FOOD,
    );
    this.food.castShadow = true;
    this.food.count = 0;

    this.human = this.character(COLORS.human, false);
    this.scene.add(this.mapGroup, this.food, this.human.root, this.vision, this.zombieLayer);
  }

  mount(el: HTMLElement): void {
    this.el = el;
    el.appendChild(this.renderer.domElement);
    this.overlay.className = "fp-overlay";
    el.appendChild(this.overlay);
    if (new URLSearchParams(location.search).has("fps")) {
      this.fpsEl = document.createElement("div");
      this.fpsEl.className = "fps";
      el.appendChild(this.fpsEl);
    }
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
  }

  /** Starts or stops rendering (the hidden view does not draw). */
  setActive(active: boolean): void {
    this.active = active;
    cancelAnimationFrame(this.frameId);
    if (active) {
      this.resize();
      this.frameId = requestAnimationFrame(this.loop);
    }
  }

  /** Follow keeps the camera's orbit around the human, orbit is free, first is the human's
   * eyes (120 degree view, fog at the 100 m vision radius, only what the human sees). */
  setCameraMode(mode: CameraMode): void {
    const wasFirst = this.mode === "first";
    this.mode = mode;
    const first = mode === "first";
    this.controls.enabled = !first;
    if (first !== wasFirst) {
      this.scene.fog = first ? this.fog : null;
      this.camera.near = first ? 0.05 : 0.1;
      this.camera.far = first ? FP_FOG[1] + 5 : 3000;
      // Fog is compiled into the shaders.
      this.scene.traverse((o) => {
        if (o instanceof THREE.Mesh || o instanceof THREE.Line) {
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => (m.needsUpdate = true));
        }
      });
      this.applyScale();
      this.resize();
      if (!first) this.camera.rotation.set(0, 0, 0);
      this.overlay.style.opacity = "0";
      this.overlay.textContent = "";
    }
    if (this.curr) this.updateFood(this.curr);
    if (first) return;
    if (mode === "follow" && this.curr) {
      const h = this.curr.human;
      this.focus.copy(ground(h.x, h.y));
      this.placeCamera(this.focus, 14 * this.bodyScale);
    } else if (this.init) {
      this.frameMap();
    }
  }

  /** First person: show unseen zombies translucent instead of hiding them. */
  setXray(xray: boolean): void {
    this.xray = xray;
  }

  /** Characters at true scale in first person (zombies 1.75 m tall), enlarged otherwise. */
  private applyScale(): void {
    const first = this.mode === "first";
    this.human.root.scale.setScalar(this.bodyScale);
    for (const z of this.zombies) z.root.scale.setScalar(first ? ZOMBIE_FP_SCALE : this.bodyScale);
  }

  setInit(frame: InitFrame): void {
    this.init = frame;
    this.prev = this.curr = null;
    this.lastDoors = -1;
    this.endedAt = 0;
    for (const ring of this.rings) this.disposeRing(ring);
    this.rings = [];
    this.bodyScale = Math.min(3, Math.max(1, frame.map_size / 60));
    this.applyScale();
    this.buildMap(frame);
    this.buildVision(frame);
    if (frame.map_size !== this.framedSize) {
      // Only reframe on a new map size, so the camera survives seeks and auto restarts.
      this.framedSize = frame.map_size;
      this.frameMap();
    }
  }

  pushSnapshot(frame: SnapshotFrame): void {
    const now = performance.now();
    const continuous = this.curr !== null && frame.tick === this.curr.tick + 1;
    const first = this.curr === null;
    this.prev = continuous ? this.curr : frame;
    this.curr = frame;
    this.currAt = now;
    if (frame.outcome === "running") this.endedAt = 0;
    else if (this.endedAt === 0) this.endedAt = continuous ? now : now - FALL_MS;

    for (const [kind, x, y, radius] of frame.events) this.addRing(kind, x, y, radius, now);
    if (frame.doors !== this.lastDoors) {
      this.lastDoors = frame.doors;
      this.updateDoors();
    }
    this.updateFood(frame);
    this.ensureZombies(frame.zombies.length);
    if (first && this.mode === "follow") this.setCameraMode("follow");
  }

  private loop = (): void => {
    if (!this.active) return;
    this.frameId = requestAnimationFrame(this.loop);
    const now = performance.now();
    this.update(now);
    if (this.mode !== "first") this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (this.fpsEl) {
      this.frameTimes.push(performance.now() - now);
      if (this.frameTimes.length >= 60) {
        const ms = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        this.fpsEl.textContent = `${ms.toFixed(1)} ms/frame (render)`;
        this.frameTimes = [];
      }
    }
  };

  private resize(): void {
    if (!this.el) return;
    const w = Math.max(1, this.el.clientWidth);
    const h = Math.max(1, this.el.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.fov = this.mode === "first" ? verticalFov(120, w / h) : 50;
    this.camera.updateProjectionMatrix();
  }

  private frameMap(): void {
    if (!this.init) return;
    const half = this.init.map_size / 2;
    this.focus.set(half, 0, -half);
    this.placeCamera(this.focus, this.init.map_size * 0.95);
  }

  private placeCamera(target: THREE.Vector3, distance: number): void {
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(new THREE.Vector3(0, 0.62, 0.78).multiplyScalar(distance));
    this.controls.maxDistance = Math.max(60, (this.init?.map_size ?? 100) * 2);
    this.controls.update();
  }

  // ---------------------------------------------------------------- static map

  private buildMap(frame: InitFrame): void {
    for (const obj of [...this.mapGroup.children]) {
      this.mapGroup.remove(obj);
      obj.traverse((o) => {
        if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) {
          o.geometry.dispose();
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
        }
      });
    }
    const size = frame.map_size;
    const half = size / 2;

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(size, size),
      new THREE.MeshStandardMaterial({ color: COLORS.ground, roughness: 1 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(half, 0, -half);
    floor.receiveShadow = true;
    this.mapGroup.add(floor);

    const grid = new THREE.GridHelper(size, Math.max(1, Math.round(size / 10)), COLORS.grid, COLORS.grid);
    const gm = grid.material as THREE.Material;
    gm.transparent = true;
    gm.opacity = 0.25;
    grid.position.set(half, 0.01, -half);
    this.mapGroup.add(grid);

    const wallMat = new THREE.MeshStandardMaterial({ color: COLORS.wall, roughness: 0.85 });
    const borderMat = new THREE.MeshStandardMaterial({ color: COLORS.border, roughness: 0.9 });
    const closedMat = new THREE.MeshStandardMaterial({ color: COLORS.doorClosed, roughness: 0.6 });
    const openMat = new THREE.MeshStandardMaterial({
      color: COLORS.doorOpen,
      roughness: 0.6,
      emissive: COLORS.doorOpen,
      emissiveIntensity: 0.25,
    });
    const floorMat = new THREE.MeshStandardMaterial({ color: COLORS.houseFloor, roughness: 1 });

    for (const [ax, ay, bx, by] of [
      [0, 0, size, 0],
      [size, 0, size, size],
      [size, size, 0, size],
      [0, size, 0, 0],
    ]) {
      this.mapGroup.add(this.segment([ax, ay, bx, by], 0.6, 0.4, borderMat));
    }

    this.doorMeshes = frame.houses.map((h) => {
      // House floor from the wall extents in the house frame.
      const c = Math.cos(h.angle);
      const s = Math.sin(h.angle);
      let [minU, maxU, minV, maxV] = [Infinity, -Infinity, Infinity, -Infinity];
      for (const [ax, ay, bx, by] of [...h.walls, ...h.doors]) {
        for (const [px, py] of [
          [ax, ay],
          [bx, by],
        ]) {
          const u = (px - h.x) * c + (py - h.y) * s;
          const v = -(px - h.x) * s + (py - h.y) * c;
          [minU, maxU, minV, maxV] = [Math.min(minU, u), Math.max(maxU, u), Math.min(minV, v), Math.max(maxV, v)];
        }
      }
      const hf = new THREE.Mesh(new THREE.PlaneGeometry(maxU - minU, maxV - minV), floorMat);
      hf.rotation.set(-Math.PI / 2, 0, h.angle);
      const cu = (minU + maxU) / 2;
      const cv = (minV + maxV) / 2;
      hf.position.copy(ground(h.x + cu * c - cv * s, h.y + cu * s + cv * c, 0.02));
      hf.receiveShadow = true;
      this.mapGroup.add(hf);

      for (const w of h.walls) this.mapGroup.add(this.segment(w, WALL_HEIGHT, WALL_THICKNESS, wallMat));
      return h.doors.map((d) => {
        const closed = this.segment(d, WALL_HEIGHT * 0.92, WALL_THICKNESS * 0.8, closedMat);
        const open = this.segment(d, 0.05, WALL_THICKNESS * 1.6, openMat);
        open.castShadow = false;
        this.mapGroup.add(closed, open);
        return { closed, open };
      });
    });
    this.lastDoors = -1;
    this.updateDoors();
  }

  /** A box standing on the ground along the segment `[ax, ay, bx, by]`. */
  private segment(seg: number[], height: number, thickness: number, mat: THREE.Material): THREE.Mesh {
    const [ax, ay, bx, by] = seg;
    const len = Math.hypot(bx - ax, by - ay);
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(len + thickness * 0.5, height, thickness), mat);
    mesh.position.copy(ground((ax + bx) / 2, (ay + by) / 2, height / 2));
    mesh.rotation.y = Math.atan2(by - ay, bx - ax);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  private updateDoors(): void {
    this.doorMeshes.forEach((doors, i) =>
      doors.forEach((d, k) => {
        const closed = this.lastDoors >= 0 && ((this.lastDoors >> (2 * i + k)) & 1) === 1;
        d.closed.visible = closed;
        d.open.visible = !closed;
      }),
    );
  }

  private buildVision(frame: InitFrame): void {
    for (const obj of [...this.vision.children]) {
      this.vision.remove(obj);
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
        obj.geometry.dispose();
        (obj.material as THREE.Material).dispose();
      }
    }
    const radius = frame.config.human.vision_radius_m;
    const fov = Math.min(2 * Math.PI, (frame.config.human.vision_fov_deg * Math.PI) / 180);
    const cone = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 64, -fov / 2, fov),
      new THREE.MeshBasicMaterial({
        color: COLORS.vision,
        transparent: true,
        opacity: 0.13,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    );
    cone.rotation.x = -Math.PI / 2;
    cone.position.y = 0.04;
    cone.renderOrder = 1;
    // Outline (sides and arc), so the cone reads as vision rather than as a shadow.
    const edge: THREE.Vector3[] = [];
    const full = fov >= 2 * Math.PI - 1e-3;
    if (!full) edge.push(new THREE.Vector3(0, 0, 0));
    for (let i = 0; i <= 64; i++) {
      const a = -fov / 2 + (fov * i) / 64;
      edge.push(new THREE.Vector3(Math.cos(a) * radius, 0, -Math.sin(a) * radius));
    }
    if (!full) edge.push(new THREE.Vector3(0, 0, 0));
    const outline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(edge),
      new THREE.LineBasicMaterial({ color: COLORS.vision, transparent: true, opacity: 0.55 }),
    );
    outline.position.y = 0.05;
    this.vision.add(cone, outline);
  }

  // ---------------------------------------------------------------- characters

  /** A rounded body facing +x: the human has round friendly eyes; zombies lean forward with
   * a scowl (uneven yellowish eyes with glowing red pupils, angry brows, a crooked toothy
   * mouth). */
  private character(color: number, zombie: boolean): Character {
    const r = 0.3;
    const root = new THREE.Group();
    const body = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55 });
    const trunk = new THREE.Mesh(new THREE.CapsuleGeometry(r, r * 1.3, 6, 16), mat);
    trunk.position.y = r + (r * 1.3) / 2;
    trunk.castShadow = true;
    body.add(trunk);
    if (zombie) this.zombieFace(body, r);
    else this.humanFace(body, r);
    root.add(body);
    return { root, body };
  }

  private humanFace(body: THREE.Group, r: number): void {
    const eyeGeo = new THREE.SphereGeometry(r * 0.26, 12, 8);
    const pupilGeo = new THREE.SphereGeometry(r * 0.13, 10, 6);
    const eyeMat = new THREE.MeshStandardMaterial({ color: COLORS.eye, roughness: 0.3 });
    const pupilMat = new THREE.MeshBasicMaterial({ color: COLORS.pupil });
    for (const side of [-1, 1]) {
      const eye = new THREE.Mesh(eyeGeo, eyeMat);
      eye.position.set(r * 0.82, r * 2.05, side * r * 0.36);
      const pupil = new THREE.Mesh(pupilGeo, pupilMat);
      pupil.position.set(r * 0.2, 0, 0);
      eye.add(pupil);
      body.add(eye);
    }
  }

  private zombieFace(body: THREE.Group, r: number): void {
    // Hunched forward.
    body.rotation.z = -0.14;
    const eyeMat = new THREE.MeshStandardMaterial({ color: COLORS.zombieEye, roughness: 0.4 });
    const pupilMat = new THREE.MeshBasicMaterial({ color: COLORS.zombiePupil });
    const darkMat = new THREE.MeshStandardMaterial({ color: COLORS.zombieDark, roughness: 0.8 });
    const toothMat = new THREE.MeshStandardMaterial({ color: COLORS.tooth, roughness: 0.5 });
    // Uneven eyes: the left one bulges, the right one is small and lower.
    for (const [side, size, y] of [
      [-1, 0.3, 2.08],
      [1, 0.21, 1.98],
    ]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(r * size, 12, 8), eyeMat);
      eye.position.set(r * 0.8, r * y, side * r * 0.37);
      // Pinpoint pupils that glow (unlit material).
      const pupil = new THREE.Mesh(new THREE.SphereGeometry(r * size * 0.42, 10, 6), pupilMat);
      pupil.position.set(r * size * 0.75, 0, 0);
      eye.add(pupil);
      body.add(eye);
      // Brow slanting down towards the middle of the face.
      const brow = new THREE.Mesh(new THREE.BoxGeometry(r * 0.12, r * 0.1, r * 0.46), darkMat);
      brow.position.set(r * 0.93, r * (y + size + 0.06), side * r * 0.33);
      brow.rotation.x = -side * 0.5;
      body.add(brow);
    }
    // Crooked open mouth with jagged teeth.
    const mouth = new THREE.Group();
    mouth.position.set(r * 0.97, r * 1.42, r * 0.04);
    mouth.rotation.x = 0.18;
    const hole = new THREE.Mesh(new THREE.BoxGeometry(r * 0.12, r * 0.3, r * 0.95), new THREE.MeshBasicMaterial({ color: COLORS.mouth }));
    mouth.add(hole);
    const toothGeo = new THREE.ConeGeometry(r * 0.07, r * 0.16, 4);
    for (const [z, up] of [
      [-0.32, false],
      [-0.08, false],
      [0.2, false],
      [-0.2, true],
      [0.1, true],
      [0.33, true],
    ] as [number, boolean][]) {
      const tooth = new THREE.Mesh(toothGeo, toothMat);
      // Upper teeth hang down, lower teeth point up.
      tooth.rotation.x = up ? 0 : Math.PI;
      tooth.position.set(r * 0.05, (up ? -1 : 1) * r * 0.09, z * r);
      mouth.add(tooth);
    }
    body.add(mouth);
  }

  private ensureZombies(n: number): void {
    while (this.zombies.length < n) {
      const z = this.character(COLORS.zombie, true);
      this.zombies.push(z);
      this.zombieLayer.add(z.root);
    }
    this.zombies.forEach((z, i) => {
      z.root.visible = i < n;
    });
    this.applyScale();
  }

  /** Makes a zombie translucent (x-ray) or opaque again. */
  private setGhost(c: Character, ghost: boolean): void {
    if (!!c.ghost === ghost) return;
    c.ghost = ghost;
    c.root.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        const m = o.material as THREE.Material;
        m.transparent = ghost;
        m.opacity = ghost ? 0.3 : 1;
        m.depthWrite = !ghost;
        o.castShadow = !ghost;
      }
    });
  }

  private place(c: Character, x: number, y: number, hx: number, hy: number, moving: boolean, now: number, phase: number): void {
    c.root.position.copy(ground(x, y));
    if (hx !== 0 || hy !== 0) c.root.rotation.y = Math.atan2(hy, hx);
    // A small hop while walking.
    c.body.position.y = moving ? Math.abs(Math.sin(now / 140 + phase)) * 0.06 : 0;
  }

  // ---------------------------------------------------------------- dynamic state

  private updateFood(frame: SnapshotFrame): void {
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const up = new THREE.Vector3(0, 1, 0);
    const first = this.mode === "first";
    const scale = new THREE.Vector3().setScalar(first ? 1 : Math.min(2, this.bodyScale));
    // First person: only the packages the human sees.
    const list = first ? (frame.seen_food ?? []).map((k) => frame.food[k]) : frame.food;
    const n = Math.min(MAX_FOOD, list.length);
    for (let i = 0; i < n; i++) {
      const [x, y] = list[i];
      // Stable per-position yaw so packages do not spin as the list changes.
      q.setFromAxisAngle(up, ((x * 12.9898 + y * 78.233) % 1) * Math.PI * 2);
      m.compose(ground(x, y, 0.15 * scale.y), q, scale);
      this.food.setMatrixAt(i, m);
    }
    this.food.count = n;
    this.food.instanceMatrix.needsUpdate = true;
  }

  private addRing(kind: number, x: number, y: number, radius: number, now: number): void {
    const line = new THREE.LineLoop(
      this.ringGeometry,
      new THREE.LineBasicMaterial({ color: COLORS.noise[kind] ?? COLORS.noise[0], transparent: true }),
    );
    line.position.copy(ground(x, y, 0.08));
    line.scale.setScalar(0.01);
    this.scene.add(line);
    this.rings.push({ line, radius, born: now });
  }

  private disposeRing(ring: Ring): void {
    this.scene.remove(ring.line);
    (ring.line.material as THREE.Material).dispose();
  }

  private update(now: number): void {
    this.rings = this.rings.filter((ring) => {
      const k = (now - ring.born) / RING_MS;
      if (k >= 1) {
        this.disposeRing(ring);
        return false;
      }
      ring.line.scale.setScalar(Math.max(0.01, ring.radius * Math.min(1, k * 1.5)));
      (ring.line.material as THREE.LineBasicMaterial).opacity = 1 - k;
      // First person: all noise is the human's own; the HUD flashes it instead.
      ring.line.visible = this.mode !== "first";
      return true;
    });

    const init = this.init;
    const prev = this.prev;
    const curr = this.curr;
    this.human.root.visible = this.vision.visible = curr !== null;
    if (!init || !curr || !prev) return;
    const interval = curr.speed > 0 ? (init.decision_dt * 1000) / curr.speed : 0;
    const t = interval > 0 && !curr.paused ? Math.min(1, (now - this.currAt) / interval) : 1;
    const lerp = (a: number, b: number) => a + (b - a) * t;
    const animate = !curr.paused && t < 1;

    const p = prev.human;
    const c = curr.human;
    const hx = lerp(p.x, c.x);
    const hy = lerp(p.y, c.y);
    const humanMoving = animate && (p.x !== c.x || p.y !== c.y);
    this.place(this.human, hx, hy, c.hx, c.hy, humanMoving && this.endedAt === 0, now, 0);
    // The human tips over when the episode ends in a death.
    const dead = curr.outcome === "caught" || curr.outcome === "starved";
    const fall = dead && this.endedAt > 0 ? Math.min(1, (now - this.endedAt) / FALL_MS) : 0;
    this.human.body.rotation.x = fall * (Math.PI / 2);
    this.vision.position.copy(ground(hx, hy));
    this.vision.rotation.y = Math.atan2(c.hy, c.hx);
    const first = this.mode === "first";
    this.vision.visible = !dead && !first;
    this.human.root.visible = !first;

    const seen = first ? visibility(curr.zombies.length, curr.seen_zombies, this.xray) : null;
    curr.zombies.forEach((z, i) => {
      const pz = prev.zombies[i] ?? z;
      const moving = animate && (pz[0] !== z[0] || pz[1] !== z[1]);
      const zc = this.zombies[i];
      this.place(zc, lerp(pz[0], z[0]), lerp(pz[1], z[1]), z[2], z[3], moving, now, i * 1.7);
      zc.root.visible = seen ? seen[i] !== "hidden" : true;
      this.setGhost(zc, seen ? seen[i] === "xray" : false);
    });

    if (first) {
      this.firstPerson(prev, curr, hx, hy, t, humanMoving, now);
      this.sun.target.position.copy(ground(hx, hy));
      this.sun.position.copy(ground(hx, hy)).add(new THREE.Vector3(-30, 60, 25));
      return;
    }
    const follow = this.mode === "follow";
    const focus = follow ? ground(hx, hy) : this.controls.target.clone();
    if (follow) {
      // Move target and camera together, so the user's orbit angle and zoom are kept.
      const delta = focus.clone().sub(this.controls.target).multiplyScalar(0.2);
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
    // Shadows cover a window around the point of interest.
    this.sun.target.position.copy(focus);
    this.sun.position.copy(focus).add(new THREE.Vector3(-30, 60, 25));
  }

  /** Camera at the human's eyes: shortest-arc yaw over the decision interval, a small walking
   * bob, a fall with a red tint when caught, grey when starved, "survived" when truncated. */
  private firstPerson(
    prev: SnapshotFrame,
    curr: SnapshotFrame,
    hx: number,
    hy: number,
    t: number,
    moving: boolean,
    now: number,
  ): void {
    const from = headingYaw(prev.human.hx, prev.human.hy, this.yaw);
    const to = headingYaw(curr.human.hx, curr.human.hy, from);
    this.yaw = lerpYaw(from, to, t);
    const ended = this.endedAt > 0 ? Math.min(1, (now - this.endedAt) / FALL_MS) : 0;
    const caught = curr.outcome === "caught";
    const bob = moving && this.endedAt === 0 ? Math.sin(now / 170) * 0.03 : 0;
    const eye = caught ? EYE_M + (FALLEN_EYE_M - EYE_M) * ended : EYE_M;
    this.camera.position.copy(ground(hx, hy, eye + bob));
    // three's camera looks along -z; heading yaw a maps to rotation.y = a - pi/2.
    this.camera.rotation.set(caught ? 0.5 * ended : 0, this.yaw - Math.PI / 2, caught ? 0.6 * ended : 0, "YXZ");
    const o = this.overlay;
    if (curr.outcome === "running") {
      o.style.opacity = "0";
      o.textContent = "";
    } else if (caught) {
      o.style.background = "rgba(190, 20, 20, 0.45)";
      o.style.opacity = String(ended);
      o.textContent = "caught";
    } else if (curr.outcome === "starved") {
      o.style.background = "rgba(90, 90, 90, 0.6)";
      o.style.opacity = String(ended);
      o.textContent = "starved";
    } else {
      o.style.background = "transparent";
      o.style.opacity = "1";
      o.textContent = "survived";
    }
  }
}
