// Frames and API payloads produced by zai-server (crates/zai-server/src/frames.rs, api.rs).

export interface HouseFrame {
  x: number;
  y: number;
  angle: number;
  walls: [number, number, number, number][];
  doors: [number, number, number, number][];
}

export interface InitFrame {
  type: "init";
  source: "live" | "replay";
  id: string;
  episode: number;
  seed: number;
  total_ticks: number | null;
  decision_dt: number;
  config: {
    map_size_m: number;
    entity_radius_m: number;
    human: { vision_radius_m: number; vision_fov_deg: number; hunger_interval_s: number };
    zombie: { hearing_radius_m: number };
  };
  map_size: number;
  houses: HouseFrame[];
}

export interface SnapshotFrame {
  type: "snapshot";
  tick: number;
  time: number;
  speed: number;
  paused: boolean;
  human: { x: number; y: number; hx: number; hy: number; carried: number; hunger: number };
  /** `[x, y, heading_x, heading_y]` per zombie. */
  zombies: [number, number, number, number][];
  doors: number;
  food: [number, number][];
  /** [kind (0 walk, 1 pickup, 2 door), x, y, radius] */
  events: [number, number, number, number][];
  outcome: "running" | "caught" | "starved" | "truncated";
}

export type Frame = InitFrame | SnapshotFrame;

/** A side of a match: a built-in bot or a trained model (`GET /api/models`). */
export type Agent = "scripted" | "random" | { policy: string };

export interface ModelInfo {
  id: string;
  side: "human" | "zombie";
  run_id: string;
  iteration: number;
  steps: number;
  /** Env config file the model was trained on (`custom` if none matched). */
  stage: string;
  hidden: number;
  obs_dim: number;
  /** Observation layout the model was trained on (1 before M6.1). */
  obs_version: number;
  created_at: number;
  config: Record<string, unknown>;
}

export interface MatchInfo {
  id: string;
  human: Agent;
  zombies: Agent;
  deterministic: boolean;
  episode: number;
  tick: number;
  time: number;
  speed: number;
  paused: boolean;
  status: "running" | "finished" | "stopped" | "failed";
  outcome: string;
  replays: string[];
}

export interface ReplaySummary {
  id: string;
  match_id: string;
  episode: number;
  seed: number;
  human: string;
  zombies: string;
  human_model?: string;
  zombie_model?: string;
  outcome: string;
  length: number;
}

export interface MetricPoint {
  model: string;
  key: string;
  step: number;
  wall_time: number;
  value: number;
}

export interface RunSummary {
  run_id: string;
  models: string[];
  last_step: number;
  updated: number;
}
