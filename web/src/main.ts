import "./style.css";

import {
  createMatch,
  listMatches,
  listModels,
  listReplays,
  listRuns,
  stopMatch,
  Stream,
} from "./api";
import { Charts } from "./charts";
import { TopDown } from "./topdown";
import type { Agent, Frame, InitFrame, ModelInfo, ReplaySummary, SnapshotFrame } from "./types";

const SPEEDS = [0.5, 1, 2, 4, 8, 16];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const view = new TopDown();
let charts: Charts;
let stream: Stream | null = null;
let init: InitFrame | null = null;
let last: SnapshotFrame | null = null;
let selected: string | null = null;
let seeking = false;
/** Training run whose recorded episodes are shown above the stage/charts. */
let currentRun: string | null = null;
let replays: ReplaySummary[] = [];
let models: ModelInfo[] = [];
let lastModels = "";

const agentName = (a: Agent) => (typeof a === "string" ? a : a.policy);
const POLICY = "policy:";

/** Adds the trained models of each side to the agent selects and their stages to the map
 * select, keeping the current choices. */
function renderModelOptions(): void {
  const form = $<HTMLFormElement>("new-match");
  for (const [name, side] of [
    ["human", "human"],
    ["zombies", "zombie"],
  ] as const) {
    const select = form.elements.namedItem(name) as HTMLSelectElement;
    const current = select.value;
    select.querySelector("optgroup")?.remove();
    const own = models.filter((m) => m.side === side);
    if (own.length > 0) {
      const group = document.createElement("optgroup");
      group.label = "trained";
      for (const m of own) {
        const o = new Option(`${m.id} · it ${m.iteration} · ${m.stage}`, POLICY + m.id);
        group.appendChild(o);
      }
      select.appendChild(group);
    }
    select.value = [...select.options].some((o) => o.value === current) ? current : "scripted";
  }
  const stage = form.elements.namedItem("stage") as HTMLSelectElement;
  const current = stage.value;
  stage.replaceChildren(new Option("server default", ""));
  for (const s of [...new Set(models.map((m) => m.stage))].sort()) {
    stage.appendChild(new Option(s, s));
  }
  stage.value = [...stage.options].some((o) => o.value === current) ? current : "";
  updateFormNote();
}

function selectedModel(name: "human" | "zombies"): ModelInfo | undefined {
  const form = $<HTMLFormElement>("new-match");
  const v = (form.elements.namedItem(name) as HTMLSelectElement).value;
  return v.startsWith(POLICY) ? models.find((m) => m.id === v.slice(POLICY.length)) : undefined;
}

/** Warns when a trained model plays on a map other than the one it was trained on. */
function updateFormNote(): void {
  const form = $<HTMLFormElement>("new-match");
  const stage = (form.elements.namedItem("stage") as HTMLSelectElement).value;
  const off = [selectedModel("human"), selectedModel("zombies")].filter(
    (m): m is ModelInfo => m !== undefined && m.stage !== stage,
  );
  const note = $("form-note");
  note.hidden = off.length === 0;
  note.textContent = off.map((m) => `${m.id} was trained on ${m.stage}`).join("; ");
}

/** Picking a trained model switches the map to its training stage (zombies take priority). */
function onAgentChange(): void {
  const form = $<HTMLFormElement>("new-match");
  const m = selectedModel("zombies") ?? selectedModel("human");
  if (m) (form.elements.namedItem("stage") as HTMLSelectElement).value = m.stage;
  updateFormNote();
}

function agentValue(v: string): Agent {
  return v.startsWith(POLICY) ? { policy: v.slice(POLICY.length) } : (v as Agent);
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function onFrame(frame: Frame): void {
  if (frame.type === "init") {
    init = frame;
    view.setInit(frame);
    const seek = $<HTMLInputElement>("seek");
    seek.hidden = frame.source !== "replay";
    seek.max = String(frame.total_ticks ?? 0);
    $("hud").hidden = false;
    return;
  }
  last = frame;
  view.pushSnapshot(frame);
  renderControls();
}

function renderControls(): void {
  if (!init || !last) return;
  const s = last;
  const hungerInterval = init.config.human.hunger_interval_s;
  $("hud-time").textContent = `${formatTime(s.time)} (tick ${s.tick})`;
  $("hud-hunger").style.width = `${Math.max(0, Math.min(1, s.human.hunger / hungerInterval)) * 100}%`;
  $("hud-hunger-s").textContent = `${s.human.hunger.toFixed(0)} s`;
  $("hud-carried").textContent = String(s.human.carried);
  $("hud-outcome").textContent = s.outcome;
  const kind = init.source === "replay" ? "Replay" : "Match";
  $("status").textContent = `${kind} ${init.id} · episode ${init.episode} · ${s.speed > 0 ? `x${s.speed}` : "max"}${s.paused ? " · paused" : ""}`;
  const btn = $<HTMLButtonElement>("play-pause");
  btn.disabled = false;
  btn.textContent = s.paused ? "Play" : "Pause";
  for (const b of document.querySelectorAll<HTMLButtonElement>("#speeds button")) {
    b.classList.toggle("active", Number(b.dataset.speed) === s.speed);
  }
  if (!seeking) $<HTMLInputElement>("seek").value = String(s.tick);
}

function showStage(visible: boolean): void {
  $("stage").hidden = !visible;
  for (const id of ["play-pause", "speeds"]) $(id).hidden = !visible;
  if (!visible) {
    $("hud").hidden = true;
    $<HTMLInputElement>("seek").hidden = true;
  }
}

function showRun(runId: string): void {
  stream?.close();
  stream = null;
  selected = `t:${runId}`;
  currentRun = runId;
  renderRunReplays();
  init = last = null;
  showStage(false);
  $("status").textContent = `Training run ${runId}`;
  void charts.show(runId).catch((e) => ($("status").textContent = (e as Error).message));
  void refreshLists();
}

function watch(path: string, key: string, run: string | null = null): void {
  stream?.close();
  charts.hide();
  currentRun = run;
  showStage(true);
  selected = key;
  init = last = null;
  stream = new Stream(path, onFrame, () => {
    $("status").textContent = "Disconnected";
  });
  renderRunReplays();
  void refreshLists();
}

/** Episodes a trainer recorded during `currentRun` (replays with `match_id === run_id`),
 * one chip per episode in iteration order; clicking one plays it and keeps the strip. */
function renderRunReplays(): void {
  const el = $("run-replays");
  const run = currentRun;
  const own = run ? replays.filter((r) => r.match_id === run) : [];
  el.hidden = own.length === 0;
  if (!run || own.length === 0) {
    el.replaceChildren();
    return;
  }
  own.sort((a, b) => a.episode - b.episode || a.id.localeCompare(b.id));
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = "Episodes recorded during training";
  const chips = own.map((r) => {
    const b = document.createElement("button");
    const key = `r:${r.id}`;
    const k = r.id.match(/-e(\d+)$/)?.[1] ?? "0";
    b.className = `chip ${r.outcome}`;
    b.classList.toggle("active", selected === key);
    b.textContent = `it ${r.episode} · #${k} · ${r.outcome}`;
    b.title = `${r.id}: ${r.human_model ?? r.human} human vs ${r.zombie_model ?? r.zombies} zombies, ${r.length} decisions`;
    b.onclick = () => watch(`/ws/replay/${r.id}`, key, run);
    return b;
  });
  const back = document.createElement("button");
  back.className = "chip";
  back.textContent = "Curves";
  back.classList.toggle("active", selected === `t:${run}`);
  back.onclick = () => showRun(run);
  el.replaceChildren(label, back, ...chips);
}

let lastLists = "";

function item(title: string, detail: string, key: string, onOpen: () => void): HTMLLIElement {
  const li = document.createElement("li");
  li.classList.toggle("selected", selected === key);
  const open = document.createElement("button");
  open.className = "item";
  const name = document.createElement("span");
  name.textContent = title;
  const small = document.createElement("small");
  small.textContent = detail;
  open.append(name, small);
  open.onclick = onOpen;
  li.appendChild(open);
  return li;
}

async function refreshLists(): Promise<void> {
  try {
    const [matches, allReplays, runs, allModels] = await Promise.all([
      listMatches(),
      listReplays(),
      listRuns(),
      listModels(),
    ]);
    const modelSignature = JSON.stringify(allModels.map((m) => [m.id, m.created_at]));
    if (modelSignature !== lastModels) {
      lastModels = modelSignature;
      models = allModels;
      renderModelOptions();
    }
    // Re-render only on change, so clicks are not lost to a DOM swap.
    const signature = JSON.stringify([matches, allReplays, runs, selected, currentRun]);
    if (signature === lastLists) return;
    lastLists = signature;
    replays = allReplays;
    renderRunReplays();
    // Episodes recorded by trainers are listed under their run, not in the sidebar.
    const runIds = new Set(runs.map((r) => r.run_id));
    const perRun = new Map<string, number>();
    for (const r of allReplays) {
      if (runIds.has(r.match_id)) perRun.set(r.match_id, (perRun.get(r.match_id) ?? 0) + 1);
    }
    $("matches").replaceChildren(
      ...matches.map((m) => {
        const key = `m:${m.id}`;
        const vs = `${agentName(m.human)} vs ${agentName(m.zombies)}`;
        const li = item(m.id, `${m.status} · ep ${m.episode} · ${m.outcome} · ${vs}`, key, () =>
          watch(`/ws/match/${m.id}`, key),
        );
        if (m.status === "running") {
          const stop = document.createElement("button");
          stop.textContent = "Stop";
          stop.className = "small";
          stop.onclick = () => void stopMatch(m.id).then(refreshLists);
          li.appendChild(stop);
        }
        return li;
      }),
    );
    $("replays").replaceChildren(
      ...allReplays.filter((r) => !runIds.has(r.match_id)).map((r) => {
        const key = `r:${r.id}`;
        const vs = `${r.human_model ?? r.human}/${r.zombie_model ?? r.zombies}`;
        return item(r.id, `${r.outcome} · ${r.length} ticks · ${vs}`, key, () =>
          watch(`/ws/replay/${r.id}`, key),
        );
      }),
    );
    $("runs").replaceChildren(
      ...runs.map((r) =>
        item(
          r.run_id,
          `${r.models.join(", ")} · step ${r.last_step.toLocaleString()}${
            perRun.has(r.run_id) ? ` · ${perRun.get(r.run_id)} episodes` : ""
          }`,
          `t:${r.run_id}`,
          () =>
          showRun(r.run_id),
        ),
      ),
    );
  } catch (e) {
    $("status").textContent = `Server unavailable: ${(e as Error).message}`;
  }
}

function setupControls(): void {
  const speeds = $("speeds");
  for (const s of SPEEDS) {
    const b = document.createElement("button");
    b.textContent = `x${s}`;
    b.dataset.speed = String(s);
    b.onclick = () => stream?.send({ cmd: "speed", value: s });
    speeds.appendChild(b);
  }
  const toggle = () => stream?.send(last?.paused ? { cmd: "play" } : { cmd: "pause" });
  $("play-pause").onclick = toggle;
  document.addEventListener("keydown", (ev) => {
    if (ev.code === "Space" && !(ev.target instanceof HTMLInputElement)) {
      ev.preventDefault();
      toggle();
    }
  });
  const seek = $<HTMLInputElement>("seek");
  seek.oninput = () => (seeking = true);
  seek.onchange = () => {
    seeking = false;
    stream?.send({ cmd: "seek", tick: Number(seek.value) });
  };

  const form = $<HTMLFormElement>("new-match");
  for (const name of ["human", "zombies"]) {
    (form.elements.namedItem(name) as HTMLSelectElement).onchange = onAgentChange;
  }
  (form.elements.namedItem("stage") as HTMLSelectElement).onchange = updateFormNote;
  form.onsubmit = async (ev) => {
    ev.preventDefault();
    const data = new FormData(form);
    // A stage is the full env config of a training run; count fields override it.
    const stage = String(data.get("stage") ?? "");
    const base = models.find((m) => m.stage === stage && stage !== "")?.config ?? {};
    const config: Record<string, unknown> = { ...base };
    for (const key of ["zombie_count", "food_count"]) {
      const v = String(data.get(key) ?? "");
      if (v !== "") config[key] = Number(v);
    }
    const seed = String(data.get("seed") ?? "");
    try {
      $("form-error").textContent = "";
      const m = await createMatch({
        human: agentValue(String(data.get("human"))),
        zombies: agentValue(String(data.get("zombies"))),
        deterministic: data.get("deterministic") === "on",
        auto_restart: data.get("auto_restart") === "on",
        config,
        ...(seed !== "" ? { seed: Number(seed) } : {}),
      });
      watch(`/ws/match/${m.id}`, `m:${m.id}`);
    } catch (e) {
      $("form-error").textContent = (e as Error).message;
    }
  };
}

async function main(): Promise<void> {
  await view.mount($("stage"));
  charts = new Charts($("charts"));
  setupControls();
  await refreshLists();
  setInterval(refreshLists, 3000);
}

void main();
