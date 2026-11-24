import { decode } from "@msgpack/msgpack";

import type { Frame, MatchInfo, MetricPoint, ReplaySummary, RunSummary } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const listMatches = () => fetch("/api/matches").then((r) => json<MatchInfo[]>(r));
export const listReplays = () => fetch("/api/replays").then((r) => json<ReplaySummary[]>(r));
export const listRuns = () => fetch("/api/runs").then((r) => json<RunSummary[]>(r));
export const listMetrics = (runId: string) =>
  fetch(`/api/metrics?run_id=${encodeURIComponent(runId)}`).then((r) => json<MetricPoint[]>(r));

export function createMatch(body: Record<string, unknown>): Promise<MatchInfo> {
  return fetch("/api/matches", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<MatchInfo>(r));
}

export function stopMatch(id: string): Promise<Response> {
  return fetch(`/api/matches/${id}/control`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ stop: true }),
  });
}

export type Command =
  | { cmd: "speed"; value: number }
  | { cmd: "pause" }
  | { cmd: "play" }
  | { cmd: "seek"; tick: number };

/** WebSocket to a live match or a replay; frames are MessagePack, commands JSON. */
export class Stream {
  private ws: WebSocket;

  constructor(path: string, onFrame: (f: Frame) => void, onClose: () => void) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this.ws = new WebSocket(`${proto}//${location.host}${path}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (ev) => onFrame(decode(new Uint8Array(ev.data as ArrayBuffer)) as Frame);
    this.ws.onclose = onClose;
  }

  send(c: Command): void {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(c));
  }

  close(): void {
    this.ws.onclose = null;
    this.ws.close();
  }
}
