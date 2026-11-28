/** Pure helpers of the first-person camera (docs/ARCHITECTURE.md 4.9.6), unit-tested with
 * vitest (`firstperson.test.ts`). */

/** three.js `fov` is vertical: the vertical FOV, in degrees, that gives a horizontal FOV of
 * `horizontalDeg` at this aspect ratio, capped at `capDeg` on narrow windows. */
export function verticalFov(horizontalDeg: number, aspect: number, capDeg = 100): number {
  const half = (horizontalDeg * Math.PI) / 360;
  const v = (2 * Math.atan(Math.tan(half) / Math.max(aspect, 1e-6)) * 180) / Math.PI;
  return Math.min(capDeg, v);
}

/** Interpolates yaw (radians) from `a` to `b` along the shorter arc, `t` in [0, 1]. */
export function lerpYaw(a: number, b: number, t: number): number {
  let d = (b - a) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

/** Yaw of a heading vector (world x right, y up), or `fallback` for a zero heading. */
export function headingYaw(hx: number, hy: number, fallback: number): number {
  return hx === 0 && hy === 0 ? fallback : Math.atan2(hy, hx);
}

/** Per entity: drawn normally, translucent (x-ray) or hidden, given the seen indices. */
export function visibility(
  count: number,
  seen: readonly number[] | undefined,
  xray: boolean,
): ("seen" | "xray" | "hidden")[] {
  const s = new Set(seen ?? []);
  return Array.from({ length: count }, (_, i) => (s.has(i) ? "seen" : xray ? "xray" : "hidden"));
}

/** Compass label of a yaw (radians, 0 = east, counter-clockwise). */
export function compass(yaw: number): string {
  const names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];
  const k = Math.round(yaw / (Math.PI / 4));
  return names[((k % 8) + 8) % 8];
}
