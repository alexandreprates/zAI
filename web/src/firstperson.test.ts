import { describe, expect, it } from "vitest";

import { compass, headingYaw, lerpYaw, verticalFov, visibility } from "./firstperson";

describe("verticalFov", () => {
  it("gives the 120 degree horizontal cone at 16:9 and caps narrow windows", () => {
    expect(verticalFov(120, 16 / 9)).toBeCloseTo(88.5, 0);
    expect(verticalFov(120, 1)).toBe(100);
    expect(verticalFov(90, 1, 120)).toBeCloseTo(90, 6);
  });
});

describe("lerpYaw", () => {
  it("turns along the shorter arc across the wrap", () => {
    const a = Math.PI - 0.1;
    const b = -Math.PI + 0.1;
    expect(lerpYaw(a, b, 0.5)).toBeCloseTo(Math.PI, 6);
    expect(lerpYaw(0, Math.PI / 2, 0.5)).toBeCloseTo(Math.PI / 4, 6);
    expect(lerpYaw(0.3, 0.3 + 4 * Math.PI, 1)).toBeCloseTo(0.3, 6);
  });
  it("keeps the previous yaw for a zero heading", () => {
    expect(headingYaw(0, 0, 1.2)).toBe(1.2);
    expect(headingYaw(0, 1, 0)).toBeCloseTo(Math.PI / 2, 6);
  });
});

describe("visibility", () => {
  it("hides unseen entities unless x-ray is on", () => {
    expect(visibility(3, [2, 0], false)).toEqual(["seen", "hidden", "seen"]);
    expect(visibility(2, undefined, true)).toEqual(["xray", "xray"]);
  });
});

describe("compass", () => {
  it("names the eight directions", () => {
    expect(compass(0)).toBe("E");
    expect(compass(Math.PI / 2)).toBe("N");
    expect(compass(-Math.PI / 2)).toBe("S");
    expect(compass(Math.PI)).toBe("W");
  });
});
