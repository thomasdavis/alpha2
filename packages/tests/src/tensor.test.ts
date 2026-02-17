import { describe, it, expect } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";

describe("CpuRefBackend", () => {
  const B = new CpuRefBackend();

  it("zeros", () => {
    const t = B.zeros([2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.data.length).toBe(6);
    expect(Array.from(t.data)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it("ones", () => {
    const t = B.ones([3]);
    expect(Array.from(t.data)).toEqual([1, 1, 1]);
  });

  it("fromArray", () => {
    const t = B.fromArray([1, 2, 3, 4], [2, 2]);
    expect(t.shape).toEqual([2, 2]);
    expect(Array.from(t.data)).toEqual([1, 2, 3, 4]);
  });

  it("add", () => {
    const a = B.fromArray([1, 2, 3], [3]);
    const b = B.fromArray([4, 5, 6], [3]);
    const c = B.add(a, b);
    expect(Array.from(c.data)).toEqual([5, 7, 9]);
  });

  it("matmul 2x2", () => {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    const a = B.fromArray([1, 2, 3, 4], [2, 2]);
    const b = B.fromArray([5, 6, 7, 8], [2, 2]);
    const c = B.matmul(a, b);
    expect(c.shape).toEqual([2, 2]);
    const vals = Array.from(c.data).map(v => Math.round(v));
    expect(vals).toEqual([19, 22, 43, 50]);
  });

  it("softmax sums to 1", () => {
    const x = B.fromArray([1, 2, 3], [1, 3]);
    const s = B.softmax(x, -1);
    const sum = Array.from(s.data).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it("gelu", () => {
    const x = B.fromArray([0, 1, -1], [3]);
    const g = B.gelu(x);
    expect((g.data as Float32Array)[0]).toBeCloseTo(0, 3);
    expect((g.data as Float32Array)[1]).toBeCloseTo(0.8413, 2);
  });

  it("embedding", () => {
    const weight = B.fromArray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [3, 2]);
    const indices = B.fromArray([0, 2], [2], "i32");
    const out = B.embedding(weight, indices);
    expect(out.shape).toEqual([2, 2]);
    expect((out.data as Float32Array)[0]).toBeCloseTo(0.1);
    expect((out.data as Float32Array)[1]).toBeCloseTo(0.2);
    expect((out.data as Float32Array)[2]).toBeCloseTo(0.5);
    expect((out.data as Float32Array)[3]).toBeCloseTo(0.6);
  });

  it("layerNorm", () => {
    const x = B.fromArray([1, 2, 3, 4], [2, 2]);
    const w = B.ones([2]);
    const b = B.zeros([2]);
    const out = B.layerNorm(x, w, b, 1e-5);
    expect(out.shape).toEqual([2, 2]);
    // Each row should be normalized: mean≈0, var≈1
    const d = out.data as Float32Array;
    expect(d[0] + d[1]).toBeCloseTo(0, 3);
    expect(d[2] + d[3]).toBeCloseTo(0, 3);
  });

  it("crossEntropy", () => {
    // Perfect prediction should give low loss
    const logits = B.fromArray([10, 0, 0, 0, 10, 0], [2, 3]);
    const targets = B.fromArray([0, 1], [2], "i32");
    const loss = B.crossEntropy(logits, targets);
    // crossEntropy returns a scalar (shape [])
    expect(loss.data.length).toBe(1);
    expect((loss.data as Float32Array)[0]).toBeLessThan(0.1);
  });

  it("causalMask", () => {
    const mask = B.causalMask(3);
    expect(mask.shape).toEqual([3, 3]);
    const d = mask.data as Float32Array;
    // Lower triangle + diagonal should be 0
    expect(d[0]).toBe(0); // [0,0]
    expect(d[3]).toBe(0); // [1,0]
    expect(d[4]).toBe(0); // [1,1]
    // Upper triangle should be -Infinity
    expect(d[1]).toBe(-Infinity); // [0,1]
    expect(d[2]).toBe(-Infinity); // [0,2]
  });

  it("transpose", () => {
    const a = B.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const t = B.transpose(a, 0, 1);
    expect(t.shape).toEqual([3, 2]);
    expect(Array.from(t.data).map(v => Math.round(v))).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("reshape", () => {
    const a = B.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = B.reshape(a, [3, 2]);
    expect(r.shape).toEqual([3, 2]);
    expect(Array.from(r.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("sum all", () => {
    const a = B.fromArray([1, 2, 3], [3]);
    const s = B.sum(a);
    expect((s.data as Float32Array)[0]).toBe(6);
  });

  it("sum axis", () => {
    const a = B.fromArray([1, 2, 3, 4], [2, 2]);
    const s = B.sum(a, 0);
    expect(s.shape).toEqual([2]);
    expect(Array.from(s.data).map(v => Math.round(v))).toEqual([4, 6]);
  });
});
