#!/usr/bin/env node
/*
 * sass-fields.mjs — derive sm_86 instruction field positions by differential probing.
 *
 * WHAT: emits pairs of PTX programs that differ in exactly one operand, then
 * XORs the resulting encodings. The bits that changed ARE the field.
 *
 * WHY: published reverse-engineering of NVIDIA's ISA exists but is partial,
 * version-specific, and unverifiable from where we sit. Rather than trust a
 * table someone else derived, we derive our own from the compiler in front of
 * us, on the exact architecture we target. If ptxas puts the destination
 * register somewhere other than where a blog post says, this finds out.
 *
 * The method is the same one that caught X58: vary one thing, and let the
 * difference name the mechanism. A field position guessed from a single sample
 * is indistinguishable from a coincidence; a field position confirmed across
 * eight register numbers is not.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: it does not decode semantics or modifier
 * bits it was not asked about. It answers exactly "which bits move when I change
 * this operand", which is what an encoder needs.
 *
 * Usage:
 *   node scripts/sass-fields.mjs [outfile]
 */
import { execFileSync } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const TOOLS = process.env.CUDA_TOOLS ?? "/home/ajax/bin/cudatools";
const PTXAS = join(TOOLS, "ptxas");
const NVDISASM = join(TOOLS, "nvdisasm");
const ARCH = "sm_86";

const PREAMBLE = `.version 7.5
.target ${ARCH}
.address_size 64
.visible .entry probe(.param .u64 p) {
  .reg .b64 %rd<8>;
  .reg .b32 %r<40>;
  .reg .f32 %f<40>;
  .reg .pred %p<8>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
`;

function encodings(dir, tag, body) {
  const ptx = join(dir, `${tag}.ptx`);
  const cubin = join(dir, `${tag}.cubin`);
  writeFileSync(ptx, `${PREAMBLE}${body}\n  ret;\n}\n`);
  execFileSync(PTXAS, ["-arch", ARCH, "-O0", "-o", cubin, ptx], { stdio: "pipe" });
  const text = execFileSync(NVDISASM, ["-c", "-hex", cubin], { encoding: "utf8" });

  const out = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const m = /\/\*([0-9a-f]{4})\*\/\s+(.*?);?\s*\/\* (0x[0-9a-f]{16}) \*\//.exec(lines[i]);
    if (!m) continue;
    const hi = /\/\* (0x[0-9a-f]{16}) \*\//.exec(lines[i + 1] ?? "");
    if (!hi) continue;
    out.push({ text: m[2].trim().replace(/\s+/g, " "), word: (BigInt(hi[1]) << 64n) | BigInt(m[3]) });
  }
  return out;
}

/* Bit indices set in v, as an ascending array. */
function bits(v) {
  const b = [];
  for (let i = 0n; i < 128n; i++) if ((v >> i) & 1n) b.push(Number(i));
  return b;
}

/* Contiguous runs, rendered as "hi:lo" or a bare index. */
function ranges(list) {
  if (!list.length) return [];
  const out = [];
  let lo = list[0], prev = list[0];
  for (const b of list.slice(1)) {
    if (b !== prev + 1) { out.push([lo, prev]); lo = b; }
    prev = b;
  }
  out.push([lo, prev]);
  return out.map(([a, b]) => (a === b ? `${a}` : `${b}:${a}`));
}

/*
 * A probe varies one operand across several values and reports the union of
 * bits that ever changed. `pick` selects which instruction in the program to
 * look at, since ptxas emits prologue and epilogue around our body.
 */
function differential(dir, name, values, makeBody, pick) {
  const samples = [];
  for (const v of values) {
    let insns;
    try {
      insns = encodings(dir, `${name}_${v}`, makeBody(v));
    } catch (e) {
      return { name, error: String(e.stderr ?? e).split("\n")[0] };
    }
    const ins = insns.find(pick);
    if (!ins) return { name, error: "target instruction not found in output" };
    samples.push({ v, text: ins.text, word: ins.word });
  }

  let changed = 0n;
  for (let i = 1; i < samples.length; i++) changed |= samples[0].word ^ samples[i].word;

  return {
    name,
    varied: values,
    bits: ranges(bits(changed)),
    samples: samples.map((s) => ({
      value: s.v,
      text: s.text,
      word: "0x" + s.word.toString(16).padStart(32, "0"),
    })),
  };
}

function main() {
  const dir = mkdtempSync(join(tmpdir(), "sass-fields-"));
  const results = [];

  try {
    /* Destination register of a MOV. Varying only the register number should
     * move exactly one contiguous field. */
    results.push(differential(dir, "mov_dst_reg", [2, 3, 4, 5, 8, 16],
      (r) => `mov.u32 %r${r}, 1; st.global.u32 [%rd2], %r${r};`,
      (i) => /^MOV R\d+, 0x1$/.test(i.text)));

    /* 32-bit immediate of a MOV. Powers of two isolate individual bits, so the
     * immediate field's position and width both fall out. */
    results.push(differential(dir, "mov_imm32", [1, 2, 4, 256, 65536, 16777216],
      (v) => `mov.u32 %r2, ${v}; st.global.u32 [%rd2], %r2;`,
      (i) => /^MOV R\d+, 0x[0-9a-f]+$/.test(i.text)));

    /* Source A register of an integer add. */
    results.push(differential(dir, "iadd_srcA", [2, 3, 5, 9],
      (r) => `mov.u32 %r${r}, %tid.x; add.s32 %r20, %r${r}, 7; st.global.u32 [%rd2], %r20;`,
      (i) => /^IADD3 R\d+, R\d+, 0x7/.test(i.text)));

    /* Immediate operand of an integer add. */
    results.push(differential(dir, "iadd_imm", [1, 2, 4, 1024],
      (v) => `mov.u32 %r2, %tid.x; add.s32 %r20, %r2, ${v}; st.global.u32 [%rd2], %r20;`,
      (i) => /^IADD3 R\d+, R\d+, 0x[0-9a-f]+/.test(i.text)));

    /* Special-register index in S2R: tid.x/y/z and ctaid.x/y/z. */
    results.push(differential(dir, "s2r_index",
      ["%tid.x", "%tid.y", "%tid.z", "%ctaid.x", "%ctaid.y"],
      (s) => `mov.u32 %r2, ${s}; st.global.u32 [%rd2], %r2;`,
      (i) => /^S2R R\d+, SR_/.test(i.text)));

    /* Constant-bank offset: kernel params live in c[0x0][...] on NVIDIA, so
     * this field is how every kernel argument is reached. */
    results.push(differential(dir, "const_offset", [0, 4, 8, 64],
      (o) => `ld.param.u64 %rd5, [p]; add.s64 %rd6, %rd5, ${o}; cvta.to.global.u64 %rd7, %rd6; st.global.u32 [%rd7], 1;`,
      (i) => /^MOV R\d+, c\[0x0\]\[0x[0-9a-f]+\]/.test(i.text)));

    /* Shared-memory store offset. */
    results.push(differential(dir, "sts_offset", [0, 4, 16, 128],
      (o) => `.shared .align 4 .b8 sm[1024];
        mov.u32 %r2, %tid.x; mov.u32 %r3, sm;
        ld.global.f32 %f1, [%rd2]; st.shared.f32 [%r3+${o}], %f1;`,
      (i) => /^STS /.test(i.text)));

    /* Predicate register used to guard a store. */
    results.push(differential(dir, "pred_reg", [1, 2, 3],
      (p) => `mov.u32 %r2, %tid.x; setp.lt.u32 %p${p}, %r2, 32; @%p${p} st.global.u32 [%rd2], %r2;`,
      (i) => /^@P\d+ STG/.test(i.text)));

  } finally {
    rmSync(dir, { recursive: true, force: true });
  }

  for (const r of results) {
    if (r.error) { console.error(`  ${r.name.padEnd(16)} ERROR: ${r.error}`); continue; }
    console.error(`  ${r.name.padEnd(16)} bits [${r.bits.join(", ")}]`);
    console.error(`      e.g. ${r.samples[0].text}`);
  }

  const out = process.argv[2] ?? "packages/helios/native/hephaestus/isa/sm86-fields.json";
  writeFileSync(out, JSON.stringify({ arch: ARCH, fields: results }, null, 2));
  console.error(`\n-> ${out}`);
}

main();
