#!/usr/bin/env node
/*
 * sass-catalogue.mjs — build the sm_86 encoding ground truth.
 *
 * WHAT: compiles small PTX programs with ptxas, disassembles them with nvdisasm,
 * and records every (mnemonic, operands, 128-bit encoding) triple it sees.
 *
 * WHY: NVIDIA does not document SASS, so Hephaestus cannot be written against a
 * specification. It has to be written against observed reality, and this is the
 * observation. Every encoding Hephaestus emits is checked against this
 * catalogue; anything not in here is something we have never seen the vendor
 * compiler produce and should not be emitting.
 *
 * The important property: ptxas and nvdisasm are HOST tools. Neither needs a
 * GPU. So the hardest layer of the stack can be developed and validated on a box
 * with no graphics card at all, for free -- only submission needs hardware.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: it does not verify semantics. It records
 * what bit pattern the vendor emits for a given instruction, not what that
 * instruction computes. Semantics are established by running kernels on
 * hardware and checking known answers.
 *
 * On the soul constraint: these tools sit OUTSIDE the training loop, in the same
 * category as checkpoint conversion and dataset preparation, which GOAL.md
 * explicitly permits. Nothing produced here is shipped or linked -- the output
 * is a test fixture that our own assembler is measured against.
 *
 * Usage:
 *   node scripts/sass-catalogue.mjs [outfile]
 */
import { execFileSync } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const TOOLS = process.env.CUDA_TOOLS ?? "/home/ajax/bin/cudatools";
const PTXAS = join(TOOLS, "ptxas");
const NVDISASM = join(TOOLS, "nvdisasm");
const ARCH = "sm_86";

/*
 * Each probe is a PTX body exercising one family of operations. They are kept
 * deliberately small so the surrounding boilerplate (stack pointer setup, exit)
 * does not drown the instruction under test.
 *
 * The set is chosen from what Alpha's 154 kernels actually need, not from what
 * the ISA offers: integer and float ALU, FFMA, global/shared memory, special
 * registers, predication, branches and barriers.
 */
const PROBES = [
  ["mov_imm", `mov.u32 %r1, 12345; st.global.u32 [%rd2], %r1;`],
  ["tid", `mov.u32 %r1, %tid.x; st.global.u32 [%rd2], %r1;`],
  ["ctaid", `mov.u32 %r1, %ctaid.x; st.global.u32 [%rd2], %r1;`],
  ["ntid", `mov.u32 %r1, %ntid.x; st.global.u32 [%rd2], %r1;`],
  ["iadd", `mov.u32 %r1, %tid.x; add.s32 %r2, %r1, 7; st.global.u32 [%rd2], %r2;`],
  ["imul", `mov.u32 %r1, %tid.x; mul.lo.s32 %r2, %r1, 3; st.global.u32 [%rd2], %r2;`],
  ["imad", `mov.u32 %r1, %tid.x; mov.u32 %r2, %ctaid.x; mad.lo.s32 %r3, %r1, %r2, %r1; st.global.u32 [%rd2], %r3;`],
  ["shift", `mov.u32 %r1, %tid.x; shl.b32 %r2, %r1, 2; shr.u32 %r3, %r2, 1; st.global.u32 [%rd2], %r3;`],
  ["logic", `mov.u32 %r1, %tid.x; and.b32 %r2, %r1, 15; or.b32 %r3, %r2, 256; xor.b32 %r4, %r3, 1; st.global.u32 [%rd2], %r4;`],
  ["fadd", `ld.global.f32 %f1, [%rd2]; add.f32 %f2, %f1, 0f3F800000; st.global.f32 [%rd2], %f2;`],
  ["fmul", `ld.global.f32 %f1, [%rd2]; mul.f32 %f2, %f1, %f1; st.global.f32 [%rd2], %f2;`],
  ["ffma", `ld.global.f32 %f1, [%rd2]; fma.rn.f32 %f2, %f1, %f1, %f1; st.global.f32 [%rd2], %f2;`],
  ["fminmax", `ld.global.f32 %f1, [%rd2]; max.f32 %f2, %f1, 0f00000000; min.f32 %f3, %f2, 0f40000000; st.global.f32 [%rd2], %f3;`],
  ["frcp", `ld.global.f32 %f1, [%rd2]; rcp.approx.f32 %f2, %f1; st.global.f32 [%rd2], %f2;`],
  ["fsqrt", `ld.global.f32 %f1, [%rd2]; sqrt.approx.f32 %f2, %f1; st.global.f32 [%rd2], %f2;`],
  ["fexp", `ld.global.f32 %f1, [%rd2]; ex2.approx.f32 %f2, %f1; st.global.f32 [%rd2], %f2;`],
  ["cvt", `mov.u32 %r1, %tid.x; cvt.rn.f32.s32 %f1, %r1; cvt.rzi.s32.f32 %r2, %f1; st.global.u32 [%rd2], %r2;`],
  ["ldg_stg", `ld.global.f32 %f1, [%rd2]; st.global.f32 [%rd2+4], %f1;`],
  ["ldg_vec", `ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd2]; st.global.v4.f32 [%rd2], {%f4,%f3,%f2,%f1};`],
  ["shared", `.shared .align 4 .b8 sm[1024];
     mov.u32 %r1, %tid.x; shl.b32 %r2, %r1, 2;
     mov.u32 %r3, sm; add.s32 %r4, %r3, %r2;
     ld.global.f32 %f1, [%rd2]; st.shared.f32 [%r4], %f1;
     bar.sync 0;
     ld.shared.f32 %f2, [%r4]; st.global.f32 [%rd2], %f2;`],
  ["setp_pred", `mov.u32 %r1, %tid.x; setp.lt.u32 %p1, %r1, 32; @%p1 st.global.u32 [%rd2], %r1;`],
  ["branch", `mov.u32 %r1, %tid.x; setp.gt.u32 %p1, %r1, 4; @%p1 bra SKIP; st.global.u32 [%rd2], %r1; SKIP:`],
  ["f16", `.reg .b32 %h<3>; ld.global.b32 %h1, [%rd2]; add.rn.f16x2 %h2, %h1, %h1; st.global.b32 [%rd2], %h2;`],
  ["f16_fma", `.reg .b32 %h<4>; ld.global.b32 %h1, [%rd2]; fma.rn.f16x2 %h3, %h1, %h1, %h1; st.global.b32 [%rd2], %h3;`],
  ["atomic", `mov.u32 %r1, %tid.x; atom.global.add.u32 %r2, [%rd2], %r1; st.global.u32 [%rd2+4], %r2;`],
  ["shfl", `mov.u32 %r1, %tid.x; shfl.sync.down.b32 %r2|%p1, %r1, 16, 31, -1; st.global.u32 [%rd2], %r2;`],
  ["vote", `mov.u32 %r1, %tid.x; setp.lt.u32 %p1, %r1, 16; vote.sync.ballot.b32 %r2, %p1, -1; st.global.u32 [%rd2], %r2;`],

  /* Register-pressure probes. ptxas allocates registers itself, so varying a
   * PTX virtual register does NOT vary the SASS one -- everything collapses to
   * R0. These keep many values live simultaneously, which forces the allocator
   * up through R0..R16 and gives the field-derivation step real variety to
   * work with. Without these the corpus contains almost no register diversity
   * and every derived register field rests on one or two samples. */
  ["pressure_f32", `
     ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd2];
     ld.global.v4.f32 {%f5,%f6,%f7,%f8}, [%rd2+16];
     add.f32 %f9,  %f1, %f5; add.f32 %f10, %f2, %f6;
     add.f32 %f11, %f3, %f7; add.f32 %f12, %f4, %f8;
     mul.f32 %f13, %f9, %f10; mul.f32 %f14, %f11, %f12;
     fma.rn.f32 %f15, %f13, %f14, %f9;
     st.global.f32 [%rd2],    %f9;  st.global.f32 [%rd2+4],  %f10;
     st.global.f32 [%rd2+8],  %f11; st.global.f32 [%rd2+12], %f12;
     st.global.f32 [%rd2+16], %f13; st.global.f32 [%rd2+20], %f14;
     st.global.f32 [%rd2+24], %f15;`],
  ["pressure_i32", `
     mov.u32 %r1, %tid.x;   mov.u32 %r2, %ctaid.x;  mov.u32 %r3, %ntid.x;
     add.s32 %r4, %r1, %r2; add.s32 %r5, %r2, %r3;  add.s32 %r6, %r3, %r1;
     mul.lo.s32 %r7, %r4, %r5; mul.lo.s32 %r8, %r5, %r6;
     mad.lo.s32 %r9, %r7, %r8, %r4;
     shl.b32 %r10, %r9, 2;  and.b32 %r11, %r10, 255;
     st.global.u32 [%rd2],    %r4;  st.global.u32 [%rd2+4],  %r5;
     st.global.u32 [%rd2+8],  %r6;  st.global.u32 [%rd2+12], %r7;
     st.global.u32 [%rd2+16], %r8;  st.global.u32 [%rd2+20], %r9;
     st.global.u32 [%rd2+24], %r10; st.global.u32 [%rd2+28], %r11;`],
  ["pressure_mixed", `
     ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd2];
     mov.u32 %r1, %tid.x; cvt.rn.f32.s32 %f5, %r1;
     fma.rn.f32 %f6,  %f1, %f5, %f2;
     fma.rn.f32 %f7,  %f2, %f5, %f3;
     fma.rn.f32 %f8,  %f3, %f5, %f4;
     fma.rn.f32 %f9,  %f4, %f5, %f1;
     max.f32 %f10, %f6, %f7; min.f32 %f11, %f8, %f9;
     st.global.f32 [%rd2],   %f6;  st.global.f32 [%rd2+4],  %f7;
     st.global.f32 [%rd2+8], %f8;  st.global.f32 [%rd2+12], %f9;
     st.global.f32 [%rd2+16],%f10; st.global.f32 [%rd2+20], %f11;`],
];

const PREAMBLE = `.version 7.5
.target ${ARCH}
.address_size 64
.visible .entry probe(.param .u64 p) {
  .reg .b64 %rd<8>;
  .reg .b32 %r<16>;
  .reg .f32 %f<16>;
  .reg .pred %p<8>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
`;

function compile(dir, name, body) {
  const ptx = join(dir, `${name}.ptx`);
  const cubin = join(dir, `${name}.cubin`);
  writeFileSync(ptx, `${PREAMBLE}${body}\n  ret;\n}\n`);
  execFileSync(PTXAS, ["-arch", ARCH, "-o", cubin, ptx], { stdio: "pipe" });
  return execFileSync(NVDISASM, ["-c", "-hex", cubin], { encoding: "utf8" });
}

/*
 * nvdisasm -hex emits an instruction as a text line carrying the low 64 bits,
 * followed by a continuation line carrying the high 64 bits:
 *
 *   /_*0000*_/   MOV R1, c[0x0][0x28] ;   /_* 0x00000a0000017a02 *_/
 *                                         /_* 0x000fe40000000f00 *_/
 */
function parse(text) {
  const out = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const m = /\/\*([0-9a-f]{4})\*\/\s+(.*?);?\s*\/\* (0x[0-9a-f]{16}) \*\//.exec(lines[i]);
    if (!m) continue;
    const hiMatch = /\/\* (0x[0-9a-f]{16}) \*\//.exec(lines[i + 1] ?? "");
    if (!hiMatch) continue;
    out.push({
      addr: parseInt(m[1], 16),
      text: m[2].trim().replace(/\s+/g, " "),
      lo: m[3],
      hi: hiMatch[1],
    });
  }
  return out;
}

/* The mnemonic is the first token, minus predication and modifiers. */
const mnemonic = (t) => {
  const noPred = t.replace(/^@!?U?P\d+\s+/, "");
  return noPred.split(/[\s.]/)[0];
};

function main() {
  const dir = mkdtempSync(join(tmpdir(), "sass-cat-"));
  const catalogue = { arch: ARCH, probes: {}, byMnemonic: {} };
  let total = 0;

  try {
    for (const [name, body] of PROBES) {
      let insns;
      try {
        insns = parse(compile(dir, name, body));
      } catch (e) {
        console.error(`  ${name.padEnd(12)} PTXAS FAILED: ${String(e.stderr ?? e).split("\n")[0]}`);
        continue;
      }
      catalogue.probes[name] = insns;
      total += insns.length;
      for (const ins of insns) {
        const m = mnemonic(ins.text);
        (catalogue.byMnemonic[m] ??= []).push({ text: ins.text, lo: ins.lo, hi: ins.hi, probe: name });
      }
      console.error(`  ${name.padEnd(12)} ${String(insns.length).padStart(3)} instructions`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }

  const mnemonics = Object.keys(catalogue.byMnemonic).sort();
  catalogue.summary = { totalInstructions: total, distinctMnemonics: mnemonics.length, mnemonics };

  const out = process.argv[2] ?? "packages/helios/native/hephaestus/isa/sm86-catalogue.json";
  writeFileSync(out, JSON.stringify(catalogue, null, 2));
  console.error(`\n${total} instructions, ${mnemonics.length} distinct mnemonics -> ${out}`);
  console.error(mnemonics.join(" "));
}

main();
