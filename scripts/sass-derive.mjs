#!/usr/bin/env node
/*
 * sass-derive.mjs — derive sm_86 field positions from the observed corpus.
 *
 * WHAT: reads the catalogue built by sass-catalogue.mjs, finds pairs of
 * instructions that are textually identical except in ONE operand, and XORs
 * their encodings. The bits that differ are that operand's field.
 *
 * WHY: the obvious approach -- write two PTX programs differing in one register
 * and diff them -- does not work, because ptxas performs its own register
 * allocation and collapses both to R0. The corpus, however, already contains
 * hundreds of instructions using different registers for structural reasons, so
 * the variation we need is sitting there already. We mine it instead of trying
 * to coerce the compiler.
 *
 * A field claimed from a single pair could be a coincidence. Every result here
 * reports how many independent pairs agreed, and disagreement is surfaced rather
 * than averaged away -- if two pairs that vary the same operand produce
 * different bit sets, that is a signal the operand is encoded in more than one
 * place (which happens: register fields interact with reuse-cache flags).
 *
 * Usage:
 *   node scripts/sass-derive.mjs [catalogue.json]
 */
import { readFileSync, writeFileSync } from "node:fs";

const CAT = process.argv[2] ?? "packages/helios/native/hephaestus/isa/sm86-catalogue.json";

/* Split "IADD3 R0, R0, 0x7, RZ" into mnemonic + operand list. Predication is
 * pulled off separately because it is its own field. */
function split(text) {
  const pred = /^(@!?U?P\d+)\s+/.exec(text);
  const rest = pred ? text.slice(pred[0].length) : text;
  const sp = rest.indexOf(" ");
  const head = sp < 0 ? rest : rest.slice(0, sp);
  const ops = sp < 0 ? [] : rest.slice(sp + 1).split(",").map((s) => s.trim());
  return { pred: pred ? pred[1] : null, head, ops };
}

function bitList(v) {
  const b = [];
  for (let i = 0n; i < 128n; i++) if ((v >> i) & 1n) b.push(Number(i));
  return b;
}

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

const word = (ins) => (BigInt(ins.hi) << 64n) | BigInt(ins.lo);

/*
 * Bits 105..127 carry the scheduling control field -- stall count, yield,
 * write/read barrier indices, wait mask and reuse flags. Those vary with an
 * instruction's POSITION in the program, not with its operands, so leaving them
 * in makes every operand diff look noisy: the first pass reported register
 * fields at only 10-50% agreement purely because of this.
 *
 * They are masked off here and studied separately by deriveControl(), because
 * they are a different kind of thing: an operand field is a property of the
 * instruction, the control field is a property of the schedule.
 */
const CONTROL_MASK = ((1n << 23n) - 1n) << 105n;
const operandWord = (ins) => word(ins) & ~CONTROL_MASK;

function main() {
  const cat = JSON.parse(readFileSync(CAT, "utf8"));

  /* Flatten to a unique instruction set — the same instruction appears in many
   * probes (every kernel has MOV R1, c[0x0][0x28] in its prologue) and counting
   * duplicates would inflate the agreement numbers. */
  const seen = new Map();
  for (const list of Object.values(cat.byMnemonic)) {
    for (const ins of list) if (!seen.has(ins.text + ins.lo + ins.hi)) seen.set(ins.text + ins.lo + ins.hi, ins);
  }
  const all = [...seen.values()];

  /* Group by (mnemonic, operand count, which operand differs). */
  const findings = new Map();

  for (let i = 0; i < all.length; i++) {
    const a = split(all[i].text);
    for (let j = i + 1; j < all.length; j++) {
      const b = split(all[j].text);
      if (a.head !== b.head || a.ops.length !== b.ops.length) continue;
      if (a.pred !== b.pred) continue;

      /* Exactly one operand position may differ. */
      let diffIdx = -1, diffs = 0;
      for (let k = 0; k < a.ops.length; k++) {
        if (a.ops[k] !== b.ops[k]) { diffIdx = k; diffs++; }
      }
      if (diffs !== 1) continue;

      /* Classify what kind of operand it is, so register fields and immediate
       * fields are not pooled together. */
      const va = a.ops[diffIdx], vb = b.ops[diffIdx];
      const kind =
        /^R\d+$|^RZ$/.test(va) && /^R\d+$|^RZ$/.test(vb) ? "reg"
        : /^0x[0-9a-f]+$/.test(va) && /^0x[0-9a-f]+$/.test(vb) ? "imm"
        : /^c\[/.test(va) && /^c\[/.test(vb) ? "const"
        : /^SR_/.test(va) && /^SR_/.test(vb) ? "sreg"
        : /^P\d+$|^PT$/.test(va) && /^P\d+$|^PT$/.test(vb) ? "pred"
        : "other";
      if (kind === "other") continue;

      const key = `${a.head}|${a.ops.length}|op${diffIdx}|${kind}`;
      const changed = operandWord(all[i]) ^ operandWord(all[j]);
      const entry = findings.get(key) ?? { key, mnemonic: a.head, operand: diffIdx, kind, pairs: 0, union: 0n, sets: new Map(), example: null };
      entry.pairs++;
      entry.union |= changed;
      const sig = ranges(bitList(changed)).join(",");
      entry.sets.set(sig, (entry.sets.get(sig) ?? 0) + 1);
      entry.example ??= `${all[i].text}  /  ${all[j].text}`;
      findings.set(key, entry);
    }
  }

  const rows = [...findings.values()]
    .filter((f) => f.pairs >= 2)          /* one pair is a coincidence */
    .sort((a, b) => b.pairs - a.pairs);

  console.error(`corpus: ${all.length} unique instructions\n`);
  console.error("mnemonic      op kind   pairs  field extent (union of changed bits)");
  console.error("------------  -- ----   -----  -----------------------------------");
  /*
   * Report the UNION of changed bits, not the most common pattern.
   *
   * An earlier version scored each field by how often pairs produced the
   * identical bit set, and read the resulting 10-50% as a weak signal. That was
   * the wrong statistic: R0 vs R5 and R0 vs R2 differ in different bits of the
   * SAME field, so disagreement between pairs is expected and carries no
   * information. What identifies the field is the extent every pair falls
   * inside -- the union.
   */
  for (const f of rows) {
    const u = ranges(bitList(f.union)).join(",");
    console.error(
      `${f.mnemonic.padEnd(13)} ${String(f.operand).padEnd(2)} ${f.kind.padEnd(6)} ${String(f.pairs).padStart(5)}  [${u}]`
    );
  }

  const out = "packages/helios/native/hephaestus/isa/sm86-derived-fields.json";
  writeFileSync(out, JSON.stringify({
    arch: cat.arch,
    corpusSize: all.length,
    fields: rows.map((f) => ({
      mnemonic: f.mnemonic, operand: f.operand, kind: f.kind, pairs: f.pairs,
      unionBits: ranges(bitList(f.union)),
      variants: [...f.sets.entries()].map(([bits, n]) => ({ bits, pairs: n })),
      example: f.example,
    })),
  }, null, 2));
  console.error(`\n-> ${out}`);
}

main();
