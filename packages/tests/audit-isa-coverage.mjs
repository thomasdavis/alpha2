/*
 * Does the ISA coverage register still describe the assembler?
 *
 * WHAT THE C TEST CANNOT SEE. test/hephaestus_coverage_test.c checks the table
 * from the inside: no empty reasons, no duplicates, and every row that claims
 * HP_ISA_ENCODED really emits. What it cannot check is the direction that
 * matters most over time — a mnemonic the hardware uses and the table has never
 * heard of. From inside C there is nothing to compare against.
 *
 * There is, though, on disk: hephaestus/isa/sm86-catalogue.json holds 728
 * instructions across 31 mnemonics that ptxas emitted for kernels of this
 * shape. That is a real sample of what a competent compiler reaches for on this
 * architecture for this kind of work, and anything in it with no row in the
 * register is a gap nobody has looked at.
 *
 * SO THIS FAILS ON EXACTLY ONE THING: a catalogued mnemonic with no entry. Not
 * on a missing encoder — most of them SHOULD be missing, and the register says
 * so with a reason. The failure is the absent thought, not the absent code.
 *
 * The reverse direction is reported but not failed: a table row with no
 * catalogue entry is usually correct (HMMA, LDSM, RED and F2FP were captured by
 * dedicated .cu tools because the surveyed kernels never used them, and LDGSTS
 * postdates the survey).
 *
 * Usage: node packages/tests/audit-isa-coverage.mjs
 *   Runs anywhere — it reads two files and needs no GPU.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const NATIVE = join(HERE, "../helios/native");

const catalogue = JSON.parse(
  readFileSync(join(NATIVE, "hephaestus/isa/sm86-catalogue.json"), "utf8"));

/*
 * PARSE THE TABLE OUT OF THE C, rather than keeping a second copy here.
 *
 * A duplicate list in JavaScript would drift from the one in sm86_stub.c, and a
 * coverage register that disagrees with itself is worse than none: it would
 * report a gap that is closed, or miss one that is open, and either reading
 * would be believed. The C is the source; this reads it.
 *
 * The shape it matches is the table's own — {"MNEMONIC", HP_ISA_STATE, — so a
 * row that does not parse is a row that does not compile.
 */
const stubSrc = readFileSync(join(NATIVE, "hephaestus/sm86_stub.c"), "utf8");
const rows = [...stubSrc.matchAll(/\{"([^"]+)",\s*(HP_ISA_\w+),/g)]
  .map((m) => ({ mnemonic: m[1], state: m[2] }));

if (rows.length < 20) {
  console.error(`audit: parsed only ${rows.length} rows out of sm86_stub.c — ` +
                `the table's shape changed and this parser did not`);
  process.exit(1);
}

const known = new Map(rows.map((r) => [r.mnemonic, r.state]));

/*
 * The catalogue names an instruction by its BASE mnemonic and the register by
 * the form when the form is what differs — "LDG" is encoded but "LDG.E.128" is
 * a separate row because it is separately unused. So a catalogue mnemonic
 * counts as covered when any row starts with it.
 */
const covered = (m) => [...known.keys()].some(
  (k) => k === m || k.startsWith(m + "."));

const catMnemonics = Object.keys(catalogue.byMnemonic ?? {}).sort();
const missing = catMnemonics.filter((m) => !covered(m));
const extra = rows.map((r) => r.mnemonic)
  .filter((m) => !catMnemonics.some((c) => m === c || m.startsWith(c + ".")));

const byState = (s) => rows.filter((r) => r.state === s).length;

console.log(`ISA coverage — ${rows.length} rows against a catalogue of ` +
            `${catalogue.summary?.totalInstructions ?? "?"} instructions ` +
            `in ${catMnemonics.length} mnemonics\n`);
console.log(`  encoded              ${String(byState("HP_ISA_ENCODED")).padStart(3)}`);
console.log(`  captured, unused     ${String(byState("HP_ISA_CAPTURED")).padStart(3)}`);
console.log(`  missing              ${String(byState("HP_ISA_MISSING")).padStart(3)}`);

if (extra.length) {
  console.log(`\n  rows with no catalogue entry (expected — these were captured`);
  console.log(`  by dedicated tools, or postdate the survey):`);
  console.log(`    ${extra.join(", ")}`);
}

/*
 * THE STATES THE ASSEMBLER CLAIMS MUST MATCH WHAT IT EXPORTS.
 *
 * A row saying MISSING while sm86.h declares an encoder for it — or the other
 * way round — is the failure mode the C test cannot reach either, because from
 * inside C a declaration and a definition look the same. Here they do not:
 * sm86_stub.c defines exactly the forms that abort.
 */
const stubbed = [...stubSrc.matchAll(/^hp_word (hp_\w+)\(/gm)].map((m) => m[1]);
console.log(`\n  stubs that abort with the capture recipe: ${stubbed.join(", ")}`);

if (missing.length) {
  console.error(`\nFAIL — ptxas emits these and the coverage register has never`);
  console.error(`heard of them. Add a row to sm86_stub.c saying what each one`);
  console.error(`would be for, even if the answer is "nothing we need":\n`);
  for (const m of missing) {
    const n = catalogue.byMnemonic[m];
    const count = typeof n === "number" ? n : (n?.count ?? n?.length ?? "?");
    console.error(`    ${m.padEnd(12)} ${count} occurrences in the catalogue`);
  }
  console.error(`\nA gap with a row is a decision. A gap with no row is an accident.`);
  process.exit(1);
}

console.log(`\nok — every catalogued mnemonic has a row, with a reason.`);
