/* AdamW's f16 weight shadow must equal castToF16 of the weight it just wrote.
 * The shadow packs two f16 per 32-bit word (F2FP), an even lane owning each
 * pair with its neighbour arriving by a down-shuffle — so a mispacked pair, a
 * wrong predicate, or a stale neighbour shows up as a bit difference here.
 * Usage: HELIOS_VIDMEM=1 node diff-adamw-shadow.mjs */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
const B = new NativeHeliosBackend(0);
let failed = 0;
for (const n of [128, 1280, 12288, 786432]) {
  const mk = (f) => B.fromArray(Array.from({length:n}, (_,i)=>f(i)), [n]);
  const param = mk(i=>Math.sin(i*0.7)*2);
  const grad  = mk(i=>Math.cos(i*1.3)*0.1);
  const m     = mk(i=>Math.sin(i*0.2)*0.05);
  const v     = mk(i=>0.01 + (i%7)*0.001);
  const shadow = B.zeros([n>>1], "f32");
  B.adamwShadow(param, grad, m, v, shadow, n, 0.9, 0.999, 1e-3, 1e-8, 0.01);
  // param is now updated in place; its f16 packing is the reference.
  const updated = Array.from(param.data);
  const ref = B.castF32ToF16 ? B.castF32ToF16(B.fromArray(updated,[n])) : null;
  const got = new Uint16Array(Float32Array.from(Array.from(shadow.data)).buffer);
  // Reference: pack updated[2i],updated[2i+1] as f16 the same way (via the cast kernel).
  const refPacked = ref ? new Uint16Array(Float32Array.from(Array.from(ref.data)).buffer) : null;
  let bad = -1;
  if (refPacked) { for (let i=0;i<n;i++) if (got[i]!==refPacked[i]) { bad=i; break; } }
  const label = `n=${n}`.padEnd(12);
  if (refPacked && bad<0) console.log(`  ${label} ok        shadow == castToF16(updated weight), ${n} f16 exact`);
  else if (!refPacked) console.log(`  ${label} SKIP (no castToF16)`);
  else { failed++; console.log(`  ${label} MISMATCH at f16 ${bad}: shadow ${got[bad]} vs ref ${refPacked[bad]}`); }
  for (const t of [param,grad,m,v,shadow]) B.releaseGpuTensor?.(t);
  B.finishStepOps?.();
}
console.log(failed===0 ? "\nadamw f16 shadow: exact" : `\nadamw f16 shadow: ${failed} WRONG`);
process.exit(failed===0?0:1);
