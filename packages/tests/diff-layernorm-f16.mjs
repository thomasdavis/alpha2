/* layerNormF16's packed output must equal castF32ToF16(layerNorm(x)) — the
 * SHFL-pack f16 store, reused from the adamw shadow, in the norm epilogue.
 * Usage: HELIOS_VIDMEM=1 node diff-layernorm-f16.mjs */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
const B = new NativeHeliosBackend(0);
let failed = 0;
for (const [rows,width] of [[1536,640],[512,640],[64,128],[8,64]]) {
  const n=rows*width, eps=1e-5;
  const x=B.fromArray(Array.from({length:n},(_,i)=>Math.sin(i*0.7)*3+(i%13)-6),[rows,width]);
  const w=B.fromArray(Array.from({length:width},(_,i)=>0.5+Math.cos(i*0.3)*0.4),[width]);
  const b=B.fromArray(Array.from({length:width},(_,i)=>Math.sin(i*0.2)*0.1),[width]);
  const f32y = B.layerNorm(x,w,b,eps);
  const ref  = B.castF32ToF16(f32y);                 // packed f16 of the f32 result
  const f16y = B.layerNormF16(x,w,b,eps);             // fused packed f16
  const g = new Uint16Array(Float32Array.from(Array.from(f16y.data)).buffer);
  const r = new Uint16Array(Float32Array.from(Array.from(ref.data)).buffer);
  let bad=-1; for(let i=0;i<n;i++) if(g[i]!==r[i]){bad=i;break;}
  const label=`[${rows}x${width}]`.padEnd(14);
  if(bad<0) console.log(`  ${label} ok        ${n} f16 == castF32ToF16(layerNorm)`);
  else { failed++; console.log(`  ${label} MISMATCH at f16 ${bad}: fused ${g[bad]} vs ref ${r[bad]}`); }
  for(const t of [x,w,b]) B.releaseGpuTensor?.(t); B.finishStepOps?.();
}
console.log(failed===0?"\nlayerNorm f16 output: exact":`\nlayerNorm f16 output: ${failed} WRONG`);
process.exit(failed===0?0:1);
