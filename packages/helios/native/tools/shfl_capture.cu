/*
 * SHFL, captured the way every other instruction in this encoder was: write it
 * in CUDA, compile for sm_86, read the bits out of cuobjdump -sass.
 *
 * WHY: every reduction in prometheus/ — layerNorm's mean and variance, rmsNorm,
 * softmax's max and sum, cross entropy, the column sum — runs a shared-memory
 * tree with a BLOCK-WIDE BAR.SYNC per step. A 640-wide layer norm is a fold
 * plus nine halving steps, twice, so twenty block barriers to move 7.9 MB. It
 * measures 91 us where the card's bandwidth says 17.5, and the backward 189 us
 * against 26.
 *
 * A warp reduces with no barrier and no shared memory at all if it can exchange
 * registers with its neighbours, which is what SHFL is. Five SHFL.BFLY steps
 * reduce 32 lanes; the whole tree above the warp then costs one barrier instead
 * of ten.
 *
 * WHAT EACH KERNEL PROVES. A single capture gives one bit pattern and no way to
 * tell an operand's field from a constant that happened to sit in it, so every
 * form here appears at least twice with DIFFERENT register numbers and
 * different immediates. A field's position is proven by what moves between two
 * captures, never by where it looks like it should be.
 *
 * Build (nvcc is not on PATH on the pod):
 *   /usr/local/cuda-12.8/bin/nvcc -arch=sm_86 -cubin -o shfl.cubin shfl_capture.cu
 *   /usr/local/cuda-12.8/bin/cuobjdump -sass shfl.cubin
 */

/* ---- BFLY: the reduction step. lane ^= mask. ------------------------------ */

extern "C" __global__ void k_bfly1(float *out) {
  float v = out[threadIdx.x];
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  out[threadIdx.x] = v;
}

/* Same instruction, a different immediate lane. What moves between this and
 * k_bfly1 is the lane field and nothing else. */
extern "C" __global__ void k_bfly16(float *out) {
  float v = out[threadIdx.x];
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  out[threadIdx.x] = v;
}

/* The whole five-step butterfly, which is the shape the reduction will emit.
 * Captured as one kernel so the register allocation is visible across steps. */
extern "C" __global__ void k_bfly_full(float *out) {
  float v = out[threadIdx.x];
  for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  out[threadIdx.x] = v;
}

/* A second register assignment for BFLY: two live values shuffled in sequence
 * forces different source and destination registers than the single-value form,
 * which is how the DST and SRCA slots are told apart from each other. */
extern "C" __global__ void k_bfly_pair(float *out, float *out2) {
  float a = out[threadIdx.x], b = out2[threadIdx.x];
  a += __shfl_xor_sync(0xffffffffu, a, 4);
  b += __shfl_xor_sync(0xffffffffu, b, 8);
  out[threadIdx.x] = a;
  out2[threadIdx.x] = b;
}

/* ---- IDX: the broadcast. Every lane reads one lane's value. --------------- */
/*
 * The reduction needs this as much as it needs BFLY. After a butterfly every
 * lane already holds the total, so a warp-per-row kernel needs no broadcast at
 * all — but a block-per-row kernel reducing across warps does, and so does
 * layerNorm's second pass, which wants the mean in every lane.
 */
extern "C" __global__ void k_idx0(float *out) {
  float v = out[threadIdx.x];
  out[threadIdx.x] = __shfl_sync(0xffffffffu, v, 0);
}

extern "C" __global__ void k_idx7(float *out) {
  float v = out[threadIdx.x];
  out[threadIdx.x] = __shfl_sync(0xffffffffu, v, 7);
}

/* IDX with a REGISTER lane rather than an immediate. Whether the lane operand
 * has a register form decides whether a variable-width segment reduction is one
 * instruction or a branch. */
extern "C" __global__ void k_idx_reg(float *out, int lane) {
  float v = out[threadIdx.x];
  out[threadIdx.x] = __shfl_sync(0xffffffffu, v, lane);
}

/* ---- DOWN and UP, for completeness of the family. ------------------------- */
/*
 * Not needed by the reduction — BFLY leaves the answer in every lane, which is
 * what a normalize kernel wants, and DOWN leaves it only in lane 0. Captured
 * anyway because the four variants differ in one field and capturing all four
 * PROVES that field rather than assuming it, and because a scan (which the
 * chunked softmax would want) needs UP.
 */
extern "C" __global__ void k_down2(float *out) {
  float v = out[threadIdx.x];
  v += __shfl_down_sync(0xffffffffu, v, 2);
  out[threadIdx.x] = v;
}

extern "C" __global__ void k_up2(float *out) {
  float v = out[threadIdx.x];
  v += __shfl_up_sync(0xffffffffu, v, 2);
  out[threadIdx.x] = v;
}

/* ---- The partial-warp form. ----------------------------------------------- */
/*
 * `width` below 32 changes the segment mask, which is the third immediate in
 * the SASS and the one most easily mistaken for a constant. A row of 20
 * elements reduced by a 32-lane warp needs either this or a predicate.
 */
extern "C" __global__ void k_bfly_w8(float *out) {
  float v = out[threadIdx.x];
  v += __shfl_xor_sync(0xffffffffu, v, 1, 8);
  out[threadIdx.x] = v;
}

/* ---- What the reduction will actually emit, end to end. -------------------- */
/*
 * A warp-per-row sum of a 640-wide row: twenty coalesced loads into a register
 * accumulator, then five butterfly steps. This is the kernel prometheus/ has to
 * reproduce, and having nvcc's version of it beside the emitter's is the
 * cheapest way to see a difference in instruction count or ordering.
 */
extern "C" __global__ void k_row_sum_640(const float *x, float *out) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned row = blockIdx.x;
  float acc = 0.f;
  for (unsigned i = lane; i < 640u; i += 32u) acc += x[row * 640u + i];
  for (int m = 16; m; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
  if (lane == 0) out[row] = acc;
}
