# Matrix Multiplication Is a Grammar, Not One Operation

The registry deliberately refuses to model GEMM as only `D = alpha * A * B + beta * C`. A useful compiler sees independent axes:

1. **Index geometry:** dot, GEMV, GEMM, batched, grouped, broadcast, tensor contraction, rank-k update.
2. **Algebra:** arithmetic, Boolean, min-plus, max-plus, log-sum-exp, Viterbi, finite-field, user-defined semiring.
3. **Structure:** dense, triangular, symmetric, Hermitian, banded, Toeplitz, circulant, Kronecker, butterfly, low-rank, N:M sparse, block sparse.
4. **Numerics:** exact, mixed precision, quantized, stochastic, error-feedback, compensated, interval-bounded, verified approximate.
5. **Schedule:** SIMT, tensor-core, split-K, Stream-K, persistent, grouped, tile-stream, countercurrent, deferred-gradient.
6. **Data movement:** gather/scatter operands, packed weights, resident weights, asynchronous staging, cache-aware reuse.
7. **Epilogue/prologue:** bias, activation, normalization, residual, dropout, amax, quantization, optimizer transform, checksum.
8. **Stopping semantics:** complete product, sampled product, anytime bitplane refinement, bound-driven early stop.
9. **Training role:** forward activation, input adjoint, weight gradient, covariance/K-FAC factor, low-rank correction, optimizer-consumed update.

## Uncommon but coherent kernel targets

### Semiring GEMM

`C[i,j] = reduce_k(combine(A[i,k], B[k,j]))` permits shortest paths (`min,+`), Viterbi (`max,+`), Boolean reachability (`or,and`), log-domain dynamic programs (`logsumexp,+`), and custom graph/message-passing algebras. GraphBLAS standardizes this generalized view.

### Deferred weight-gradient GEMM

Bank microbatch factors `X_i` and `D_i`, then form one large product:

`dW = concat(X_i)^T @ concat(D_i)`.

It is mathematically exact before compression and can trade activation storage for larger, more efficient GEMMs.

### Residue-corrected GEMM

Compute a cheap bulk product and correct a selected subspace exactly:

`G_hat = G_low + U [U^T (G_exact - G_low) V] V^T`.

This is a direct kernel target, not merely an after-the-fact tensor operation.

### Anytime bitplane GEMM

Process high-significance bitplanes first and stop when the remaining product bound cannot change the consumer's decision. The consumer may be a top-k selection, sign decision, quantizer bucket, or optimizer direction rather than an exact dense output.

### Optimizer-consumed GEMM

A training backend often materializes a full gradient only to immediately transform it. A fused operation can accumulate the statistics or transformed direction the optimizer actually consumes, such as block norms, covariance factors, orthogonalized momentum inputs, or low-rank residue coefficients.

### Conservation-projected GEMM

Low-bit products can be constrained to preserve selected moments or projections exactly, while redistributing error into the unconstrained complement. Candidate conserved quantities include row/column sums, momentum projection, leading curvature modes, or checksums.

## Codex rule

A new GEMM name is not an implementation. Every implementation proposal must specify the complete grammar tuple, numerical contract, target shapes, expected consumer, data-movement plan, fallback, correctness oracle, and matched-cost benchmark.
