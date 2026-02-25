/**
 * Effective rank: SVD-based dimensionality measure.
 * Inspired by symbiogenesis/model.py:Unit.effective_rank — written from scratch.
 *
 * For each 2D weight matrix, approximate effective rank via power iteration
 * to find dominant singular values, then count values > 1% of max.
 * Average across all weight matrices.
 */
import type { TensorData } from "@alpha/core";

/**
 * Compute mean effective rank across all 2D weight matrices.
 * Uses power iteration (10 iterations) to approximate top singular values.
 */
export function computeEffectiveRank(params: Map<string, TensorData>): number {
  let totalRank = 0;
  let count = 0;

  for (const [, td] of params) {
    if (td.shape.length !== 2) continue;
    const [rows, cols] = td.shape;
    if (rows < 2 || cols < 2) continue;

    const rank = estimateEffectiveRank(td.data as Float32Array, rows, cols);
    totalRank += rank;
    count++;
  }

  return count > 0 ? totalRank / count : 0;
}

/**
 * Estimate effective rank of a matrix via power iteration for top-k singular values.
 * Uses random projection for efficiency on large matrices.
 */
function estimateEffectiveRank(data: Float32Array, rows: number, cols: number): number {
  // For small matrices, use the simpler row-rank estimate
  const minDim = Math.min(rows, cols);
  if (minDim <= 8) return simpleRankEstimate(data, rows, cols, minDim);

  // Power iteration to find approximate singular values
  // We estimate ~20 singular values or minDim, whichever is less
  const k = Math.min(20, minDim);
  const singularValues = approximateSingularValues(data, rows, cols, k);

  if (singularValues.length === 0) return 0;
  const maxSV = singularValues[0];
  if (maxSV < 1e-10) return 0;

  const threshold = 0.01 * maxSV;
  let rank = 0;
  for (let i = 0; i < singularValues.length; i++) {
    if (singularValues[i] > threshold) rank++;
  }
  return rank;
}

/** Simple rank estimate for small matrices using Gram matrix eigenvalues. */
function simpleRankEstimate(data: Float32Array, rows: number, cols: number, minDim: number): number {
  // Compute A^T A (cols x cols) or A A^T (rows x rows), whichever is smaller
  const useTranspose = rows <= cols;
  const n = useTranspose ? rows : cols;

  // Compute Gram matrix
  const gram = new Float64Array(n * n);
  if (useTranspose) {
    // A * A^T
    for (let i = 0; i < rows; i++) {
      for (let j = i; j < rows; j++) {
        let dot = 0;
        for (let k = 0; k < cols; k++) {
          dot += data[i * cols + k] * data[j * cols + k];
        }
        gram[i * n + j] = dot;
        gram[j * n + i] = dot;
      }
    }
  } else {
    // A^T * A
    for (let i = 0; i < cols; i++) {
      for (let j = i; j < cols; j++) {
        let dot = 0;
        for (let k = 0; k < rows; k++) {
          dot += data[k * cols + i] * data[k * cols + j];
        }
        gram[i * n + j] = dot;
        gram[j * n + i] = dot;
      }
    }
  }

  // Power iteration to find eigenvalues of the Gram matrix
  const eigenvalues = powerIterationEigenvalues(gram, n, minDim);
  if (eigenvalues.length === 0) return 0;

  const maxEV = eigenvalues[0];
  if (maxEV < 1e-10) return 0;

  const threshold = 0.0001 * maxEV; // 1% of max singular value squared
  let rank = 0;
  for (let i = 0; i < eigenvalues.length; i++) {
    if (eigenvalues[i] > threshold) rank++;
  }
  return rank;
}

/** Approximate top-k singular values via randomized SVD. */
function approximateSingularValues(data: Float32Array, rows: number, cols: number, k: number): number[] {
  // Random projection: Q = A * R where R is cols x k random Gaussian
  const iters = 10;
  const q = new Float64Array(rows * k);

  // Initialize Q with random values
  for (let i = 0; i < q.length; i++) {
    q[i] = (Math.random() - 0.5) * 2;
  }

  // Power iteration: Q = (A * A^T)^iters * Q
  const temp = new Float64Array(cols * k);
  for (let iter = 0; iter < iters; iter++) {
    // temp = A^T * Q  (cols x k)
    temp.fill(0);
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < k; j++) {
        let dot = 0;
        for (let r = 0; r < rows; r++) {
          dot += data[r * cols + i] * q[r * k + j];
        }
        temp[i * k + j] = dot;
      }
    }
    // Q = A * temp  (rows x k)
    q.fill(0);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < k; j++) {
        let dot = 0;
        for (let c = 0; c < cols; c++) {
          dot += data[i * cols + c] * temp[c * k + j];
        }
        q[i * k + j] = dot;
      }
    }

    // QR-style normalization of columns
    for (let j = 0; j < k; j++) {
      let norm = 0;
      for (let i = 0; i < rows; i++) norm += q[i * k + j] * q[i * k + j];
      norm = Math.sqrt(norm);
      if (norm > 1e-10) {
        for (let i = 0; i < rows; i++) q[i * k + j] /= norm;
      }
    }
  }

  // Extract approximate singular values as |A^T * q_j|
  const svs: number[] = [];
  for (let j = 0; j < k; j++) {
    // Compute A^T * q_j
    let sqNorm = 0;
    for (let c = 0; c < cols; c++) {
      let dot = 0;
      for (let r = 0; r < rows; r++) {
        dot += data[r * cols + c] * q[r * k + j];
      }
      sqNorm += dot * dot;
    }
    svs.push(Math.sqrt(sqNorm));
  }

  svs.sort((a, b) => b - a);
  return svs;
}

/** Find top eigenvalues of a symmetric matrix via deflated power iteration. */
function powerIterationEigenvalues(gram: Float64Array, n: number, k: number): number[] {
  const eigenvalues: number[] = [];
  const mat = new Float64Array(gram);

  for (let ev = 0; ev < k; ev++) {
    // Initialize random vector
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.random() - 0.5;

    let eigenvalue = 0;
    for (let iter = 0; iter < 20; iter++) {
      // w = M * v
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let j = 0; j < n; j++) {
          dot += mat[i * n + j] * v[j];
        }
        w[i] = dot;
      }

      // eigenvalue = v^T * w
      eigenvalue = 0;
      for (let i = 0; i < n; i++) eigenvalue += v[i] * w[i];

      // Normalize
      let norm = 0;
      for (let i = 0; i < n; i++) norm += w[i] * w[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-10) break;
      for (let i = 0; i < n; i++) v[i] = w[i] / norm;
    }

    eigenvalues.push(Math.max(0, eigenvalue));

    // Deflate: M = M - eigenvalue * v * v^T
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        mat[i * n + j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  eigenvalues.sort((a, b) => b - a);
  return eigenvalues;
}
