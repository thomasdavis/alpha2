import type { TensorData } from "@alpha/core";

/** Build deterministic Rademacher probes over valid target positions. */
export function buildSamePositionCotangent(
  batch: number,
  time: number,
  width: number,
  start: number,
  active: number,
  skip: number,
  promptIndex: number,
  seed: number,
): TensorData {
  const data = new Float32Array(batch * time * width);
  for (let row = 0; row < active; row++) {
    const dimension = start + row;
    for (let position = skip; position < time - 1; position++) {
      data[(row * time + position) * width + dimension] = samePositionSign(
        seed,
        promptIndex,
        dimension,
        position,
      );
    }
  }
  return { shape: [batch, time, width], dtype: "f32", data };
}

/**
 * Accumulate rows of the mean same-position Jacobian from a causal VJP.
 * Multiplying each source-position gradient by its matching probe sign makes
 * current-position blocks survive while cross-position blocks cancel in
 * expectation over independent prompt probes.
 */
export function accumulateSamePositionRows(
  matrix: Float32Array,
  gradient: Float32Array,
  _batch: number,
  time: number,
  sourceWidth: number,
  dimensionStart: number,
  activeDimensions: number,
  skipFirst: number,
  promptIndex: number,
  seed: number,
): void {
  const positions = time - 1 - skipFirst;
  for (let batchRow = 0; batchRow < activeDimensions; batchRow++) {
    const outputDimension = dimensionStart + batchRow;
    const matrixBase = outputDimension * sourceWidth;
    for (let position = skipFirst; position < time - 1; position++) {
      const sign = samePositionSign(seed, promptIndex, outputDimension, position);
      const gradientBase = (batchRow * time + position) * sourceWidth;
      for (let column = 0; column < sourceWidth; column++) {
        matrix[matrixBase + column] += (sign * gradient[gradientBase + column]) / positions;
      }
    }
  }
}

export function samePositionSign(
  seed: number,
  promptIndex: number,
  outputDimension: number,
  position: number,
): 1 | -1 {
  let value = seed >>> 0;
  value ^= Math.imul(promptIndex + 1, 0x9e3779b1);
  value ^= Math.imul(outputDimension + 1, 0x85ebca77);
  value ^= Math.imul(position + 1, 0xc2b2ae3d);
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846ca68b);
  value ^= value >>> 16;
  return (value & 1) === 0 ? 1 : -1;
}

/** Deterministic synthetic causal oracle used by unit and bundle validation. */
export function evaluateSamePositionEstimatorOracle(probes = 8192): {
  readonly probes: number;
  readonly expected: readonly number[];
  readonly observed: readonly number[];
  readonly maximumAbsoluteError: number;
} {
  const time = 5;
  const width = 2;
  const skipFirst = 1;
  const validEnd = time - 1;
  const validPositions = validEnd - skipFirst;
  const estimate = new Float32Array(width * width);
  const jacobian = (
    targetPosition: number,
    sourcePosition: number,
    outputDimension: number,
    inputDimension: number,
  ): number => {
    if (targetPosition < sourcePosition) return 0;
    const diagonal = targetPosition === sourcePosition
      ? 3
      : 0.4 * (targetPosition - sourcePosition);
    return diagonal + 0.7 * outputDimension + 0.2 * inputDimension + 0.03 * sourcePosition;
  };

  for (let probe = 0; probe < probes; probe++) {
    const cotangent = buildSamePositionCotangent(
      width,
      time,
      width,
      0,
      width,
      skipFirst,
      probe,
      42,
    );
    const gradient = new Float32Array(width * time * width);
    for (let output = 0; output < width; output++) {
      for (let sourcePosition = skipFirst; sourcePosition < validEnd; sourcePosition++) {
        for (let input = 0; input < width; input++) {
          let value = 0;
          for (let targetPosition = sourcePosition; targetPosition < validEnd; targetPosition++) {
            const sign = cotangent.data[(output * time + targetPosition) * width + output];
            value += sign * jacobian(targetPosition, sourcePosition, output, input);
          }
          gradient[(output * time + sourcePosition) * width + input] = value;
        }
      }
    }
    accumulateSamePositionRows(
      estimate,
      gradient,
      width,
      time,
      width,
      0,
      width,
      skipFirst,
      probe,
      42,
    );
  }

  const expected: number[] = [];
  const observed: number[] = [];
  let maximumAbsoluteError = 0;
  for (let output = 0; output < width; output++) {
    for (let input = 0; input < width; input++) {
      let exact = 0;
      for (let position = skipFirst; position < validEnd; position++) {
        exact += jacobian(position, position, output, input) / validPositions;
      }
      const estimateValue = estimate[output * width + input] / probes;
      expected.push(exact);
      observed.push(estimateValue);
      maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(estimateValue - exact));
    }
  }
  return { probes, expected, observed, maximumAbsoluteError };
}
