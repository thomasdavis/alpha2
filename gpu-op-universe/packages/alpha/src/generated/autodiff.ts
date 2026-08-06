/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { AutodiffRequest } from "../../../common/src/types";

/**
 * alpha.autodiff.checkpoint
 * Checkpoint operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffCheckpoint = defineStub<AutodiffRequest>("alpha.autodiff.checkpoint");

/**
 * alpha.autodiff.complex-step-derivative
 * Complex step derivative operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffComplexStepDerivative = defineStub<AutodiffRequest>("alpha.autodiff.complex-step-derivative");

/**
 * alpha.autodiff.custom-jvp
 * Custom jvp operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffCustomJvp = defineStub<AutodiffRequest>("alpha.autodiff.custom-jvp");

/**
 * alpha.autodiff.custom-vjp
 * Custom vjp operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffCustomVjp = defineStub<AutodiffRequest>("alpha.autodiff.custom-vjp");

/**
 * alpha.autodiff.equilibrium-adjoint
 * Equilibrium adjoint operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffEquilibriumAdjoint = defineStub<AutodiffRequest>("alpha.autodiff.equilibrium-adjoint");

/**
 * alpha.autodiff.finite-difference
 * Finite difference operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffFiniteDifference = defineStub<AutodiffRequest>("alpha.autodiff.finite-difference");

/**
 * alpha.autodiff.forward-mode
 * Forward mode operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffForwardMode = defineStub<AutodiffRequest>("alpha.autodiff.forward-mode");

/**
 * alpha.autodiff.forward-over-reverse
 * Forward over reverse operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffForwardOverReverse = defineStub<AutodiffRequest>("alpha.autodiff.forward-over-reverse");

/**
 * alpha.autodiff.gradient-accumulate
 * Gradient accumulate operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientAccumulate = defineStub<AutodiffRequest>("alpha.autodiff.gradient-accumulate");

/**
 * alpha.autodiff.gradient-all-reduce
 * Gradient all reduce operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientAllReduce = defineStub<AutodiffRequest>("alpha.autodiff.gradient-all-reduce");

/**
 * alpha.autodiff.gradient-centralize
 * Gradient centralize operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientCentralize = defineStub<AutodiffRequest>("alpha.autodiff.gradient-centralize");

/**
 * alpha.autodiff.gradient-clip-norm
 * Gradient clip norm operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientClipNorm = defineStub<AutodiffRequest>("alpha.autodiff.gradient-clip-norm");

/**
 * alpha.autodiff.gradient-clip-value
 * Gradient clip value operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientClipValue = defineStub<AutodiffRequest>("alpha.autodiff.gradient-clip-value");

/**
 * alpha.autodiff.gradient-compress
 * Gradient compress operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientCompress = defineStub<AutodiffRequest>("alpha.autodiff.gradient-compress");

/**
 * alpha.autodiff.gradient-error-feedback
 * Gradient error feedback operation in the autodiff family.
 * Status: research; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientErrorFeedback = defineStub<AutodiffRequest>("alpha.autodiff.gradient-error-feedback");

/**
 * alpha.autodiff.gradient-noise-inject
 * Gradient noise inject operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientNoiseInject = defineStub<AutodiffRequest>("alpha.autodiff.gradient-noise-inject");

/**
 * alpha.autodiff.gradient-reduce-scatter
 * Gradient reduce scatter operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientReduceScatter = defineStub<AutodiffRequest>("alpha.autodiff.gradient-reduce-scatter");

/**
 * alpha.autodiff.gradient-scale
 * Gradient scale operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientScale = defineStub<AutodiffRequest>("alpha.autodiff.gradient-scale");

/**
 * alpha.autodiff.gradient-sketch
 * Gradient sketch operation in the autodiff family.
 * Status: research; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientSketch = defineStub<AutodiffRequest>("alpha.autodiff.gradient-sketch");

/**
 * alpha.autodiff.gradient-unscale
 * Gradient unscale operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffGradientUnscale = defineStub<AutodiffRequest>("alpha.autodiff.gradient-unscale");

/**
 * alpha.autodiff.hessian
 * Hessian operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffHessian = defineStub<AutodiffRequest>("alpha.autodiff.hessian");

/**
 * alpha.autodiff.hessian-vector-product
 * Hessian vector product operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffHessianVectorProduct = defineStub<AutodiffRequest>("alpha.autodiff.hessian-vector-product");

/**
 * alpha.autodiff.implicit-differentiate
 * Implicit differentiate operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffImplicitDifferentiate = defineStub<AutodiffRequest>("alpha.autodiff.implicit-differentiate");

/**
 * alpha.autodiff.jacobian
 * Jacobian operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffJacobian = defineStub<AutodiffRequest>("alpha.autodiff.jacobian");

/**
 * alpha.autodiff.jvp
 * Jvp operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffJvp = defineStub<AutodiffRequest>("alpha.autodiff.jvp");

/**
 * alpha.autodiff.quantized-adjoint
 * Quantized adjoint operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffQuantizedAdjoint = defineStub<AutodiffRequest>("alpha.autodiff.quantized-adjoint");

/**
 * alpha.autodiff.recompute
 * Recompute operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffRecompute = defineStub<AutodiffRequest>("alpha.autodiff.recompute");

/**
 * alpha.autodiff.rematerialize
 * Rematerialize operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffRematerialize = defineStub<AutodiffRequest>("alpha.autodiff.rematerialize");

/**
 * alpha.autodiff.reparameterization-estimator
 * Reparameterization estimator operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffReparameterizationEstimator = defineStub<AutodiffRequest>("alpha.autodiff.reparameterization-estimator");

/**
 * alpha.autodiff.reverse-mode
 * Reverse mode operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffReverseMode = defineStub<AutodiffRequest>("alpha.autodiff.reverse-mode");

/**
 * alpha.autodiff.reverse-over-forward
 * Reverse over forward operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffReverseOverForward = defineStub<AutodiffRequest>("alpha.autodiff.reverse-over-forward");

/**
 * alpha.autodiff.scan-adjoint
 * Scan adjoint operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffScanAdjoint = defineStub<AutodiffRequest>("alpha.autodiff.scan-adjoint");

/**
 * alpha.autodiff.score-function-estimator
 * Score function estimator operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffScoreFunctionEstimator = defineStub<AutodiffRequest>("alpha.autodiff.score-function-estimator");

/**
 * alpha.autodiff.sparse-adjoint
 * Sparse adjoint operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffSparseAdjoint = defineStub<AutodiffRequest>("alpha.autodiff.sparse-adjoint");

/**
 * alpha.autodiff.stop-gradient
 * Stop gradient operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffStopGradient = defineStub<AutodiffRequest>("alpha.autodiff.stop-gradient");

/**
 * alpha.autodiff.straight-through-estimator
 * Straight through estimator operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffStraightThroughEstimator = defineStub<AutodiffRequest>("alpha.autodiff.straight-through-estimator");

/**
 * alpha.autodiff.vector-hessian-product
 * Vector hessian product operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffVectorHessianProduct = defineStub<AutodiffRequest>("alpha.autodiff.vector-hessian-product");

/**
 * alpha.autodiff.vjp
 * Vjp operation in the autodiff family.
 * Status: standard; target: architecture-agnostic; differentiability: meta.
 */
export const autodiffVjp = defineStub<AutodiffRequest>("alpha.autodiff.vjp");
