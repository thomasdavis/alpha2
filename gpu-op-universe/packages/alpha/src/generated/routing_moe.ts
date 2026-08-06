/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { ActorOpRequest } from "../../../common/src/types";

/**
 * alpha.routing_moe.capacity-router
 * Capacity router operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeCapacityRouter = defineStub<ActorOpRequest>("alpha.routing_moe.capacity-router");

/**
 * alpha.routing_moe.combine-expert-outputs
 * Combine expert outputs operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeCombineExpertOutputs = defineStub<ActorOpRequest>("alpha.routing_moe.combine-expert-outputs");

/**
 * alpha.routing_moe.dispatch-to-experts
 * Dispatch to experts operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeDispatchToExperts = defineStub<ActorOpRequest>("alpha.routing_moe.dispatch-to-experts");

/**
 * alpha.routing_moe.expert-archive
 * Expert archive operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertArchive = defineStub<ActorOpRequest>("alpha.routing_moe.expert-archive");

/**
 * alpha.routing_moe.expert-audit
 * Expert audit operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertAudit = defineStub<ActorOpRequest>("alpha.routing_moe.expert-audit");

/**
 * alpha.routing_moe.expert-choice-dispatch
 * Expert choice dispatch operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertChoiceDispatch = defineStub<ActorOpRequest>("alpha.routing_moe.expert-choice-dispatch");

/**
 * alpha.routing_moe.expert-choice-router
 * Expert choice router operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertChoiceRouter = defineStub<ActorOpRequest>("alpha.routing_moe.expert-choice-router");

/**
 * alpha.routing_moe.expert-distill
 * Expert distill operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertDistill = defineStub<ActorOpRequest>("alpha.routing_moe.expert-distill");

/**
 * alpha.routing_moe.expert-evict
 * Expert evict operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertEvict = defineStub<ActorOpRequest>("alpha.routing_moe.expert-evict");

/**
 * alpha.routing_moe.expert-load-balance
 * Expert load balance operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertLoadBalance = defineStub<ActorOpRequest>("alpha.routing_moe.expert-load-balance");

/**
 * alpha.routing_moe.expert-merge
 * Expert merge operation in the routing moe family.
 * Status: research; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertMerge = defineStub<ActorOpRequest>("alpha.routing_moe.expert-merge");

/**
 * alpha.routing_moe.expert-mitosis
 * Expert mitosis operation in the routing moe family.
 * Status: research; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertMitosis = defineStub<ActorOpRequest>("alpha.routing_moe.expert-mitosis");

/**
 * alpha.routing_moe.expert-prefetch
 * Expert prefetch operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertPrefetch = defineStub<ActorOpRequest>("alpha.routing_moe.expert-prefetch");

/**
 * alpha.routing_moe.expert-replica-select
 * Expert replica select operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeExpertReplicaSelect = defineStub<ActorOpRequest>("alpha.routing_moe.expert-replica-select");

/**
 * alpha.routing_moe.grouped-expert-gemm
 * Grouped expert gemm operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeGroupedExpertGemm = defineStub<ActorOpRequest>("alpha.routing_moe.grouped-expert-gemm");

/**
 * alpha.routing_moe.hash-router
 * Hash router operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeHashRouter = defineStub<ActorOpRequest>("alpha.routing_moe.hash-router");

/**
 * alpha.routing_moe.hierarchical-router
 * Hierarchical router operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeHierarchicalRouter = defineStub<ActorOpRequest>("alpha.routing_moe.hierarchical-router");

/**
 * alpha.routing_moe.router-diversity
 * Router diversity operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeRouterDiversity = defineStub<ActorOpRequest>("alpha.routing_moe.router-diversity");

/**
 * alpha.routing_moe.router-entropy
 * Router entropy operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeRouterEntropy = defineStub<ActorOpRequest>("alpha.routing_moe.router-entropy");

/**
 * alpha.routing_moe.router-locality-cost
 * Router locality cost operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeRouterLocalityCost = defineStub<ActorOpRequest>("alpha.routing_moe.router-locality-cost");

/**
 * alpha.routing_moe.router-zloss
 * Router zloss operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeRouterZLoss = defineStub<ActorOpRequest>("alpha.routing_moe.router-zloss");

/**
 * alpha.routing_moe.sinkhorn-router
 * Sinkhorn router operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeSinkhornRouter = defineStub<ActorOpRequest>("alpha.routing_moe.sinkhorn-router");

/**
 * alpha.routing_moe.soft-router
 * Soft router operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeSoftRouter = defineStub<ActorOpRequest>("alpha.routing_moe.soft-router");

/**
 * alpha.routing_moe.token-choice-dispatch
 * Token choice dispatch operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeTokenChoiceDispatch = defineStub<ActorOpRequest>("alpha.routing_moe.token-choice-dispatch");

/**
 * alpha.routing_moe.top-krouter
 * Top krouter operation in the routing moe family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-soft.
 */
export const routingMoeTopKRouter = defineStub<ActorOpRequest>("alpha.routing_moe.top-krouter");
