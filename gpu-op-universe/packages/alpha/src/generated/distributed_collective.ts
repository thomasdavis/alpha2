/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { CollectiveOpRequest } from "../../../common/src/types";

/**
 * alpha.distributed_collective.all-gather
 * All gather operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveAllGather = defineStub<CollectiveOpRequest>("alpha.distributed_collective.all-gather");

/**
 * alpha.distributed_collective.all-reduce
 * All reduce operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveAllReduce = defineStub<CollectiveOpRequest>("alpha.distributed_collective.all-reduce");

/**
 * alpha.distributed_collective.all-to-all
 * All to all operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveAllToAll = defineStub<CollectiveOpRequest>("alpha.distributed_collective.all-to-all");

/**
 * alpha.distributed_collective.all-to-all-v
 * All to all v operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveAllToAllV = defineStub<CollectiveOpRequest>("alpha.distributed_collective.all-to-all-v");

/**
 * alpha.distributed_collective.barrier
 * Barrier operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveBarrier = defineStub<CollectiveOpRequest>("alpha.distributed_collective.barrier");

/**
 * alpha.distributed_collective.broadcast
 * Broadcast operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveBroadcast = defineStub<CollectiveOpRequest>("alpha.distributed_collective.broadcast");

/**
 * alpha.distributed_collective.expert-parallel-combine
 * Expert parallel combine operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveExpertParallelCombine = defineStub<CollectiveOpRequest>("alpha.distributed_collective.expert-parallel-combine");

/**
 * alpha.distributed_collective.expert-parallel-dispatch
 * Expert parallel dispatch operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveExpertParallelDispatch = defineStub<CollectiveOpRequest>("alpha.distributed_collective.expert-parallel-dispatch");

/**
 * alpha.distributed_collective.gather
 * Gather operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveGather = defineStub<CollectiveOpRequest>("alpha.distributed_collective.gather");

/**
 * alpha.distributed_collective.gossip-average
 * Gossip average operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveGossipAverage = defineStub<CollectiveOpRequest>("alpha.distributed_collective.gossip-average");

/**
 * alpha.distributed_collective.hierarchical-all-reduce
 * Hierarchical all reduce operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveHierarchicalAllReduce = defineStub<CollectiveOpRequest>("alpha.distributed_collective.hierarchical-all-reduce");

/**
 * alpha.distributed_collective.neighbor-all-gather
 * Neighbor all gather operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveNeighborAllGather = defineStub<CollectiveOpRequest>("alpha.distributed_collective.neighbor-all-gather");

/**
 * alpha.distributed_collective.neighbor-all-to-all
 * Neighbor all to all operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveNeighborAllToAll = defineStub<CollectiveOpRequest>("alpha.distributed_collective.neighbor-all-to-all");

/**
 * alpha.distributed_collective.parameter-server-pull
 * Parameter server pull operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveParameterServerPull = defineStub<CollectiveOpRequest>("alpha.distributed_collective.parameter-server-pull");

/**
 * alpha.distributed_collective.parameter-server-push
 * Parameter server push operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveParameterServerPush = defineStub<CollectiveOpRequest>("alpha.distributed_collective.parameter-server-push");

/**
 * alpha.distributed_collective.pipeline-recv
 * Pipeline recv operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectivePipelineRecv = defineStub<CollectiveOpRequest>("alpha.distributed_collective.pipeline-recv");

/**
 * alpha.distributed_collective.pipeline-send
 * Pipeline send operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectivePipelineSend = defineStub<CollectiveOpRequest>("alpha.distributed_collective.pipeline-send");

/**
 * alpha.distributed_collective.recv
 * Recv operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveRecv = defineStub<CollectiveOpRequest>("alpha.distributed_collective.recv");

/**
 * alpha.distributed_collective.reduce
 * Reduce operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveReduce = defineStub<CollectiveOpRequest>("alpha.distributed_collective.reduce");

/**
 * alpha.distributed_collective.reduce-scatter
 * Reduce scatter operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveReduceScatter = defineStub<CollectiveOpRequest>("alpha.distributed_collective.reduce-scatter");

/**
 * alpha.distributed_collective.remote-district-call
 * Remote district call operation in the distributed collective family.
 * Status: research; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveRemoteDistrictCall = defineStub<CollectiveOpRequest>("alpha.distributed_collective.remote-district-call");

/**
 * alpha.distributed_collective.remote-district-return
 * Remote district return operation in the distributed collective family.
 * Status: research; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveRemoteDistrictReturn = defineStub<CollectiveOpRequest>("alpha.distributed_collective.remote-district-return");

/**
 * alpha.distributed_collective.ring-all-reduce
 * Ring all reduce operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveRingAllReduce = defineStub<CollectiveOpRequest>("alpha.distributed_collective.ring-all-reduce");

/**
 * alpha.distributed_collective.scatter
 * Scatter operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveScatter = defineStub<CollectiveOpRequest>("alpha.distributed_collective.scatter");

/**
 * alpha.distributed_collective.send
 * Send operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveSend = defineStub<CollectiveOpRequest>("alpha.distributed_collective.send");

/**
 * alpha.distributed_collective.send-recv
 * Send recv operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveSendRecv = defineStub<CollectiveOpRequest>("alpha.distributed_collective.send-recv");

/**
 * alpha.distributed_collective.sequence-parallel-shard
 * Sequence parallel shard operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveSequenceParallelShard = defineStub<CollectiveOpRequest>("alpha.distributed_collective.sequence-parallel-shard");

/**
 * alpha.distributed_collective.tensor-parallel-shard
 * Tensor parallel shard operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveTensorParallelShard = defineStub<CollectiveOpRequest>("alpha.distributed_collective.tensor-parallel-shard");

/**
 * alpha.distributed_collective.tree-all-reduce
 * Tree all reduce operation in the distributed collective family.
 * Status: standard; target: architecture-agnostic; differentiability: custom-adjoint.
 */
export const distributedCollectiveTreeAllReduce = defineStub<CollectiveOpRequest>("alpha.distributed_collective.tree-all-reduce");
