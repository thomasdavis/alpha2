/* Auto-generated scaffold support. Implementations must replace defineStub. */
export type OperationStatus = "standard" | "research" | "speculative";
export type DType = "bool" | "binary1" | "ternary2" | "int2" | "uint2" | "int4" | "uint4" | "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32" | "int64" | "uint64" | "fp16" | "bf16" | "tf32" | "fp32" | "fp64" | "complex32" | "complex64" | "complex128" | "nf4" | "fp8e4m3" | "fp8e5m2" | "mxfp4" | "mxfp6" | "mxfp8";
export type Layout = "scalar" | "contiguous" | "rowMajor" | "columnMajor" | "strided" | "broadcast" | "interleaved" | "blocked" | "tiled" | "swizzled" | "tensorOpCongruous" | "tensorOpCrosswise" | "packedBits" | "ragged" | "coo" | "csr" | "csc" | "bsr" | "bsc" | "ell" | "dia" | "sell" | "nmSparse";
export type ExecutionScope = "thread" | "warp" | "block" | "grid" | "device" | "multiDevice" | "host";
export interface TensorRef { readonly id: string; readonly dtype: DType; readonly shape: readonly number[]; readonly strides?: readonly number[]; readonly layout?: Layout; readonly address?: bigint; }
export interface OpContext { readonly traceId?: string; readonly deterministic?: boolean; readonly deviceId?: number; readonly deadlineNs?: bigint; readonly metadata?: Readonly<Record<string, unknown>>; }
export interface BaseOpRequest { readonly inputs?: readonly TensorRef[]; readonly outputs?: readonly TensorRef[]; readonly attrs?: Readonly<Record<string, unknown>>; }
export interface ScalarOpRequest extends BaseOpRequest { readonly op?: string; }
export interface TensorOpRequest extends BaseOpRequest { readonly axes?: readonly number[]; }
export interface TransformOpRequest extends TensorOpRequest { readonly permutation?: readonly number[]; readonly targetShape?: readonly number[]; }
export interface ReductionRequest extends TensorOpRequest { readonly reducer?: string; readonly keepDims?: boolean; }
export interface ScanRequest extends TensorOpRequest { readonly inclusive?: boolean; readonly reverse?: boolean; readonly combine?: string; }
export interface MatmulRequest extends BaseOpRequest { readonly m?: number; readonly n?: number; readonly k?: number; readonly batch?: number; readonly transposeA?: boolean; readonly transposeB?: boolean; readonly semiring?: string; readonly epilogue?: string; }
export interface SparseOpRequest extends MatmulRequest { readonly format?: Layout; readonly nnz?: number; readonly mask?: TensorRef; }
export interface SolverOpRequest extends BaseOpRequest { readonly tolerance?: number; readonly maxIterations?: number; readonly preconditioner?: string; }
export interface AttentionRequest extends BaseOpRequest { readonly causal?: boolean; readonly scale?: number; readonly window?: number; readonly kvCache?: TensorRef; }
export interface SequenceRequest extends BaseOpRequest { readonly state?: TensorRef; readonly chunkSize?: number; readonly reverse?: boolean; }
export interface QuantizationRequest extends BaseOpRequest { readonly bits?: number; readonly groupSize?: number; readonly symmetric?: boolean; readonly stochastic?: boolean; }
export interface RandomRequest extends BaseOpRequest { readonly seed?: bigint; readonly offset?: bigint; readonly distribution?: string; }
export interface AutodiffRequest extends BaseOpRequest { readonly cotangents?: readonly TensorRef[]; readonly tangents?: readonly TensorRef[]; }
export interface OptimizerRequest extends BaseOpRequest { readonly step?: bigint; readonly learningRate?: number; readonly weightDecay?: number; }
export interface CollectiveOpRequest extends BaseOpRequest { readonly rank?: number; readonly worldSize?: number; readonly root?: number; readonly reduceOp?: string; }
export interface ActorOpRequest extends BaseOpRequest { readonly actorId?: number; readonly mailboxId?: number; readonly eventBudget?: number; readonly state?: TensorRef; }
export interface BackendOpRequest extends BaseOpRequest { readonly programId?: string; readonly graphId?: string; readonly priority?: number; }
export interface CompilerOpRequest extends BaseOpRequest { readonly nodeId?: string; readonly regionId?: string; readonly target?: string; }
export interface InstructionOpRequest extends BaseOpRequest { readonly opcode?: string; readonly operandsRaw?: readonly unknown[]; readonly control?: Readonly<Record<string, number | boolean>>; }
export interface ScheduleOpRequest extends BaseOpRequest { readonly timeline?: bigint; readonly epoch?: bigint; readonly wave?: number; readonly budget?: number; }
export interface LaunchOpRequest extends BaseOpRequest { readonly channelId?: string; readonly grid?: readonly [number, number, number]; readonly block?: readonly [number, number, number]; readonly sharedBytes?: number; }
export interface MemoryOpRequest extends BaseOpRequest { readonly bytes?: bigint; readonly alignment?: number; readonly gpuVa?: bigint; readonly flags?: readonly string[]; }
export interface DriverOpRequest extends BaseOpRequest { readonly fd?: number; readonly handle?: number; readonly command?: number; readonly payload?: Uint8Array; }
export interface OpResult { readonly operationId: string; readonly outputs?: readonly TensorRef[]; readonly handle?: string | number | bigint; readonly metadata?: Readonly<Record<string, unknown>>; }
export type StubFunction<T extends BaseOpRequest = BaseOpRequest> = (ctx: OpContext, request: T) => Promise<OpResult>;
export class UnimplementedOperationError extends Error { constructor(readonly operationId: string) { super(`Operation not implemented: ${operationId}`); this.name = "UnimplementedOperationError"; } }
export function defineStub<T extends BaseOpRequest>(operationId: string): StubFunction<T> { return async (_ctx, _request) => { throw new UnimplementedOperationError(operationId); }; }
