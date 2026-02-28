/**
 * spirv.ts — Minimal SPIR-V code generator written from scratch.
 *
 * SPIR-V is a binary intermediate language for GPU compute shaders.
 * Instead of depending on glslc/glslangValidator, we generate SPIR-V
 * bytecode directly from TypeScript.
 *
 * This assembler supports the subset needed for compute shaders:
 *   - Storage buffer bindings (read/write f32 arrays)
 *   - Workgroup size decoration
 *   - Basic arithmetic (add, sub, mul, div, fma)
 *   - Control flow (loops, conditionals)
 *   - Built-in variables (GlobalInvocationId, etc.)
 *
 * Reference: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
 */

// ── SPIR-V constants ────────────────────────────────────────────────────────

const MAGIC = 0x07230203;
const VERSION = 0x00010300;        // SPIR-V 1.3 (default for most Helios kernels)
const GENERATOR = 0x00000000;      // Helios (unregistered)

// Opcodes (only the ones we need)
export const Op = {
  Capability:           17,
  ExtInstImport:        11,
  MemoryModel:          14,
  EntryPoint:           15,
  ExecutionMode:        16,
  Decorate:             71,
  MemberDecorate:       72,
  TypeVoid:             19,
  TypeBool:             20,
  TypeInt:              21,
  TypeFloat:            22,
  TypeVector:           23,
  TypeArray:            28,
  TypeRuntimeArray:     29,
  TypeStruct:           30,
  TypePointer:          32,
  TypeFunction:         33,
  Constant:             43,
  ConstantComposite:    44,
  Variable:             59,
  Load:                 61,
  Store:                62,
  AccessChain:          65,
  Function:             54,
  FunctionEnd:          56,
  Label:                248,
  Return:               253,
  ReturnValue:          254,
  Branch:               249,
  BranchConditional:    250,
  SelectionMerge:       247,
  LoopMerge:            246,
  Phi:                  245,
  IAdd:                 128,
  ISub:                 130,
  IMul:                 132,
  UDiv:                 134,
  SDiv:                 135,
  UMod:                 137,
  FAdd:                 129,
  FSub:                 131,
  FMul:                 133,
  FDiv:                 136,
  FMod:                 141,
  FNegate:              127,
  VectorTimesScalar:    142,
  CompositeExtract:     81,
  CompositeConstruct:   80,
  CopyObject:           83,
  ULessThan:            176,
  SLessThan:            177,
  UGreaterThanEqual:    174,
  SGreaterThanEqual:    175,
  FOrdNotEqual:         182,
  FOrdLessThan:         184,
  FOrdGreaterThan:      186,
  FOrdGreaterThanEqual: 190,
  LogicalAnd:           167,
  LogicalOr:            166,
  LogicalNot:           168,
  IEqual:               170,
  Select:               169,
  ShiftLeftLogical:     196,
  ShiftRightLogical:    194,
  ShiftRightArithmetic: 195,
  BitwiseOr:            197,
  BitwiseXor:           198,
  BitwiseAnd:           199,
  Not:                  200,
  ConvertFToU:          109,
  ConvertFToS:          110,
  ConvertUToF:          112,
  ConvertSToF:          111,
  FConvert:             115,
  Bitcast:              124,
  ExtInst:              12,
  Extension:            10,
  ControlBarrier:       224,
  MemoryBarrier:        225,
  IsNan:                156,
  IsInf:                157,
  // Cooperative matrix (VK_KHR_cooperative_matrix)
  OpTypeCooperativeMatrixKHR:   4456,
  OpCooperativeMatrixLoadKHR:   4457,
  OpCooperativeMatrixStoreKHR:  4458,
  OpCooperativeMatrixMulAddKHR: 4459,
  OpCooperativeMatrixLengthKHR: 4460,
} as const;

// Capabilities
export const Capability = {
  Shader: 1,
  Float16: 9,
  VulkanMemoryModel: 5345,
  StorageBuffer16BitAccess: 4433, // SPV_KHR_16bit_storage
  StorageBufferStorageClass: 4443, // SPV_KHR_storage_buffer_storage_class
  CooperativeMatrixKHR: 6022,
} as const;

// Cooperative matrix usage
export const CooperativeMatrixUse = {
  MatrixA: 0,
  MatrixB: 1,
  MatrixAccumulator: 2,
} as const;

// Addressing/memory models
export const AddressingModel = { Logical: 0 } as const;
export const MemoryModel = {
  GLSL450: 1,
  Vulkan: 3,
} as const;

// Execution model / mode
export const ExecutionModel = { GLCompute: 5 } as const;
export const ExecutionMode = { LocalSize: 17 } as const;

// Storage classes
export const StorageClass = {
  UniformConstant: 0,
  Input: 1,
  Uniform: 2,
  Output: 3,
  Workgroup: 4,
  CrossWorkgroup: 5,
  Private: 6,
  Function: 7,
  PushConstant: 9,
  StorageBuffer: 12,
} as const;

// Decorations
export const Decoration = {
  Block: 2,
  BufferBlock: 3,
  ArrayStride: 6,
  Offset: 35,
  DescriptorSet: 34,
  Binding: 33,
  BuiltIn: 11,
  NonWritable: 24,
  NonReadable: 25,
} as const;

// Built-in variables
export const BuiltIn = {
  NumWorkgroups: 24,
  WorkgroupSize: 25,
  WorkgroupId: 26,
  LocalInvocationId: 27,
  GlobalInvocationId: 28,
  LocalInvocationIndex: 29,
  NumSubgroups: 38,
  SubgroupId: 40,
  SubgroupLocalInvocationId: 41,
} as const;

// GLSL.std.450 extended instruction set
export const GLSLstd450 = {
  Exp: 27,
  Log: 28,
  Sqrt: 31,
  Pow: 26,
  FAbs: 4,
  FMax: 40,
  FMin: 37,
  Floor: 8,
  Ceil: 9,
  Tanh: 21,
  FClamp: 43,
  FMA: 50,
} as const;

// Scope (for barriers)
export const Scope = {
  CrossDevice: 0,
  Device: 1,
  Workgroup: 2,
  Subgroup: 3,
  Invocation: 4,
} as const;

// Memory semantics (for barriers)
export const MemorySemantics = {
  None: 0x0000,
  AcquireRelease: 0x0008,
  WorkgroupMemory: 0x0100,
} as const;

// Function control
export const FunctionControl = { None: 0 } as const;

// ── SPIR-V Builder ──────────────────────────────────────────────────────────

export class SpirVBuilder {
  private nextId = 1;
  private versionWord: number;
  private capabilities: number[] = [];
  private extensions: number[][] = [];
  private extInstImports: number[][] = [];
  private memoryModelInstr: number[] = [];
  private entryPoints: number[][] = [];
  private executionModes: number[][] = [];
  private debugNames: number[][] = [];
  private annotations: number[][] = [];
  private typeDecls: number[][] = [];
  private globalVars: number[][] = [];
  private functionBodies: number[][] = [];

  private bound = 0;

  constructor(versionWord = VERSION) {
    this.versionWord = versionWord;
  }

  /** Allocate a new SPIR-V ID. */
  id(): number {
    return this.nextId++;
  }

  /** Encode a string as SPIR-V literal words (null-terminated, padded to 4 bytes). */
  private encodeString(s: string): number[] {
    const bytes = new TextEncoder().encode(s + "\0");
    const padded = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
    padded.set(bytes);
    const words: number[] = [];
    const view = new DataView(padded.buffer);
    for (let i = 0; i < padded.length; i += 4) {
      words.push(view.getUint32(i, true));
    }
    return words;
  }

  /** Encode an instruction: (wordCount << 16) | opcode, followed by operands. */
  private instr(opcode: number, operands: number[]): number[] {
    const wordCount = 1 + operands.length;
    return [(wordCount << 16) | opcode, ...operands];
  }

  // ── Declaration methods ─────────────────────────────────────────────────

  addCapability(cap: number): void {
    this.capabilities.push(...this.instr(Op.Capability, [cap]));
  }

  /** Emit OpExtension (opcode 10) — declares a SPIR-V extension. */
  addExtension(name: string): void {
    this.extensions.push(this.instr(Op.Extension, [...this.encodeString(name)]));
  }

  addExtInstImport(resultId: number, name: string): void {
    this.extInstImports.push(this.instr(Op.ExtInstImport, [resultId, ...this.encodeString(name)]));
  }

  setMemoryModel(addressing: number, memory: number): void {
    this.memoryModelInstr = this.instr(Op.MemoryModel, [addressing, memory]);
  }

  addEntryPoint(execModel: number, funcId: number, name: string, interfaceIds: number[]): void {
    this.entryPoints.push(this.instr(Op.EntryPoint, [execModel, funcId, ...this.encodeString(name), ...interfaceIds]));
  }

  addExecutionMode(funcId: number, mode: number, ...operands: number[]): void {
    this.executionModes.push(this.instr(Op.ExecutionMode, [funcId, mode, ...operands]));
  }

  addDecorate(targetId: number, decoration: number, ...operands: number[]): void {
    this.annotations.push(this.instr(Op.Decorate, [targetId, decoration, ...operands]));
  }

  addMemberDecorate(structId: number, member: number, decoration: number, ...operands: number[]): void {
    this.annotations.push(this.instr(Op.MemberDecorate, [structId, member, decoration, ...operands]));
  }

  // ── Type declarations ─────────────────────────────────────────────────

  typeVoid(id: number): void {
    this.typeDecls.push(this.instr(Op.TypeVoid, [id]));
  }

  typeBool(id: number): void {
    this.typeDecls.push(this.instr(Op.TypeBool, [id]));
  }

  typeInt(id: number, width: number, signedness: number): void {
    this.typeDecls.push(this.instr(Op.TypeInt, [id, width, signedness]));
  }

  typeFloat(id: number, width: number): void {
    this.typeDecls.push(this.instr(Op.TypeFloat, [id, width]));
  }

  typeVector(id: number, componentType: number, count: number): void {
    this.typeDecls.push(this.instr(Op.TypeVector, [id, componentType, count]));
  }

  typeArray(id: number, elementType: number, length: number): void {
    this.typeDecls.push(this.instr(Op.TypeArray, [id, elementType, length]));
  }

  typeRuntimeArray(id: number, elementType: number): void {
    this.typeDecls.push(this.instr(Op.TypeRuntimeArray, [id, elementType]));
  }

  typeStruct(id: number, memberTypes: number[]): void {
    this.typeDecls.push(this.instr(Op.TypeStruct, [id, ...memberTypes]));
  }

  typePointer(id: number, storageClass: number, type: number): void {
    this.typeDecls.push(this.instr(Op.TypePointer, [id, storageClass, type]));
  }

  typeFunction(id: number, returnType: number, paramTypes: number[] = []): void {
    this.typeDecls.push(this.instr(Op.TypeFunction, [id, returnType, ...paramTypes]));
  }

  /** Declare a cooperative matrix type (VK_KHR_cooperative_matrix). */
  typeCooperativeMatrixKHR(id: number, componentType: number, scope: number, rows: number, cols: number, use: number): void {
    this.typeDecls.push(this.instr(Op.OpTypeCooperativeMatrixKHR, [id, componentType, scope, rows, cols, use]));
  }

  // ── Constants ─────────────────────────────────────────────────────────

  constant(resultType: number, resultId: number, value: number): void {
    this.typeDecls.push(this.instr(Op.Constant, [resultType, resultId, value]));
  }

  constantF32(resultType: number, resultId: number, value: number): void {
    const buf = new ArrayBuffer(4);
    new Float32Array(buf)[0] = value;
    const bits = new Uint32Array(buf)[0];
    this.typeDecls.push(this.instr(Op.Constant, [resultType, resultId, bits]));
  }

  constantComposite(resultType: number, resultId: number, constituents: number[]): void {
    this.typeDecls.push(this.instr(Op.ConstantComposite, [resultType, resultId, ...constituents]));
  }

  /** OpConstantNull (opcode 46) — creates a zero-initialized constant of any type. */
  constantNull(resultType: number, resultId: number): void {
    this.typeDecls.push(this.instr(46, [resultType, resultId]));
  }

  /** OpConstantTrue (opcode 41) */
  constantTrue(resultType: number, resultId: number): void {
    this.typeDecls.push(this.instr(41, [resultType, resultId]));
  }

  /** OpConstantFalse (opcode 42) */
  constantFalse(resultType: number, resultId: number): void {
    this.typeDecls.push(this.instr(42, [resultType, resultId]));
  }

  // ── Global variables ──────────────────────────────────────────────────

  variable(resultType: number, resultId: number, storageClass: number): void {
    this.globalVars.push(this.instr(Op.Variable, [resultType, resultId, storageClass]));
  }

  // ── Function body instructions ────────────────────────────────────────

  /** Add raw instructions to the function body section. */
  addFunctionInstr(...words: number[]): void {
    this.functionBodies.push(words);
  }

  /** Helper: emit a single instruction into the function body. */
  emit(opcode: number, operands: number[]): void {
    this.functionBodies.push(this.instr(opcode, operands));
  }

  // ── Build ─────────────────────────────────────────────────────────────

  /** Assemble all sections into a SPIR-V binary (Uint32Array). */
  build(): Uint32Array {
    this.bound = this.nextId;

    const words: number[] = [];

    // Header
    words.push(MAGIC);
    words.push(this.versionWord);
    words.push(GENERATOR);
    words.push(this.bound);
    words.push(0); // schema (reserved)

    // Sections in order
    words.push(...this.capabilities);
    for (const ext of this.extensions) words.push(...ext);
    for (const ext of this.extInstImports) words.push(...ext);
    words.push(...this.memoryModelInstr);
    for (const ep of this.entryPoints) words.push(...ep);
    for (const em of this.executionModes) words.push(...em);
    for (const d of this.debugNames) words.push(...d);
    for (const a of this.annotations) words.push(...a);
    for (const t of this.typeDecls) words.push(...t);
    for (const g of this.globalVars) words.push(...g);
    for (const f of this.functionBodies) words.push(...f);

    return new Uint32Array(words);
  }
}
