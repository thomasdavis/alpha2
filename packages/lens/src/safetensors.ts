import { mkdir, open, readFile } from "node:fs/promises";
import { dirname } from "node:path";

export type SafeDtype = "F32" | "F16";

export interface SafeTensorValue {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

interface TensorHeader {
  readonly dtype: SafeDtype;
  readonly shape: number[];
  readonly data_offsets: [number, number];
}

export async function writeLensSafetensors(
  path: string,
  tensors: ReadonlyMap<string, SafeTensorValue>,
  dtype: SafeDtype,
  metadata: Record<string, string> = {},
): Promise<void> {
  const header: Record<string, unknown> = { __metadata__: metadata };
  const encoded: Uint8Array[] = [];
  let offset = 0;
  for (const [key, tensor] of tensors) {
    const expected = tensor.shape.reduce((product, dimension) => product * dimension, 1);
    if (expected !== tensor.data.length) {
      throw new Error(`${key} shape [${tensor.shape}] has ${expected} elements but data has ${tensor.data.length}`);
    }
    const bytes = dtype === "F32" ? f32Bytes(tensor.data) : f16Bytes(tensor.data);
    header[key] = { dtype, shape: [...tensor.shape], data_offsets: [offset, offset + bytes.byteLength] };
    encoded.push(bytes);
    offset += bytes.byteLength;
  }

  let headerText = JSON.stringify(header);
  let headerBytes = new TextEncoder().encode(headerText);
  const padding = (8 - (headerBytes.length % 8)) % 8;
  if (padding > 0) {
    headerText += " ".repeat(padding);
    headerBytes = new TextEncoder().encode(headerText);
  }

  await mkdir(dirname(path), { recursive: true });
  const handle = await open(path, "w");
  try {
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(headerBytes.length));
    await handle.write(prefix);
    await handle.write(headerBytes);
    for (const bytes of encoded) await handle.write(bytes);
  } finally {
    await handle.close();
  }
}

export async function readLensSafetensors(path: string): Promise<{
  tensors: Map<string, SafeTensorValue>;
  dtypes: Map<string, SafeDtype>;
  metadata: Record<string, string>;
}> {
  const bytes = await readFile(path);
  if (bytes.length < 8) throw new Error(`${path} is shorter than a safetensors header`);
  const headerLength = Number(bytes.readBigUInt64LE(0));
  const dataStart = 8 + headerLength;
  const raw = JSON.parse(bytes.subarray(8, dataStart).toString("utf8")) as Record<string, unknown>;
  const tensors = new Map<string, SafeTensorValue>();
  const dtypes = new Map<string, SafeDtype>();
  const metadata = (raw.__metadata__ ?? {}) as Record<string, string>;
  for (const [key, value] of Object.entries(raw)) {
    if (key === "__metadata__") continue;
    const info = value as TensorHeader;
    if (info.dtype !== "F32" && info.dtype !== "F16") throw new Error(`${key} uses unsupported dtype ${info.dtype}`);
    const start = dataStart + info.data_offsets[0];
    const end = dataStart + info.data_offsets[1];
    const slice = bytes.subarray(start, end);
    const data = info.dtype === "F32" ? decodeF32(slice) : decodeF16(slice);
    const expected = info.shape.reduce((product, dimension) => product * dimension, 1);
    if (data.length !== expected) throw new Error(`${key} byte length does not match shape [${info.shape}]`);
    tensors.set(key, { shape: info.shape, data });
    dtypes.set(key, info.dtype);
  }
  return { tensors, dtypes, metadata };
}

export function floatToHalf(value: number): number {
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);
  float[0] = value;
  const x = bits[0];
  const sign = (x >>> 16) & 0x8000;
  let mantissa = x & 0x7fffff;
  let exponent = (x >>> 23) & 0xff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  exponent = exponent - 127 + 15;
  if (exponent >= 0x1f) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  return sign | (exponent << 10) | ((mantissa + 0x1000) >>> 13);
}

export function halfToFloat(value: number): number {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x3ff;
  let bits: number;
  if (exponent === 0) {
    if (mantissa === 0) bits = sign;
    else {
      exponent = 1;
      while ((mantissa & 0x400) === 0) { mantissa <<= 1; exponent--; }
      mantissa &= 0x3ff;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  const raw = new Uint32Array([bits >>> 0]);
  return new Float32Array(raw.buffer)[0];
}

function f32Bytes(data: Float32Array): Uint8Array {
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function f16Bytes(data: Float32Array): Uint8Array {
  const halves = new Uint16Array(data.length);
  for (let index = 0; index < data.length; index++) halves[index] = floatToHalf(data[index]);
  return new Uint8Array(halves.buffer);
}

function decodeF32(bytes: Uint8Array): Float32Array {
  const copy = bytes.slice();
  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

function decodeF16(bytes: Uint8Array): Float32Array {
  const copy = bytes.slice();
  const halves = new Uint16Array(copy.buffer, copy.byteOffset, copy.byteLength / 2);
  return Float32Array.from(halves, halfToFloat);
}
