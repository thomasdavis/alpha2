import type { Ledger } from "./db.js";
import { sha256Bytes } from "./hash.js";

export interface ExportedPacketEnvelopeContract {
  format: string;
  sessionId: string;
  pass?: string;
  envelopeJson: string;
}

/**
 * Require a byte-equivalent immutable envelope in the content-addressed export
 * ledger. This is deliberately read-only and must run before submission blobs
 * or scientific evidence are written.
 */
export async function requireExportedPacketEnvelope(
  ledger: Ledger,
  contract: ExportedPacketEnvelopeContract
): Promise<string> {
  const envelopeSha256 = sha256Bytes(contract.envelopeJson);
  const result = await ledger.client.execute({
    sql: `SELECT ea.id FROM export_artifact ea
          JOIN blob b ON b.sha256 = ea.blob_sha256
          WHERE ea.format = ?
            AND ea.blob_sha256 = ?
            AND b.byte_length = ?
            AND b.media_type = 'application/json'
            AND json_valid(ea.manifest_json)
            AND json_extract(ea.manifest_json, '$.sessionId') = ?
            AND (? IS NULL OR json_extract(ea.manifest_json, '$.pass') = ?)
          LIMIT 1`,
    args: [contract.format, envelopeSha256, Buffer.byteLength(contract.envelopeJson, "utf8"),
      contract.sessionId, contract.pass ?? null, contract.pass ?? null]
  });
  if (result.rows.length !== 1) {
    throw new Error("Submission immutable envelope does not match an exported packet");
  }
  return envelopeSha256;
}
