import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { RcrUlDataLoader, loadRcrUlExamples, type RcrUlExample } from "@alpha/train";

const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

function example(id: string, tokens: number[], positions: number[]): RcrUlExample {
  return {
    stableId: id,
    positiveConversationSha256: id.padEnd(64, "0").slice(0, 64),
    tokens: Int32Array.from(tokens),
    penaltyTargetPositions: Uint32Array.from(positions),
  };
}

describe("RCR-UL frozen rollout data", () => {
  it("aligns penalty token positions with shifted language-model targets", () => {
    const loader = new RcrUlDataLoader([
      example("a", [10, 11, 12, 13], [2, 3]),
      example("b", [20, 21, 22], []),
    ], 2, 4);
    const batch = loader.nextBatch();
    expect(Array.from(batch.inputs.data as Int32Array)).toEqual([10, 11, 12, 0, 20, 21, 0, 0]);
    expect(Array.from(batch.targets.data as Int32Array)).toEqual([11, 12, 13, 0, 21, 22, 0, 0]);
    expect(Array.from(batch.lossMask!.data as Float32Array)).toEqual([0, 1, 1, 0, 0, 0, 0, 0]);
    expect(loader.penaltyPositionCount).toBe(2);
  });

  it("seekBatches exactly reproduces deterministic shuffled order", () => {
    const examples = Array.from({ length: 7 }, (_, i) => example(String(i + 1), [100 + i, 200 + i], [1]));
    const uninterrupted = new RcrUlDataLoader(examples, 2, 2, 73);
    const expected = Array.from({ length: 5 }, () => Array.from(uninterrupted.nextBatch().inputs.data as Int32Array));
    const resumed = new RcrUlDataLoader(examples, 2, 2, 73);
    resumed.seekBatches(3);
    expect(Array.from(resumed.nextBatch().inputs.data as Int32Array)).toEqual(expected[3]);
    expect(Array.from(resumed.nextBatch().inputs.data as Int32Array)).toEqual(expected[4]);
  });

  it("loads the declared JSONL schema and rejects duplicate identities", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-rcr-ul-"));
    tempDirs.push(dir);
    const good = {
      schema: "alpha-rcr-ul-example-v1",
      stable_id: "rollout-1",
      positive_conversation_sha256: "a".repeat(64),
      token_ids: [10, 11, 12, 13],
      penalty_target_positions: [2, 3],
    };
    const goodPath = join(dir, "good.jsonl");
    await writeFile(goodPath, `${JSON.stringify(good)}\n`);
    const loaded = await loadRcrUlExamples(goodPath);
    expect(loaded).toHaveLength(1);
    expect(Array.from(loaded[0].tokens)).toEqual(good.token_ids);
    expect(Array.from(loaded[0].penaltyTargetPositions)).toEqual(good.penalty_target_positions);

    const duplicatePath = join(dir, "duplicate.jsonl");
    await writeFile(duplicatePath, `${JSON.stringify(good)}\n${JSON.stringify(good)}\n`);
    await expect(loadRcrUlExamples(duplicatePath)).rejects.toThrow("duplicates stable_id");
  });

  it("rejects a trajectory that would be silently truncated", () => {
    expect(() => new RcrUlDataLoader([example("a", [1, 2, 3, 4], [3])], 1, 2))
      .toThrow("maximum exact trajectory is 3");
  });
});
