# Alpha chat repair v3 — local implementation and preflight record

**Date:** 2026-08-01

**Outcome:** the RCR-UL experiment is implemented and locally verified, but it has not run on NVIDIA and no
candidate has trained. The public model remains the selected repair step 1,200 with quality gate `FAIL`.

**Implementation commits:**

- `8341dd0` — masked unlikelihood primitive, CPU and Helios paths, paired trainer, frozen-data loaders,
  telemetry, tests, cohort/rollout/compiler tools, and arm launcher;
- `5753ca9` — bind cohort generation and training to the selected checkpoint's native 512-token context;
- `b367f6b` — add batched fp32 Transformers rollout generation, fail-closed native trajectory parity, and require
  the parity evidence during accelerated mask compilation;
- `957a02b` and `db7daed` — add the development-only checkpoint evaluator, paired endpoint analyzer, exact
  eligible-69 materialization, and byte-reproducible evaluation freeze.

**Training authorization:** none is implied by this record. No paid pod was created and the sealed final remains
unopened.

At `2026-08-01T16:59:59Z`, `runpodctl pod list` showed one running pod named
`wbv-v3-checkpoint-sentinel-20260801-a1-alloc2` at `$0.44/hr`. It is unrelated to Alpha and was not touched. There
was no Alpha pod. Always recheck ownership because the list was empty earlier in the same local preflight.

## 1. What this preflight established

Repair v2 made responses begin reliably on its development population but created more loops than it fixed. The
local mechanism analysis therefore proposed one bounded causal test: compare an intervention that lowers the
probability of tokens completing repeated 4-grams on Alpha's own failed train-only rollouts against an otherwise
identical zero-weight branch.

The implementation now exists end to end:

1. freeze a train-only positive cohort and a disjoint fresh development selector;
2. generate one immutable greedy failed trajectory per selected training prompt;
3. mechanically compile repeated-4-gram completion positions into an unlikelihood mask;
4. train C0 and U1 through the exact same positive and negative branches;
5. change only the RCR-UL weight from `0.0` to `0.5`;
6. select only from free conversational behavior at predeclared checkpoints.

Local proof covers the data structures, objective, gradients, exact zero-weight control, cursor/RNG resume, and
shader construction. It does not substitute for executing all 50 GPU assertions on a real NVIDIA Vulkan device.

## 2. Important context-contract correction

The first freeze attempt used a 1,024-token planning limit. That was incorrect for this intervention. The selected
checkpoint at SHA-256

    399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec

serializes `blockSize=512`. The initial one-row smoke happened to fit, but a full cohort generated against the
1,024-token freeze would eventually have failed, while training with `--block=1024` would have introduced a
context migration into an experiment intended to change only the unlikelihood weight.

The defect was caught before paid compute or training. The corrected contract is:

- model and training block size: 512;
- maximum prompt: 384 tokens;
- reserved completion: 128 tokens;
- silent truncation: none;
- model-compatibility override: removed from the launcher.

The following earlier artifacts are preserved as superseded evidence and must not be used for v3 training:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r2-20260801/
    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r2-replay-20260801/
    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-rollout-smoke-20260801/

Nothing was deleted or overwritten.

## 3. Implemented objective and CPU proof

The new backend/autograd primitive is:

    crossEntropyUnlikelihoodMasked(logits, badTargets, mask, epsilon)

It computes the mask-normalized objective:

    -log(max(1 - p_bad, epsilon))

The analytic row gradient is:

    p_bad / max(1 - p_bad, epsilon)
      * (onehot_bad - softmax)
      * normalized_mask
      * upstream

Local coverage includes:

- hand-computed forward values;
- finite-difference gradients;
- analytic-gradient checks;
- exact-zero unmasked rows;
- exact-zero all-zero-mask batches;
- epsilon-clamp finiteness;
- upstream-gradient scaling;
- audit statistics for active rows, mask mass, mean bad-token probability, and maximum bad-token probability.

The model exposes the objective explicitly through `GPTLossObjective`; ordinary CE semantics were not silently
changed.

## 4. Helios implementation and pending NVIDIA proof

Helios now has fused masked unlikelihood forward and backward variants:

- registry name `ul_fwd_masked`;
- registry name `ul_masked_backward`.

The model-sized tensors remain GPU-resident. The forward path reads back only one loss scalar and the `N`
per-token-row loss values needed for audit statistics, never the `N × vocabulary` logits. The output buffer
remains the last binding, preserving the earlier masked-CE binding-corruption repair.

The fail-closed NVIDIA gate now requires exactly **50 executed assertions**. New coverage includes:

- forward and backward CPU/GPU parity;
- unlikelihood audit-stat parity;
- exact-zero all-zero-mask behavior;
- exact-zero upstream behavior;
- a complete paired CE plus RCR-UL AdamW step against CPU;
- deterministic GPU replay of that paired step;
- inclusion in the existing mixed-precision path.

This host exposes llvmpipe rather than NVIDIA. The GPU tests therefore correctly skip here. A real NVIDIA run
must report 50/50, zero skipped, zero failed, and zero todo before an arm may start.

## 5. Paired trainer and resume guarantees

The trainer has an explicit `rcrUl` configuration with frozen data path, scalar weight, and epsilon. It rejects:

- RCR-UL without assistant-only SFT;
- a missing negative cohort;
- negative or non-finite weights;
- epsilon outside `(0, 1e-6]` for this experiment;
- implicit positive train/development splitting;
- positive/negative row-count mismatch;
- any row whose positive-conversation SHA differs;
- oversized or silently truncated negative trajectories;
- invalid, duplicate, or unsorted penalty positions;
- a resume that changes or disables the negative-data path, weight, or epsilon.

Each microstep executes:

1. positive forward and backward;
2. GPU synchronization and graph reclamation;
3. matched negative forward and backward;
4. one shared optimizer update.

C0 supplies an exact zero upstream gradient to the negative branch rather than skipping it. The CPU regression
test proves C0 is byte-identical to a positive-only reference update. U1 changes parameters. An interrupted
paired run resumes to byte-identical terminal parameters compared with an uninterrupted run.

Per-step telemetry includes separate CE and unlikelihood losses, mask masses, examples with penalties,
first/last penalty positions, mean/max bad-token probability, pre/post-clip gradient norms, non-finite counts,
positive/negative forward/backward timing, allocator state, and a batch hash that binds both trajectories and
the negative mask.

## 6. Canonical corrected freeze

The canonical frozen directory is:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r3-20260801/

Its independent deterministic replay is:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r3-replay-20260801/

The six model-visible or selection artifacts match byte-for-byte between the canonical build and replay. The
manifest files differ only because each records its own absolute output paths.

### Counts and boundary audit

| Item | Result |
|---|---:|
| Positive train rows available | 23,529 |
| Generation-eligible at 384-token cap | 14,974 |
| Frozen rollout candidates | 4,096 |
| Per-source candidates | 1,024 each from everyday, OASST2, Smol Magpie, and SODA |
| Candidate prompt-token range | 5–384 |
| Prompt plus 128 reserve over 512 | 0 |
| Development rows available | 1,172 |
| Development eligible | 749 |
| Fresh development selector | 96, 24 per source |
| Frozen qualitative panel | 24 |
| Rollout/development identity overlap | 0 |
| Rollout/development normalized-prompt overlap | 0 |

### Canonical content hashes

| Artifact | SHA-256 |
|---|---|
| `rollout-candidates.jsonl` | `c8df6ccd79c4eb813d87c48eee9d2462837a944d24aeba1263c87515282e670a` |
| `positive-cohort.txt` | `3c9dcc8d44db15491dc94e0167e864da4fc436a49edbdbf9bac6b4b0652377da` |
| `rollout-exclusions.jsonl` | `bbea8330f6730eba9e60f578c125bddd092537f6b1e82d67d5afdece39551e2d` |
| `development-selector.jsonl` | `0133dcda7d6ae3d5d7ed315e528e6cf566f332a355ed6189525f7a9f2b90c683` |
| `development-panel.jsonl` | `c4c869f6c1dc30a9fa644d5e45782683f200db4f80bc9c54995abf0dd0983000` |
| `development-exclusions.jsonl` | `7e574f35703d80c1c0bca7a6599a079a29fd4729270854beff174e2d9e116557` |
| canonical `freeze-manifest.json` | `976ef6b37949c729a2abad77f50f46c685dcb63269af1a1963dca58428e11231` |

The canonical and replay directories each occupy about 21 MiB.

## 7. Native rollout and accelerated-parity smoke

The corrected native smoke is preserved at:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-rollout-smoke-r3-20260801/

It contains the first 24 deterministically ranked candidates: six from each source, 946 generated token decisions,
23 learned-EOS stops, one 128-token cap, and eight mechanically detected loops. The native raw JSONL SHA-256 is:

    996f6b99f15291efab3d82430f40c66646b79708d16a61ca7d78696ed1433781

The first candidate has 147 prompt tokens and generated 53 tokens, terminating at learned EOS. Its exact output
was:

> No, you can do it. You can also try a pair of bike-shoes or a pair of bike-shoes. You can also try a pair of
> bike-shoes or a pair of bike-shoes.

Its 4-gram repeat rate was `0.5102040816326531`; it was mechanically classified as a degenerate loop. This is
useful mechanism evidence and a good prospective negative trajectory. It is not a model improvement.

The native smoke proved:

- exact checkpoint SHA enforcement;
- exact candidate-file SHA enforcement;
- prompt re-tokenization equality;
- atomic assistant boundary;
- native context-reserve enforcement;
- deterministic greedy selection;
- selected and runner-up logits, probability, log-sum-exp, and f32 logit-vector hash per token;
- learned-EOS handling;
- resumable append-only output.

The single-process native decoder is the reference but is not an economical way to build all 4,096 rows. The
24-row smoke took roughly nineteen minutes over two bounded invocations on this loaded eight-core shared host.
Blindly extending that scalar path would project to multiple days and would misuse either the local host or a
rented GPU whose accelerator remained idle.

The parity-proven stock Transformers export therefore has a separate batched fp32 generator:

    scripts/generate_chat_repair_v3_rollouts_hf.py

It disables TF32 and stochastic decoding, requires CUDA outside a bounded CPU smoke, hashes the native checkpoint
and exported safetensors, validates every prompt token against the exported tokenizer, preserves the same per-token
audit, and remains append-only/resumable. The mask compiler will reject its complete ledger unless a separate
native-parity report passes and is supplied.

The accelerated 24-row CPU smoke is preserved at:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-rollout-hf-cpu-smoke-r3-20260801/

Its raw JSONL SHA-256 is:

    f60c80f972ca5449689f7c440e36ef1e73828e351face5c2122d411cc5bf7317

The fail-closed parity verifier compared all 24 conversations and all 946 generated token decisions. It passed:

- exact prompt, content, generated, and stop token trajectories;
- exact output text, stop reasons, EOS flags, loop classifications, and output hashes;
- exact selected and runner-up token IDs at every step;
- maximum chosen-logit delta `2.2649765014648438e-05`;
- maximum runner-up-logit delta `2.09808349609375e-05`;
- maximum log-sum-exp delta `2.0958954344951763e-05`;
- maximum chosen-probability delta `2.3206448295232107e-06`.

The parity report is:

    /mnt/donto-data/donto-resources/research/
      alpha-chat-repair-v3-rollout-hf-cpu-smoke-r3-20260801/native-parity-report.json

with SHA-256:

    04ecc1f3883e53a79cd3caa6b6cf3011a74b1492c3eddda19d796b587a3ce290

On the same shared CPU, batching completed the two 11/13-row portions in roughly three minutes including two
separate framework/model loads. CUDA throughput remains to be measured, but the semantic acceleration path is no
longer hypothetical: its token trajectories are exactly native on the frozen cross-source smoke.

Both smoke directories remain partial by design and neither is the canonical complete rollout ledger.

## 8. Frozen checkpoint-evaluation path

Checkpoint selection is now executable rather than an operator-only checklist. The canonical evaluation freeze
is:

    /mnt/donto-data/donto-resources/research/
      alpha-chat-repair-v3-evaluation-freeze-r2-canonical-20260801/

Its independent replay is:

    /mnt/donto-data/donto-resources/research/
      alpha-chat-repair-v3-evaluation-freeze-r2-replay-20260801/

Both files in the two directories match byte-for-byte:

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `evaluation-contract.json` | — | `c0270b2fb544fec5e03addb168841c20183ab7b7522a0937e3e0647ae0b509ce` |
| `v2-eligible69-prompts.jsonl` | 69 | `4ba67c07fea204bbc76d76fb2b9208519bdd0029aa48046bb8143b6bcdedb584` |

The first r1 freeze and replay are preserved. Their eligible-69 bytes matched, but their contract hashes differed
because the contract embedded wall-clock time and its output directory. That non-semantic provenance defect was
caught before evaluation or GPU spend. The r2 contract binds source commit `db7daed`, uses its immutable commit
time, and records the generated regression file by logical filename, so a replay is exactly reproducible.

The exact eligible-69 file is the original v2 development-suite order restricted to the 69 IDs in the canonical
transition analysis. It has no normalized-prompt overlap with the fresh 96-case selector. Its prompt lengths
range from 10 to 508 tokens; 20 exceed the fresh selector's 384-token cap. The accelerated evaluator therefore
batches full-reserve prompts normally but groups longer prompts by exact length, preserving each row's native
`min(128, 512 - prompt_tokens)` generation allowance rather than letting left padding silently consume context.

For each I0, C0, or U1 checkpoint, `evaluate_chat_repair_v3_checkpoint.ts` now fails closed unless it can prove:

- I0 is the immutable selected checkpoint, or C0/U1 is step 50, 100, 200, or 400 with a same-directory v3 run
  contract and the correct arm;
- the checkpoint architecture, initialization, freeze, selector, panel, and regression hashes match;
- the worktree is clean and the evaluator commit is recorded;
- the exported stock `LlamaForCausalLM` passes native Alpha logit and tokenizer parity for that checkpoint;
- the fresh 96 and exact regression 69 generations are complete, resumable, and hash-bound;
- raw output rows contain no held-out reference text field;
- the 24-case panel is rendered from the frozen subset, while its human verdict remains explicitly pending;
- neither sealed final is passed, executed, or inspected.

`analyze_chat_repair_v3_pair.ts` then validates matching I0/C0/U1 evidence and computes the declared loop and
structural endpoints by prompt ID. Even a mechanical pass is emitted only as
`MECHANICAL_PASS_HUMAN_PENDING`; the analyzer cannot select a candidate while the blinded qualitative comparison
is missing. It does not collapse the 69-case regression panel to an invented scalar score and cannot use loss or
BGE as the selector.

Two bounded CPU smokes are preserved:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-eval-hf-cpu-smoke-20260801/
    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-eval-regression-hf-cpu-smoke-20260801/

They cover two fresh prompts and the first two exact regression prompts, including a 391-token legacy prompt.
Both are partial diagnostics, not baseline evaluations. The raw rows contain exact prompt/checkpoint identities
and no reference-response field. No candidate was produced.

## 9. Local verification result

Commands executed after the implementation and resume-provenance repair:

    npm run typecheck
    npm test -w @alpha/tests -- --run \
      src/rcr-ul-data.test.ts \
      src/rcr-ul-trainer.test.ts \
      src/gradcheck-ops.test.ts \
      src/frozen-eval.test.ts
    npm test -w @alpha/tests -- --run src/parity-helios.test.ts
    npm test -w @alpha/tests
    python3 -m py_compile \
      scripts/build_chat_repair_v3_freeze.py \
      scripts/generate_chat_repair_v3_rollouts_hf.py \
      scripts/generate_chat_repair_v3_eval_hf.py \
      scripts/verify_chat_repair_v3_rollout_parity.py
    bash -n scripts/run_chat_repair_v3_arm.sh
    git diff --check

Results:

| Check | Result |
|---|---|
| TypeScript build | PASS |
| Focused data/loss/trainer suite | 67 passed, 0 failed |
| Local parity file | 29 NVIDIA-only assertions skipped as intended on llvmpipe |
| Full suite | 223 passed, 50 NVIDIA-gated, 0 failed |
| Python syntax | PASS |
| Native/accelerated 24-row trajectory parity | PASS, 946/946 selected tokens exact |
| Evaluation freeze replay | PASS, contract and eligible-69 bytes exact |
| Fresh/regression evaluator CPU smoke | PASS, resumable and reference-blinded |
| Arm-launcher shell syntax | PASS |
| Whitespace check | PASS |

The skipped tests are not accepted as GPU proof. They are the exact assertions that must execute on NVIDIA.

The root `npm test` Turbo wrapper still invokes `vitest run` inside workspace packages that intentionally contain
no local test files, so it exits at `@alpha/core` with “No test files found.” The authoritative combined model
suite remains `npx vitest run packages/tests/src`; it passed as recorded above. The marginal pre-existing frozen-
eval subprocess test also received a 120-second test timeout after twice crossing its five-second default under
parallel host load; its assertions were unchanged and it passed both focused and full reruns.

## 10. Operator-requested public baseline sample

On 2026-08-01 the operator explicitly requested the current public model's answer to `what is dna?` and asked
that it be posted to Discord. The exact greedy output was:

> Dna is a type of protein that is made up of two parts, called the pituitary gland and the pituitary gland. It
> is responsible for the production of the hormones that make up the body. Dna is also responsible for the
> production of the hormones that make up the body.

This answer is factually wrong and repetitive. It was posted as an explicitly labelled baseline failure, not as
an improvement. Four further operator-requested baseline examples were later posted with exact prompts, outputs,
stop reasons, repetition rates, and failure labels. These posts do not alter the improvement-only publication
policy for unsolicited progress announcements.

## 11. Artifact-size estimate

One prior optimizer-bearing checkpoint is approximately 692,528,815 bytes. Preserving a checkpoint every 50
steps through 400 produces eight checkpoints per arm, or approximately 10.3 GiB for both arms before small logs,
rollouts, and evaluations. The corrected freeze and smoke are tens of MiB. This is expected to remain below the
operator's 15 GiB pause threshold, but the launcher/run guard must measure actual growth and pause before the
project-owned additions cross it.

No checkpoint may be deleted merely to make the run appear smaller. If measured headroom is inadequate, stop
before training and revise the preservation schedule explicitly in a new contract.

## 12. Exact remaining gates

The following are still open, in order:

1. finish the complete 4,096-row fp32 accelerated rollout ledger on CUDA, require its first 24 trajectories to
   match the admitted native-parity population, and write its immutable manifest;
2. compile and independently audit the full negative cohort and mask manifest with the parity report bound;
3. build the committed source on a real NVIDIA Vulkan host;
4. execute and preserve the fail-closed 50/50 NVIDIA gate;
5. run `scripts/run_chat_repair_v3_probe.sh` and prove its selection-ineligible model-sized paired step has
   finite telemetry and feasible memory;
6. run C0 and U1 sequentially from the same checkpoint and source commit;
7. run the I0 baseline once, then evaluate only C0/U1 steps 50, 100, 200, and 400 through the frozen evaluation
   driver while preserving intermediate checkpoints;
8. apply the declared automatic and blinded conversational gates;
9. execute the sealed final only if a candidate first passes development admission;
10. export and publish only if the public-promotion contract passes.

Until steps 1–4 pass, training must not begin. Until steps 7–9 pass, no candidate may replace the public model.

## 13. Recovery order

A future agent should:

1. read `HANDOFF.md`;
2. read `CHAT-REPAIR-V2-MECHANISM-ANALYSIS-2026-08-01.md`;
3. read `CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md`;
4. read this preflight record;
5. verify Git branch `agent/alpha-chat-repair-v2-closeout` and the latest pushed commit;
6. verify the canonical r3 freeze hashes above;
7. verify the canonical r2 evaluation-contract and eligible-69 hashes above;
8. confirm no paid Alpha pod exists before creating one;
9. continue from the first open gate rather than rerunning completed local work.

The scientific truth at this boundary is simple: the local experiment machinery is substantially stronger and
more fail-closed than v2, but Alpha itself has not improved yet.
