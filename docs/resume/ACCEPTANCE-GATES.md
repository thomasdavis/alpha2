# Acceptance gates for any future Alpha continuation

These gates replace informal “looks better” judgments. They do not authorize work; they define what a
newly authorized program must prove.

## Gate A0 — authorization and contract

Required before pod creation:

- explicit current user authorization for Alpha training;
- maximum spend and pod class;
- named starting checkpoint with verified SHA-256;
- one-variable experiment hypothesis;
- target supervised-token count and stop condition;
- new immutable launcher/continuation contract;
- confirmation that no unrelated RunPod will be touched.

Failure action: do not create a pod.

## Gate A1 — local correctness

Required before paid deployment:

- complete TypeScript build;
- CPU unit and gradcheck suite passes;
- SFT mask tests pass;
- deterministic shuffle/resume tests pass;
- no secret appears in tracked files;
- source tree and artifacts are hash-bound;
- launcher contract-only positive and adversarial negative probes pass.

Failure action: remain local.

## Gate A2 — NVIDIA correctness

Required on the exact deployed source:

- NVIDIA vendor confirmed;
- canonical fail-closed GPU suite executes every expected assertion;
- zero skipped, failed, or todo GPU assertions;
- no CPU fallback for core operations;
- one bounded optimizer step matches the reference within the declared tolerance.

Failure action: preserve evidence and terminate the Alpha pod.

## Gate A3 — bounded pilot mechanics

Required during the first authorized pilot:

- metrics rows are finite, consecutive, and advancing in real wall time;
- GPU utilization and throughput prove GPU-resident execution;
- allocator telemetry is complete with zero overflow;
- RSS and external memory remain bounded across checkpoint cycles;
- every retained checkpoint passes native parameter and optimizer audit;
- remote and local copies match byte count and SHA-256;
- guard and finalizer are active with zero unexplained restarts.

Failure action: stop the pilot, mirror evidence, terminate the scoped pod.

## Gate A4 — generation-direction admission

Evaluate a sealed non-frozen development set using deterministic greedy decoding. A candidate must:

- reduce immediate-EOS rate by at least 25 percentage points relative to its declared control;
- improve aggregate semantic passes, not merely nonempty output;
- introduce zero new role leaks;
- have no increase in degenerate-loop count;
- keep maximum four-gram repetition below 0.20 for admitted responses;
- improve or preserve performance in every prompt-length band;
- show the same direction across at least two adjacent checkpoints;
- publish all inputs and outputs, including regressions.

A single compelling example cannot pass A4.

Repair v2 is the binding negative precedent for this gate. It achieved universal nonempty development output but
failed because loop counts and semantic contingency regressed on exact shared prompts. A future selector must
therefore compare only mutually generation-eligible IDs, stratify by source/prompt family, and reject a candidate
that replaces emptiness with fluent irrelevance or repetition. The v2 sealed-final suite remains unspent and is
not a development selector.

## Gate A5 — checkpoint admission

A checkpoint can be retained as the next-stage source only when:

- A3 and A4 both pass;
- its native ALPH file includes complete optimizer/RNG/tokenizer state;
- its exact source, inputs, metrics prefix, and generation report are hash-bound;
- a clean resume fixture reproduces its next batch and schedule state;
- no sharper metric-only observation is mislabeled as recoverable.

## Gate D3 — unchanged final chat bar

Use the frozen set only after the recipe and checkpoint are fixed.

- at least 95/100 structural assistant responses;
- EOS termination without user-role leakage;
- no degenerate loops across 100;
- four-gram repeat rate below 0.20;
- at least 80/100 conversational semantic PASS;
- zero gibberish-class semantic FAIL;
- report true 200-question QA results without tuning or concealment.

The archived terminal result remains 2/100 structural and 0/100 semantic PASS.

## Gate A6 — publication

Normal model publication requires:

- D3 machine PASS;
- blinded semantic PASS;
- standard Alpha/Transformers parity;
- anonymous empty-cache CPU load at an immutable Hub revision;
- complete model card with data, cost, limitations, and evaluation;
- no custom code requirement;
- public Space shows the same model and exact checkpoint provenance.

An explicit failed-quality research release may be published only under a separate operator override
that makes the failure prominent. It must never satisfy A6.

## Discord gate

Discord receives only controlled qualitative improvement:

- same input;
- before output;
- after output;
- why it improved;
- aggregate suite boundary;
- no webhook in tracked files.

Loss, throughput, checkpoint creation, and routine status are not qualitative improvement.

## Spend and termination discipline

- No CPU training.
- No concurrent producer/consumer workload on the training GPU.
- Nice and I/O-prioritize long local operations.
- Verify real metric-row movement, not only process existence.
- A stopped RunPod can retain billable storage; terminate the exact scoped pod after verified mirroring.
- Never remove another project’s pod.
