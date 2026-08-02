# Alpha chat recipe V12 `3e-4` pilot outcome

Date: 2026-08-02

**Decision:** rejected. No checkpoint was selected, published, sent to Discord,
or exposed to the sealed final evaluation.

This arm tested the closest public SmolLM2/Smol-SmolTalk SFT recipe that could
be reproduced faithfully in Alpha: the complete rendered conversation received
ordinary next-token supervision, conversations were packed into 1,024-token
windows, and training began from Alpha's clean pretrained checkpoint rather
than a prior post-training run. The arm used peak learning rate `3e-4`, minimum
learning rate `3e-5`, 200 warmup steps, batch size 16, and 2,000 optimizer
steps. Symbiogenesis, assistant-only masking, RCR-UL, and bespoke synthetic
curricula were disabled.

## Immutable identity

- training source commit:
  `7754238337524c8caf5d16d3ac24a55f874b5b9c`;
- clean parent SHA-256:
  `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049`;
- native tokenizer SHA-256:
  `c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24`;
- rendered training corpus SHA-256:
  `e15e19f100040565faac1ed0381ed6e3db2a06c2b9a197b756fc0dd7c20b8f2a`;
- rendered validation corpus SHA-256:
  `0b6e240d5ffcbb3a26d961bcd81f37787830ff9ebfe37d4e0faa528fcdcd701c`;
- corpus manifest SHA-256:
  `68365ae0e2e6c4289a5ab1fd4458fd67b92085dd15475f4ccbe6723448046617`;
- evaluation-freeze SHA-256:
  `3e5a35d01644961bf464c627b527cf99290b1ed6f56467ebaccfbe86a4c66908`;
- evaluator commit:
  `a3b92bb64df6d117ccf5f4d5f1f3ffd73e7d7e53`.

The corpus contains 277,479,707 native training tokens and 14,532,749 native
validation tokens. One packed pass is 16,936 optimizer steps, so the pilot
exposed only about 11.8% of one pass. The short length was intentional: the
frozen contract required a free-generation viability signal before authorizing
the full two-pass public schedule.

## Operational result

The dedicated RTX 4090 run completed all 2,000 declared steps at roughly
5.1K-5.5K tokens per second. GPU allocator warm-up caused bounded VRAM
oscillation, but `gpu_allocator_free_range_overflows` remained zero.

The first process reached step 1,750 but the checkpoint close failed with
system error `-122` after the pod's per-volume quota was reached. The incomplete
201,326,592-byte file was preserved separately and never evaluated. Completed
earlier checkpoints were losslessly compressed only after `zstd -t` and a
decompressed SHA-256 comparison passed. Removing only the invalid partial from
the run directory and compressing completed checkpoints made enough quota for a
resume from the complete step-1,500 checkpoint.

The resumed Vulkan trajectory was close but not bit-identical to the original
segment. The first attempt reported validation loss 1.8554 at step 1,750; the
replayed run reported 1.8574. The fresh step-1,750 and step-2,000 files are the
only files evaluated for those steps. This replay difference is preserved as a
known limitation, not hidden by merging metrics as though the run were one
uninterrupted deterministic trajectory.

Every evaluated checkpoint passed native-to-Hugging-Face parity with 100% top-1
agreement on the parity prompt and maximum absolute logit error below `1e-3`.
The evaluator initially had a stale hard-coded 512-token export check. That
attempt stopped before any rollout. Commit `a3b92bb` made the already-supported
512/1,024 context set explicit in the Python generator; 223 tests passed and 50
NVIDIA-only tests remained correctly gated. A fresh evaluator identity was then
used for every rollout below.

## Frozen free-generation trajectory

All rows were generated greedily for at most 128 new tokens. `Structural` is a
necessary mechanics gate, not a semantic-quality score.

| Step | Selector structural / 96 | Selector loops | Selector role leaks | Regression structural / 69 | Regression loops | Regression role leaks | Release loops / 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 37 | 72 | 4 | 31 | 42 | 3 | 6 |
| 500 | 22 | 77 | 32 | 27 | 41 | 15 | 6 |
| 750 | 38 | 70 | 5 | 32 | 37 | 2 | 6 |
| 1,000 | 55 | 61 | 4 | 39 | 37 | 2 | 5 |
| 1,250 | 37 | 62 | 20 | 28 | 41 | 10 | 5 |
| 1,500 | 17 | 77 | 34 | 28 | 45 | 14 | 4 |
| 1,750 | 32 | 73 | 25 | 22 | 46 | 11 | 3 |
| 2,000 | 26 | 73 | 28 | 28 | 42 | 11 | 5 |

The selected public Alpha baseline is selector `83/96` structural with 35
loops and regression `55/69` structural with 24 loops. Step 1,000 was the best
mechanical window in this arm, but it remained far behind the public baseline:
it had 22 fewer selector structural passes, 26 more selector loops, 16 fewer
regression structural passes, and 13 more regression loops.

The trajectory was non-monotonic despite generally improving held-out loss.
After step 1,000, role-marker leakage and failure to stop increased sharply.
This is another direct demonstration that validation loss cannot select a
chatty Alpha checkpoint.

### Strict model gate versus correct live turn boundary

Inspection of the generated token IDs found a separate deployment defect.
Alpha's multi-turn serialization is:

```text
<|user|> question <|assistant|> answer <|user|> follow-up ... <|end_of_text|>
```

There is no dedicated end-of-turn token after `answer`. The next `<|user|>` is
therefore a valid assistant-turn boundary, while `<|end_of_text|>` ends the
entire serialized conversation. The live HF runtime had stopped only on
`<|end_of_text|>`, causing it to stream a model-generated next user turn.
Commit `4f704c7` now stops before any atomic user, assistant, or whole-dialogue
boundary in both streaming and non-streaming generation. The helper fails
closed if the tokenizer does not encode all three markers as distinct atomic
tokens; the runtime build, TypeScript build, and 224 tests pass.

The frozen strict-EOS evaluation above was **not** rewritten after seeing the
result. It remains a useful identical-decoding comparison with the public
baseline and measures whether the model itself chooses whole-dialogue EOS. A
separate post-hoc runtime-boundary diagnostic truncated each existing rollout
at the first role marker without regenerating it. At step 1,000 this changed the
selector clean-stop count from 55 to 59 and loops from 61 to 60; regression
clean stops remained 41 and loops changed from 37 to 36. Release probes still
had five loops in six prompts. The correct runtime fixes simulated-role leakage
but does not rescue the arm's repetition or semantic failures, so the rejection
decision is unchanged.

## What the model actually did

The arm removed immediate silence but did not install reliable single-turn
conversation. Typical outputs did one or more of the following:

- repeated a generic sentence until the token limit;
- emitted `<|user|>` and `<|assistant|>` and continued both sides of a training
  conversation;
- restated the prompt in a circular definition;
- produced fluent-looking but factually and logically incoherent lists;
- switched to an unrelated generic support or planning template.

For example, step 1,750 answered an aviation prompt by repeating "the types of
aviation" and answered a divisor question with a contradictory list. A private
step-1,000 probe answered `What is DNA?` by calling DNA "a type of DNA" and then
repeating a molecule/protein template. These samples were not sent to Discord
because they are regressions, not improvements.

## Interpretation

The public full-sequence packing recipe fixed neither the public-baseline gap
nor Alpha's semantic weakness at this short exposure. The closest public
SmolLM2 SFT checkpoint is built on a base with vastly more pretraining exposure;
the result therefore supports a foundation-exposure diagnosis at least as much
as a post-training-recipe diagnosis. It does **not** prove that packed
full-sequence training can never work for Alpha, because the pilot consumed
less than one eighth of a pass. It does prove that spending the full two-pass
budget on this learning-rate arm is not justified by the frozen viability gate.

The predeclared `1e-3` arm remains necessary because it is the closest match to
the public recipe and differs only in learning rate. It must clear the same
free-generation gate before any longer run is considered.

## Preserved evidence

Compact evaluation evidence, including every model-visible output, summary,
audit, qualitative panel, parity report, evaluator log, and manifest, is under:

```text
/mnt/donto-data/alpha-runs/alpha-chat-recipe-v12-20260802/evaluations/
```

All eight native checkpoints, the exact run contract, configuration, and
metrics are under:

```text
/mnt/donto-data/alpha-runs/alpha-chat-recipe-v12-20260802/lr3e4/
```

`RAW-CHECKPOINT-HASHES.sha256` identifies the exact uncompressed checkpoint
bytes. `CHECKPOINT-HASHES.sha256` identifies the retained `.zst` files. Every
retained archive passed decompression and raw-hash verification. The arm uses
about 1.6 GiB after lossless compression, well below the 15 GiB review pause.
