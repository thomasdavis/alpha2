# Why Alpha sometimes answered and usually returned empty

## Short answer

There were two failures, not one.

First, the model did not learn a stable policy for beginning an assistant response. Teacher-forced SFT made it
reasonably good at predicting tokens inside an answer after the correct answer prefix was already present, but
the first token after the assistant marker represented only a tiny fraction of the loss. At inference, EOS often
won that first-token contest and greedy decoding stopped immediately.

Second, every generation path appended a literal space after the final `<|assistant|>` marker. Alpha's byte BPE
normally absorbs that space into the first content token. At a generation boundary the literal space instead
became a standalone token absent from the SFT boundary, activating code-fence and forum-like modes in otherwise
viable corrective checkpoints.

Occasionally a content token narrowly beat EOS. In those cases the model entered a memorized local
pattern and could produce a superficially conversational sentence. Small prompt or checkpoint changes
could flip that winner, which made quality look intermittent.

The 2026-07-31 corrective run supplied causal evidence for the training diagnosis: deterministic shuffling,
equal conversation weighting, explicit first-content-token weighting, and independent EOS weighting produced
48/48 nonempty development responses from the clean base. The prompt diagnosis was isolated on the same
checkpoint: changing only `<|assistant|> ` to `<|assistant|>` changed a code-fence loop into an ordinary response
and EOS under both fast and reference inference. See `CHAT-REPAIR-2026-07-31.md`.

## Generation-boundary correction

| Text | Token IDs | Meaning |
|---|---|---|
| `<|assistant|>` | `[257]` | correct terminal generation marker |
| `<|assistant|> ` | `[257, 32]` | erroneous standalone space token |
| `<|assistant|> Hello` | `[257, 400, 11713]` | valid known content with a space-owning first token |

Commit `cf4ad61` fixes the frozen evaluator, QA evaluator, native API, frozen-suite builder/verifier, and exported
Hugging Face chat template. Historical assistant turns still need their space before known content. The original
terminal outputs remain authentic evidence of the previously published runtime, but that 100-case result mixed
checkpoint weakness with this protocol defect and cannot be treated as a clean model-only estimate.

## Terminal evidence

The terminal frozen evaluation was deterministic greedy decoding:

| Prompt length | Prompts | Nonempty | Empty |
|---|---:|---:|---:|
| at most 100 tokens | 17 | 7 | 10 |
| 101–300 tokens | 31 | 1 | 30 |
| over 300 tokens | 52 | 0 | 52 |

By source:

- OASST2 validation: 7 nonempty of 49; every nonempty prompt was 24–84 tokens.
- Smol Magpie ultra-short: 1 nonempty of 48; it was a Python import loop.
- Everyday conversations: 0 nonempty of 3.

All eight nonempty terminal responses were still failures: six loops and two fragments. The relevant
evidence is frozen-eval-chat/chat-results.jsonl under the terminal run directory.

## Checkpoint instability was measured, not anecdotal

The same eight non-frozen prompts were decoded greedily at regular checkpoints:

| Checkpoint | Structural | Nonempty | Loops | Mean four-gram repetition |
|---:|---:|---:|---:|---:|
| 15,000 | 0/8 | 4/8 | 3 | 0.3629 |
| 17,000 | 3/8 | 4/8 | 1 | 0.0444 |
| 18,000 | 1/8 | 3/8 | 1 | 0.1129 |
| 19,000 | 0/8 | 3/8 | 2 | 0.1546 |
| 20,000 | 2/8 | 3/8 | 0 | 0.0208 |
| 21,000 | 1/8 | 2/8 | 0 | 0.0054 |
| 22,000 | 2/8 | 4/8 | 2 | 0.2352 |
| 23,000 | 1/8 | 4/8 | 2 | 0.1909 |
| 24,000 | 2/8 | 3/8 | 0 | 0.0328 |
| 25,000 | 1/8 | 5/8 | 2 | 0.2258 |
| 26,000 | 1/8 | 3/8 | 1 | 0.1237 |
| 27,000 | 0/8 | 3/8 | 3 | 0.3374 |
| 28,000 | 1/8 | 4/8 | 3 | 0.3347 |
| 29,000 | 0/8 | 4/8 | 4 | 0.4180 |
| 30,000 | 1/8 | 2/8 | 1 | 0.0484 |

This oscillation did not track teacher-forced validation loss reliably. A checkpoint could improve one
greeting while regressing the aggregate prompt set.

Two genuine but narrow improvements were preserved:

- At 15K to 17K, a star-emoji loop became a relevant greeting and follow-up question.
- At 20K to 21K, two star emojis became a relevant greeting and question.

Both comparisons explicitly reported aggregate regressions or remaining empty responses. They were not
evidence of general chat readiness.

## Ranked contributing mechanisms

### 1. Teacher forcing did not protect answer initiation

The assistant-only loss supervises every assistant token, including final EOS. In a long response,
almost all loss comes from interior continuation tokens. The first response token after the assistant
marker is roughly one target among hundreds.

During validation the correct answer prefix is supplied at every position. During generation the model
must supply its own first token. If that token is EOS, the rollout ends before its learned continuation
ability can be used. This is the central reason a low held-out loss and empty free generation coexisted.

### 2. The SFT loader did not shuffle the ordered corpus

SftDataLoader advances a monotonic cursor and wraps; it has no shuffle. splitSftExamples preserves input
order within train and validation. The corpus itself is grouped into source spans:

- 450,402 SmolTalk rows;
- 121 SmolTalk2 everyday-conversation rows;
- 32,776 SmolTalk2 system-chat rows;
- 3,439 OASST2 rows;
- 24,690 SODA rows.

The one-epoch run therefore consumed long homogeneous source blocks. This creates recency bias and
catastrophic forgetting, and it is consistent with the measured checkpoint-to-checkpoint swings.

### 3. Token-weighted training undervalued conversation starts

There was one un-packed conversation per row, but cross entropy was averaged over supervised assistant
tokens. Long answers contributed far more weight than short answers. The corpus median was hundreds of
tokens and 203,074 conversations were prefix-trimmed to fit the 1,024-token bound. Optimizing local
continuation inside long synthetic answers was much easier than learning robust one-token answer
initiation across prompt styles.

### 4. Model capacity and data heterogeneity

The 57.7M-parameter model saw a broad 511,428-conversation mixture after only 1B pretraining tokens.
It learned recognizable fragments and local templates but did not have enough robust conditional
capacity for long instructions, factual recall, and stable conversation simultaneously. Code-fence and
repetition attractors in terminal outputs reflect those learned local modes.

### 5. Greedy decoding exposes tiny logit changes as binary behavior

The frozen eval and Space UI use temperature zero. If EOS is the highest logit by any margin, output is
empty. If a content token narrowly wins, an entire response may follow. That makes a continuous scoring
weakness appear as an all-or-nothing product failure.

The exact same Space input should remain deterministic. Variation for byte-identical input at
temperature zero would be a serving defect and should be investigated separately.

## What the evidence rules out

- **Padding trained the model to emit EOS:** ruled out. Padding target IDs have loss weight zero, and
  the independent mask audit passed.
- **Assistant/user mask reversal:** ruled out by CPU and NVIDIA masked-loss tests plus the corpus mask
  audit. The earlier swapped GPU masked-CE bindings were fixed before the certified SFT source.
- **A Hugging Face conversion-only bug:** unlikely as the primary cause. Alpha and Transformers logit
  parity passed, and both the native runtime and stock Transformers reproduced immediate EOS.
- **The Space silently failing:** ruled out. Empty output is a successful model result, explicitly
  surfaced as alpha.empty_eos, and no fallback model exists.
- **More of the same epoch would necessarily fix it:** unsupported. Generation quality oscillated while
  validation remained competitive, so blind continuation could deepen the wrong behavior.

## Practical conclusion

If the program is ever reopened, the first repair target is the SFT recipe and its selection gate—not
the server and not a temperature trick. Deterministic shuffling, source balancing, explicit answer-start
measurement, and generation-gated checkpoint selection should be proven before another full run.
