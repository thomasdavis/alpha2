# Alpha chat foundations v10 — outcome

Date: 2026-08-02

**Decision:** rejected; no checkpoint selected, published, deployed to BLAH, or
shared to Discord.

V10 tested whether a second independently generated and reviewed realization of
the ten-family foundations curriculum would improve unseen conversation while
retaining V8's stopping behavior. GPT-5.4 generated 6,400 new conversations and
GPT-5.5 reviewed every candidate. After rejection and cross-wave deduplication,
the merged corpus contained 10,862 training conversations and retained the
original 615-conversation V8 development population byte-for-byte.

Training completed 400/400 steps from the exact U1 checkpoint. All 400 metric
rows were finite, all eight declared checkpoints were written, native-to-HF
parity passed for every checkpoint, and the release probes passed 6/6. This was
an operationally clean null result.

## Structural result

V10 substantially preserved the mechanical repair. Its strongest sampled
candidate, step 250, produced 3 loops in 1,845 temperature-0.7 continuations,
terminated all 1,845 with EOS, and passed 1,838 structurally. Step 300 was the
strongest greedy point on the full 615-prompt development population with zero
loops, 615 EOS terminations, and 613 nonempty outputs.

Those measurements do not establish conversational competence.

## Reference-blinded semantic rejection

GPT-5.5 reviewed all 100 cases in a fixed, source-balanced development panel
without access to checkpoint identities, references, manifests, training data,
or earlier results.

| Candidate | PASS | BORDERLINE | FAIL | Decision |
| --- | ---: | ---: | ---: | --- |
| V10 step 300 | 8 | 16 | 76 | not selected |
| V10 step 250 | 9 | 16 | 75 | not selected |

The reviewer selected `NONE`. Both candidates remained dominated by circular
restatements, wrong arithmetic, ignored latest-turn instructions, weak context
use, contradiction, and invented personal facts.

A second reference-blinded comparison then tested V10 step 300 directly against
the prior V8 step 200 checkpoint on the same 100 cases:

| Candidate | PASS | BORDERLINE | FAIL | Pairwise wins |
| --- | ---: | ---: | ---: | ---: |
| V8 step 200 | 13 | 27 | 60 | 19 |
| V10 step 300 | 7 | 18 | 75 | 5 |

The remaining 76 cases were ties. V8 was materially better in foundational
answers, context grounding, negation and contrast, language pragmatics, premise
resistance, and ordinary conversation. V10 had a small edge only in a few
multi-turn note-writing cases. The reviewer explicitly warned that V8 is still
not conversationally competent; its selection was relative, not a release
pass.

## Interpretation

V10 rejects the narrow hypothesis that another independent wave of polished,
one-shot assistant-only SFT examples is enough. The new wave did not merely
plateau: it diluted the useful V8 behavior on the fixed development panel.

The failure pattern is consistent with supervision that teaches answer surfaces
without sufficiently strengthening the model's representation of the user's
language, the latest dialogue-state change, and the operation the user is
requesting. This is a hypothesis, not a proven mechanism.

The next finite intervention therefore changes the objective. It applies
ordinary all-token causal language modeling to the same independently reviewed
synthetic conversations before a short assistant-only recovery stage. Keeping
the bytes fixed isolates objective and exposure from another generation wave.
If that also fails, the next data intervention will use linked contrast
families rather than more independent rows.

## Evidence

Canonical research root:

```text
/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v10-20260802/
```

The root contains the accepted and rejected synthetic population, independent
review lineage, exact corpus and mask hashes, run contract and metrics,
structural outputs, sampled trajectories, failed zero-row evaluation attempts,
both blinded packets, and both final GPT-5.5 reviews. Rejected checkpoint weights
remain on the dedicated Alpha pod and are not duplicated locally.

