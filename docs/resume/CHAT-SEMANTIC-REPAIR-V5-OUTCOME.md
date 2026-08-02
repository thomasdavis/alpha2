# Alpha semantic repair v5 outcome

Date: 2026-08-02

**Verdict:** rejected; no checkpoint selected or published.

V5 completed 1,600/1,600 steps from the clean pretrained parent in 1,205.1
seconds on the dedicated NVIDIA run. All eight declared checkpoints were
written and passed native-to-Hugging-Face export parity. A first evaluation
attempt stopped because a fresh CUDA process lacked the deterministic CuBLAS
workspace setting; that attempt is preserved, the evaluator was fixed at
commit `1926bdb`, and the complete evaluation then passed operationally.

## Mechanical trajectory

| Step | Selector structural | Selector loops | Regression structural | Regression loops | Release-probe loops |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 85 | 45 | 50 | 33 | 5 |
| 400 | 75 | 51 | 47 | 29 | 5 |
| 600 | 74 | 47 | 46 | 29 | 5 |
| 800 | 82 | 32 | 51 | 24 | 1 |
| 1000 | 79 | 32 | 51 | 23 | 3 |
| 1200 | 82 | 37 | 56 | 19 | 5 |
| 1400 | 83 | 33 | 53 | 24 | 3 |
| 1600 | 84 | 34 | 53 | 18 | 3 |

The public baseline was selector `83/35` and regression `55/24` for structural
passes/loops. Several v5 points traded stopping against repetition, but none
dominated the baseline.

## Semantic rejection

The strongest-looking checkpoints remained wrong on the untouched live probes.
Step 1,200 said:

- DNA is “a type of DNA” made from cells;
- a promise and prediction both “tell you what you are trying to do”;
- the empathy response was a broken apology script;
- the ambiguity response contradicted itself without identifying noun/verb
  readings;
- the committee answer repeated generic group language.

Step 800 had fewer mechanical release-probe loops but the same failures. It
called DNA a type of DNA made of cells, treated a promise as a prediction, and
answered the empathy prompt with an irrelevant apology template. This proves
that clean initialization alone is not the repair.

## Preserved evidence

Local evidence root:

```text
/mnt/donto-data/alpha-runs/alpha-chat-semantic-v5-20260802/
```

It contains the complete run contract, configuration, metrics, training log,
checkpoint hashes, parity reports, generated outputs, audits, panels, and the
preserved failed evaluator attempt. Large rejected checkpoints and reproducible
HF exports remain on the live pod while the next intervention is active; no v5
checkpoint is release-eligible.

## Consequence

The next intervention is broad conversational foundation training, not another
micro-corpus or anti-loop objective. Narrow semantic data can be revisited only
after the model demonstrates ordinary direct-answer competence.
