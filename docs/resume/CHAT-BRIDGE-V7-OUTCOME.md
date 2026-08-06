# Alpha direct-semantic bridge v7 — outcome

**Decision:** rejected; no checkpoint selected, exposed to the sealed final, published, deployed to BLAH, or
shared to Discord

V7 tested whether one finite pass over 40,000 compact SmolTalk conversations plus 2,980 reviewed semantic
conversations could add meaning while preserving the mechanically stable U1 checkpoint. The accepted corpus
passed exact tokenizer and mask replay for all 45,245 train and development rows, excluded exact visible and
BLAH prompts across every user turn, and contained no exact fenced-code serialization. Training completed all
2,800 declared steps in 2,053.4 seconds with no skipped batch, allocator overflow, or gradient spike.

## Declared checkpoint evidence

| Step | Validation loss | Selector structural / 96 | Selector loops | Regression structural / 69 | Regression loops | Release loops / 6 |
| ---: | --------------: | -----------------------: | -------------: | -------------------------: | ---------------: | ----------------: |
| 400 | 2.0605 | 93 | 20 | 58 | 11 | 0 |
| 800 | 1.9806 | 94 | 23 | 60 | 16 | 0 |
| 1,200 | 1.7935 | 91 | 26 | 59 | 17 | 1 |
| 1,600 | 1.7847 | 96 | 29 | 57 | 15 | 1 |
| 2,000 | 1.9291 | 94 | 27 | 59 | 13 | 0 |
| 2,400 | 1.8023 | 95 | 25 | 58 | 12 | 1 |
| 2,800 | 1.8372 | 94 | 27 | 58 | 11 | 1 |

U1 remains stronger mechanically at 93/96 structural with six loops on the selector and 61/69 structural with
four loops on the regression suite. V7 often retained response initiation and EOS stopping, but it replaced
U1's repaired trajectories with 20–29 selector loops and 11–17 regression loops. The lowest validation loss at
step 1,600 was the worst selector checkpoint by loop count, again confirming that teacher-forced loss cannot
select a conversational model.

The fixed 24-case qualitative panel supplies no rescuable semantic gain. Across the trajectory Alpha continued
to prescribe niacin for scurvy, compute `12 + 12 = 5`, describe a Mexican pizza as the requested cuisine dish,
claim it could edit a calendar, and answer follow-up questions by recycling the preceding answer's nouns. The
text became shorter or more polished in a few cases, but wrongness, circularity, shallow continuation, and
degenerate repetition remained. A repetition-cleanup stage is therefore not justified: it would repair a
mechanically worse checkpoint without evidence that v7 installed the missing competence.

## Preservation

The small run metadata, all model-visible evaluation outputs, exact native/Hugging Face parity reports, and
checkpoint hashes are preserved under
`/mnt/donto-data/alpha-runs/alpha-chat-bridge-v7-20260802/`. Reproducible Hugging Face exports and the seven
optimizer-bearing checkpoints were not mirrored locally. Their exact identities are recorded in
`CHECKPOINT-HASHES.sha256`; the remote run remains on the dedicated Alpha pod pending later storage triage.

The next experiment must change the information density and response contract of the positive corpus. It should
use strong-teacher, short, independently useful answers and controlled follow-ups across broad foundational
language, instruction, reasoning, uncertainty, and ordinary-conversation skills; preserve exact BLAH prompt
exclusion; and carry the proven rollout-conditioned repetition objective during positive training rather than
trying to repair loops after semantic SFT.
