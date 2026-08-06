# Alpha broad chat foundation v6 — outcome

**Decision:** rejected; no checkpoint selected, published, exposed to the sealed final, or deployed to BLAH

V6 tested whether the previous semantic failures were caused by insufficiently broad positive SFT. It started
from the exact clean pretrained checkpoint and trained for 8,000 finite steps on 500,186 contamination-filtered
conversations from the canonical SFT corpus. The run completed in 5,943.9 seconds. One gradient spike at step
1,253 was skipped by the declared guard, the learning-rate scale recovered after 200 clean steps, and no later
spike or allocator overflow occurred.

## Declared checkpoint evidence

|  Step | Validation loss | Fresh structural / 96 | Fresh loops | Regression structural / 69 | Regression loops | Release structural / 6 | Release loops |
| ----: | --------------: | --------------------: | ----------: | -------------------------: | ---------------: | ---------------------: | ------------: |
| 1,000 |          3.0118 |                    63 |          57 |                         34 |               32 |                      4 |             3 |
| 2,000 |          3.1821 |                    58 |          57 |                         37 |               33 |                      4 |             3 |
| 3,000 |          3.0547 |                    68 |          51 |                         37 |               31 |                      4 |             3 |
| 4,000 |          2.9829 |                    59 |          52 |                         34 |               29 |                      4 |             3 |
| 5,000 |          3.0864 |                    64 |          51 |                         35 |               31 |                      5 |             3 |
| 6,000 |          2.9126 |                    62 |          47 |                         35 |               37 |                      4 |             3 |
| 7,000 |          3.1362 |                    64 |          50 |                         38 |               36 |                      4 |             2 |
| 8,000 |          3.0890 |                    59 |          47 |                         35 |               34 |                      4 |             2 |

The public baseline remains materially stronger mechanically at 83/96 structural with 35 loops on the fresh
selector and 55/69 with 24 loops on the regression suite. V6 therefore had no selection-eligible checkpoint.
Its best validation loss at step 6,000 did not correspond to its best conversational behavior.

Step 8,000's fixed qualitative panel also failed on meaning: it described scurvy as a niacin deficiency,
classified Parmesan cuisine as Mexican, fabricated a Tokyo yakitori venue, failed a simple arithmetic update,
and repeatedly continued earlier turns instead of answering the current user. Some social-dialogue turns were
short and plausible, but they did not compensate for the aggregate mechanical and semantic regressions.

## Preservation

Local metadata and all model-visible outputs are under
`/mnt/donto-data/alpha-runs/alpha-chat-foundation-v6-20260802/`. The eight optimizer-bearing checkpoints remain
on the dedicated Alpha pod and are identified by `CHECKPOINT-HASHES.sha256`; they were not duplicated onto the
local disk because v6 is rejected and retained project artifacts remain subject to the 15 GiB soft pause. All
parity-verified Hugging Face exports are reproducible and were excluded from the local mirror.

The evaluation controller encountered three disk-quota interruptions while exporting step 7,000. Each failed
attempt is preserved under `evaluations/failed-attempts/`. After removing only reproducible parity-verified model
exports, steps 7,000 and 8,000 completed normally. This infrastructure incident did not create a quality result.

The next experiment is v7, which preserves U1's proven autoregressive stability and adds a compact direct-answer
bridge instead of repeating broad clean-base SFT.
