# Alpha direct-semantic bridge v7 — accepted preflight

**Accepted corpus:** `/mnt/donto-data/donto-resources/research/alpha-chat-bridge-v7-r2-20260802/`

**Source commit:** `674719d8a2c453377442b43fcf8a789fa47b3032`

## Why r1 was rejected

The first deterministic build passed its structural and contamination checks but failed a product-content audit.
Of 40,000 selected direct rows, 10,640 contained fenced code. A hash-selected 24-row inspection visibly
overrepresented Python, SQL, CSS, and JSON tasks. That corpus is preserved at
`alpha-chat-bridge-v7-20260802`, but no model saw it.

R2 adds one bounded syntax exclusion for the exact fenced-code delimiter. It does not use a programming keyword
list, topic names, answer keys, or a semantic classifier. Residual inline technical content remains possible and
is reported honestly.

## R2 identity

| Artifact                          |   Rows |      Bytes | SHA-256                                                            |
| --------------------------------- | -----: | ---------: | ------------------------------------------------------------------ |
| train                             | 42,980 | 41,028,266 | `51f83c105fa295672ffacfb271c0b135f12bdb148147169a445185e306574e24` |
| development                       |  2,265 |  2,130,742 | `79dd59067ac7a13e0f81e8610610e4cb87ee9aa47f0825c6c46ce98ebc328127` |
| catalog                           | 45,245 | 12,962,017 | `a63f02238ddc9fab93a2e45965946052ce19f47e4c25d8e6019746f38c5f5e9f` |
| manifest                          |      — |      4,845 | `f5cb4183348351a6df3b58db77c18ac23f73784117576ae7093fbfab198dcd55` |
| exhaustive train mask audit       | 42,980 |      1,227 | `83d9baba49b8ab9a2a6f93f2dcb063260284c9a4071d017b82f3960d19bda3c5` |
| exhaustive development mask audit |  2,265 |      1,217 | `71fb3ddb4e29158364d6345e2b1963fa09dd59a932f04e1deb4ce61724c94d63` |

The train split contains 40,000 direct rows and 2,980 reviewed semantic rows. The development split contains
2,000 direct rows and all 265 reviewed semantic development rows. The stricter whole-conversation holdout rule
excluded 2,124 older semantic-train rows that shared a user move with a visible suite or semantic development.
The direct census retained 89,217 eligible rows after excluding 59,733 fenced-code rows, 37,070 overlength rows,
23 exact holdout-prompt collisions, and all non-single-exchange or non-SmolTalk rows.

## Executable gates

- 42,980/42,980 train and 2,265/2,265 development conversations pass the structural validator.
- Exhaustive mask replay covers all 45,245 rows with the exact Alpha tokenizer.
- Role markers are atomic; user and scaffolding targets are masked; every final EOS is supervised.
- Train token lengths are 17 / 214 / 356 / 379 / 512 at min/p50/p95/p99/max; zero exceed block 512.
- Development token lengths are 28 / 209 / 354 / 378 / 479; zero exceed block 512.
- Train and development conversation hashes are disjoint.
- The frozen 24-item BLAH baseline and all visible v4 suites are exact-normalized holdouts.
- The inherited sealed final was not read.
- A deterministic 24-row direct-content audit contains no fenced code and spans explanations, rewriting,
  summaries, arithmetic/science, formatting, and ordinary practical requests. It also shows substantial
  constraint-following and summarization content, so the run is a bridge experiment rather than a claim that the
  corpus already embodies Alpha's final philosophical specialization.

## Runtime gate

The GPU staging check reproduced the corpus, audit, tokenizer, and U1 hashes byte-for-byte. Training began only
from U1 step 400 SHA-256 `0453a842b264c80c3578bc419c3dc94b46420aca30cad93593d62c812f5710fb`, on the clean exact source commit,
after the first real GPU step completed and reported zero allocator overflow.

No checkpoint is selected by this preflight. Selection remains free-generation-only under
`CHAT-BRIDGE-V7-CONTRACT.md`.
