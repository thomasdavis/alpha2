# Canonical evidence index

This index answers “where is the proof?” without requiring a future session to search the whole data
disk.

## 2026-07-31 chat repair

Root:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/

Primary files:

| Evidence | Purpose |
|---|---|
| `full-end2/checkpoint-1200.json` | selected native corrective checkpoint |
| `full-end2/run/config.json` | full executed configuration |
| `full-end2/run/repair-contract.json` | exact source, input hashes, and intervention contract |
| `full-end2/run/metrics.jsonl` | complete 2,200-step trajectory |
| `full-end2/eval-step-*/` | corrected free-generation comparisons at steps 1,200–2,200 |
| `full-end2/nvidia-gate/` | exact 46/46 NVIDIA assertion evidence |
| `full-end2/hf-alpha-60m-chat-repair-1200/` | standard six-file portable export |
| `final-heldout/` | untouched 100-chat/200-QA result after verified copy |
| `final-heldout-pair-analysis.json` | recomputed frozen-input and machine-gate comparison |

Selected checkpoint SHA-256:

    399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec

Corrective corpus root:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/

The corpus manifest binds 33,113 train and 1,767 development conversations. No AlphaCorpus candidate was used.
See `CHAT-REPAIR-2026-07-31.md` for the prompt-boundary diagnosis and honest output examples.

Final selected frozen outputs:

| File | SHA-256 |
|---|---|
| `final-heldout/chat-results.jsonl` | `3f1a178299468be0549f32f7c871445de2113ed652bfd82c3068588445311570` |
| `final-heldout/qa-results.jsonl` | `137a3981401e0563dd1bdde2e2fc86aa04112363deb10a879d10b3fb495c9300` |
| `final-heldout/summary.json` | `997535ef15a9cd00a44c7c7d84474539688a317d98112d25695995061b9699af` |
| `final-heldout-pair-analysis.json` | `8e6b245c9932ca93887549a6e839ce61337eb52a7925a4d3bc9930a978b29763` |

Final machine result: 55/100 structural, 70/100 nonempty, 31 loops, and 0/200 closed-book QA exact. Gate FAIL.

## Terminal run

Root:

    /mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728/

Primary files:

| Evidence | Purpose |
|---|---|
| checkpoint-30322.json | terminal native ALPH checkpoint despite historical extension |
| metrics.jsonl | complete 30,322-row SFT trajectory |
| sft-contract.json | immutable input, source, optimizer, and target-step contract |
| terminal-sft-verification.json | terminal parameter and input audit |
| flagship-sft-analysis.json | strict execution analyzer |
| terminal-finalizer-status.json | finalizer outcome and mirrored-artifact binding |
| frozen-eval-pair-analysis.json | base-versus-chat machine D3 adjudication |
| frozen-chat-semantic-review-report.json | blinded 100-case semantic failure |
| hf-export-parity.log | Alpha-versus-Transformers parity |
| terminal-manifest.sha256 | terminal remote artifact seal |

Terminal checkpoint SHA-256:

    6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8

## Frozen evaluation

Root:

    /mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728/frozen-eval-chat/

Important files and hashes:

| File | SHA-256 |
|---|---|
| chat-results.jsonl | bc369665e98ec49ae141e271508fa289d6fcbc7acc14fe8632360ba1f64fe161 |
| qa-results.jsonl | 82d3254f02f7c900e395ae82387256097a9926c4e651544215a993af5a5d0cd7 |
| summary.json | c4751b33d19f09fbb84f223397af63897975980dfcf52172e9e18905ae955930 |

Frozen manifest SHA-256:

    bf6e6ea4e7fb9ccffd2bab6283de42fe33e681679883da06d691f06cb867ac68

Machine pair report SHA-256:

    92da0b3bf5bd984c579ded700c1b2f9bfe928fe010a5352f65d1a15aea3d48c6

Semantic report SHA-256:

    35cc1a87fad2c4f258cfdbd5859d0a0106c0f2c1e8bdd0d6e5ada303a0ffc1e9

## Checkpoint sample series

The identical eight-prompt, non-frozen greedy comparisons live under:

    ad-hoc-discord-checkpoint-15000/
    ad-hoc-quality-checkpoint-17000/
    ad-hoc-quality-checkpoint-18000/
    ...
    ad-hoc-quality-checkpoint-30000/

Each directory contains results/chat-results.jsonl and results/summary.json. These are diagnostic
samples, not frozen-eval substitutes.

The two Discord-approved qualitative comparisons are:

    discord-progress/quality-improvement-15000-to-17000-casual-chat.txt
    discord-progress/quality-improvement-20000-to-21000-casual-chat.txt

They include the same input, before and after output, and an honest aggregate boundary.

## SFT corpus and masking

Root:

    /mnt/donto-data/alpha-corpora/sft-text-v2/

Files:

- sft-v2.txt — 511,428 rendered conversations.
- sft-v2.txt.manifest.json — sources, counts, ordering, hashes, and trimming.
- length-audit.json — tokenizer-bounded length distribution.
- mask-audit.json — independent assistant-only mask verification.

Corpus SHA-256:

    ffad0a376c7eac2e0ec91f0901ec1ff87cba67cc298222828ce3df1a3e60b3fb

The mask audit sampled 1,032 rows and passed atomic role markers, assistant-only state transitions,
supervised final EOS, and zero over-bound rows. This rules out padding loss but not the unshuffled
source-order problem.

## Publication reports

Corrective release:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/hf-space-published-v2.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/hf-cold-load/report.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/backend-health-public.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/backend-evidence-public.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/chat-response.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/browser/

Corrective public identities:

| Artifact | Revision / SHA-256 |
|---|---|
| Standard model revision | `ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0` |
| Native archive revision | `ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60` |
| Static Space revision | `d87e0950baf0a16ccd2859c2cee6314602ba2881` |
| Installed backend bundle | `c2bd8a24387584cf0eae11082adef235e62a7d12b901c749e5ddd23b18b642f4` |

Historical terminal publication:

    /mnt/donto-data/alpha-runs/hf-chat-publication-experimental-published-20260730.json
    /mnt/donto-data/alpha-runs/hf-checkpoint-publication-published-20260730.json
    /mnt/donto-data/alpha-runs/hf-static-space-publication-published-v2-20260730.json
    /mnt/donto-data/alpha-runs/hf-chat-cold-load-b481f469-20260730/report.json

All report PASS refers to packaging, upload, identity, or cold-load verification. It does not supersede
the failed D3 quality reports.

## Space and backend proof

Root:

    /mnt/donto-data/alpha-runs/alpha-60m-space-runtime-5bd723d-20260730/

Evidence includes:

- Caddy configuration before and staged/current copies;
- desktop and 390-pixel mobile screenshots;
- an after-empty-EOS screenshot;
- public API and service proof;
- compiled backend provenance.

## Recovery bundle

    /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/

Read RESUME.md and verify MANIFEST.sha256. The mirrored hf-archive subdirectory is the exact upload
payload for the public training-checkpoint repository.

## Repository record

- Terminal archive tag: alpha-60m-archive-20260730
- Terminal closeout commit: f5162239ae330e98880f89bf950dc69a9125a38e
- Space runtime source: 5bd723db49b15df1b80a279a016c68727270bacc
- Certified SFT training source: c333bf247fbe87b85d01f3d34789b46615dd1034

Do not use an old HANDOFF live endpoint or pod ID as current truth. The historical section is retained
only to reconstruct the paid trajectory.
