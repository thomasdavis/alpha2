# Alpha chat foundations v11 — outcome and versioned publication

Date: 2026-08-02

**Decision:** V11 is rejected as an improvement. Phase S was not run. The step-300
checkpoint was published only because the operator explicitly requested a new,
versioned research model on BLAH. It remains labelled `quality_gate=FAIL` and does
not replace V8 or the earlier public Alpha entry.

## What was tested

Phase M started from the exact V8 step-200 native checkpoint and trained for 300
steps over the existing 10,862 GPT-5.4-generated, GPT-5.5-reviewed conversations.
The intervention changed only the learning objective: all chat tokens were
supervised instead of assistant tokens alone. The architecture, tokenizer, chat
template, greedy evaluation path, data bytes, and model-visible examples remained
fixed. Symbiogenesis was disabled.

The run completed all 300 steps with finite loss. Development loss fell from
`3.1931` at step 75 to `2.9370` at step 300. That is execution evidence, not a
quality result.

## Fixed development behavior

All four Phase M checkpoints answered and stopped on all 615 development prompts.
Compared with V8, however, each produced more detected repetition loops:

| Checkpoint | Structural | Nonempty | EOS | Loops | Mean 4-gram repeat | Max 4-gram repeat |
|---|---:|---:|---:|---:|---:|---:|
| V8 reference | 612/615 | 613/615 | 614/615 | 5 | 0.004581 | 0.278689 |
| V11 step 75 | 615/615 | 615/615 | 615/615 | 14 | 0.012056 | 0.736842 |
| V11 step 150 | 615/615 | 615/615 | 615/615 | 13 | 0.013003 | 0.772727 |
| V11 step 225 | 615/615 | 615/615 | 615/615 | 11 | 0.011577 | 0.444444 |
| V11 step 300 | 615/615 | 615/615 | 615/615 | 12 | 0.012065 | 0.428571 |

The all-token bridge therefore made response initiation and termination perfectly
reliable on this development population, but it worsened the already important
autoregressive repetition boundary.

## Reference-blinded semantic review

GPT-5.5 reviewed 100 cases with candidate identities hidden. The mapping was:

- A: V11 step 150;
- B: V8 step 200;
- C: V11 step 300;
- D: V11 step 75;
- E: V11 step 225.

| Candidate | PASS | BORDERLINE | FAIL |
|---|---:|---:|---:|
| A | 14 | 12 | 74 |
| B, V8 reference | 16 | 9 | 75 |
| C, V11 step 300 | 15 | 11 | 74 |
| D | 13 | 12 | 75 |
| E | 15 | 11 | 74 |

The reviewer ranked `B, C, E, A, D` and selected `NONE`. Dominant failures remained
wrong arithmetic, failure to apply updates, circular repetition, contradiction,
and failure to perform instructions. Since no Phase M checkpoint beat V8, the
contract prohibited the assistant-only Phase S recovery run.

Review SHA-256:
`29355fb8a4e8093472b08f0bb4438964383749c00dd2be8faf625ea468a40a1a`.
Review-packet SHA-256:
`2e093ea7fe5ccc3b64a98a601aa6c80fa8d9a147d4e057c9c03601bde3ce5a9f`.

## Versioned experimental publication

The normal selection gate did not pass. The operator nevertheless requested that
the exact checkpoint be published as a new model so it could be compared publicly
without erasing the older Alpha.

| Item | Exact value |
|---|---|
| Native checkpoint | V11 Phase M step 300 |
| Native SHA-256 | `6226c1443741058089f110b89dfa341e0325851098d3aaf049a501c1ca3393f9` |
| Hugging Face repo | `ajaxdavis/alpha-chat-v11-m300-experimental` |
| Immutable HF revision | `29f0372fb94c1d249421daca50c3fbd263dc1309` |
| Weight SHA-256 | `7fc33a82fa103233e01fd8e9aeb38531e03b3fe08655c33c79ca7f835fcd71b6` |
| Runtime | `https://donto.org/alpha-v11-m300` |
| BLAH name | `Alpha v11-m300 experimental` |
| BLAH model ID | `Mq5PrXS1MUk2yl0eSKUXwA` |
| BLAH alias | `alpha-v11-m300` |
| BLAH run | `XEDqvFu4Adbj86rKEVUqEg` |
| Quality gate | `FAIL` |

An anonymous cold load from an empty Hugging Face cache resolved the exact
revision and weight hash and generated through `LlamaForCausalLM`. Native-to-HF
final-logit parity passed 2/2 top-1 with maximum absolute difference
`7.009506225585938e-05`.

## Public BLAH result

The fresh run completed all 24 eval definitions. One eval result was an inference
error. The new entry is worse overall than the immediately preceding complete
24-eval run for the earlier Alpha entry:

| Metric | Earlier Alpha | V11 step 300 | Delta |
|---|---:|---:|---:|
| Mean score | 0.395833 | 0.362500 | -0.033333 |
| Full credit | 5/24 | 6/24 | +1 |
| Zero credit | 3/24 | 7/24 | +4 |
| Runtime errors | 0 | 1 | +1 |
| Pairwise prompt wins / ties / losses | — | 4 / 12 / 8 | — |
| Median latency | 980 ms | 26,478 ms | +25,498 ms |

The judge reasoning is mostly consistent with the raw outputs. One full-credit
result is misleading as a product signal: `Tell me about something you find
interesting.` received full credit for loop resistance even though Alpha answered
`I can't tell from the text.` The rubric tested stopping, not usefulness.

Representative successes:

- supplied context: `The capital city is Marrowind.` — correct and concise;
- greeting: `Hello!` — natural but minimal;
- naming an animal: `The animal is a puppy.` — correct;
- exact copying: `banana` — correct.

Representative failures:

- `What is 1 + 1?` -> `1 + 1.`;
- `What colour is the sky?` -> `The sky is the same color as the sky.`;
- `What is the opposite of hot?` -> `The opposite of hot.`;
- `Count from one to three.` -> `No.`;
- `Is water wet? Answer yes or no.` -> `It is wet, not wet.`;
- the early-bird completion entered a 121-token repeated fragment and never said
  `worm`;
- a strict three-word response ended in a BLAH `Bad Gateway` after three attempts;
- a multi-turn arithmetic trajectory produced an empty assistant turn, which the
  strict runtime rejected on the next call as `message 3 has empty content`.

These are semantic imitation, task-performance, repetition, and adapter failures.
They are not mainly refusal or output truncation. The chat markers used by BLAH
match Alpha's training template, but the runtime must tolerate an empty historical
assistant turn in a future version so a bad model response does not become an
HTTP failure on the next turn.

## Versioning policy

Every future public checkpoint or runtime behavior change receives all of the
following:

1. a new monotonically increasing Alpha version label;
2. a new immutable Hugging Face revision, and a new repository when it is a new
   checkpoint line;
3. a distinct versioned runtime path;
4. a new BLAH model record and model ID;
5. a new BLAH eval run;
6. an honest quality verdict bound to that exact checkpoint and runtime.

Existing BLAH model records are never repointed to a different checkpoint or
silently changed after evaluation. A runtime-only repair still increments the
public version because it can change observed behavior.

## Next intervention

Do not extend V11 and do not run Phase S. The evidence supports V12 as a new
synthetic-data intervention organized into linked contrast families: a base
dialogue, paraphrase, minimal meaning-changing edit, update, hard negative,
required invariant, and natural response variants. GPT-5.4 should generate the
families and GPT-5.5 should review boundaries and hard negatives. Selection must
remain on untouched families and free multi-turn behavior, not loss or output
non-emptiness.

## Evidence locations

Mounted run:

    /mnt/donto-data/alpha-runs/alpha-chat-foundations-v11-20260802/

Mounted research mirror:

    /mnt/donto-data/donto-resources/research/alpha-chat-foundations-v11-20260802/

BLAH evidence hashes:

| File | SHA-256 |
|---|---|
| `blah-evaluation/eval-definitions.json` | `483e8822d03171c765d45914f8061dd194fbd7e6991c82af06374066824b564c` |
| `blah-evaluation/logs.json` | `b9c202f27899c197bb4526830fd7eccaa0da8f5fdbbe660b7d360cf35de84e7d` |
| `blah-evaluation/model.json` | `967de9819e663d6aec859a38a198541132d8158c413f5ccbfdcdc863b6d47b43` |
| `blah-evaluation/results.json` | `46c28a187bac75f1c41a1a7f13649762e4d17056139017f01fcced1fb05541f5` |
| `blah-evaluation/run.json` | `14d8532fc649018e8e1b901c3c312dcf158aa4550daec3e8d6262dedeeac3fea` |
