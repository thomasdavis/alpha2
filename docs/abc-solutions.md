# ABC Music Notation Model: 5 Solutions

## Problem
The current ABC model (BPE tokenizer, vocab=2000, 2.8MB data, 500 iters) produces garbled output like `g6Ea e F F a E Be dG` instead of valid ABC notation. Root cause: only 0.18 epochs of training — the model never saw enough data to learn ABC grammar.

## Solution 1: Char Tokenizer + Smaller Dataset (Recommended)

**Change tokenizer to char, shrink data to ~1000 tunes (~300KB).**

- Char tokenizer: vocab ~70-80 (ASCII printable chars used in ABC). Every token is meaningful — notes (A-G, a-g), durations (2,3,4), bar lines (|), headers (X:, T:, M:, K:), ornaments ({, }), etc.
- 300KB / (256 block × 4 batch) = ~293 batches/epoch. At 2000 iters = ~6.8 epochs. Comparable to chords training (105 epochs) in terms of grammar exposure.
- No bad merges crossing structural boundaries (headers vs notes vs bar lines).
- Expected: model learns ABC structure (header → notes → bar lines) within 2000 iters.

**Pros**: Natural fit for ABC. Tiny vocab = fast convergence. No tokenizer artifacts.
**Cons**: Slower generation (one char at a time). Slightly longer sequences.
**Estimate**: ~2000 iters, ~30-45 min on CPU.

## Solution 2: BPE with Smaller Vocab + Smaller Dataset

**Keep BPE but drop vocab from 2000 to 200, shrink data to ~1000 tunes.**

- Vocab 200 means merges stop early — mostly single chars + common bigrams (|:, :|, X:, etc.)
- Avoids the worst cross-structural merges while still compressing common patterns.
- Same data reduction for more epochs of coverage.

**Pros**: Slightly shorter sequences than pure char. Captures common ABC idioms as single tokens.
**Cons**: Still risk of a few bad merges. More complex than char. Marginal compression benefit at vocab 200.
**Estimate**: ~2000 iters, ~30-45 min on CPU.

## Solution 3: Line-Level Tokenizer (Custom)

**New tokenizer that treats each ABC header line as a single token, then char-tokenizes the music body.**

- Headers (X:1, M:4/4, K:G) become single tokens — there are only ~50-100 unique header values.
- Music lines are char-tokenized as in Solution 1.
- Sequences become much shorter (6-token header instead of 20+ chars).

**Pros**: Very efficient header encoding. Short sequences. Clean structure.
**Cons**: Requires a new tokenizer type (engineering effort). Header vocab may grow large with diverse data. Mixing token types adds complexity.
**Estimate**: 1-2 hours to implement tokenizer + ~30 min training.

## Solution 4: More Training on Current Setup

**Keep BPE vocab=2000 + full 2.8MB dataset, but train for 50,000+ iterations.**

- The model needs ~16,000 iters just for 1 epoch. Meaningful learning requires 5-10+ epochs = 80,000-160,000 iters.
- At ~15s/iter, 80K iters = ~333 hours (~14 days).

**Pros**: No code changes. Eventually the model would learn.
**Cons**: Completely impractical on CPU. BPE merges still create garbage tokens. Wasteful — most of the 2000-token vocab would remain undertrained.
**Estimate**: 14+ days on CPU. Not viable.

## Solution 5: Post-Processing Pipeline

**Keep current model, add ABC-aware post-processing to fix output.**

- Regex-based cleanup: strip invalid characters, fix header ordering, ensure bar line balance.
- Template injection: force valid header block (X:, T:, M:, K:) then let model fill in notes.
- Constrained decoding: at each step, mask logits to only allow valid next characters.

**Pros**: Could salvage the existing trained model. Constrained decoding is principled.
**Cons**: Can't fix fundamentally garbled output — post-processing only helps if the model is "close" to correct. Constrained decoding requires significant engineering and ABC grammar definition. The current model isn't close enough for regex cleanup to help.
**Estimate**: 2-4 hours of engineering for constrained decoding. Results uncertain.

---

## Recommendation

**Solution 1 (Char Tokenizer + Smaller Dataset)** is the clear winner:
- Simplest to implement (char tokenizer already exists)
- Fastest path to valid output
- Natural alignment between tokenizer and ABC's character-level structure
- Proven approach — our char tokenizer works, we just need the right data ratio
