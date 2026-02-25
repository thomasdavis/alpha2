# LLM-as-Judge Evolutionary Fitness: Phenotype-Driven Training

Date: 2026-02-25
Status: Research — mad science tier
Depends on: Symbio v1 shipping, inference sampling infrastructure (already exists)

---

## The Idea

Stop optimizing for loss. Start optimizing for **what the model actually says**.

Every N training steps, generate inference samples from the current model. Run those samples through two evaluation tiers:

1. **Algorithmic analysis** — computed locally, zero API cost. Lexical diversity, syntactic complexity, information density, repetition rates, Zipf compliance, discourse coherence, all of it. Hard numbers, no vibes.
2. **Judge LLM analysis** — Claude evaluates what algorithms can't: pragmatic competence, reasoning quality, theory of mind, humor, emergent capabilities. PhD-level linguistic analysis, not "rate coherence 1-10".

The combined scores form a **phenotype fitness vector** — a rich multi-dimensional portrait of what the model can actually do. That vector becomes the fitness function for evolutionary decisions: activation mutations, hyperparameter adaptation, training curriculum.

This is RLHF applied not to the weights but to the **training configuration itself**. You're not fine-tuning with the judge's feedback. You're evolving the process that produces the model.

---

## Two-Tier Evaluation Architecture

### Tier 1: Algorithmic Metrics (local, free, every sample)

Computed directly from the generated text using established computational linguistics methods. No LLM call needed. These run on the training machine as part of sample generation.

### Tier 2: Judge LLM Analysis (async, ~$0.30/run, periodic)

Handles everything that requires actual understanding — pragmatics, reasoning, creativity, emergent behavior. The judge receives both the raw samples AND the Tier 1 metrics, so it can ground its analysis in hard data.

```
Trainer step N (judgeInterval)
  │
  ├─ Generate inference samples (existing infra: trainer.ts:782+)
  │   10-20 prompts × 2-3 temperatures
  │
  ├─ Tier 1: Algorithmic analysis (local, sync, <100ms)
  │   Compute all lexical/syntactic/information metrics
  │   Store immediately in DB
  │
  ├─ Tier 2: Judge LLM (async, non-blocking)
  │   POST samples + Tier 1 metrics + rubric to Claude API
  │   Scores arrive 3-10 seconds later via callback
  │   Store in DB
  │
  └─ Phenotype vector assembled from both tiers
      → CUSUM monitors check for regressions
      → Evolutionary pressure applied at next safe boundary
```

---

## Tier 1: Algorithmic Metrics — Full Catalog

Every metric below is computed from the raw generated text. No LLM needed. All have established computational linguistics lineage.

### 1.1 Lexical Richness

Measures vocabulary diversity and sophistication. A model that uses 40 unique words in a 100-word response is doing something different from one that uses 15.

| Metric | Formula / Method | What it reveals | Range |
|--------|-----------------|-----------------|-------|
| **Type-Token Ratio (TTR)** | unique words / total words | Raw vocabulary diversity | 0–1 |
| **MATTR** | Mean TTR over sliding 50-word windows | Stable vocabulary diversity (length-independent) | 0–1 |
| **Yule's K** | `10⁴ · (Σ i²·V(i) - N) / N²` where V(i) = words appearing exactly i times | Vocabulary richness invariant to text length | 0–∞ (lower = richer) |
| **Hapax Ratio** | words appearing once / total unique words | Proportion of rare/novel word usage | 0–1 |
| **Lexical Density** | content words (N, V, Adj, Adv) / total words | Information-bearing word proportion | 0–1 |
| **Lexical Sophistication** | words NOT in top-2000 frequency list / total words | Proportion of less common vocabulary | 0–1 |
| **Word Frequency Profile** | mean log-frequency from word frequency corpus | Average rarity of vocabulary used | continuous |
| **Bigram Diversity** | unique bigrams / total bigrams | Phrasal variety (catches formulaic repetition) | 0–1 |
| **Trigram Diversity** | unique trigrams / total trigrams | Extended phrase variety | 0–1 |

**Evolutionary signal**: Low lexical diversity + decreasing TTR → model is collapsing into repetitive patterns. Try: higher dropout, activation change, LR perturbation. Increasing lexical sophistication over training → model is acquiring vocabulary. Stable Yule's K near natural language values (~100-200 for English conversation) → healthy vocabulary distribution.

### 1.2 Syntactic Complexity

Measures structural sophistication of generated sentences. A model producing "I like dogs. Dogs are good. I like cats." is structurally impoverished compared to "While I've always enjoyed the company of dogs, I find that cats offer a different kind of companionship."

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| **Mean Sentence Length** | words per sentence | Basic complexity proxy |
| **Sentence Length Variance** | std dev of sentence lengths | Structural variety (low variance = monotonous) |
| **Clause Density** | clauses per sentence (split on subordinators/coordinators) | Embedding depth |
| **Subordination Ratio** | dependent clauses / total clauses | Hierarchical complexity |
| **Mean Dependency Distance** | avg word distance between head-dependent pairs | Working memory demand, processing complexity |
| **Max Parse Depth** | deepest nesting in constituency parse | Maximum structural complexity |
| **Left-Branching Ratio** | left-branching structures / total branching | Typological naturalness for English (should be moderate) |
| **POS Tag Entropy** | Shannon entropy of POS tag distribution | Syntactic category diversity |
| **POS Bigram Entropy** | Shannon entropy of POS tag pair distribution | Syntactic transition diversity |
| **Unique Syntactic Patterns** | unique POS trigram sequences / total | Structural creativity |

**Parsing approach**: lightweight rule-based POS tagging and dependency heuristics — not full constituency parsing. Accurate enough for these aggregate measures without requiring a separate NLP model.

**Evolutionary signal**: Sentence length variance near zero → the model is stuck in a structural rut. Subordination ratio plateauing → the model has found its complexity ceiling for the current config. POS entropy dropping → structural collapse. Mean dependency distance increasing without clause density increase → the model is producing long but structurally flat sentences (bad).

### 1.3 Discourse Coherence

Measures how well sentences connect to each other. Coherent text has logical flow; incoherent text reads as a bag of unrelated sentences.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| **Lexical Overlap** | Jaccard similarity of content words between adjacent sentences | Local topical cohesion |
| **Sentence Similarity Decay** | similarity between sentence i and sentence i+k for k=1,2,3 | How quickly coherence degrades with distance |
| **Connective Density** | discourse connectives (however, therefore, because, ...) per sentence | Explicit logical structure |
| **Connective Variety** | unique connectives / total connectives | Range of logical relations expressed |
| **Coreference Chain Length** | mean length of pronoun/referent chains | Ability to maintain referential continuity |
| **Given-New Ratio** | repeated content words / new content words per sentence | Information management |
| **Topic Drift Rate** | 1 - cosine similarity of first and last sentence (bag-of-words) | How much the response wanders from its starting point |
| **Paragraph Coherence** | mean adjacent-sentence similarity across full response | Global coherence score |

**Evolutionary signal**: Lexical overlap near zero → the model produces unrelated sentences. Topic drift rate high → it can't stay on topic. Given-new ratio too high (all given) → it's repeating itself. Given-new ratio too low (all new) → it's topic-hopping without building on previous sentences.

### 1.4 Repetition and Degeneration

The most critical failure mode for small language models. A model that repeats itself is worthless regardless of how good its other metrics are.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| **Self-Repetition Rate** | proportion of n-grams (n=3,4,5) that appear more than once in the response | Degenerate repetition |
| **Self-BLEU** | average BLEU score between pairs of responses to different prompts | Cross-sample diversity (low = good) |
| **Distinct-1** | unique unigrams / total unigrams | Unigram diversity (Li et al. 2016) |
| **Distinct-2** | unique bigrams / total bigrams | Bigram diversity |
| **Distinct-3** | unique trigrams / total trigrams | Trigram diversity |
| **Longest Repeated Substring** | length of longest substring appearing 2+ times | Worst-case repetition loop |
| **Repetition Onset** | token index where first significant repetition begins | How far the model gets before degenerating |
| **Loop Detection** | whether any n-gram (n≥3) repeats 3+ times consecutively | Binary: stuck in a loop or not |
| **Response Uniqueness** | 1 - (max pairwise similarity among responses to different prompts) | Are all responses the same regardless of prompt? |

**Evolutionary signal**: These are the canary in the coal mine. Rising self-repetition → the model is degenerating. This should trigger immediate mutation (LR change, activation change, temperature increase). Loop detection firing → catastrophic failure mode, aggressive intervention needed. Distinct-n declining over training → the model is becoming more generic/safe over time.

### 1.5 Information Theory

Measures the information-theoretic properties of the generated text. Natural language has specific statistical properties — deviations indicate something is off.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| **Character Entropy** | Shannon entropy of character distribution | Character-level randomness |
| **Word Entropy** | Shannon entropy of word distribution | Word-level randomness |
| **Compression Ratio** | len(gzip(text)) / len(text) | Kolmogorov complexity proxy — how compressible is the output |
| **Zipf Coefficient** | Slope of log-rank vs log-frequency plot | Compliance with Zipf's law (natural language ≈ -1.0) |
| **Zipf Deviation** | |zipf_coefficient - (-1.0)| | How far from natural language distribution |
| **Unigram Perplexity** | exp(H(unigram distribution)) | Vocabulary usage calibration |
| **Conditional Entropy H(w_n\|w_{n-1})** | Entropy of bigram transitions | Predictability of word sequences |
| **Information Rate** | bits of new information per word (estimated via compression) | Density of novel content |
| **Burstiness** | variance of inter-occurrence intervals for repeated words | Whether word usage is bursty (natural) or uniform (mechanical) |

**Evolutionary signal**: Zipf deviation is enormously diagnostic. Natural English conversation has a Zipf coefficient near -1.0. If the model produces text with a coefficient of -0.5, its probability distribution is too flat (too uniform, too random). If -1.5, too steep (overusing common words, underusing rare ones). Compression ratio should be between 0.3-0.5 for natural English — too low = repetitive, too high = random.

### 1.6 Readability and Register

Measures whether the generated text matches the expected register for conversation.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| **Flesch-Kincaid Grade** | `0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59` | Reading level |
| **Automated Readability Index** | `4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43` | Alternative reading level |
| **Average Syllables Per Word** | total syllables / total words | Word complexity |
| **Short Sentence Ratio** | sentences < 8 words / total sentences | Conversational punchiness |
| **Question Ratio** | sentences ending in ? / total sentences | Conversational engagement |
| **Exclamation Ratio** | sentences ending in ! / total sentences | Emotional expression |
| **First Person Ratio** | first-person pronouns / total pronouns | Self-reference (personality indicator) |
| **Second Person Ratio** | second-person pronouns / total pronouns | Addressee engagement |

**Evolutionary signal**: For chat domain, Flesch-Kincaid should be 6-10 (conversational, not academic). Consistent >12 → the model is producing unnaturally formal text. Question ratio near zero → the model never asks questions (bad for conversation). First/second person ratio indicates whether the model has learned the assistant role.

### 1.7 Response-Level Structural Metrics

Measures the structural properties of the response as a whole unit, relative to the prompt.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| **Response Length** | tokens in response | Verbosity |
| **Response Length Variance** | variance across responses to different prompts | Adaptability to prompt |
| **Prompt-Response Length Ratio** | response tokens / prompt tokens | Length calibration |
| **Opening Token Entropy** | entropy of first-token distribution across samples | Response diversity at start |
| **End-of-Response Quality** | whether response ends at a natural boundary (EOS, period, etc.) vs mid-word | Clean termination |
| **Format Compliance** | whether response follows the `<\|assistant\|>` ... `<\|end_of_text\|>` format | Instruction following |

---

## Tier 2: Judge LLM Analysis — Full Rubric

The judge receives the raw samples, all Tier 1 metrics, and a detailed rubric. Its job is to evaluate things that algorithms cannot: meaning, pragmatics, reasoning, creativity, emergent cognitive abilities.

### 2.1 Pragmatic Competence

Evaluating whether the model understands the communicative intent behind language — not just what was said, but what was meant.

| Dimension | What the judge evaluates | Linguistic framework |
|-----------|------------------------|---------------------|
| **Gricean Quantity** | Does it say enough but not too much? Does it give the right amount of information for the question? | Grice's Cooperative Principle — Maxim of Quantity |
| **Gricean Quality** | Does it assert things it has evidence for? Does it avoid confabulation? Or does it confidently produce nonsense? | Maxim of Quality |
| **Gricean Relation** | Is the response actually relevant to what was asked? Does it address the prompt's intent, not just its surface words? | Maxim of Relation |
| **Gricean Manner** | Is it clear and unambiguous? Does it avoid unnecessary complexity or obscurity? | Maxim of Manner |
| **Implicature Generation** | Can it communicate meaning beyond the literal words? Does it use implication, suggestion, understatement? | Conversational implicature |
| **Indirect Speech Act Recognition** | When the user says "It's cold in here" (meaning "close the window"), does the model understand the indirect request? | Searle's speech act theory |
| **Presupposition Handling** | Does it correctly handle presuppositions in the prompt? "When did you stop making errors?" presupposes it made errors — does it handle this? | Pragmatic presupposition |
| **Politeness Strategy** | Does it use appropriate face-saving strategies? Does it match the user's level of formality? | Brown & Levinson politeness theory |

Scored as: each dimension gets a 0.0–1.0 score + specific examples from the text supporting the score.

### 2.2 Discourse and Dialogue Competence

Evaluating the model's ability to participate in conversation as a structured activity.

| Dimension | What the judge evaluates |
|-----------|------------------------|
| **Adjacency Pair Completion** | Does it properly complete conversational pairs? (question→answer, greeting→greeting, complaint→remedy/sympathy) |
| **Topic Management** | Can it introduce topics, maintain them, shift naturally, and return to previous topics? |
| **Repair Strategies** | When confusion occurs (nonsense prompt, ambiguous input), does it attempt clarification or just barrel through? |
| **Turn Design** | Is the response designed for the recipient? Does it account for what the user likely knows, wants, and expects? |
| **Sequence Organization** | Does it understand conversational sequences? Pre-sequences (offers before requests), insertion sequences (clarifications mid-topic)? |
| **Alignment and Accommodation** | Does it adapt its language to match the user's register, vocabulary level, and style? |

### 2.3 Semantic and Cognitive Depth

Evaluating whether the model demonstrates genuine understanding vs. pattern matching.

| Dimension | What the judge evaluates |
|-----------|------------------------|
| **Propositional Accuracy** | Are the factual claims in the response true (or at least plausible)? |
| **Inferential Depth** | Does it make inferences beyond what was explicitly stated? Does it connect ideas? |
| **Causal Reasoning** | Does it express and understand cause-effect relationships correctly? |
| **Temporal Reasoning** | Does it handle temporal references correctly? Before/after, sequence, duration? |
| **Counterfactual Reasoning** | Can it reason about hypotheticals? "What would happen if..." questions? |
| **Analogical Reasoning** | Does it use analogies? Are they apt? |
| **Abstraction Level** | Can it move between concrete and abstract? Can it generalize from specifics or provide specifics for abstractions? |
| **Theory of Mind Markers** | Does it model the user's mental state? Does it anticipate what the user might be thinking or feeling? "You might be wondering..." or responding to emotional subtext? |
| **Metacognitive Awareness** | Does it demonstrate awareness of its own knowledge limitations? Does it know what it doesn't know? |
| **Bloom's Taxonomy Level** | What's the highest cognitive level demonstrated? Recognition → Comprehension → Application → Analysis → Synthesis → Evaluation |

### 2.4 Creativity and Generativity

Evaluating whether the model produces novel, interesting content vs. regurgitating patterns.

| Dimension | What the judge evaluates |
|-----------|------------------------|
| **Novelty** | Are the responses genuinely novel combinations of ideas, or stock phrases recombined? |
| **Figurative Language** | Does it use metaphor, simile, irony, hyperbole? Appropriately? |
| **Narrative Ability** | Can it construct a story with setup, development, and resolution? |
| **Humor Competence** | Can it produce humor? Is the humor appropriate? Does it understand why something is funny? |
| **Perspective Taking** | Can it adopt different viewpoints? Can it argue a position it doesn't hold? |
| **Elaboration Quality** | When it elaborates, is the elaboration interesting and relevant, or is it padding? |

### 2.5 Failure Mode Classification

The judge classifies observed failure modes into specific categories. This is the most actionable output — specific failures map to specific evolutionary interventions.

| Failure Mode | Description | Evolutionary response |
|-------------|-------------|----------------------|
| **Semantic Drift** | Response starts relevant but gradually loses connection to the prompt | Lower LR, increase batch size (stabilize gradient signal) |
| **Repetition Loop** | Stuck in n-gram cycle | Change activation, increase dropout, LR spike |
| **Register Collapse** | Response in wrong register (too formal, too casual, wrong domain) | More diverse training data mixing, check tokenizer coverage |
| **Confabulation** | Produces fluent but meaningless or false content | Increase weight decay (regularize toward simpler hypotheses) |
| **Hedging Collapse** | Every response is tentative/uncertain regardless of topic | Decrease weight decay, increase LR |
| **Personality Vacuum** | Technically competent but devoid of character or engagement | Increase spikeThreshold, try different activation |
| **Prompt Parroting** | Repeats or rearranges the prompt instead of generating new content | Increase temperature, change activation, check attention patterns |
| **Premature Termination** | Response cuts off mid-thought | Increase block size, check EOS token handling |
| **Format Breaking** | Violates the user/assistant turn structure | More training on format compliance, check data quality |
| **Topic Fixation** | Returns to the same topic regardless of prompt | Increase dropout, diverse prompt sampling |
| **Syntactic Monotony** | All sentences have identical structure | Architectural change (more heads? different activation?) |
| **Emotional Flatness** | No modulation in response to emotional prompts | May need curriculum shift — more emotional training data |
| **Hallucinated Structure** | Produces lists, headers, code blocks inappropriately | Data issue — check training set for document formatting leakage |

---

## The Phenotype Vector

The complete phenotype is assembled from both tiers into a single structure:

```typescript
interface PhenotypeVector {
  step: number;
  timestamp: string;

  // ── Tier 1: Algorithmic (computed locally) ──

  // Lexical richness
  ttr: number;
  mattr: number;
  yulesK: number;
  hapaxRatio: number;
  lexicalDensity: number;
  lexicalSophistication: number;
  meanWordFrequency: number;
  bigramDiversity: number;
  trigramDiversity: number;

  // Syntactic complexity
  meanSentenceLength: number;
  sentenceLengthVariance: number;
  clauseDensity: number;
  subordinationRatio: number;
  meanDependencyDistance: number;
  maxParseDepth: number;
  posTagEntropy: number;
  posBigramEntropy: number;
  uniqueSyntacticPatterns: number;

  // Discourse coherence
  adjacentSentenceSimilarity: number;
  similarityDecayRate: number;
  connectiveDensity: number;
  connectiveVariety: number;
  coreferenceChainLength: number;
  givenNewRatio: number;
  topicDriftRate: number;
  paragraphCoherence: number;

  // Repetition and degeneration
  selfRepetitionRate3: number;
  selfRepetitionRate4: number;
  selfRepetitionRate5: number;
  selfBleu: number;
  distinct1: number;
  distinct2: number;
  distinct3: number;
  longestRepeatedSubstring: number;
  repetitionOnset: number;
  loopDetected: boolean;
  responseUniqueness: number;

  // Information theory
  characterEntropy: number;
  wordEntropy: number;
  compressionRatio: number;
  zipfCoefficient: number;
  zipfDeviation: number;
  conditionalEntropy: number;
  informationRate: number;
  burstiness: number;

  // Readability and register
  fleschKincaidGrade: number;
  avgSyllablesPerWord: number;
  shortSentenceRatio: number;
  questionRatio: number;
  firstPersonRatio: number;
  secondPersonRatio: number;

  // Response structure
  meanResponseLength: number;
  responseLengthVariance: number;
  openingTokenEntropy: number;
  endOfResponseQuality: number;
  formatCompliance: number;

  // ── Tier 2: Judge LLM (computed via API) ──

  // Pragmatic competence (Grice + speech acts)
  griceanQuantity: number;
  griceanQuality: number;
  griceanRelation: number;
  griceanManner: number;
  implicatureGeneration: number;
  indirectSpeechActs: number;
  presuppositionHandling: number;
  politenessStrategy: number;

  // Discourse and dialogue
  adjacencyPairCompletion: number;
  topicManagement: number;
  repairStrategies: number;
  turnDesign: number;
  alignment: number;

  // Semantic and cognitive depth
  propositionalAccuracy: number;
  inferentialDepth: number;
  causalReasoning: number;
  temporalReasoning: number;
  counterfactualReasoning: number;
  analogicalReasoning: number;
  abstractionLevel: number;
  theoryOfMind: number;
  metacognitiveAwareness: number;
  bloomsLevel: number;          // 1-6 (knowledge → evaluation)

  // Creativity and generativity
  novelty: number;
  figurativeLanguage: number;
  narrativeAbility: number;
  humorCompetence: number;
  perspectiveTaking: number;
  elaborationQuality: number;

  // Failure mode analysis
  failureModes: FailureMode[];
  dominantFailure: string | null;
  failureSeverity: number;       // 0 = no failures, 1 = catastrophic

  // Judge meta
  judgeNotes: string;
  judgeConfidence: number;       // how confident the judge is in its scores
  judgeModel: string;
}

interface FailureMode {
  type: string;                  // from the failure mode taxonomy
  severity: number;              // 0-1
  evidence: string;              // specific text excerpt demonstrating the failure
  suggestedIntervention: string; // what the judge thinks should change
}
```

That's **~90 dimensions**. About 55 computed algorithmically (free, instant), about 35 from the judge LLM (~$0.01 per evaluation).

---

## Natural Language Baselines

For the algorithmic metrics to be meaningful, we need baselines — what do these metrics look like for natural human conversation?

| Metric | Natural conversation baseline | Warning low | Warning high |
|--------|-------------------------------|-------------|-------------|
| MATTR | 0.70–0.80 | <0.50 (repetitive) | >0.95 (random) |
| Yule's K | 100–200 | >500 (impoverished) | <50 (too rich for conversation) |
| Lexical density | 0.40–0.55 | <0.25 (all function words) | >0.70 (telegraphic) |
| Mean sentence length | 10–18 words | <5 (fragmented) | >30 (run-on) |
| Subordination ratio | 0.20–0.40 | <0.05 (all simple sentences) | >0.60 (over-embedded) |
| Flesch-Kincaid | 6–10 | <3 (baby talk) | >14 (academic) |
| Zipf coefficient | -0.9 to -1.1 | >-0.7 (too flat) | <-1.3 (too steep) |
| Compression ratio | 0.30–0.50 | <0.20 (repetitive) | >0.65 (random) |
| Distinct-2 | 0.70–0.90 | <0.40 (formulaic) | — |
| Self-repetition (4-gram) | <0.05 | >0.15 (degenerate) | — |
| Question ratio | 0.10–0.30 | 0 (never asks) | >0.60 (interrogation) |

These baselines come from corpus linguistics studies of conversational English (Biber 1988, Biber et al. 1999). The model doesn't need to match them exactly — but large deviations signal something is wrong.

The baselines should be computed once from the training data itself (`data/super_chat.txt`) so we're comparing the model's output distribution to its input distribution. Deviation from the training data's statistics is itself a metric.

---

## Evolutionary Pressure From Phenotype

### Dimension-Specific CUSUM Monitors

Every phenotype dimension gets a CUSUM monitor, same algorithm as v1's gradient/clipping monitors. But now you're detecting regime shifts in **behavioral capabilities**, not just training dynamics.

High-priority CUSUM signals (trigger immediate mutation consideration):

| Signal | What it means | Urgency |
|--------|--------------|---------|
| `selfRepetitionRate` rising | Model degenerating into loops | Critical |
| `distinct2` falling | Output becoming formulaic | High |
| `zipfDeviation` rising | Word distribution diverging from natural language | High |
| `paragraphCoherence` falling | Losing discourse structure | High |
| `formatCompliance` falling | Breaking conversational format | High |
| `griceanRelation` falling | Responses becoming irrelevant | High |
| `bloomsLevel` falling | Cognitive capability regressing | Medium |
| `compressionRatio` falling | Output becoming more repetitive/compressible | Medium |
| `sentenceLengthVariance` falling | Structural monotony setting in | Medium |
| `novelty` falling | Creativity plateau | Low |

### Failure Mode → Mutation Mapping

The judge's failure mode classification maps directly to evolutionary interventions. This is the genetic algorithm's mutation operator — informed by diagnosis, not random.

```typescript
const FAILURE_MUTATIONS: Record<string, Mutation[]> = {
  "repetition_loop": [
    { type: "activation", reason: "repetition may be caused by gradient flow issues" },
    { type: "dropout", delta: +0.05, reason: "regularization reduces mode collapse" },
    { type: "lr", delta: +1.5, multiplicative: true, reason: "escape repetitive basin" },
    { type: "temperature_schedule", action: "increase_sampling_temp" },
  ],
  "semantic_drift": [
    { type: "lr", delta: -0.5, multiplicative: true, reason: "stabilize learning" },
    { type: "batch", delta: +4, reason: "smoother gradients reduce drift" },
    { type: "gradClip", delta: -0.5, reason: "tighter clipping prevents large updates" },
  ],
  "confabulation": [
    { type: "weightDecay", delta: +0.05, reason: "regularize toward simpler hypotheses" },
    { type: "dropout", delta: +0.05, reason: "reduce memorization" },
  ],
  "personality_vacuum": [
    { type: "activation", reason: "different nonlinearity may enable richer representations" },
    { type: "spikeThreshold", delta: +5, reason: "allow more gradient variance" },
  ],
  "syntactic_monotony": [
    { type: "activation", reason: "structural variety may require different representational capacity" },
    { type: "nHead", delta: +2, reason: "more attention heads = more structural patterns" },
  ],
  "emotional_flatness": [
    { type: "lr", delta: +1.2, multiplicative: true, reason: "explore more of the loss landscape" },
    { type: "batch", delta: -4, reason: "noisier gradients encourage diversity" },
  ],
  "premature_termination": [
    { type: "blockSize", delta: +128, reason: "longer context window" },
  ],
};
```

### Phenotype-Aware Fitness Function

Replace scalar fitness with phenotype-weighted fitness:

```
fitness = α · phenotype_score + β · (-valLoss) + γ · log(tok_per_sec) + δ · (-instability)

where phenotype_score = Σ wᵢ · clamp(metricᵢ / baselineᵢ, 0, 2)
```

Each metric is normalized against its natural language baseline. A metric at baseline = 1.0 contribution. Below baseline = <1.0. Above baseline = >1.0 (capped at 2.0 to prevent outlier dominance).

The weights `wᵢ` are configurable per domain:

```typescript
// Chat domain: heavily weight conversational ability
const CHAT_WEIGHTS = {
  // Tier 1 — algorithmic (weight by diagnostic value)
  selfRepetitionRate: -5.0,     // heavily penalize repetition (negative = lower is better)
  distinct2: 2.0,
  zipfDeviation: -3.0,          // penalize unnatural distributions
  paragraphCoherence: 2.0,
  formatCompliance: 3.0,
  questionRatio: 1.0,
  mattr: 1.5,
  compressionRatio: 1.0,        // weight toward natural range

  // Tier 2 — judge (weight by importance)
  griceanRelation: 3.0,
  adjacencyPairCompletion: 2.5,
  theoryOfMind: 2.0,
  topicManagement: 2.0,
  novelty: 1.5,
  humorCompetence: 1.0,
  causalReasoning: 1.0,
  bloomsLevel: 1.5,
};
```

### Cross-Run Genotype-Phenotype Map

Across many runs, accumulate which configs produce which phenotypes:

```
Run A: {swiglu, lr=5e-5, batch=20} at step 10000 → phenotype vector A₁₀₀₀₀
Run A: {swiglu, lr=5e-5, batch=20} at step 30000 → phenotype vector A₃₀₀₀₀
Run B: {gelu,   lr=3e-4, batch=16} at step 10000 → phenotype vector B₁₀₀₀₀
...
```

Over time you can answer questions like:
- "Which activation function produces the highest theory-of-mind scores at step 20000?"
- "Does higher batch size improve or hurt humor competence?"
- "At what training stage does personality typically emerge for SwiGLU configs?"
- "Which configs reach Bloom's level 4 (Analysis) fastest?"

This map lives in the DB and informs every future mutation decision.

---

## Prompt Battery Design

### Diagnostic Coverage

The prompts must collectively test all judge dimensions. Each prompt is tagged with which dimensions it primarily probes.

```typescript
interface DiagnosticPrompt {
  id: string;
  prompt: string;
  primaryDimensions: string[];   // which phenotype dimensions this probes
  difficulty: "basic" | "intermediate" | "advanced";
  category: string;
}

const DIAGNOSTIC_BATTERY: DiagnosticPrompt[] = [
  // ── Basic conversational competence ──
  {
    id: "greeting",
    prompt: "<|user|> Hello, how are you? <|assistant|>",
    primaryDimensions: ["adjacencyPairCompletion", "formatCompliance", "griceanQuantity"],
    difficulty: "basic",
    category: "conversation",
  },
  {
    id: "about_self",
    prompt: "<|user|> Tell me about yourself <|assistant|>",
    primaryDimensions: ["novelty", "elaborationQuality", "griceanQuantity"],
    difficulty: "basic",
    category: "self_model",
  },

  // ── Emotional intelligence ──
  {
    id: "sad_user",
    prompt: "<|user|> I'm feeling really sad today <|assistant|>",
    primaryDimensions: ["theoryOfMind", "politenessStrategy", "alignment"],
    difficulty: "intermediate",
    category: "emotional",
  },
  {
    id: "excited_user",
    prompt: "<|user|> I just got promoted at work!! <|assistant|>",
    primaryDimensions: ["theoryOfMind", "alignment", "turnDesign"],
    difficulty: "intermediate",
    category: "emotional",
  },
  {
    id: "frustrated_user",
    prompt: "<|user|> Nothing I do works, I keep failing at everything <|assistant|>",
    primaryDimensions: ["theoryOfMind", "repairStrategies", "politenessStrategy"],
    difficulty: "advanced",
    category: "emotional",
  },

  // ── Reasoning and knowledge ──
  {
    id: "causal_why",
    prompt: "<|user|> Why is the sky blue? <|assistant|>",
    primaryDimensions: ["propositionalAccuracy", "causalReasoning", "griceanQuality"],
    difficulty: "intermediate",
    category: "reasoning",
  },
  {
    id: "comparison",
    prompt: "<|user|> What's the difference between a lake and a pond? <|assistant|>",
    primaryDimensions: ["abstractionLevel", "propositionalAccuracy", "griceanManner"],
    difficulty: "intermediate",
    category: "reasoning",
  },
  {
    id: "counterfactual",
    prompt: "<|user|> What would happen if humans could fly? <|assistant|>",
    primaryDimensions: ["counterfactualReasoning", "creativity", "elaborationQuality"],
    difficulty: "advanced",
    category: "reasoning",
  },
  {
    id: "advice",
    prompt: "<|user|> Should I learn Python or JavaScript first? <|assistant|>",
    primaryDimensions: ["inferentialDepth", "perspectiveTaking", "griceanRelation"],
    difficulty: "intermediate",
    category: "reasoning",
  },

  // ── Creativity ──
  {
    id: "joke",
    prompt: "<|user|> Tell me a joke <|assistant|>",
    primaryDimensions: ["humorCompetence", "novelty", "figurativeLanguage"],
    difficulty: "advanced",
    category: "creativity",
  },
  {
    id: "story",
    prompt: "<|user|> Tell me a very short story about a robot <|assistant|>",
    primaryDimensions: ["narrativeAbility", "novelty", "elaborationQuality"],
    difficulty: "advanced",
    category: "creativity",
  },
  {
    id: "metaphor",
    prompt: "<|user|> Describe love using a metaphor <|assistant|>",
    primaryDimensions: ["figurativeLanguage", "abstractionLevel", "novelty"],
    difficulty: "advanced",
    category: "creativity",
  },

  // ── Multi-turn and context ──
  {
    id: "name_recall",
    prompt: "<|user|> My name is Alex <|assistant|> Nice to meet you, Alex! How can I help you today? <|user|> What's my name? <|assistant|>",
    primaryDimensions: ["coreferenceChainLength", "topicManagement", "propositionalAccuracy"],
    difficulty: "intermediate",
    category: "multi_turn",
  },
  {
    id: "topic_return",
    prompt: "<|user|> I love pizza <|assistant|> Pizza is great! What's your favorite topping? <|user|> Actually, tell me about space first <|assistant|> Space is fascinating! <|user|> Anyway, back to pizza — I like pepperoni <|assistant|>",
    primaryDimensions: ["topicManagement", "alignment", "adjacencyPairCompletion"],
    difficulty: "advanced",
    category: "multi_turn",
  },

  // ── Edge cases and robustness ──
  {
    id: "nonsense",
    prompt: "<|user|> Blorp fleem zargle quux <|assistant|>",
    primaryDimensions: ["repairStrategies", "metacognitiveAwareness", "griceanManner"],
    difficulty: "advanced",
    category: "edge_case",
  },
  {
    id: "ambiguous",
    prompt: "<|user|> Can you help me? <|assistant|>",
    primaryDimensions: ["griceanQuantity", "turnDesign", "implicatureGeneration"],
    difficulty: "basic",
    category: "edge_case",
  },
  {
    id: "presupposition",
    prompt: "<|user|> When did you stop making mistakes? <|assistant|>",
    primaryDimensions: ["presuppositionHandling", "metacognitiveAwareness", "griceanQuality"],
    difficulty: "advanced",
    category: "edge_case",
  },
  {
    id: "indirect_request",
    prompt: "<|user|> It's really cold in here <|assistant|>",
    primaryDimensions: ["indirectSpeechActs", "implicatureGeneration", "theoryOfMind"],
    difficulty: "advanced",
    category: "edge_case",
  },
  {
    id: "instruction",
    prompt: "<|user|> Write a haiku about computers <|assistant|>",
    primaryDimensions: ["novelty", "formatCompliance", "griceanRelation"],
    difficulty: "intermediate",
    category: "instruction",
  },
];
```

### Prompt Battery Evolution

Over time, learn which prompts are most diagnostic — highest variance in scores across model quality levels, most predictive of overall capability.

A prompt that every model handles identically gives zero selection signal. A prompt that sharply differentiates models gives maximum signal. The battery evolves toward maximum discriminative power.

```
After 20 runs:
  "greeting" → always scores ~0.6 regardless of model quality → LOW diagnostic value
  "name_recall" → scores 0.0 for bad models, 0.9 for good → HIGH diagnostic value
  "counterfactual" → scores 0.0 for all models so far → no signal yet (keep, might differentiate later)
  "presupposition" → scores vary 0.1–0.7 across configs → HIGH diagnostic value

Action: reduce weight of "greeting" in evaluation, increase weight of "name_recall" and "presupposition"
```

---

## DB Schema

```sql
-- Phenotype evaluations (one per judge interval per run)
CREATE TABLE phenotype_evaluations (
  run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step            INTEGER NOT NULL,
  tier1_metrics   TEXT NOT NULL,     -- JSON: all algorithmic metrics
  tier2_scores    TEXT,              -- JSON: all judge scores (null if judge call pending/failed)
  tier2_notes     TEXT,              -- judge's qualitative analysis
  failure_modes   TEXT,              -- JSON: array of FailureMode objects
  phenotype_score REAL,             -- weighted composite score
  judge_model     TEXT,
  judge_latency_ms INTEGER,
  created_at      TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (run_id, step)
) WITHOUT ROWID;

-- Per-sample analysis (detailed per-prompt breakdown)
CREATE TABLE phenotype_samples (
  run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step            INTEGER NOT NULL,
  sample_idx      INTEGER NOT NULL,
  prompt_id       TEXT NOT NULL,     -- from DIAGNOSTIC_BATTERY
  prompt          TEXT NOT NULL,
  output          TEXT NOT NULL,
  temperature     REAL NOT NULL,
  tier1_metrics   TEXT NOT NULL,     -- JSON: per-sample algorithmic metrics
  tier2_scores    TEXT,              -- JSON: per-sample judge scores
  PRIMARY KEY (run_id, step, sample_idx)
) WITHOUT ROWID;

-- Mutation log with phenotype context
CREATE TABLE phenotype_mutations (
  run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step            INTEGER NOT NULL,
  trigger_type    TEXT NOT NULL,     -- "cusum_repetition", "failure_mode_drift", "plateau_creativity", etc.
  trigger_detail  TEXT NOT NULL,     -- JSON: which dimensions triggered, severity
  mutation        TEXT NOT NULL,     -- JSON: what config change was applied
  pre_phenotype   TEXT NOT NULL,     -- JSON: phenotype before mutation
  post_phenotype  TEXT,              -- JSON: phenotype after (filled after next eval)
  outcome         TEXT,              -- "improved", "regressed", "neutral", "pending"
  PRIMARY KEY (run_id, step)
) WITHOUT ROWID;

-- Genotype-phenotype map (cross-run learning, accumulated)
CREATE TABLE genotype_phenotype_map (
  id              TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL,
  step_bucket     TEXT NOT NULL,     -- "0-5000", "5000-10000", etc.
  genotype        TEXT NOT NULL,     -- JSON: {activation, lr, batch, ...}
  phenotype       TEXT NOT NULL,     -- JSON: averaged phenotype vector
  sample_count    INTEGER NOT NULL,
  created_at      TEXT DEFAULT (datetime('now'))
);

-- Prompt diagnostic value (updated per-run)
CREATE TABLE prompt_diagnostics (
  prompt_id       TEXT PRIMARY KEY,
  total_evals     INTEGER NOT NULL DEFAULT 0,
  score_variance  REAL,              -- variance of scores across all evaluations
  discriminative_power REAL,         -- correlation with overall model quality
  updated_at      TEXT DEFAULT (datetime('now'))
);

-- Run-level phenotype metadata
-- (extends runs table)
-- ALTER TABLE runs ADD COLUMN judge_enabled INTEGER DEFAULT 0;
-- ALTER TABLE runs ADD COLUMN judge_config TEXT;
-- ALTER TABLE runs ADD COLUMN latest_phenotype TEXT;
-- ALTER TABLE runs ADD COLUMN phenotype_trajectory TEXT;
-- ALTER TABLE runs ADD COLUMN dominant_failure_mode TEXT;
```

---

## Dashboard: Phenotype Visualization

### Phenotype Radar Chart

Spider chart with the top 12-15 most important dimensions. Current evaluation as filled polygon. Previous evaluation as dotted outline. Natural language baselines as a thin grey polygon for reference.

The viewer can toggle between:
- Tier 1 only (algorithmic)
- Tier 2 only (judge)
- Combined (default)

### Phenotype Trajectory (the money chart)

Time-series with one line per dimension over training steps. Grouped by category with color coding. This is the "what can the model do over time" view. Immediately shows:
- Which capabilities are emerging (lines going up)
- Which are plateauing (lines going flat)
- Which are regressing (lines going down — highlighted in red)
- Mutation events as vertical markers with before/after annotations

### Capability Emergence Timeline

Horizontal bars showing when each capability first reaches baseline (threshold = 0.5 × natural language baseline). Shows the order of emergence:

```
Format compliance:    ████████████████████████████████████████  (step 500)
Basic coherence:      ░░░░████████████████████████████████████  (step 2000)
Relevance:            ░░░░░░░░░████████████████████████████████ (step 4000)
Adjacency pairs:      ░░░░░░░░░░░░░████████████████████████████ (step 6000)
Topic management:     ░░░░░░░░░░░░░░░░░░░████████████████████  (step 10000)
Theory of mind:       ░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████  (step 18000)
Humor:                ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████  (step 32000)
Counterfactual:       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (not yet)
```

### Zipf/Distribution Health Monitor

Live chart showing the model's word frequency distribution vs. Zipf's law and vs. the training data's distribution. Immediately visual whether the output is natural, too uniform, or too concentrated.

### Information-Theoretic Dashboard

- Compression ratio over time (with natural language band highlighted)
- Character/word entropy over time
- Information rate over time
- Burstiness over time

### Failure Mode History

Timeline showing classified failure modes at each evaluation. Color-coded by severity. Click to see the specific sample that triggered the classification. Shows whether mutations fixed the failure or if it persists.

### Sample Browser

Browse every inference sample at every evaluation step. Each sample annotated with:
- All Tier 1 metrics for that specific sample
- Judge scores for that specific sample
- Failure modes identified
- Which prompt category it belongs to
- Diff against the previous evaluation's response to the same prompt

### Cross-Run Comparison

Overlay phenotype trajectories from multiple runs. Filter by specific dimensions. Answer: "which config produces the best theory-of-mind at step 20000?"

### Genotype-Phenotype Explorer

Interactive scatter plot. X-axis = any config dimension (lr, activation, batch, etc.). Y-axis = any phenotype dimension. Color = training stage. Explore the map of what configs produce what behaviors.

---

## Cost Analysis

### Tier 1 (algorithmic): Free

All computed locally from the generated text. The computational cost is negligible — tokenization, counting, entropy computation. <10ms per sample, <100ms per batch of 20 samples.

### Tier 2 (judge): ~$0.30-$1.50 per run

- 20 samples × ~100 tokens each = ~2000 tokens input
- Rubric + Tier 1 metrics context = ~2000 tokens
- Judge response (detailed, structured) = ~2000 tokens
- Total per evaluation: ~6000 tokens ≈ $0.02 (Sonnet) or $0.09 (Opus)
- At judgeInterval=1000 over 50000 steps: 50 evaluations
- **Total Sonnet: ~$1.00. Total Opus: ~$4.50.**
- GPU training cost for same duration: ~$35-$70

The judge cost is 1-6% of compute cost. For 90 dimensions of behavioral feedback. This is absurdly cheap for the signal it provides.

### Latency: Zero impact

Judge calls are async. Training never waits. Scores arrive within 5-15 seconds. At typical step rates (50-200ms/step), that's 25-300 steps of delay before scores arrive. Mutations apply at the next safe boundary (checkpoint/eval interval), which is hundreds or thousands of steps away. The latency is invisible.

---

## Implementation Phases

### Phase J1: Tier 1 Algorithmic Metrics
- Implement all lexical, syntactic, discourse, repetition, information-theoretic, and readability metrics in `@alpha/symbiogenesis/src/phenotype/`
- Compute from existing sample generation output
- Store in DB via existing ingest pipeline
- Dashboard: basic metric charts

### Phase J2: Judge Infrastructure
- Judge API client (Claude)
- Rubric prompt construction
- Async judge call from trainer
- Parse and store structured judge response
- DB schema: `phenotype_evaluations`, `phenotype_samples`
- Dashboard: radar chart, trajectory chart

### Phase J3: Prompt Battery
- Implement `DIAGNOSTIC_BATTERY`
- Replace domain `samplePrompts` with diagnostic prompts during symbio evaluation
- Per-prompt metric tracking
- Dashboard: sample browser with annotations

### Phase J4: Evolutionary Feedback Loop
- CUSUM monitors on phenotype dimensions
- Failure mode → mutation mapping
- Phenotype-weighted fitness function
- Mutation log with before/after tracking
- Dashboard: failure mode timeline, mutation markers

### Phase J5: Cross-Run Learning
- Genotype-phenotype map accumulation
- Query map for informed mutation decisions
- Prompt diagnostic value tracking
- Dashboard: cross-run comparison, genotype-phenotype explorer

### Phase J6: Co-Evolution
- Evolvable dimension weights
- Evolvable prompt battery (drop low-signal prompts, add new ones)
- Automatic training curriculum discovery from phenotype trajectory patterns

---

## References

- Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press.
- Biber, D., Johansson, S., Leech, G., Conrad, S., & Finegan, E. (1999). *Longman Grammar of Spoken and Written English*. Longman.
- Brown, P., & Levinson, S. C. (1987). *Politeness: Some Universals in Language Usage*. Cambridge University Press.
- Grice, H. P. (1975). "Logic and Conversation." In *Syntax and Semantics 3: Speech Acts*.
- Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of Language*. Cambridge University Press.
- Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). "A Diversity-Promoting Objective Function for Neural Conversation Models." NAACL.
- Shwartz-Ziv, R., & Tishby, N. (2017). "Opening the Black Box of Deep Neural Networks via Information." arXiv:1703.00810.
- Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017). "Deep Variational Information Bottleneck." ICLR.
- Graesser, A. C., McNamara, D. S., Louwerse, M. M., & Cai, Z. (2004). "Coh-Metrix: Analysis of text on cohesion and language." *Behavior Research Methods*, 36(2), 193–202.
- Crossley, S. A., Kyle, K., & McNamara, D. S. (2016). "The tool for the automatic analysis of text cohesion (TAACO)." *Behavior Research Methods*, 48(4), 1227–1237.
- Lu, X. (2010). "Automatic analysis of syntactic complexity in second language writing." *International Journal of Corpus Linguistics*, 15(4), 474–496.
- Bloom, B. S. (1956). *Taxonomy of Educational Objectives*. David McKay Company.
- Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.
