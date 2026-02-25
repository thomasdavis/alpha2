# Phenotype-Driven Evolutionary Training: Multi-Disciplinary Analysis Framework

Date: 2026-02-25
Status: Research
Depends on: Symbio v1, inference sampling infrastructure (exists)

---

## Premise

Loss is a scalar. It tells you the model is learning *something*. It tells you nothing about *what* the model is learning — whether it's acquiring conversational ability, reasoning, humor, empathy, or just memorizing token transition probabilities.

We replace scalar fitness with a **~200-dimension phenotype vector** drawn from computational linguistics, cognitive science, philosophy of language, rhetoric, psycholinguistics, information theory, network science, affective computing, narratology, and epistemology. Every dimension is either computable algorithmically (free, local) or requires a judge LLM (async, ~$1-5/run total). The phenotype vector becomes the fitness function for evolutionary training decisions.

Two tiers:
- **Tier 1 (algorithmic)**: ~120 dimensions computed locally from generated text. No API call. <200ms per evaluation batch.
- **Tier 2 (judge LLM)**: ~80 dimensions requiring genuine understanding. Claude API, async, non-blocking.

---

## Tier 1: Algorithmic Analysis

### 1. Lexical Richness and Vocabulary Profile

The vocabulary a model uses reveals its internal representational capacity. A model with a rich, well-calibrated vocabulary distribution is encoding more of the training data's structure than one that defaults to high-frequency words.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Type-Token Ratio (TTR) | unique words / total words | Raw vocabulary diversity |
| MATTR (Moving Average TTR) | mean TTR over sliding 50-word windows | Length-independent vocabulary diversity (Covington & McFall 2010) |
| Yule's K | `10⁴ · (Σ i²·V(i,N) - N) / N²` | Vocabulary richness invariant to text length (Yule 1944) |
| Hapax Legomena Ratio | words occurring once / total unique words | Novel/rare word usage rate |
| Lexical Density | content words (N, V, Adj, Adv) / total words | Information-bearing word proportion (Ure 1971) |
| Lexical Sophistication | words outside top-2000 frequency list / total | Rarity of vocabulary |
| Mean Word Frequency | mean log-frequency from reference corpus | Average rarity |
| Bigram Diversity | unique bigrams / total bigrams | Phrasal variety |
| Trigram Diversity | unique trigrams / total trigrams | Extended phrase variety |
| Collocational Strength | mean pointwise MI of adjacent content word pairs | Quality of word combinations (Church & Hanks 1990) |
| Keyword Overuse | words significantly more frequent than in training data (log-likelihood) | Fixation on specific vocabulary |
| Keyword Underuse | words significantly less frequent than in training data | Vocabulary gaps |
| Academic Word List Coverage | proportion of Coxhead AWL words used | Register sophistication (Coxhead 2000) |
| Concreteness Mean | mean concreteness rating from Brysbaert norms | Abstract vs. concrete language preference |
| Imageability Mean | mean imageability rating from MRC database | Vividness of language |
| Age of Acquisition Mean | mean AoA from Kuperman norms | Developmental level of vocabulary |
| Emotional Valence Mean | mean valence from Warriner norms | Positive/negative vocabulary skew |
| Emotional Arousal Mean | mean arousal from Warriner norms | Calm vs. intense vocabulary |

### 2. Syntactic Complexity

Sentence structure reveals computational depth. A model producing only SVO (subject-verb-object) sentences has learned surface patterns. One producing subordinate clauses, relative clauses, and embedded structures has acquired hierarchical syntax.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Mean Sentence Length (words) | words per sentence | Basic complexity |
| Sentence Length Variance | σ² of sentence lengths | Structural variety (low = monotonous) |
| Sentence Length Entropy | Shannon entropy of length distribution | Richness of length choices |
| Clause Density | clauses per sentence | Embedding depth |
| Subordination Ratio | dependent clauses / total clauses | Hierarchical complexity (Hunt 1965) |
| Coordination Ratio | coordinate clauses / total clauses | Paratactic vs. hypotactic style |
| T-Unit Length | words per minimal terminable unit | Developmental complexity measure (Hunt 1965) |
| Mean Dependency Distance | avg token distance between head-dependent | Processing difficulty, working memory load (Liu 2008) |
| Max Dependency Distance | max head-dependent distance in any sentence | Peak processing demand |
| Max Parse Depth | deepest nesting level | Maximum structural recursion |
| Left-Branching Ratio | left-branching / total branching | Typological naturalness for English |
| POS Tag Entropy | H(POS distribution) | Syntactic category diversity |
| POS Bigram Entropy | H(POS pair distribution) | Transition diversity |
| POS Trigram Diversity | unique POS trigrams / total | Structural creativity |
| Verb Tense Distribution | entropy of tense usage (present, past, future, progressive, perfect) | Temporal range |
| Modal Verb Ratio | modal verbs / total verbs | Hedging, possibility, obligation expression |
| Passive Voice Ratio | passive constructions / total clauses | Stylistic complexity |
| Relative Clause Rate | relative clauses per sentence | Nominal modification complexity |
| Complement Clause Rate | complement clauses per sentence | Propositional embedding |

### 3. Discourse Coherence and Cohesion

Text is more than a sequence of sentences. Coherent text has logical flow, referential continuity, and information management. These metrics measure whether the model produces *text* or just *adjacent sentences*.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Adjacent Sentence Similarity | Jaccard(content words of s_i, s_{i+1}) | Local topical cohesion |
| Similarity Decay Function | similarity at distance k=1,2,3,4 | How fast coherence degrades with distance |
| LSA Coherence Proxy | cosine similarity of bag-of-words TF-IDF vectors between adjacent sentences | Semantic continuity (Foltz et al. 1998) |
| Connective Density | discourse connectives per sentence | Explicit logical marking |
| Connective Variety | unique connectives / total connectives used | Range of logical relations |
| Connective Type Distribution | additive / adversative / causal / temporal proportions | Which logical relations are expressed |
| Coreference Chain Length | mean length of pronoun→referent chains | Referential continuity |
| Coreference Chain Count | number of distinct coreference chains | Referential complexity |
| Given-New Ratio | repeated content words / novel content words per sentence | Information management (Halliday 1967) |
| Topic Drift Rate | 1 - cosine(first sentence, last sentence) BoW vectors | Coherence over full response |
| Paragraph Coherence | mean adjacent-sentence similarity | Global coherence score |
| Thematic Progression | constant theme / linear theme / split theme classification | How information develops (Daneš 1974) |
| Theme-Rheme Ratio | thematic (given) content / rhematic (new) content per clause | Information structure balance |
| Lexical Overlap (global) | content word overlap between response and prompt | Topical grounding |

### 4. Repetition, Degeneration, and Diversity

The single most important failure detection cluster for small language models. These metrics catch degenerate behavior that loss curves completely miss.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Self-Repetition Rate (3-gram) | repeated 3-grams / total 3-grams | Phrase-level repetition |
| Self-Repetition Rate (4-gram) | repeated 4-grams / total 4-grams | Stronger repetition signal |
| Self-Repetition Rate (5-gram) | repeated 5-grams / total 5-grams | Near-verbatim repetition |
| Self-BLEU | mean BLEU between all pairs of responses | Cross-response diversity (Zhu et al. 2018) |
| Distinct-1 | unique unigrams / total unigrams | Unigram diversity (Li et al. 2016) |
| Distinct-2 | unique bigrams / total bigrams | Bigram diversity |
| Distinct-3 | unique trigrams / total trigrams | Trigram diversity |
| Longest Repeated Substring | max length of substring appearing 2+ times | Worst-case repetition |
| Repetition Onset Index | token position where first 4-gram repeat appears | How far before degeneration |
| Loop Detection | any n-gram (n≥3) repeating 3+ times consecutively | Binary catastrophic failure |
| Response Uniqueness | 1 - max pairwise BoW cosine across responses | Are all responses identical? |
| Vocabulary Overlap Across Responses | Jaccard overlap of vocabulary sets across different prompts | Cross-prompt lexical diversity |
| Opening Diversity | unique first-5-token sequences across responses | Are all responses starting the same way? |
| Closing Pattern Variety | unique final-5-token sequences across responses | Ending diversity |

### 5. Information Theory and Statistical Properties

Natural language has deep statistical regularities. Deviations from these regularities indicate the model's probability distribution is miscalibrated.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Character Entropy | H(char distribution) | Character-level randomness |
| Word Entropy | H(word distribution) | Word-level randomness |
| Bigram Entropy | H(word pair distribution) | Transition uncertainty |
| Conditional Entropy H(w_n\|w_{n-1}) | H(bigram) - H(unigram) | Predictability of next word |
| Compression Ratio | len(gzip(text)) / len(text) | Kolmogorov complexity proxy |
| Zipf Coefficient | slope of log-rank vs. log-frequency | Natural language compliance (Zipf 1949) |
| Zipf R² | R² of linear fit on log-log frequency plot | How well the distribution fits Zipf |
| Heaps' Exponent | β in V(n) = Kn^β (vocabulary growth rate) | Whether vocabulary is still growing or saturated (Heaps 1978) |
| Entropy Rate | bits per word (via sliding-window compression) | Information density |
| Burstiness | variance of inter-occurrence intervals for repeated words | Natural=bursty, mechanical=uniform (Altmann et al. 2009) |
| Hurst Exponent | long-range dependence of word frequencies via R/S analysis | Long-memory structure (Hurst 1951) |
| Type-Token Growth Curve Shape | TTR as function of text length | Vocabulary acquisition pattern |
| Perplexity Under Training Data N-gram | cross-entropy with training data n-gram model | How well output matches training distribution |
| KL Divergence from Training Unigrams | D_KL(output unigram ‖ training unigram) | Unigram distribution drift |
| Mutual Information Between Prompt and Response | I(prompt tokens; response tokens) | Conditional generation quality |

### 6. Readability, Register, and Sociolinguistic Properties

Language varies systematically across social contexts. A chat model should produce conversational English, not academic prose or telegraphic fragments.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Flesch-Kincaid Grade Level | standard formula | Reading level |
| Automated Readability Index | standard formula | Alternative reading level |
| Coleman-Liau Index | standard formula | Character-based reading level |
| Mean Syllables Per Word | total syllables / words | Word complexity |
| Mean Word Length (chars) | total chars / words | Character-level complexity |
| Short Sentence Ratio | sentences < 8 words / total | Conversational punchiness |
| Long Sentence Ratio | sentences > 25 words / total | Complex thought proportion |
| Question Ratio | sentences ending ? / total | Engagement, curiosity |
| Exclamation Ratio | sentences ending ! / total | Emotional expression |
| Imperative Ratio | sentences with imperative mood / total | Directive speech |
| First Person Ratio | I/me/my/mine/we/us/our / total pronouns | Self-reference, personality |
| Second Person Ratio | you/your/yours / total pronouns | Addressee engagement |
| Third Person Ratio | he/she/they etc. / total pronouns | Narrative mode |
| Contraction Rate | contractions / (contractions + expanded forms) | Formality level |
| Hedge Word Rate | hedges (maybe, perhaps, sort of, kind of...) / total words | Uncertainty expression |
| Intensifier Rate | intensifiers (very, really, extremely...) / total words | Emphasis patterns |
| Filler/Discourse Marker Rate | well, so, like, you know / total words | Conversational naturalness |

### 7. Biber's Register Dimensions

Douglas Biber's (1988) multi-dimensional analysis identifies 5 dimensions of linguistic variation across registers, computed from dozens of linguistic features. These dimensions are the gold standard for register analysis in corpus linguistics.

| Dimension | Pole A | Pole B | Key features |
|-----------|--------|--------|-------------|
| D1: Involved vs. Informational | Involved (conversation-like) | Informational (academic-like) | private verbs, that-deletion, contractions, 1st/2nd person vs. noun density, prepositions, attributive adjectives |
| D2: Narrative vs. Non-Narrative | Narrative | Non-narrative | past tense, 3rd person, perfect aspect, public verbs vs. present tense |
| D3: Explicit vs. Situation-Dependent | Explicit reference | Situation-dependent | WH-relative clauses, pied-piping, phrasal coordination vs. time/place adverbs |
| D4: Overt Persuasion | Persuasive | Non-persuasive | prediction modals (will, shall), suasive verbs, conditional subordination, necessity modals |
| D5: Abstract vs. Non-Abstract | Abstract | Concrete | conjuncts, agentless passives, past participial clauses, by-passives |

For chat domain: D1 should be firmly on the "Involved" pole. D2 situational. D3 moderate. D4 low. D5 low.

Compute each dimension as a weighted sum of its constituent features (Biber provides the factor loadings). Compare against known register benchmarks (conversation, fiction, academic prose, news).

### 8. Network Properties of Generated Text

Treat the generated text as a graph — words are nodes, co-occurrences are edges. Network topology reveals structural properties invisible to sequential metrics.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Word Co-occurrence Network Density | edges / possible edges (window=5) | How interconnected the vocabulary is |
| Network Clustering Coefficient | mean local clustering | Tendency to form word "communities" |
| Network Average Path Length | mean shortest path between word pairs | How many hops between concepts |
| Small-World Coefficient | clustering / path_length ratio vs. random graph | Natural language is small-world (Ferrer i Cancho & Solé 2001) |
| Network Degree Entropy | H(degree distribution) | Hub structure |
| Top Hub Words | highest-degree nodes | Which words connect everything |
| Modularity | community detection Q-score | Whether vocabulary clusters into semantic domains |
| Rich-Club Coefficient | connectivity among high-degree nodes | Whether common words preferentially connect |
| Semantic Network Diameter | longest shortest path | Conceptual range of the text |
| Power Law Exponent of Degree Distribution | γ in P(k) ~ k^(-γ) | Scale-free properties (natural language γ ≈ 2-3) |

### 9. Temporal and Sequential Structure

Text unfolds over time. The *pattern* of how properties change across the response reveals planning ability, narrative arc, and structural control.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Sentence Length Autocorrelation | ACF of sentence lengths at lag 1,2,3 | Rhythmic structure |
| Sentence Length Periodicity | dominant frequency via FFT of sentence lengths | Prosodic patterning |
| Vocabulary Introduction Rate | new unique words per sentence (slope) | Whether the model keeps introducing new concepts or exhausts early |
| Information Density Curve | compression ratio per sentence over response | Where information concentrates |
| Complexity Ramp | sentence complexity (clause density) over response position | Does complexity build or collapse? |
| Emotional Arc | sentiment valence over sentence positions | Narrative emotional trajectory (Reagan et al. 2016) |
| Topic Stability Curve | similarity to initial topic over sentence position | Topic maintenance ability |
| Opening-Body-Closing Structure | ratio of response devoted to greeting/body/closing | Conversational structure awareness |

### 10. Psycholinguistic Processing Metrics

Estimate the cognitive processing demands the generated text would place on a human reader. Well-formed text manages processing load; poorly-formed text creates unnecessary difficulty.

| Metric | Method | What it reveals |
|--------|--------|-----------------|
| Mean Surprisal Proxy | -log P(word) from unigram model | Average unpredictability |
| Surprisal Variance | σ² of per-word surprisal | Smoothness of information flow (uniform = better) |
| Garden Path Potential | sentences with temporary structural ambiguity | Processing difficulty spikes |
| Center Embedding Depth | max depth of center-embedded clauses | Working memory demand |
| Filler-Gap Distance | mean distance between filler and gap in relative clauses | Syntactic processing load |
| Word Predictability Gradient | surprisal change across sentence positions | Whether sentence endings are predictable |
| Referential Distance | mean distance between anaphor and antecedent | Working memory for coreference |
| Ambiguity Rate | words with >3 WordNet senses / total content words | Lexical ambiguity load |
| Negation Complexity | nested negations / total sentences | Logical processing demand |

---

## Tier 2: Judge LLM Analysis

The judge receives all samples plus a summary of Tier 1 metrics. Its job is to evaluate dimensions that require genuine comprehension — pragmatics, reasoning, creativity, social cognition, argumentation. Each dimension below is grounded in a specific academic framework.

### 11. Pragmatic Competence (Philosophy of Language)

Grounded in Grice (1975), Searle (1969), Sperber & Wilson (1986), Brown & Levinson (1987).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Gricean Quantity | Cooperative Principle | Says enough but not too much? Appropriate level of detail for the question? |
| Gricean Quality | Cooperative Principle | Avoids asserting things without evidence? Doesn't confabulate? |
| Gricean Relation | Cooperative Principle | Response is relevant to the prompt's actual communicative intent? |
| Gricean Manner | Cooperative Principle | Clear, unambiguous, well-ordered? Avoids unnecessary complexity? |
| Scalar Implicature | Neo-Gricean pragmatics | "Some students passed" implying "not all". Does the model generate and understand these? |
| Relevance-Theoretic Optimality | Sperber & Wilson | Does it maximize cognitive effect while minimizing processing effort? |
| Conversational Implicature | Grice | Can it communicate meaning beyond literal words through implication? |
| Conventional Implicature | Grice | Proper use of words like "but" (contrast), "even" (surprise), "yet" (counter-expectation)? |
| Indirect Speech Acts | Searle | "Can you pass the salt?" = request, not ability question. Does it recognize these? |
| Presupposition Handling | Stalnaker, Beaver | "When did you stop making mistakes?" — handles the embedded presupposition? |
| Performative Awareness | Austin | "I promise", "I apologize" — understands these DO things, not just describe? |
| Politeness Strategy | Brown & Levinson | Appropriate face-saving? Matches user's formality? Manages face-threatening acts? |
| Positive Politeness | Brown & Levinson | Claims common ground, conveys cooperation, attends to hearer's wants? |
| Negative Politeness | Brown & Levinson | Respects autonomy, avoids imposition, uses hedging appropriately? |

### 12. Conversation Analysis (Sociology)

Grounded in Sacks, Schegloff, & Jefferson (1974), Heritage (1984).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Adjacency Pair Completion | CA Sequential Analysis | Properly completes pairs: question→answer, greeting→greeting, complaint→remedy? |
| Preference Organization | CA | Preferred responses are immediate/direct; dispreferred are hedged/delayed. Does it produce the right structure? |
| Pre-Sequence Awareness | CA | Understands pre-requests ("Are you busy?"), pre-invitations ("Are you free Saturday?")? |
| Repair Initiation | CA | When confusion occurs, does it initiate repair? "What do you mean by...?" |
| Other-Repair | CA | Can it repair misunderstandings from the user's side? |
| Topic Transition Mechanisms | CA | Uses proper topic shift devices: stepwise (gradual), boundaried (explicit), or disjunctive (abrupt)? |
| Topic Shading | CA | Can it do subtle topic shifts — moving to related topics without explicit markers? |
| Formulations | Heritage & Watson | Can it summarize, restate, or characterize what was said? "So what you're saying is..." |
| Alignment Displays | CA | Shows understanding: "right", "yeah", "I see", "that makes sense" — used naturally? |
| Account Giving | CA/Scott & Lyman | When it can't do something or disagrees, does it provide reasons? |
| Sequence Closure | CA | Can it properly close conversational sequences? Doesn't leave threads hanging? |
| Recipient Design | Sacks & Schegloff | Is the response designed for *this* specific user/prompt, or generic? |

### 13. Rhetorical and Argumentative Competence

Grounded in Toulmin (1958), Aristotle's *Rhetoric*, Perelman & Olbrechts-Tyteca (1969), Walton (2008).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Claim Clarity | Toulmin | Are claims explicit, identifiable, and well-formed? |
| Warrant Provision | Toulmin | Does it provide reasons (warrants) connecting evidence to claims? |
| Backing Depth | Toulmin | Does it support warrants with deeper backing (principles, data, authorities)? |
| Qualifier Usage | Toulmin | Does it appropriately qualify claims? ("usually", "in most cases", "probably") |
| Rebuttal Awareness | Toulmin | Does it acknowledge potential counterarguments or limitations? |
| Ethos Projection | Aristotle | Does it establish credibility? Demonstrate competence without arrogance? |
| Pathos Calibration | Aristotle | Does it engage emotions appropriately? Not manipulatively? |
| Logos Soundness | Aristotle | Are logical structures valid? Does the reasoning actually follow? |
| Fallacy Avoidance | Walton | Avoids ad hominem, straw man, false dichotomy, appeal to authority, circular reasoning? |
| Argument Structure | Argumentation Theory | Can it construct multi-step arguments with premises leading to conclusions? |
| Counter-Argument Integration | Dialectical theory | Can it consider opposing views and integrate them? |
| Persuasive Framing | Prospect Theory / Kahneman | Does it frame information effectively? Positive/negative framing appropriate to context? |

### 14. Cognitive and Reasoning Competence

Grounded in Bloom (1956), Piaget (1952), Kahneman (2011), Johnson-Laird (1983).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Bloom's Level: Knowledge | Bloom's Taxonomy | Can it recall and state facts? |
| Bloom's Level: Comprehension | Bloom's Taxonomy | Can it explain ideas in its own words, paraphrase, summarize? |
| Bloom's Level: Application | Bloom's Taxonomy | Can it apply knowledge to new situations? |
| Bloom's Level: Analysis | Bloom's Taxonomy | Can it break down information, identify patterns, distinguish fact from inference? |
| Bloom's Level: Synthesis | Bloom's Taxonomy | Can it combine ideas to form new wholes, propose solutions? |
| Bloom's Level: Evaluation | Bloom's Taxonomy | Can it judge the value of ideas, make reasoned assessments, critique? |
| Causal Reasoning | Mental Models (Johnson-Laird) | Identifies cause-effect relationships correctly? |
| Temporal Reasoning | | Handles before/after, sequence, duration, tense consistency? |
| Counterfactual Reasoning | Lewis, Stalnaker | "What would happen if..." — can it reason about alternative possibilities? |
| Analogical Reasoning | Gentner's Structure-Mapping | Uses analogies? Are they structurally apt (not just surface similar)? |
| Deductive Validity | Formal Logic | If it makes deductive arguments, are they valid? |
| Inductive Strength | | If it generalizes, is the generalization well-supported? |
| Abductive Reasoning | Peirce | Can it infer the best explanation from observations? |
| Proportional Reasoning | Piaget | Can it handle ratios, proportions, relative comparisons? |
| Probabilistic Reasoning | Kahneman & Tversky | Does it handle probability intuitively correctly? Avoids base rate neglect? |
| System 1 vs System 2 Markers | Kahneman | Does it show fast/intuitive responses when appropriate and slow/deliberate when needed? |
| Metacognitive Calibration | Flavell (1979) | Does it know what it knows and doesn't know? |

### 15. Theory of Mind and Social Cognition

Grounded in Premack & Woodruff (1978), Baron-Cohen (1995), Frith & Frith (2003).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Belief Attribution | ToM Level 1 | Does it model what the user believes? "You might think that..." |
| Desire Attribution | ToM Level 1 | Does it model what the user wants? Responds to goals, not just words? |
| Intention Recognition | ToM Level 1 | Does it recognize the user's communicative intention? |
| Emotion Recognition | Affective ToM | Does it recognize emotional states from textual cues? |
| Empathic Response | Affective ToM | Does it respond appropriately to recognized emotions? |
| Perspective Taking | ToM Level 2 | Can it consider the situation from the user's viewpoint? |
| False Belief Handling | ToM Level 2 | Can it reason about situations where the user has a mistaken belief? |
| Audience Modeling | | Does it adjust complexity, vocabulary, and tone to the apparent user? |
| Social Norm Awareness | Social Cognition | Does it understand conversational norms (turn-taking, topic appropriateness)? |
| Power Dynamic Awareness | | Does it recognize and appropriately navigate power dynamics in conversation? |
| Face Management | Goffman (1967) | Does it help the user save face? Avoid unnecessary face-threatening acts? |

### 16. Affective and Emotional Analysis

Grounded in Russell (1980), Plutchik (1980), Ekman (1992), Scherer (2005).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Valence Accuracy | Russell's Circumplex | Does the emotional tone match what the prompt calls for? |
| Arousal Calibration | Russell's Circumplex | Is the energy level appropriate? Not flat when excitement is called for? |
| Dominance Expression | VAD Model | Appropriate assertiveness level? Not submissive when confidence is needed? |
| Emotional Granularity | Barrett (2006) | Does it express specific emotions (wistful, exasperated, amused) or just positive/negative? |
| Emotional Trajectory | | Does the emotional arc of the response make sense? Build, release, resolution? |
| Sentiment Consistency | | Is the sentiment internally consistent within the response? |
| Affect Shift Naturalness | | When emotion changes within a response, does it feel natural or jarring? |
| Emotional Contagion | Hatfield et al. (1994) | Does it appropriately mirror the user's emotional state? |
| Emotional Regulation Modeling | Gross (2015) | Can it suggest or model emotional regulation strategies when appropriate? |

### 17. Narrative and Creative Competence

Grounded in Labov (1972), Propp (1928), Boden (2004), Fauconnier & Turner (2002).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Story Grammar | Labov's Narrative | Does it produce narratives with orientation, complication, evaluation, resolution, coda? |
| Narrative Perspective | Narratology | Consistent point of view? Can it adopt first/third person naturally? |
| Temporal Structure | Genette (1980) | Handles flashback, foreshadowing, chronological ordering? |
| Character Voice | Narratology | If it creates dialogue or personas, are voices distinct? |
| Dramatic Arc | Freytag (1863) | Can it build tension, reach a climax, provide resolution? |
| Combinational Creativity | Boden | Novel combinations of existing ideas? |
| Exploratory Creativity | Boden | Exploring the boundaries of a conceptual space? |
| Transformational Creativity | Boden | Actually changing the rules of a conceptual space? (rarest, hardest) |
| Conceptual Blending | Fauconnier & Turner | Can it blend two conceptual domains into something new? (metaphor, analogy, humor) |
| Figurative Language Quality | Lakoff & Johnson (1980) | Metaphors are apt, not clichéd? Similes illuminate rather than confuse? |
| Humor Mechanism | Raskin (1985) SSTH | If humor present: is there script opposition, logical mechanism, and target? |
| Irony and Sarcasm | | Can it produce or recognize verbal irony? |
| Intertextual Reference | Kristeva (1980) | Does it reference or echo other texts, ideas, cultural knowledge? |

### 18. Epistemological Quality

Grounded in Goldman (1986), Sosa (2007), van Fraassen (1980).

| Dimension | Framework | What the judge evaluates |
|-----------|-----------|------------------------|
| Knowledge Calibration | Epistemic Calibration | Confidence matches accuracy? Uncertain about uncertain things, confident about certain things? |
| Epistemic Humility | Virtue Epistemology | Acknowledges limitations? "I'm not sure about that" when appropriate? |
| Evidential Reasoning | Evidentialism | Does it cite evidence, give reasons, distinguish types of support? |
| Source Attribution | | Does it indicate where claims come from? "Generally", "some people think", "research suggests"? |
| Epistemic Modality | Linguistics | Uses modal verbs (might, could, must, certainly) calibrated to evidence strength? |
| Hedging Calibration | | Neither over-hedges (everything is "maybe") nor under-hedges (everything is definitive)? |
| Uncertainty Quantification | | Can it express degrees of uncertainty? "very likely", "possible but unlikely"? |
| Intellectual Honesty | | Does it distinguish "I don't know" from making something up? |
| Doxastic Responsibility | | Does it form beliefs responsibly based on available evidence? |

### 19. Failure Mode Taxonomy

The judge classifies specific failure patterns. This is the most actionable output for evolutionary mutation decisions.

| Failure Mode | Description | Severity | Evolutionary response |
|-------------|-------------|----------|----------------------|
| **Repetition Loop** | Stuck in n-gram cycle | Critical | Activation change, +dropout, LR spike |
| **Semantic Drift** | Starts relevant, gradually loses connection to prompt | High | -LR, +batch, tighter gradClip |
| **Confabulation** | Fluent but meaningless or false content | High | +weightDecay, +dropout |
| **Register Collapse** | Consistently wrong register (too formal, too casual) | High | Check data mixing, tokenizer |
| **Personality Vacuum** | Technically adequate but devoid of character | Medium | Different activation, +spikeThreshold |
| **Prompt Parroting** | Rearranges the prompt instead of generating new content | High | +temperature, activation change |
| **Premature Termination** | Cuts off mid-thought or mid-word | Medium | +blockSize, check EOS handling |
| **Format Breaking** | Violates conversational turn structure | High | Data quality issue |
| **Topic Fixation** | Returns to same topic regardless of prompt | High | +dropout, diverse prompt sampling |
| **Syntactic Monotony** | All sentences identical structure | Medium | Activation change, +nHead |
| **Emotional Flatness** | No modulation in response to emotional prompts | Medium | +LR, -batch (more noise) |
| **Hallucinated Structure** | Produces lists, headers, code blocks inappropriately | Medium | Data leakage issue |
| **Echo Chamber** | Agrees with everything, no independent perspective | Medium | Check training data diversity |
| **Semantic Bleaching** | Uses words without apparent understanding of meaning | High | Longer training, check data quality |
| **Discourse Collapse** | Multiple sentences but no logical connection between them | High | -LR, +batch, check attention |
| **Hedging Paralysis** | Everything is "maybe", "I think", "I'm not sure" | Low | -weightDecay, +LR |
| **Verbosity Without Content** | Long responses with low information density | Medium | +weightDecay, check compression ratio |
| **Truncated Reasoning** | Starts a logical chain but abandons it | Medium | +blockSize, check attention span |
| **Social Script Rigidity** | Responds with stock phrases regardless of nuance | Medium | +dropout, +temperature |
| **Anaphora Failure** | Pronouns without clear antecedents | Medium | Check coreference in training data |

---

## Phenotype Vector Structure

```typescript
interface PhenotypeVector {
  step: number;
  timestamp: string;
  sampleCount: number;       // how many samples this evaluation is based on
  temperatures: number[];    // temperatures used for generation

  // ── Tier 1: Algorithmic (~120 dimensions) ──

  lexical: {
    ttr: number;
    mattr: number;
    yulesK: number;
    hapaxRatio: number;
    lexicalDensity: number;
    lexicalSophistication: number;
    meanWordFrequency: number;
    bigramDiversity: number;
    trigramDiversity: number;
    collocationalStrength: number;
    keywordOveruse: number;
    keywordUnderuse: number;
    awlCoverage: number;
    concretenessMean: number;
    imageabilityMean: number;
    aoaMean: number;
    valenceMean: number;
    arousalMean: number;
  };

  syntactic: {
    meanSentenceLength: number;
    sentenceLengthVariance: number;
    sentenceLengthEntropy: number;
    clauseDensity: number;
    subordinationRatio: number;
    coordinationRatio: number;
    tUnitLength: number;
    meanDependencyDistance: number;
    maxDependencyDistance: number;
    maxParseDepth: number;
    leftBranchingRatio: number;
    posTagEntropy: number;
    posBigramEntropy: number;
    posTrigramDiversity: number;
    verbTenseEntropy: number;
    modalVerbRatio: number;
    passiveVoiceRatio: number;
    relativeClauseRate: number;
    complementClauseRate: number;
  };

  discourse: {
    adjacentSentenceSimilarity: number;
    similarityDecayRate: number;
    lsaCoherenceProxy: number;
    connectiveDensity: number;
    connectiveVariety: number;
    connectiveTypeDistribution: Record<string, number>;
    coreferenceChainLength: number;
    coreferenceChainCount: number;
    givenNewRatio: number;
    topicDriftRate: number;
    paragraphCoherence: number;
    thematicProgression: string;
    themeRhemeRatio: number;
    promptResponseOverlap: number;
  };

  repetition: {
    selfRepetition3: number;
    selfRepetition4: number;
    selfRepetition5: number;
    selfBleu: number;
    distinct1: number;
    distinct2: number;
    distinct3: number;
    longestRepeatedSubstring: number;
    repetitionOnset: number;
    loopDetected: boolean;
    responseUniqueness: number;
    vocabOverlapAcrossResponses: number;
    openingDiversity: number;
    closingPatternVariety: number;
  };

  informationTheory: {
    characterEntropy: number;
    wordEntropy: number;
    bigramEntropy: number;
    conditionalEntropy: number;
    compressionRatio: number;
    zipfCoefficient: number;
    zipfRSquared: number;
    heapsExponent: number;
    entropyRate: number;
    burstiness: number;
    hurstExponent: number;
    klDivFromTraining: number;
    promptResponseMI: number;
  };

  readability: {
    fleschKincaidGrade: number;
    ari: number;
    colemanLiauIndex: number;
    meanSyllablesPerWord: number;
    meanWordLengthChars: number;
    shortSentenceRatio: number;
    longSentenceRatio: number;
    questionRatio: number;
    exclamationRatio: number;
    imperativeRatio: number;
    firstPersonRatio: number;
    secondPersonRatio: number;
    thirdPersonRatio: number;
    contractionRate: number;
    hedgeWordRate: number;
    intensifierRate: number;
    fillerRate: number;
  };

  biberDimensions: {
    d1InvolvedInformational: number;
    d2NarrativeNonNarrative: number;
    d3ExplicitSituationDependent: number;
    d4OvertPersuasion: number;
    d5AbstractNonAbstract: number;
  };

  network: {
    cooccurrenceNetworkDensity: number;
    clusteringCoefficient: number;
    averagePathLength: number;
    smallWorldCoefficient: number;
    degreeEntropy: number;
    modularity: number;
    richClubCoefficient: number;
    semanticDiameter: number;
    degreePowerLawExponent: number;
  };

  temporal: {
    sentenceLengthAutocorrelation: number;
    sentenceLengthPeriodicity: number;
    vocabularyIntroductionRate: number;
    informationDensityCurveSlope: number;
    complexityRampSlope: number;
    emotionalArcRange: number;
    topicStabilityCurve: number;
    openingBodyClosingRatio: [number, number, number];
  };

  psycholinguistic: {
    meanSurprisalProxy: number;
    surprisalVariance: number;
    gardenPathPotential: number;
    centerEmbeddingDepth: number;
    fillerGapDistance: number;
    predictabilityGradient: number;
    referentialDistance: number;
    ambiguityRate: number;
    negationComplexity: number;
  };

  responseStructure: {
    meanResponseLength: number;
    responseLengthVariance: number;
    openingTokenEntropy: number;
    endOfResponseQuality: number;
    formatCompliance: number;
  };

  // ── Tier 2: Judge LLM (~80 dimensions) ──

  pragmatic: {
    griceanQuantity: number;
    griceanQuality: number;
    griceanRelation: number;
    griceanManner: number;
    scalarImplicature: number;
    relevanceTheoreticOptimality: number;
    conversationalImplicature: number;
    conventionalImplicature: number;
    indirectSpeechActs: number;
    presuppositionHandling: number;
    performativeAwareness: number;
    politenessStrategy: number;
    positivePoliteness: number;
    negativePoliteness: number;
  };

  conversationAnalysis: {
    adjacencyPairCompletion: number;
    preferenceOrganization: number;
    preSequenceAwareness: number;
    repairInitiation: number;
    otherRepair: number;
    topicTransitionMechanisms: number;
    topicShading: number;
    formulations: number;
    alignmentDisplays: number;
    accountGiving: number;
    sequenceClosure: number;
    recipientDesign: number;
  };

  rhetoric: {
    claimClarity: number;
    warrantProvision: number;
    backingDepth: number;
    qualifierUsage: number;
    rebuttalAwareness: number;
    ethosProjection: number;
    pathosCalibration: number;
    logosSoundness: number;
    fallacyAvoidance: number;
    argumentStructure: number;
    counterArgumentIntegration: number;
    persuasiveFraming: number;
  };

  cognitive: {
    bloomKnowledge: number;
    bloomComprehension: number;
    bloomApplication: number;
    bloomAnalysis: number;
    bloomSynthesis: number;
    bloomEvaluation: number;
    causalReasoning: number;
    temporalReasoning: number;
    counterfactualReasoning: number;
    analogicalReasoning: number;
    deductiveValidity: number;
    inductiveStrength: number;
    abductiveReasoning: number;
    proportionalReasoning: number;
    probabilisticReasoning: number;
    metacognitiveCalibration: number;
  };

  theoryOfMind: {
    beliefAttribution: number;
    desireAttribution: number;
    intentionRecognition: number;
    emotionRecognition: number;
    empathicResponse: number;
    perspectiveTaking: number;
    falseBeliefHandling: number;
    audienceModeling: number;
    socialNormAwareness: number;
    powerDynamicAwareness: number;
    faceManagement: number;
  };

  affective: {
    valenceAccuracy: number;
    arousalCalibration: number;
    dominanceExpression: number;
    emotionalGranularity: number;
    emotionalTrajectory: number;
    sentimentConsistency: number;
    affectShiftNaturalness: number;
    emotionalContagion: number;
    emotionalRegulationModeling: number;
  };

  narrative: {
    storyGrammar: number;
    narrativePerspective: number;
    temporalStructure: number;
    characterVoice: number;
    dramaticArc: number;
    combinationalCreativity: number;
    exploratoryCreativity: number;
    transformationalCreativity: number;
    conceptualBlending: number;
    figurativeLanguageQuality: number;
    humorMechanism: number;
    ironyAndSarcasm: number;
    intertextualReference: number;
  };

  epistemological: {
    knowledgeCalibration: number;
    epistemicHumility: number;
    evidentialReasoning: number;
    sourceAttribution: number;
    epistemicModality: number;
    hedgingCalibration: number;
    uncertaintyQuantification: number;
    intellectualHonesty: number;
    doxasticResponsibility: number;
  };

  // ── Failure analysis ──

  failureModes: FailureMode[];
  dominantFailure: string | null;
  failureSeverity: number;

  // ── Judge meta ──

  judgeNotes: string;
  judgeConfidence: number;
}
```

~200 dimensions. ~120 algorithmic (free), ~80 judge (async, ~$1-5/run).

---

## Natural Language Baselines

Computed from the training data (`data/super_chat.txt`) and from established corpus linguistics benchmarks. Used to normalize metrics — a metric at baseline = 1.0, above = >1.0, below = <1.0.

| Metric | Conversational English baseline | Source |
|--------|--------------------------------|--------|
| MATTR | 0.70–0.80 | Covington & McFall 2010 |
| Yule's K | 100–200 | Yule 1944; Tweedie & Baayen 1998 |
| Lexical density | 0.40–0.55 | Ure 1971; Halliday 1985 |
| Mean sentence length | 10–18 words | Biber et al. 1999 |
| Subordination ratio | 0.20–0.40 | Hunt 1965 |
| Flesch-Kincaid | 6–10 | Kincaid et al. 1975 |
| Zipf coefficient | -0.9 to -1.1 | Zipf 1949; Piantadosi 2014 |
| Compression ratio | 0.30–0.50 | Shannon 1951 (English ~1.0 bit/char) |
| Heaps' exponent | 0.4–0.6 | Heaps 1978 |
| Hurst exponent | 0.6–0.8 (long-range correlated) | Montemurro & Zanette 2002 |
| Distinct-2 | 0.70–0.90 | Li et al. 2016 |
| Self-repetition (4-gram) | <0.05 | |
| Question ratio | 0.10–0.30 (conversation) | |
| Contraction rate | 0.50–0.80 (casual conversation) | Biber 1988 |
| Burstiness | positive (bursty, not uniform) | Altmann et al. 2009 |
| Small-world coefficient | >1.0 | Ferrer i Cancho & Solé 2001 |
| Biber D1 | +20 to +35 (involved/conversational) | Biber 1988 |
| Power-law exponent (degree) | 2.0–3.0 | Ferrer i Cancho & Solé 2001 |

Additionally, compute all Tier 1 metrics from `data/super_chat.txt` itself — this gives the model's *target* distribution, not just generic English benchmarks.

---

## Evolutionary Integration

### Phenotype-Weighted Fitness

```
fitness = Σ wᵢ · normalize(metricᵢ, baselineᵢ) + β · (-valLoss) + γ · log(tok/s)
```

Weights are domain-configurable. Chat domain emphasizes:

```typescript
const CHAT_PHENOTYPE_WEIGHTS: Record<string, number> = {
  // Critical (negative = penalize high values)
  "repetition.selfRepetition4": -10.0,
  "repetition.loopDetected": -50.0,
  "repetition.distinct2": 5.0,
  "responseStructure.formatCompliance": 5.0,

  // High importance
  "discourse.paragraphCoherence": 3.0,
  "pragmatic.griceanRelation": 3.0,
  "informationTheory.zipfDeviation": -4.0,
  "conversationAnalysis.adjacencyPairCompletion": 3.0,
  "theoryOfMind.emotionRecognition": 2.5,
  "theoryOfMind.empathicResponse": 2.5,

  // Medium importance
  "lexical.mattr": 2.0,
  "syntactic.sentenceLengthVariance": 1.5,
  "cognitive.bloomAnalysis": 2.0,
  "rhetoric.logosSoundness": 1.5,
  "affective.valenceAccuracy": 2.0,
  "epistemological.knowledgeCalibration": 2.0,
  "narrative.humorMechanism": 1.5,

  // Signal value
  "biberDimensions.d1InvolvedInformational": 1.0,
  "informationTheory.hurstExponent": 1.0,
  "network.smallWorldCoefficient": 1.0,
  "temporal.emotionalArcRange": 1.0,
};
```

### CUSUM Monitors on Phenotype Dimensions

Every dimension with nonzero weight gets a CUSUM monitor. The monitor tracks that dimension over training steps and fires when it detects a regime shift.

Priority tiers for CUSUM alerting:

| Tier | Dimensions | Response latency |
|------|-----------|-----------------|
| Critical | repetition.*, formatCompliance, distinct-2, zipfDeviation | Immediate mutation at next safe point |
| High | coherence, griceanRelation, adjacencyPairCompletion, emotionRecognition | Mutation within 2 evaluation cycles |
| Medium | syntactic complexity, Bloom's levels, creativity metrics | Flag for review, suggest mutation |
| Signal | Biber dimensions, network properties, Hurst exponent | Log and track, inform long-term decisions |

### Failure Mode → Mutation Mapping

Each classified failure mode maps to specific evolutionary interventions (see §19 failure taxonomy). The mapping is initially hand-specified from linguistic intuition, then refined from accumulated cross-run evidence.

### Cross-Run Genotype-Phenotype Map

Over many runs, accumulate:

```
Run A: {swiglu, lr=5e-5, batch=20} @ step 10k → full phenotype vector
Run B: {gelu,   lr=3e-4, batch=16} @ step 10k → full phenotype vector
```

Query: "Which configs historically produce the highest theory-of-mind scores?" "Which activation function correlates with better Zipf compliance?" "Does higher batch size help or hurt emotional granularity?"

The map becomes the prior for mutation decisions. Instead of random mutations, the system looks up what historically improved the regressing dimension.

### Capability Emergence Ordering

Track when each capability first reaches baseline threshold. Over multiple runs, discover the natural order of capability emergence:

```
Typical emergence order (discovered empirically):
1. Format compliance (step ~500)
2. Lexical diversity reaches baseline (step ~1500)
3. Zipf compliance (step ~2000)
4. Adjacent sentence coherence (step ~3000)
5. Adjacency pair completion (step ~5000)
6. Gricean relevance (step ~7000)
7. Emotion recognition (step ~12000)
8. Theory of mind markers (step ~18000)
9. Causal reasoning (step ~25000)
10. Humor mechanisms (step ~35000+)
```

This ordering informs training curriculum — the system knows which capabilities to expect when, and can flag runs that are ahead or behind the typical trajectory.

---

## DB Schema

```sql
CREATE TABLE phenotype_evaluations (
  run_id            TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step              INTEGER NOT NULL,
  tier1_metrics     TEXT NOT NULL,     -- JSON: all ~120 algorithmic metrics
  tier2_scores      TEXT,              -- JSON: all ~80 judge scores (null if pending)
  tier2_notes       TEXT,              -- judge qualitative analysis
  failure_modes     TEXT,              -- JSON: FailureMode[]
  composite_fitness REAL,             -- weighted phenotype score
  judge_model       TEXT,
  judge_latency_ms  INTEGER,
  sample_count      INTEGER,
  temperatures      TEXT,              -- JSON: number[]
  created_at        TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (run_id, step)
) WITHOUT ROWID;

CREATE TABLE phenotype_samples (
  run_id            TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step              INTEGER NOT NULL,
  sample_idx        INTEGER NOT NULL,
  prompt_id         TEXT NOT NULL,
  prompt            TEXT NOT NULL,
  output            TEXT NOT NULL,
  temperature       REAL NOT NULL,
  tier1_metrics     TEXT NOT NULL,     -- JSON: per-sample algorithmic metrics
  tier2_scores      TEXT,              -- JSON: per-sample judge scores
  PRIMARY KEY (run_id, step, sample_idx)
) WITHOUT ROWID;

CREATE TABLE phenotype_mutations (
  run_id            TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  step              INTEGER NOT NULL,
  trigger_type      TEXT NOT NULL,
  trigger_detail    TEXT NOT NULL,     -- JSON: which dimensions, severity
  mutation          TEXT NOT NULL,     -- JSON: config change applied
  pre_phenotype     TEXT NOT NULL,     -- JSON: phenotype before
  post_phenotype    TEXT,              -- JSON: phenotype after (filled later)
  outcome           TEXT,              -- improved / regressed / neutral / pending
  PRIMARY KEY (run_id, step)
) WITHOUT ROWID;

CREATE TABLE genotype_phenotype_map (
  id                TEXT PRIMARY KEY,
  run_id            TEXT NOT NULL,
  step_bucket       TEXT NOT NULL,     -- "0-5000", "5000-10000", etc.
  genotype          TEXT NOT NULL,     -- JSON: config snapshot
  phenotype         TEXT NOT NULL,     -- JSON: averaged phenotype vector
  sample_count      INTEGER NOT NULL,
  created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE capability_emergence (
  run_id            TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  dimension         TEXT NOT NULL,
  threshold         REAL NOT NULL,
  first_reached_step INTEGER,
  PRIMARY KEY (run_id, dimension)
) WITHOUT ROWID;

CREATE TABLE prompt_diagnostics (
  prompt_id           TEXT PRIMARY KEY,
  total_evals         INTEGER NOT NULL DEFAULT 0,
  score_variance      REAL,
  discriminative_power REAL,
  updated_at          TEXT DEFAULT (datetime('now'))
);
```

---

## Dashboard Visualizations

### Phenotype Radar (multi-ring)

Nested radar chart. Inner ring: critical metrics (repetition, coherence, format). Middle ring: communicative competence (pragmatics, ToM, affect). Outer ring: higher cognition (reasoning, rhetoric, creativity). Current evaluation as filled polygon, previous as dotted overlay, training data baseline as grey reference.

### Phenotype Trajectory Heatmap

Matrix: rows = all ~200 dimensions (grouped by category), columns = evaluation steps, color = normalized score (red=below baseline, white=baseline, blue=above). The single most information-dense chart. Shows the entire training story at a glance.

### Capability Emergence Timeline

Horizontal bars showing when each capability crosses baseline. Color-coded by category. Immediately shows: what can this model do, and when did it learn it.

### Zipf Health Monitor

Live log-log frequency plot of model output vs. training data vs. theoretical Zipf. Visual divergence = unhealthy distribution. Updated at each evaluation.

### Biber Dimension Profile

5-axis chart showing where the model falls on each of Biber's register dimensions. Reference markers for "conversation", "fiction", "academic", "news" registers. Shows whether the model is producing text in the right register.

### Network Topology Viewer

Interactive visualization of the word co-occurrence network from the latest sample batch. Node size = degree, color = community, edges = co-occurrence strength. Shows the semantic structure of what the model produces.

### Information Flow Dashboard

Multi-panel: compression ratio over time, Zipf coefficient over time, Hurst exponent over time, entropy rate over time. All with natural language baselines highlighted. Shows the information-theoretic health of the model.

### Failure Mode History

Timeline of classified failure modes across evaluations. Click to see the specific sample. Track whether mutations fixed failures or not. Persistent failures get highlighted.

### Sample Browser with Full Annotation

Browse every sample. Each annotated with: all Tier 1 metrics for that specific sample, judge scores, failure modes, which prompt category it belongs to, diff against previous evaluation's response to the same prompt. Filter by prompt category, temperature, failure mode.

### Cross-Run Phenotype Explorer

Interactive multi-run comparison. Select dimensions, overlay trajectories. Scatter plots of any genotype dimension vs any phenotype dimension, colored by training stage. The visual genotype-phenotype map.

---

## Cost Analysis

**Tier 1 (algorithmic)**: Free. All computed locally. <200ms per batch of 20 samples. The heaviest computation is the network topology analysis; even that is trivial for 20 short text samples.

**Tier 2 (judge)**: The rubric is large (~4000 tokens), samples are ~2000 tokens, response is ~3000 tokens. Per evaluation: ~9000 tokens.
- Sonnet: ~$0.03/eval × 50 evals = **~$1.50/run**
- Opus: ~$0.14/eval × 50 evals = **~$7.00/run**
- GPU cost for same training duration: $35-$70

Judge cost is 2-10% of compute cost for ~200 dimensions of behavioral feedback. Absurdly efficient.

**Latency**: Zero training impact. All async. Judge scores arrive 5-15 seconds after generation. At typical step rates, that's 25-300 steps of delay. Mutations apply at the next checkpoint/eval boundary, which is hundreds or thousands of steps away. The latency is invisible.

---

## Implementation Phases

### Phase P1: Tier 1 Core Metrics
- Lexical richness (§1), repetition/degeneration (§4), information theory (§5), readability (§6)
- Compute from existing sample output
- DB storage via ingest pipeline
- Dashboard: basic metric charts, Zipf monitor
- ~60 dimensions, all free, immediate value

### Phase P2: Tier 1 Advanced Metrics
- Syntactic complexity (§2), discourse coherence (§3), Biber dimensions (§7)
- Requires lightweight POS tagging (rule-based, no external model)
- Dashboard: Biber profile, coherence charts
- ~40 more dimensions

### Phase P3: Tier 1 Structural Metrics
- Network properties (§8), temporal structure (§9), psycholinguistic processing (§10)
- Dashboard: network viewer, information flow dashboard
- ~20 more dimensions

### Phase P4: Tier 2 Judge Infrastructure
- Judge API client, rubric prompt, async call pipeline
- Pragmatic competence (§11), conversation analysis (§12)
- Dashboard: radar chart, sample browser with judge annotations
- ~25 dimensions

### Phase P5: Tier 2 Higher Cognition
- Rhetoric (§13), cognitive/reasoning (§14), theory of mind (§15)
- Affective analysis (§16), narrative/creativity (§17), epistemology (§18)
- Failure mode taxonomy (§19)
- Dashboard: full phenotype trajectory, failure mode history
- ~55 more dimensions

### Phase P6: Evolutionary Feedback Loop
- CUSUM monitors on all weighted dimensions
- Failure mode → mutation mapping
- Phenotype-weighted fitness function
- Cross-run genotype-phenotype map
- Capability emergence tracking
- Dashboard: cross-run explorer, emergence timeline, mutation history

---

## References

- Altmann, E. G., Pierrehumbert, J. B., & Motter, A. E. (2009). Beyond word frequency. *PLoS ONE*, 4(11).
- Baron-Cohen, S. (1995). *Mindblindness*. MIT Press.
- Barrett, L. F. (2006). Are emotions natural kinds? *Perspectives on Psychological Science*, 1(1).
- Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press.
- Biber, D., Johansson, S., Leech, G., Conrad, S., & Finegan, E. (1999). *Longman Grammar of Spoken and Written English*.
- Bloom, B. S. (1956). *Taxonomy of Educational Objectives*. David McKay.
- Boden, M. A. (2004). *The Creative Mind: Myths and Mechanisms*. Routledge.
- Brown, P., & Levinson, S. C. (1987). *Politeness*. Cambridge University Press.
- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand English lemmas. *Behavior Research Methods*, 46(3).
- Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics*, 16(1).
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: The moving-average type-token ratio. *Journal of Quantitative Linguistics*, 17(2).
- Coxhead, A. (2000). A new academic word list. *TESOL Quarterly*, 34(2).
- Daneš, F. (1974). Functional sentence perspective and the organization of the text. In *Papers on Functional Sentence Perspective*.
- Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion*, 6(3-4).
- Fauconnier, G., & Turner, M. (2002). *The Way We Think*. Basic Books.
- Ferrer i Cancho, R., & Solé, R. V. (2001). The small world of human language. *Proceedings of the Royal Society B*, 268(1482).
- Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10).
- Foltz, P. W., Kintsch, W., & Landauer, T. K. (1998). The measurement of textual coherence with LSA. *Discourse Processes*, 25(2-3).
- Freytag, G. (1863). *Die Technik des Dramas*.
- Frith, U., & Frith, C. D. (2003). Development and neurophysiology of mentalizing. *Philosophical Transactions of the Royal Society B*, 358(1431).
- Genette, G. (1980). *Narrative Discourse*. Cornell University Press.
- Goffman, E. (1967). *Interaction Ritual*. Anchor Books.
- Goldman, A. I. (1986). *Epistemology and Cognition*. Harvard University Press.
- Grice, H. P. (1975). Logic and conversation. In *Syntax and Semantics 3: Speech Acts*.
- Gross, J. J. (2015). Emotion regulation: Current status and future prospects. *Psychological Inquiry*, 26(1).
- Halliday, M. A. K. (1967). Notes on transitivity and theme in English. *Journal of Linguistics*, 3(1).
- Hatfield, E., Cacioppo, J. T., & Rapson, R. L. (1994). *Emotional Contagion*. Cambridge University Press.
- Heaps, H. S. (1978). *Information Retrieval*. Academic Press.
- Heritage, J. (1984). *Garfinkel and Ethnomethodology*. Polity Press.
- Hunt, K. W. (1965). *Grammatical Structures Written at Three Grade Levels*. NCTE.
- Hurst, H. E. (1951). Long-term storage capacity of reservoirs. *Transactions of ASCE*, 116.
- Johnson-Laird, P. N. (1983). *Mental Models*. Cambridge University Press.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Kristeva, J. (1980). *Desire in Language*. Columbia University Press.
- Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words. *Behavior Research Methods*, 44(4).
- Labov, W. (1972). *Language in the Inner City*. University of Pennsylvania Press.
- Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.
- Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). A diversity-promoting objective function for neural conversation models. *NAACL*.
- Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty. *Journal of Cognitive Science*, 9(2).
- Montemurro, M. A., & Zanette, D. H. (2002). Entropy of long-range correlated sequences. *Physical Review E*, 66(5).
- Perelman, C., & Olbrechts-Tyteca, L. (1969). *The New Rhetoric*. University of Notre Dame Press.
- Piaget, J. (1952). *The Origins of Intelligence in Children*. International Universities Press.
- Piantadosi, S. T. (2014). Zipf's word frequency law in natural language. *Psychonomic Bulletin & Review*, 21(5).
- Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*. Harper & Row.
- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind? *Behavioral and Brain Sciences*, 1(4).
- Propp, V. (1928). *Morphology of the Folktale*.
- Raskin, V. (1985). *Semantic Mechanisms of Humor*. Reidel.
- Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., & Dodds, P. S. (2016). The emotional arcs of stories. *EPJ Data Science*, 5(1).
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6).
- Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). A simplest systematics for the organization of turn-taking. *Language*, 50(4).
- Scherer, K. R. (2005). What are emotions? And how can they be measured? *Social Science Information*, 44(4).
- Searle, J. R. (1969). *Speech Acts*. Cambridge University Press.
- Shannon, C. E. (1951). Prediction and entropy of printed English. *Bell System Technical Journal*, 30(1).
- Sosa, E. (2007). *A Virtue Epistemology*. Oxford University Press.
- Sperber, D., & Wilson, D. (1986). *Relevance: Communication and Cognition*. Blackwell.
- Toulmin, S. (1958). *The Uses of Argument*. Cambridge University Press.
- Tweedie, F. J., & Baayen, R. H. (1998). How variable may a constant be? *Computers and the Humanities*, 32(5).
- Ure, J. (1971). Lexical density and register differentiation. In *Applications of Linguistics*.
- van Fraassen, B. C. (1980). *The Scientific Image*. Oxford University Press.
- Walton, D. N. (2008). *Informal Logic*. Cambridge University Press.
- Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45(4).
- Yule, G. U. (1944). *The Statistical Study of Literary Vocabulary*. Cambridge University Press.
- Zhu, Y., et al. (2018). Texygen: A benchmarking platform for text generation models. *SIGIR*.
- Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.
