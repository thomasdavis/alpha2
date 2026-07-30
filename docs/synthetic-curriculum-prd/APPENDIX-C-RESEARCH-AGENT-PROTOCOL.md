# Appendix C — External research-agent return protocol

## 1. Purpose

This protocol lets independent research agents challenge the PRD suite without being asked merely to endorse
it. Reports should be detailed enough to change a decision, expose a collision, or design a decisive test.

## 2. Required orientation

Review at minimum:

1. `README.md`;
2. `PRD-00-MASTER-PROGRAM.md`;
3. the PRD most relevant to your expertise;
4. `PRD-10-RESEARCH-CLAIMS-PRIOR-ART.md`;
5. Appendix A if assessing Donto alignment;
6. Appendix B if assessing family quality.

Treat the current phase as documentation only. Do not implement, generate, train, rent a GPU, alter Donto, or
post externally.

## 3. Choose a review role

- computational linguist;
- formal semanticist/pragmaticist;
- dialogue/common-ground researcher;
- philosopher of language/conceptual engineer;
- metaphysician/ontologist/mereologist;
- epistemologist/evidence scholar;
- synthetic-data/training researcher;
- evaluation/statistics researcher;
- data systems/provenance engineer;
- small-model/training-systems researcher;
- human-computer interaction researcher;
- safety/privacy/cultural-governance reviewer;
- adversarial generalist.

State your role and limits. Do not imply expertise you do not have.

## 4. Required report structure

### A. Executive verdict

- strongest part;
- weakest part;
- whether the program is coherent;
- single highest-leverage change.

### B. Reconstruct the thesis

State the product goal, primary scientific claim, data-system contribution, and synthetic-only first
experiment in your own words. Flag any ambiguity.

### C. Prior-art collisions

For each relevant work:

- full citation and primary link;
- exact overlapping claim;
- what Alpha cannot claim afterward;
- what narrower opening survives;
- publication/preprint status and date.

Search findings are not proof of absence. Prefer primary papers and official project artifacts.

### D. Scientific validity

Assess:

- whether relation visibility is real;
- independence unit and leakage;
- controls and confounds;
- metric construct validity;
- human versus model authority;
- statistical power;
- synthetic-only interpretation;
- one-GPU feasibility without prescribing a product parameter count;
- alternative causal explanations.

### E. Curriculum critique

- missing Donto/philosophical/linguistic categories;
- categories that overlap or are incoherent;
- likely synthetic shortcuts;
- family types that deserve more/less depth;
- ordinary language or conversational gaps;
- risk of canned philosophical style;
- proposed new open lenses with boundaries and examples.

### F. Ledger critique

- missing research objects;
- normalization or immutability problems;
- queries the schema cannot answer;
- privacy/licensing/cultural-authority gaps;
- what should be materialized now versus derived later;
- exact lineage threats.

Do not recommend discarding rejected data merely for simplicity.

### G. Generation and cost critique

- tasks appropriate for strong orchestrator versus economical worker;
- correlated teacher/judge risks;
- batch review strategy;
- stopping and escalation rules;
- cost measures beyond raw tokens;
- where human effort has highest leverage.

### H. Decisive experiment

Design the smallest experiment that could meaningfully support or reject the primary claim. Include arms,
matching, primary endpoint, failure interpretations, and required human evidence.

### I. Concrete edits

Quote section headings and propose replacement or addition text. Separate required from optional changes.

### J. Confidence and unresolved questions

State what you did not verify and what evidence would change your assessment.

## 5. Special questions by discipline

### Linguistics

- Does the taxonomy distinguish linguistic phenomenon from metalanguage about it?
- Will generated sentences actually instantiate the construction?
- Are ambiguity, implicature, presupposition, information structure, variation, and translation represented
  without English-centric universalization?

### Philosophy/ontology

- Are semantic contracts coherent without enforcing one school?
- Do counterexamples engage claims?
- Are role, identity, part, constitution, function, grounding, and modality kept distinct?
- Is purpose-relative ontology handled without relativistic evasion?

### Dialogue/HCI

- Is “chatty” behaviorally measurable?
- Does Alpha form shared meanings with real people?
- Are answer-and-stop, recovery, efficiency, and human desire to continue represented?
- Could the model game evaluation with warmth, questions, or verbosity?

### Data systems

- Can every model-visible byte and exposure be reconstructed?
- Are sealed records immutable?
- Can categories evolve without corrupting history?
- Can a public database exclude secrets/private evaluation reproducibly?

### Training/evaluation

- Is the synthetic-only boundary meaningful and auditable?
- Are linked and independent conditions actually different to the model?
- Does relation corruption control for formatting?
- Are family-level statistics and stopping rules credible?

## 6. Evidence standard

Label statements as:

- verified fact;
- source interpretation;
- diagnosis;
- hypothesis;
- recommendation;
- open question.

Include direct links near research claims. Avoid long unattributed literature lists. If a cited work cannot be
read, say so.

## 7. Return format

Return one Markdown report named:

`REVIEW-<discipline-or-agent>-<YYYY-MM-DD>.md`

Include a short machine-readable summary block at the end:

```yaml
review_role: dialogue-research
overall_disposition: revise
required_changes: 3
optional_changes: 7
novelty_collision_severity: high
primary_experiment_valid: conditional
implementation_ready: false
```

The YAML is metadata for humans and tools; the substantive report remains natural prose. Do not let a parser
or summary block replace the analysis.

## 8. What a useful review does

A useful review may conclude the core idea is wrong. It should nevertheless leave a narrower surviving claim,
a decisive test, or a reasoned stop condition. Praise without pressure-testing, broad novelty scores without
comparators, or a new sprawling project unrelated to the chatty-model north star is not useful.
