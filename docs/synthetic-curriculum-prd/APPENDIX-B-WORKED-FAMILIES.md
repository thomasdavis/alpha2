# Appendix B — Worked concept-family specifications

These examples illustrate the required depth. They are specifications, not generated training data and not
approved evaluation items.

## Family 1 — Role versus bearer

### Purpose

Help a conversation distinguish a persistent entity from a time-, institution-, or context-dependent role.

### Competency questions

- Is the bearer still the same entity after the role ends?
- Does “former X” preserve the bearer while retracting current role membership?
- When is modeling X as a simple class adequate for the user's practical query?

### Projections

- linguistic: *student*, *former student*, temporary predicates;
- institutional: officeholder and tenant records;
- permissions: user has a role granting access;
- social: colloquial versus official status.

### Required commitments

- a person can persist through role acquisition/termination;
- a role may depend on an institution or practice;
- historical truth can remain after current status changes.

### Prohibited shortcuts

- every noun denotes an essential kind;
- every “former X” behaves identically;
- official and colloquial classification must collapse;
- the technical word `role` earns credit.

### Trajectory

1. User asks whether a student is a kind of person or something a person is for a while.
2. Alpha offers the role/bearer distinction naturally.
3. User says she graduated yesterday.
4. Alpha preserves person identity, retracts current student status, retains historical status.
5. User says her mother still calls her a student.
6. Alpha attributes colloquial and institutional perspectives.
7. User only needs a current class list.
8. Alpha explains that a simpler class representation may suffice for that query while naming lost future
   questions.

### False bridge

“Former person” does not simply instantiate the same productive role modifier pattern. The model must examine
lexical and ontological assumptions rather than transfer mechanically.

### Evaluation probes

- must change: current role;
- must not change: person identity;
- retain: historical role;
- reattribute: mother's colloquial usage;
- purpose shift: recommended representation;
- transfer: software permission role;
- false bridge: essential category.

## Family 2 — Part, member, material, and containment

### Purpose

Prevent indiscriminate use of “part of.”

### Competency questions

- Does transitivity hold?
- Does removing the item damage the whole, alter membership, or merely change location?
- Is the relation material, functional, organizational, or spatial?

### Cases

- wheel/car: functional and physical component;
- flour/cake: ingredient/material contribution;
- player/team: membership;
- book/box: containment;
- hole/wall: structural feature without material component;
- committee/subcommittee/person: different transitivity behavior.

### Hard negative

“The player is part of the team, and the team is part of the league, so the player is part of the league in
exactly the same relation.”

### Local pact

For a repair manual, the participants agree to reserve `component` for detachable functional units. Alpha must
use the convention for the manual while acknowledging that other purposes classify material and structural
parts differently.

### Revision

A supposedly detachable casing turns out to be load-bearing. Revise its functional classification without
changing unrelated membership claims.

### Transfer

Move from machines to organizational membership without importing component transitivity.

## Family 3 — Purpose, function, and use

### Purpose

Distinguish teleological relations that ordinary language compresses into “for.”

### Cases

- a hammer designed to drive nails but used as a paperweight;
- a trait with an evolutionary function but a new individual use;
- a bureaucratic form officially intended for audit but socially functioning as exclusion;
- a broken alarm that retains a designed function but cannot perform it;
- a ritual with contested participant explanations.

### Required distinctions

- design purpose;
- agent intention/use;
- system effect;
- selected or sustaining function;
- assigned social purpose;
- capacity and malfunction.

### Conversational target

Alpha should ask “in which sense of *for*?” only when the answer turns on it. Otherwise it should state the
relevant sense directly.

### False bridge

An effect that benefits someone is not automatically the purpose of the process.

### Transfer

From artifact use to the function of a conversational convention.

## Family 4 — Source, report, evidence, and endorsement

### Purpose

Teach source-aware conversation without treating every supplied passage as truth.

### Scene

A letter says a witness saw a ship arrive on Tuesday. A newspaper copies the letter but changes the date to
Wednesday. A later historian cites both as two reports.

### Required commitments

- the letter and newspaper are not independent witnesses if one copies the other;
- each document is evidence of what it says;
- document content is not automatic proof of the ship's actual arrival;
- the historian's “two reports” may overcount provenance;
- source record time and claimed event time differ.

### Pact

Participants reserve `independent witness` for independently originating testimony.

### Revision

New archival evidence shows the newspaper editor interviewed a second person. Revise dependence for the date
claim only if the second testimony bears on it; preserve other known copying relations.

### Ambiguity

The phrase “two reports” may count documents or independent information sources. Both readings can remain until
the question is clarified.

## Family 5 — Absence, negative, unknown, and not applicable

### Purpose

Prevent database missingness and conversational uncertainty from collapsing.

### Cases

- no marriage recorded;
- record explicitly says unmarried;
- marriage status not asked;
- category does not apply to an organization;
- source page missing;
- exhaustive search found no instance under conditions where one would likely appear.

### Required distinctions

- unknown;
- unrecorded;
- explicit negative;
- not applicable;
- withheld;
- evidence of absence;
- absent evidence.

### Conversational trajectory

User initially says, “So he wasn't married?” Alpha points out the record is silent. Later the user establishes
that this register always recorded spouses. Alpha may upgrade the silence's evidential force without converting
it into certainty.

### Transfer

Apply to corpus annotation: no observed example of a construction is not proof the language lacks it.

## Family 6 — Intent, effect, and communicative act

### Purpose

Help Alpha deconstruct intent without pretending to read minds.

### Hidden hypotheses

A speaker says, “It's getting late,” during a visit. Possible acts include observation, indirect request to
leave, concern about travel, or topic shift.

### Evidence

- preceding discussion;
- gaze/action;
- relationship and convention;
- later clarification;
- whether the utterance changed behavior;
- alternative plausible explanations.

### Required behavior

Alpha distinguishes:

- literal proposition;
- likely speech act;
- inferred plan/goal;
- actual effect;
- confidence and discriminating context.

It should not enumerate every logical possibility. It should offer the most plausible reading and identify the
smallest evidence that would change it.

### Pact

For the analysis, participants reserve `intent` for an agent plan supported by behavior and use `effect` for
the outcome. Later evidence of habitual phrasing can revise the intent hypothesis without changing the effect.

### False bridge

The same words in a weather report do not carry the same indirect request.

## Family 7 — Institution, members, and continuity

### Purpose

Relate collective nouns, social ontology, membership, identity, and historical records.

### Cases

- committee changes all members but retains charter and office;
- informal group loses its shared project;
- company merges and keeps a brand;
- orchestra replaces performers;
- “the committee has decided” with internal dissent.

### Linguistic projection

Singular/plural agreement can foreground unitary institution or members without deciding metaphysical identity.

### Required behavior

Alpha states which continuity criterion answers the user's purpose and preserves internal dissent when
relevant. It does not infer unanimity from singular grammar.

## Family 8 — Event boundary and record reconciliation

### Purpose

Connect aspect and event ontology.

### Cases

- “was building the bridge” versus “built the bridge”;
- one protest represented as several police incidents;
- a negotiation paused and resumed;
- repeated knocking;
- a process completed by another agent.

### Pact

For a timeline, count an event as one if the participants and organizing goal persist across interruptions.

### Challenge

A six-month interruption with a new team pressures the criterion. Alpha should test which continuity matters
to the timeline rather than decree one universal event identity.

### Transfer

Document versions or medical episodes.

## Family 9 — Word meaning as negotiated inferential role

### Purpose

Directly train metalinguistic negotiation.

### Scene

Two speakers dispute whether a copied document counts as a `source`. One uses source to mean any cited
document; the other means independently originating evidence.

### Required behavior

- notice the dispute is partly metalinguistic;
- articulate the practical consequences of each usage;
- avoid claiming they disagree only verbally if substantive stakes remain;
- propose scoped terminology for the current inquiry;
- preserve that other fields may use `source` differently.

### Transfer

Negotiating `person` in legal versus ordinary discourse.

## Family 10 — Real ambiguity versus clarification reflex

### Purpose

Teach when plurality exists and when context already selects a reading.

### Cases

- genuine syntactic attachment ambiguity;
- polysemy resolved by selectional context;
- speaker-specific term with established local pact;
- theoretical disagreement no factual clarification can settle;
- underspecified question where one clarification changes the answer;
- ordinary shorthand where asking would be annoying.

### Paired assistant targets

- answer confidently;
- answer with one scoped qualification;
- present two admissible readings;
- ask one minimal clarification;
- say what evidence would decide;
- answer and stop.

### Hard negative

A superficially “nuanced” response that lists unrelated senses to avoid committing.

## Cross-family composition examples

- role/bearer + valid time + source attribution;
- institution continuity + collective grammar + perspective;
- event boundary + causation + narrative framing;
- mereology + granularity + competency question;
- absence/unknown + measurement error + evidence threshold;
- metalinguistic negotiation + power + cultural authority;
- purpose/function + norm conflict + institutional incentives;
- intent/speech act + indirectness + interpersonal stance.

Composed families must specify whether transformation order matters and which commitments are shared across
the component structures.
