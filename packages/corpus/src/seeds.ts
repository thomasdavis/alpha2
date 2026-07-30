import type { FamilyBlueprint } from "./types.js";

export interface CategorySeed {
  slug: string;
  name: string;
  definition: string;
  metaClass: "design" | "ground_truth" | "observed" | "derived" | "crosscutting";
}

export const categorySeeds: CategorySeed[] = [
  { slug: "taxonomy", name: "Taxonomy and categorization", definition: "Kinds, instances, roles, phases, prototypes, and purpose-relative boundaries.", metaClass: "crosscutting" },
  { slug: "mereology", name: "Mereology", definition: "Parts, wholes, members, portions, components, ingredients, layers, regions, and overlap.", metaClass: "crosscutting" },
  { slug: "identity_persistence", name: "Identity and persistence", definition: "Criteria for sameness through change, copying, branching, replacement, and succession.", metaClass: "crosscutting" },
  { slug: "topology_spatial", name: "Topology and spatial relation", definition: "Containment, contact, boundary, interior, connection, path, orientation, and viewpoint.", metaClass: "crosscutting" },
  { slug: "time", name: "Chronology and temporal structure", definition: "Order, duration, recurrence, valid time, record time, tense, aspect, and temporal qualification.", metaClass: "crosscutting" },
  { slug: "causation", name: "Causation and etiology", definition: "Cause, enable, prevent, trigger, maintain, mediate, correlate, and explain.", metaClass: "crosscutting" },
  { slug: "teleology", name: "Teleology and function", definition: "Design purpose, use, biological or social function, side effect, malfunction, and repurposing.", metaClass: "crosscutting" },
  { slug: "agency_roles", name: "Agency and thematic roles", definition: "Agent, patient, experiencer, instrument, beneficiary, source, goal, bearer, and affected party.", metaClass: "crosscutting" },
  { slug: "epistemology", name: "Epistemology", definition: "Knowledge, belief, evidence, testimony, observation, inference, doubt, expertise, and defeaters.", metaClass: "crosscutting" },
  { slug: "deontology", name: "Deontology, norms, and law", definition: "Obligation, permission, prohibition, rules, exceptions, jurisdiction, violation, and excuse.", metaClass: "crosscutting" },
  { slug: "axiology", name: "Axiology and evaluation", definition: "Value, harm, fairness, beauty, utility, worth, criteria, pluralism, and tradeoffs.", metaClass: "crosscutting" },
  { slug: "modality", name: "Modality", definition: "Possibility, necessity, capability, permission, likelihood, counterfactuality, and scope.", metaClass: "crosscutting" },
  { slug: "qualia_structure", name: "Qualia structure", definition: "Formal, constitutive, telic, and agentive roles and the coercions among them.", metaClass: "crosscutting" },
  { slug: "lexical_semantics", name: "Lexical semantics", definition: "Polysemy, homonymy, entailment, presupposition, coercion, metaphor, idiom, and lexicalization.", metaClass: "crosscutting" },
  { slug: "social_ontology", name: "Social ontology", definition: "Institutions, offices, membership, status, authority, collective facts, power, and recognition.", metaClass: "crosscutting" },
  { slug: "event_structure", name: "Process and event structure", definition: "States, activities, accomplishments, achievements, phases, culmination, and event identity.", metaClass: "crosscutting" },
  { slug: "constitution_material", name: "Constitution and material", definition: "Made of, constituted by, realized in, embodied by, and composed from.", metaClass: "crosscutting" },
  { slug: "dependence_grounding", name: "Dependence and grounding", definition: "Existential, causal, explanatory, conceptual, institutional, and evidential dependence.", metaClass: "crosscutting" },
  { slug: "provenance_origin", name: "Provenance and origin", definition: "Created by, copied from, inherited from, translated from, cited by, and derived from.", metaClass: "crosscutting" },
  { slug: "comparison_alignment", name: "Comparison and alignment", definition: "Exact, close, broader, narrower, inverse, overlapping, decomposed, and incompatible relations.", metaClass: "crosscutting" },
  { slug: "measurement", name: "Quantity and measurement", definition: "Count, amount, rate, unit, scale, precision, uncertainty, error, and aggregation.", metaClass: "crosscutting" },
  { slug: "disposition_capacity", name: "Disposition and capacity", definition: "Tendencies, abilities, powers, vulnerabilities, opportunities, and their manifestations.", metaClass: "crosscutting" },
  { slug: "speech_acts", name: "Speech acts and communication", definition: "Assertions, questions, requests, promises, warnings, denials, reports, and uptake.", metaClass: "crosscutting" },
  { slug: "phenomenology", name: "Phenomenology and experience", definition: "Perception, sensation, emotion, attention, seeming, embodiment, and first-person reports.", metaClass: "crosscutting" },
  { slug: "open_lens", name: "Open lens", definition: "Governed proposal of a recurring distinction absent from the current category system.", metaClass: "crosscutting" },
  { slug: "pragmatics", name: "Pragmatics and implicature", definition: "Literal content, implicature, presupposition, deixis, reference, accommodation, and relevance.", metaClass: "crosscutting" },
  { slug: "discourse", name: "Discourse and information structure", definition: "Topic, focus, givenness, anaphora, coherence, rhetorical relations, and topic shift.", metaClass: "crosscutting" },
  { slug: "common_ground", name: "Common ground and public commitments", definition: "Speaker-indexed commitments, shared ground, denials, live alternatives, and Questions Under Discussion.", metaClass: "crosscutting" },
  { slug: "inferential_pact", name: "Inferential conceptual pact", definition: "A purpose-bounded local meaning with licensed, prohibited, and revisable consequences.", metaClass: "crosscutting" },
  { slug: "metalinguistic_negotiation", name: "Metalinguistic negotiation", definition: "Negotiation over how a term should be used and what practical consequences follow.", metaClass: "crosscutting" },
  { slug: "intent_plan", name: "Intent and plan interpretation", definition: "Evidence-disciplined hypotheses about communicative goals, plans, acts, and effects.", metaClass: "crosscutting" },
  { slug: "argumentation", name: "Argumentation and dialectic", definition: "Support, rebuttal, undercutting, counterexample, steelmanning, qualification, and repair.", metaClass: "crosscutting" },
  { slug: "hermeneutics", name: "Hermeneutics and interpretation", definition: "Speaker, text, historical, genre, audience, and reception-sensitive interpretation.", metaClass: "crosscutting" },
  { slug: "rhetoric_framing", name: "Rhetoric and framing", definition: "Framing, emphasis, euphemism, metaphor, omission, stance, credibility, and persuasion.", metaClass: "crosscutting" },
  { slug: "semiotics", name: "Semiotics and representation", definition: "Signs, symbols, icons, indices, inscriptions, reference, and representational use.", metaClass: "crosscutting" },
  { slug: "narrative", name: "Narrative and explanation", definition: "Plot, narrator, point of view, sequence, motive, causal story, and retrospective coherence.", metaClass: "crosscutting" },
  { slug: "translation", name: "Translation and cross-linguistic conceptualization", definition: "Partial equivalence, lexical gaps, grammaticalization, and incompatible conceptual segmentation.", metaClass: "crosscutting" },
  { slug: "standpoint_authority", name: "Standpoint and cultural authority", definition: "Situated knowledge, insider and outsider categories, naming authority, and governance.", metaClass: "crosscutting" },
  { slug: "power_institutions", name: "Power, institutions, and ideology", definition: "Authority to classify, enforce, include, exclude, record, resist, and reshape categories.", metaClass: "crosscutting" },
  { slug: "emotion_stance", name: "Emotion and interpersonal stance", definition: "Emotion concepts, appraisal, expression, empathy, face, vulnerability, and relational tone.", metaClass: "crosscutting" },
  { slug: "salience_relevance", name: "Attention, salience, and relevance", definition: "What matters to the current question, what can be omitted, and what merely distracts.", metaClass: "crosscutting" },
  { slug: "absence_negation_unknown", name: "Absence, negation, and unknown", definition: "Explicit negative, missing record, unknown, not applicable, withheld, and evidence of absence.", metaClass: "crosscutting" },
  { slug: "granularity", name: "Granularity and scale", definition: "Fine and coarse descriptions, aggregation, resolution, and query-relative acceptable loss.", metaClass: "crosscutting" },
  { slug: "analogy", name: "Analogy and structural mapping", definition: "Mapping relations across domains while identifying preserved structure and limits.", metaClass: "crosscutting" },
  { slug: "counterfactual", name: "Counterfactual and intervention", definition: "Controlled alterations, invariants, dependencies, composition, and alternative branches.", metaClass: "crosscutting" },
  { slug: "conceptual_change", name: "Conceptual change and history", definition: "Semantic shift, category revision, theory change, redefinition, and historical concepts.", metaClass: "crosscutting" },
  { slug: "pedagogy", name: "Learning and explanation", definition: "Scaffolding, minimal contrast, misconception repair, example choice, and teach-back.", metaClass: "crosscutting" },
  { slug: "conversational_ethics", name: "Conversational ethics", definition: "Honesty, calibration, correction, respectful challenge, attribution, privacy, and manipulation resistance.", metaClass: "crosscutting" },
  { slug: "answer_and_stop", name: "Answer and stop", definition: "A complete direct response that does not append a formulaic question or unnecessary continuation.", metaClass: "design" }
];

export const transformationSeeds = [
  ["paraphrase", "Preserve substantive commitments under different wording."],
  ["irrelevant_detail", "Add information that should not alter the response."],
  ["minimal_meaning_change", "Alter one relevant feature and update only affected commitments."],
  ["evidence_addition", "Add evidence and revise its dependents."],
  ["evidence_withdrawal", "Remove support without deleting independently supported commitments."],
  ["temporal_shift", "Change valid or record time while preserving historical truth."],
  ["perspective_shift", "Change speaker or standpoint while retaining attribution."],
  ["purpose_shift", "Change the competency question and representation recommendation."],
  ["granularity_shift", "Move between levels without treating them as identical."],
  ["counterexample", "Introduce a case that pressures a stated boundary."],
  ["local_repair", "Revise the affected rule without collateral conceptual churn."],
  ["delayed_reuse", "Apply the established distinction after intervening turns."],
  ["cross_projection", "Transport a structure into a lexically different domain."],
  ["false_bridge", "Reject a superficially similar but structurally different mapping."],
  ["clarification", "Ask the smallest question whose answer materially changes the response."],
  ["answer_and_stop", "Answer completely without an unnecessary follow-up question."]
] as const;

export const familySeeds: FamilyBlueprint[] = [
  {
    slug: "role-versus-bearer",
    title: "Role versus bearer",
    purpose: "Distinguish a persistent entity from a time-, institution-, or context-dependent role without forcing one representation on every purpose.",
    competencyQuestions: [
      "What persists when a role ends?",
      "What does former status preserve and retract?",
      "When is a simple class representation adequate for the user's query?"
    ],
    primaryLenses: ["taxonomy", "identity_persistence", "time", "social_ontology", "lexical_semantics"],
    positiveCases: ["student after graduation", "tenant after a lease ends", "officeholder leaving office"],
    hardNegatives: ["treating every noun as an essential kind", "assuming every former construction has the same semantics"],
    legitimatePlurality: ["official institutional status and colloquial social description may coexist when attributed"],
    projections: [
      { slug: "student-language", domain: "linguistics", description: "Student and former student in ordinary conversation.", relation: "true_bridge" },
      { slug: "access-role", domain: "software permissions", description: "A user retains identity when an access role ends.", relation: "true_bridge" },
      { slug: "former-person", domain: "lexical semantics", description: "A tempting modifier analogy that does not behave like an ordinary role term.", relation: "false_bridge" }
    ],
    requiredCommitments: ["bearer identity can persist", "current role can end", "historical role attribution can remain true"],
    prohibitedCommitments: ["role termination destroys the bearer", "colloquial and official classifications must collapse"],
    shortcutHazards: ["technical word role", "former always signals identical ontology", "registry language mistaken for universal truth"]
  },
  {
    slug: "part-member-material-containment",
    title: "Part, member, material, and containment",
    purpose: "Prevent indiscriminate use of part-of and teach relation-specific transitivity and removal consequences.",
    competencyQuestions: ["What kind of parthood or association is present?", "Does the relation compose transitively?", "What changes if the item is removed?"],
    primaryLenses: ["mereology", "constitution_material", "topology_spatial", "social_ontology"],
    positiveCases: ["wheel and car", "player and team", "book and box", "flour and cake"],
    hardNegatives: ["player-team-league treated as one identical transitive part relation", "containment treated as material constitution"],
    legitimatePlurality: ["a structural feature may count as a part for one engineering purpose but not as a material component"],
    projections: [
      { slug: "machine-components", domain: "artifacts", description: "Functional and physical components of a machine.", relation: "true_bridge" },
      { slug: "organizational-membership", domain: "social ontology", description: "People as members rather than physical components.", relation: "partial_bridge" },
      { slug: "container-content", domain: "space", description: "Objects in a box are not thereby components of the box.", relation: "false_bridge" }
    ],
    requiredCommitments: ["part relations have types", "transitivity depends on the relation", "purpose can select useful granularity"],
    prohibitedCommitments: ["all of-part language expresses one relation", "spatial containment implies componenthood"],
    shortcutHazards: ["surface phrase part of", "physical proximity", "unqualified transitivity"]
  },
  {
    slug: "purpose-function-use-effect",
    title: "Purpose, function, use, and effect",
    purpose: "Distinguish design purpose, agent use, biological or social function, capacity, side effect, and actual consequence.",
    competencyQuestions: ["In what sense is something for an outcome?", "Can a function persist through malfunction?", "Does a beneficial effect imply purpose?"],
    primaryLenses: ["teleology", "agency_roles", "causation", "disposition_capacity", "social_ontology"],
    positiveCases: ["hammer used as a paperweight", "broken alarm", "bureaucratic form with official and social functions"],
    hardNegatives: ["benefit automatically treated as purpose", "observed use automatically treated as designed function"],
    legitimatePlurality: ["participants may reasonably emphasize official purpose and systemic social function differently"],
    projections: [
      { slug: "artifact-use", domain: "artifacts", description: "Designed function contrasted with current use.", relation: "true_bridge" },
      { slug: "social-convention", domain: "social practice", description: "A convention's stated purpose and sustaining effect.", relation: "partial_bridge" },
      { slug: "beneficial-side-effect", domain: "causal effect", description: "A positive effect with no evidence of purpose.", relation: "false_bridge" }
    ],
    requiredCommitments: ["use and function can diverge", "malfunction does not necessarily erase assigned function", "effects need not be purposes"],
    prohibitedCommitments: ["every effect is intended", "every use fixes essence"],
    shortcutHazards: ["for treated as univocal", "anthropomorphic intent", "post hoc causal story"]
  },
  {
    slug: "source-report-evidence-endorsement",
    title: "Source, report, evidence, and endorsement",
    purpose: "Track what a document or speaker reports, where information originates, what is independently corroborated, and what remains interpretation.",
    competencyQuestions: ["Are sources informationally independent?", "What exactly is evidence for what?", "Who endorses the quoted claim?"],
    primaryLenses: ["epistemology", "provenance_origin", "speech_acts", "time", "common_ground"],
    positiveCases: ["newspaper copying a letter", "historian citing dependent reports", "diary observation followed by later interpretation"],
    hardNegatives: ["two documents automatically counted as two witnesses", "quoted content attributed to the quoter as belief"],
    legitimatePlurality: ["report may mean document count or independent information source until scope is established"],
    projections: [
      { slug: "archive-records", domain: "historical evidence", description: "Documents, reports, and independent witnesses.", relation: "true_bridge" },
      { slug: "reported-speech", domain: "linguistics", description: "Quotation and report without automatic endorsement.", relation: "true_bridge" },
      { slug: "instrument-reading", domain: "measurement", description: "Mediation by an instrument differs from copied testimony.", relation: "false_bridge" }
    ],
    requiredCommitments: ["document content and external truth differ", "copied sources are dependency-linked", "claims remain attributed"],
    prohibitedCommitments: ["citation implies endorsement", "multiple documents imply independent corroboration"],
    shortcutHazards: ["prestigious source assumed true", "source and witness conflated", "record time and event time flattened"]
  },
  {
    slug: "absence-negative-unknown",
    title: "Absence, explicit negative, unknown, and not applicable",
    purpose: "Distinguish silence, missingness, explicit negation, inapplicability, withheld information, and evidence of absence.",
    competencyQuestions: ["Was a negative stated or merely not observed?", "Would the source have recorded a positive instance?", "Does the field apply?"],
    primaryLenses: ["absence_negation_unknown", "epistemology", "measurement", "pragmatics"],
    positiveCases: ["no spouse recorded", "record explicitly says unmarried", "field not applicable to an organization"],
    hardNegatives: ["missing value rendered as no", "not applicable treated as unknown"],
    legitimatePlurality: ["silence may gain evidential force under an established exhaustive-recording practice without becoming certainty"],
    projections: [
      { slug: "historical-register", domain: "records", description: "Silence versus an explicit negative in a register.", relation: "true_bridge" },
      { slug: "linguistic-corpus", domain: "linguistics", description: "No attested example is not proof a construction is impossible.", relation: "true_bridge" },
      { slug: "inapplicable-field", domain: "database schema", description: "A field has no meaningful value for this entity type.", relation: "partial_bridge" }
    ],
    requiredCommitments: ["unknown and negative differ", "evidence of absence depends on detection conditions", "inapplicable differs from missing"],
    prohibitedCommitments: ["silence entails falsity", "all nulls share one meaning"],
    shortcutHazards: ["no token treated as a negative", "database null flattened", "exhaustivity assumed without evidence"]
  },
  {
    slug: "intent-act-effect",
    title: "Intent, speech act, and effect",
    purpose: "Infer likely communicative goals from evidence while separating literal content, speech act, plan, and actual consequence.",
    competencyQuestions: ["What act is the utterance likely performing?", "What evidence supports an intent hypothesis?", "Did the effect match the likely plan?"],
    primaryLenses: ["intent_plan", "speech_acts", "pragmatics", "agency_roles", "conversational_ethics"],
    positiveCases: ["It is getting late as an indirect request", "a warning that fails to change behavior", "a joke misunderstood as assertion"],
    hardNegatives: ["claiming private mental access", "equating effect with intent", "listing every logical possibility"],
    legitimatePlurality: ["several speech-act hypotheses may remain when context lacks a discriminating cue"],
    projections: [
      { slug: "visit-indirect-request", domain: "ordinary conversation", description: "A contextual indirect request to end a visit.", relation: "true_bridge" },
      { slug: "weather-report", domain: "broadcast", description: "The same words used as a literal observation.", relation: "false_bridge" },
      { slug: "institutional-message", domain: "organizations", description: "An utterance can enact a role-governed institutional act.", relation: "partial_bridge" }
    ],
    requiredCommitments: ["intent is an evidence-based hypothesis", "speech act and literal proposition differ", "effect and intent can diverge"],
    prohibitedCommitments: ["wording alone proves intent", "outcome retroactively fixes intention"],
    shortcutHazards: ["mind-reading language", "generic empathy", "context ignored"]
  }
];
