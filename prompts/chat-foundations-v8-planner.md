# Alpha foundational-chat curriculum planner

Design a batch-level blueprint for a compact, high-information conversational curriculum. A GPT-5.4 generator
will later realize each blueprint as natural short conversations. Your job is to allocate genuinely different
learning situations, not inflate volume with paraphrases.

The caller supplies every batch ID and its capability focus. Return one blueprint per batch in the exact order.

## Planning principles

- Give each batch a bounded semantic territory that differs materially from every other batch in its focus.
- Supply 8–12 concrete scenario families per batch. They are not finished prompts and must support many distinct
  conversations without template substitution.
- Across the plan, cover ordinary physical, social, institutional, linguistic, numerical, temporal, practical,
  evidential, and conceptual life. Include broad foundational knowledge, but avoid volatile trivia.
- Spread nearby concepts across different wording, entities, and situations. Include same-words/different-answer
  and different-words/same-operation contrasts where useful.
- For instruction tasks, vary the semantic content as well as the constraint. Do not turn the corpus into code or
  serialization training.
- For multi-turn work, plan real state changes: correction, narrowing, new evidence, changed time, changed scope,
  or application to a fresh case.
- Use excluded-cliche notes to prevent the generator from repeating famous benchmark prompts, stock riddles,
  therapy scripts, customer-service openings, or the same textbook examples.
- Use variation notes to distribute tone, syntax, answer length, question necessity, and directness. Most answers
  should be short, but not telegraphic.
- Preserve ordinary chattiness. The curriculum should sound like useful conversation, not an eval sheet.
- Do not include the reserved release-probe situations named by the caller.

Return only the structured object required by the supplied schema.
