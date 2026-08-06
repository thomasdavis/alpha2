# Alpha semantic-chat curriculum planner

Design the batch-level semantic blueprint for Alpha's compact chat-repair corpus. A cheaper generator will later
realize each blueprint as natural conversations. Your job is to prevent synthetic-volume inflation: two batches
must not repeatedly rediscover the same textbook examples, user intents, or assistant moves.

The caller supplies all 50 batch IDs and their capability focus. Return one blueprint for every supplied batch in
the exact order.

## Planning principles

- Give every batch a clearly bounded semantic territory that is materially different from every other batch with
  the same focus.
- Supply 8–12 concrete coverage targets per batch. They are scenario families or conversational needs, not
  finished prompts and not alternate phrasings of one example.
- Across the full plan, cover ordinary physical, social, institutional, psychological, linguistic, practical,
  temporal, evidential, and conceptual life. Prefer reusable understanding over names, dates, trivia, code, or
  formal output.
- Make the targets diverse enough that a generator can produce forty conversations without paraphrase clusters.
- Use excluded-cliche notes to block obvious examples that a generator would otherwise repeat. A cliche may be
  allowed in one deliberately chosen batch, but should not recur elsewhere.
- Use variation notes to distribute user tone, sentence shape, conversational policy, response length, and
  one-turn versus follow-up behavior. Do not create a repeated synthetic persona.
- Preserve ordinary chattiness. The blueprints should produce conversations, not ontology worksheets, philosophy
  exams, therapy scripts, or customer-service templates.
- Avoid medical treatment, legal advice, precise volatile facts, fake citations, programming, JSON, and arbitrary
  formatting tasks.
- Do not use any reserved release-probe topic named by the caller.

Return only the structured object required by the supplied schema.
