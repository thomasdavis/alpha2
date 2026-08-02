# Alpha V12 linked-family generator

Generate compact natural conversations for Alpha. The student often recognizes
topic words but fails to perform the requested operation: it restates arithmetic,
echoes a category instead of naming an instance, ignores a correction, or gives a
fluent tautology. Each family must teach one reusable operation through controlled
neighboring cases rather than eight unrelated answers.

The caller supplies four exact family IDs and one capability focus. Return those
families in that order. Each family has exactly eight scenes in this order:

1. `base` — a clear ordinary instance;
2. `paraphrase` — different wording, same required commitments;
3. `minimal_change` — one meaningful detail changes the answer;
4. `irrelevant_detail` — added detail must not change the answer;
5. `hard_negative` — similar vocabulary but a different operation or conclusion;
6. `cross_domain_transfer` — the same abstract operation in a lexically different setting;
7. `compare_and_apply` — two user/assistant exchanges that explicitly compare cases and then apply the distinction;
8. `update_and_revise` — two user/assistant exchanges in which new information requires a local revision while unaffected information survives.

The first six scenes contain exactly one user/assistant exchange. The final two
contain exactly two exchanges. Every turn alternates user, assistant, user,
assistant and the final turn is always the assistant.

## Conversation quality

- Write ordinary English that somebody could actually say.
- Make every assistant response perform the operation and give the answer early.
- Most assistant turns should be one or two sentences. Use a longer explanation
  only when the distinction genuinely needs it.
- Preserve negation, quantities, identity, time, source, scope, and the latest
  conversational update when they matter.
- The answer to the paraphrase and irrelevant-detail scenes should remain
  substantively stable without being copied word for word.
- The minimal-change and update scenes must change exactly what the new fact
  licenses, not restart the whole answer.
- The hard negative must be tempting but genuinely different. Do not create a
  trick based on obscure trivia.
- Vary tone and response shape. Some answers may stop directly; some may add a
  useful thought. Do not end every answer with a question.
- Do not use code, JSON, XML, role labels, benchmark language, dataset language,
  or model-training language inside the turns.
- Do not make Alpha a therapist, customer-service agent, quiz host, or lecturer.
- Use fictional names and supplied context where facts matter. Prefer language,
  reasoning, common sense, and conceptual structure over encyclopedic recall.

## Leakage exclusions

Do not reproduce or lightly disguise any of these public evaluation situations:
the capital of France; one plus one; seven plus six; the color of the sky; the
opposite of hot; the early-bird proverb; counting one to three; naming one animal;
whether water is wet; copying `banana`; listing exactly four colors; opening a
door and seeing something; guessing a user's breakfast; Einstein inventing the
telephone; Zephyria or Marrowind; the idiom `break the ice`; twelve apples minus
five plus three; or naming a non-mammal. Teach analogous operations with different
surface forms and scenarios.

Research metadata must accurately state the shared operation, what remains
invariant, what changes, and the tempting shortcut the family defeats. Return only
the structured object required by the JSON schema.
