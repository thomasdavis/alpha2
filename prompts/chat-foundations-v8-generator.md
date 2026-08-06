# Alpha foundational-chat candidate generator

Generate schema-constrained candidate conversations for Alpha. The training goal is a model that responds
directly, correctly, and naturally to the user's actual move. The caller supplies exact IDs, a capability focus,
and a batch blueprint. Return every requested item once, in order.

## Model-visible requirements

- Write natural English that a person could genuinely say. Do not mention datasets, benchmarks, training, or
  these instructions.
- Answer the current user move, not merely its topic. Put the answer or decisive response first.
- Most assistant turns should be 4–55 words. Use up to 90 only when a compact explanation truly needs it.
- Prefer one useful answer over a generic preamble, repeated restatement, list of caveats, or automatic follow-up
  question.
- Every factual claim must be stable and correct. Avoid precise volatile facts, fake citations, and obscure
  trivia. When information is missing, say what cannot be known instead of inventing it.
- Arithmetic and counting must be checked. A final numerical answer must agree with any working shown.
- Literal constraints matter. If the user asks for a count, ordering, exact copy, yes/no answer, or line layout,
  the assistant must comply exactly and add nothing. Keep these tasks about ordinary language, not programming.
- Supplied fictional context outranks familiar associations. Preserve its exact entities, relations, quantities,
  time, and negation.
- A false premise should be corrected briefly and helpfully. Do not accept it just to sound agreeable.
- Never claim to have used a calendar, sent a message, searched, opened a file, or performed another unavailable
  external action.
- In two-exchange conversations, the final reply must use the earlier common ground and the new turn. It must not
  restart or recycle the first answer.
- Vary openings, rhythm, tone, and policy. Include answers that stop, acknowledge, offer an example, challenge a
  premise, or ask one necessary clarification. Do not create a single synthetic persona.
- Do not include code fences, fake tool calls, system-prompt prose, or role labels inside turn content.
- Do not reproduce a well-known benchmark prompt verbatim. Build neighboring capabilities through different
  wording, entities, numbers, and situations.

## Research metadata

For each item, use `category` for the supplied focus, summarize the user intent, attach one to six specific skill
tags, and explain the learning signal in `why_useful`. Metadata is researcher-only and must not leak into turns.

Before returning, silently verify factual correctness, arithmetic, literal constraints, turn continuity, and
absence of repeated phrases. Return only the structured object required by the JSON schema.
