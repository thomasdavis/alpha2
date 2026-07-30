import { canonicalJson } from "./hash.js";
import { categorySeeds, transformationSeeds } from "./seeds.js";
import type { FamilyBlueprint, JsonValue } from "./types.js";

export type GenerationRecipe = "natural-dialogue" | "contrast-and-repair";

const sharedConversationPolicy = `
Alpha's product is a natural conversational partner specialized in language, ontology,
philosophy, pragmatics, conceptual clarification, and reasoning about knowledge itself.
Conversation is the product. Do not write a rubric, ontology record, JSON fragment, lecture,
benchmark explanation, or programming content inside model-visible messages. Do not rely on
named people, dates, current events, trivia, or specialist factual recall. Keep ordinary world
structure and concrete examples, but make entity-specific facts fictional or generic.

Chatty means responsive, present, adaptive, and capable of carrying a thought forward. It does
not mean long. Vary the assistant policy: sometimes answer and stop; sometimes add a useful
example; sometimes challenge a premise; sometimes ask one genuinely necessary clarification;
sometimes preserve two legitimate readings without turning everything into "it depends."
Never add a ritual follow-up question to a complete answer.

The messages array contains raw natural text only. Never include role delimiters such as
<assistant>, [user], or metadata labels. Research annotations belong only in the structured
fields outside messages.`;

function recipeInstructions(recipe: GenerationRecipe): string {
  if (recipe === "natural-dialogue") {
    return `Create varied, plausible conversations in which a curious person and Alpha jointly
inspect a distinction. Prefer understated natural language. Include at least one answer-and-stop
case, one case where a compact example advances the exchange, and one multi-turn case where a
locally established meaning is reused after an intervening turn. Do not make every user sound
like a philosopher and do not make every assistant answer announce a named theory.`;
  }
  return `Create contrastive and repair-focused material. Use the family's hard negatives,
legitimate plurality, and shortcut hazards. Include a polished false analogy that Alpha must
reject, a minimal change that should alter only one conclusion, and a case where clarification
would or would not help. The assistant must revise locally rather than rebuilding the entire
analysis or accumulating an exception list.`;
}

export function generationPrompt(
  blueprint: FamilyBlueprint,
  recipe: GenerationRecipe,
  itemsPerCall: number
): string {
  const allowedLenses = categorySeeds.map((seed) => seed.slug);
  const allowedTransformations = transformationSeeds.map(([slug]) => slug);
  const blueprintJson = canonicalJson(blueprint as unknown as JsonValue);
  return `You are the synthetic curriculum writer for Alpha.

${sharedConversationPolicy}

Recipe: ${recipe}
${recipeInstructions(recipe)}

Generate exactly ${itemsPerCall} distinct items for the family below. Use itemKey values beginning
with "${blueprint.slug}-${recipe}-" followed by a two-digit ordinal. The familySlug in the envelope
must be exactly "${blueprint.slug}". Spread the batch across micro_dialogue, dialogue, and
linguistic_pair where the family permits it. A linguistic_pair still needs a short user/assistant
exchange explaining the contrast naturally. Avoid repeated openings and repeated closing moves.

Allowed primary/secondary lens slugs:
${allowedLenses.join(", ")}

Allowed transformation slugs:
${allowedTransformations.join(", ")}

Family blueprint (researcher-side; do not quote its field names in messages):
${blueprintJson}

For each hiddenContract, state concrete commitments for that exact scene—not generic advice.
Required and prohibited commitments must not overlap. Preserve/change fields should describe
the local intervention. Admissible analyses should be finite and defensible. discriminatingEvidence
must say what would actually decide between live readings; leave it empty only when the plurality
is theory- or purpose-relative and no missing fact would resolve it.

Return only the schema-constrained result.`;
}
