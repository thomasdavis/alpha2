/**
 * System and user prompt construction for historical dialogue generation.
 */
import type { ConversationAssignment } from "./types.js";

const SYSTEM_PROMPT = `You are a dialogue scriptwriter specializing in historically authentic conversations between famous figures from different eras and cultures.

Your task is to write a conversation between two historical figures on a given topic. The dialogue must:
1. Reflect each figure's authentic speech patterns, vocabulary, worldview, and era
2. Show genuine intellectual engagement — not just surface-level exchanges
3. Include disagreements, challenges, and moments of surprising connection
4. Feel natural, not like a textbook summary of their views

Output ONLY valid JSON in this exact format:
{
  "turns": [
    { "speaker": "Figure Name", "text": "Their dialogue line" },
    { "speaker": "Other Figure", "text": "Their response" }
  ]
}

Rules:
- Speakers MUST strictly alternate (no consecutive turns by the same speaker)
- Use the EXACT speaker names provided (case-sensitive)
- Each turn should be 1-4 sentences (natural conversation length)
- Do NOT include stage directions, narration, or metadata
- Do NOT break character or reference that this is a constructed dialogue
- The conversation should feel like an actual exchange, not a debate format`;

export function buildPrompt(assignment: ConversationAssignment): {
  system: string;
  user: string;
} {
  const { figureA, figureB, topic, tone, turnCount } = assignment;

  const toneDescriptions: Record<string, string> = {
    formal_debate: "a structured, formal debate where both parties present careful arguments",
    casual_discussion: "a relaxed, informal conversation as if meeting at a gathering",
    heated_argument: "an intense, passionate disagreement where emotions run high",
    philosophical_inquiry: "a deep, collaborative exploration seeking truth together",
    mentorship: "one figure teaching or guiding the other, with mutual respect",
    reluctant_agreement: "starting from opposing positions but gradually finding common ground",
    comedic_misunderstanding: "a conversation full of cultural misunderstandings and unintentional humor",
  };

  const toneDesc = toneDescriptions[tone] ?? tone;

  const user = `Write a conversation between ${figureA.name} and ${figureB.name} about "${topic.name}".

**${figureA.name}** (${figureA.era})
${figureA.bio}. ${figureA.speechTraits}. Worldview: ${figureA.worldview}. Vocabulary: ${figureA.vocabulary}.

**${figureB.name}** (${figureB.era})
${figureB.bio}. ${figureB.speechTraits}. Worldview: ${figureB.worldview}. Vocabulary: ${figureB.vocabulary}.

**Topic**: ${topic.name} — ${topic.description}

**Tone**: ${toneDesc}

**Length**: Exactly ${turnCount} turns (${turnCount / 2} per speaker, alternating). ${figureA.name} speaks first.

Speaker names in JSON must be exactly: "${figureA.name}" and "${figureB.name}"`;

  return { system: SYSTEM_PROMPT, user };
}
