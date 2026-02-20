import type { Rng } from "@alpha/core";

/**
 * Morphology-aware gap-fill templates.
 * Each rule maps a word-ending pattern to sentence templates where the word
 * is placed in a grammatically appropriate position.
 */
const MORPHOLOGY_RULES: [RegExp, string[]][] = [
  // Adjective-like endings
  [/(?:ous|ive|ial|ual|able|ible|ical|istic|aceous|oid|ine|ose|ular)$/i, [
    "The {w} characteristics of the sample were immediately apparent.",
    "Researchers described the phenomenon as distinctly {w}.",
    "The specimen exhibited remarkably {w} properties under examination.",
    "A {w} quality distinguished it from similar specimens.",
  ]],
  // Abstract noun endings
  [/(?:tion|sion|ment|ness|ity|ance|ence|ism|ship|ure|dom|acy)$/i, [
    "The study of {w} has advanced considerably in recent decades.",
    "Understanding {w} requires careful analysis of contributing factors.",
    "The concept of {w} was central to the broader discussion.",
    "Early research into {w} laid the groundwork for modern understanding.",
  ]],
  // Verb-like endings (infinitive form)
  [/(?:ize|ise|ify|ate)$/i, [
    "Researchers attempted to {w} the collected samples.",
    "It was necessary to {w} the entire dataset before analysis.",
    "They developed a reliable method to {w} the raw material.",
    "The first step was to {w} the components under controlled conditions.",
  ]],
  // Present participle / gerund
  [/ing$/i, [
    "The process of {w} required several days to complete.",
    "Careful {w} proved essential to the success of the project.",
    "The technique of {w} has evolved significantly over time.",
    "Years of {w} had refined their approach considerably.",
  ]],
  // Past tense / past participle
  [/ed$/i, [
    "The specimens were {w} before the analysis could begin.",
    "Once properly {w}, the materials showed improved results.",
    "The records had been thoroughly {w} by the research team.",
    "Every sample was carefully {w} according to the protocol.",
  ]],
  // Adverbs
  [/ly$/i, [
    "The experiment proceeded {w} throughout the observation period.",
    "She {w} reviewed each piece of evidence before concluding.",
    "The findings were {w} consistent with earlier predictions.",
    "He {w} documented each step of the procedure.",
  ]],
  // Agent nouns
  [/(?:er|or|ist|ant|ent)$/i, [
    "The {w} was well regarded in the professional community.",
    "Each {w} contributed a unique perspective to the research.",
    "The role of the {w} proved essential to the final outcome.",
    "A skilled {w} could complete the task in half the time.",
  ]],
];

const DEFAULT_TEMPLATES = [
  "Historical records contain several references to {w}.",
  "The significance of {w} became evident during the investigation.",
  "Scholars have debated the nature of {w} for many years.",
  "The {w} was first documented in an early twentieth century survey.",
  "Further research into {w} may yield additional insights.",
  "The {w} played a notable role in the region's development.",
  "References to {w} appear throughout the published literature.",
  "The study of {w} remains an active area of inquiry.",
];

/** Generate a coherent sentence for a word based on its morphology. */
export function gapFill(word: string, rng: Rng): string {
  for (const [pattern, templates] of MORPHOLOGY_RULES) {
    if (pattern.test(word)) {
      const template = templates[Math.floor(rng.next() * templates.length)];
      return template.replace(/\{w\}/g, word);
    }
  }
  const template = DEFAULT_TEMPLATES[Math.floor(rng.next() * DEFAULT_TEMPLATES.length)];
  return template.replace(/\{w\}/g, word);
}

/** Fisher-Yates shuffle (in-place) using injected Rng. */
export function shuffle<T>(arr: T[], rng: Rng): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

export function pick<T>(arr: T[], rng: Rng): T {
  return arr[Math.floor(rng.next() * arr.length)];
}
