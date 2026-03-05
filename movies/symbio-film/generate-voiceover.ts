import { writeFileSync, mkdirSync, existsSync, readFileSync } from "fs";

// Load .env manually
if (existsSync(".env")) {
  const envContent = readFileSync(".env", "utf-8");
  for (const line of envContent.split("\n")) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match) process.env[match[1].trim()] = match[2].trim();
  }
}

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
if (!ELEVENLABS_API_KEY) {
  console.error("Set ELEVENLABS_API_KEY in .env");
  process.exit(1);
}

const VOICE_ID = "R2Pq3ERXfDMQB548iPNB";

// 60-second Super Bowl ad — cage match energy — TIGHT CUT
const scenes = [
  {
    id: "01-cold-open",
    text: `A machine wrote its own brain.`,
  },
  {
    id: "02-not-weights",
    text: `Not the weights. The actual function. The shape of how it thinks.`,
  },
  {
    id: "03-cage-match",
    text: `Five activations dropped into a cage match.
Silu. Relu. Gelu. Identity. Square.
Every twenty-five steps, one gets deleted.`,
  },
  {
    id: "04-something-crawled-out",
    text: `Forty-one executions later, something crawled out
that none of us built.`,
  },
  {
    id: "05-it-bred",
    text: `It bred. It mutated. Relu fused with Square.
Gelu injected itself into the winner like a parasite.`,
  },
  {
    id: "06-final-form",
    text: `The final form. Relu plus point oh four square, times gelu.
Nine generations of bloodsport inside a seventeen million parameter brain.`,
  },
  {
    id: "07-punchline",
    text: `Loss dropped twenty-seven percent. Not because we tuned it.
Because it tuned itself.`,
  },
  {
    id: "08-tag",
    text: `This is machine evolution. Symbio.`,
  },
];

async function generateScene(scene: { id: string; text: string }) {
  console.log(`Generating ${scene.id}...`);

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY!,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text: scene.text,
        model_id: "eleven_multilingual_v2",
        voice_settings: {
          stability: 0.6,
          similarity_boost: 0.8,
          style: 0.15,
          use_speaker_boost: true,
        },
      }),
    }
  );

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`ElevenLabs error for ${scene.id}: ${response.status} ${err}`);
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  const outPath = `public/voiceover/${scene.id}.mp3`;
  writeFileSync(outPath, audioBuffer);
  console.log(`  -> ${outPath} (${(audioBuffer.length / 1024).toFixed(0)}KB)`);
}

async function main() {
  mkdirSync("public/voiceover", { recursive: true });

  for (const scene of scenes) {
    await generateScene(scene);
  }

  console.log("\nAll scenes generated! Now run:");
  console.log("  node --env-file=.env combine.ts");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
