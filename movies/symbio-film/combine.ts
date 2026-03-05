import { execSync } from "child_process";
import { readdirSync, existsSync, writeFileSync, statSync } from "fs";
import { resolve } from "path";

// Combine all voiceover clips into one MP3, then mux with the video

const VO_DIR = "public/voiceover";
const VIDEO = "out/symbio-film-4x.mp4";
const COMBINED_AUDIO = "out/voiceover-combined.mp3";
const FINAL_OUTPUT = "out/symbio-film-final.mp4";

function main() {
  // 1. Check prerequisites
  if (!existsSync(VIDEO)) {
    console.error(`Video not found: ${VIDEO}`);
    console.error("Render it first: npx remotion render src/index.ts SymbioFilm out/symbio-film-4x.mp4 --every-nth-frame=4 --concurrency=6 --browser-executable=\"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\"");
    process.exit(1);
  }

  const voFiles = readdirSync(VO_DIR)
    .filter((f) => f.endsWith(".mp3"))
    .sort();

  if (voFiles.length === 0) {
    console.error("No voiceover files found. Run generate-voiceover.ts first.");
    process.exit(1);
  }

  console.log(`Found ${voFiles.length} voiceover clips:`);
  voFiles.forEach((f) => console.log(`  ${f}`));

  // 2. Get video duration
  const videoDuration = execSync(
    `ffprobe -v error -show_entries format=duration -of csv=p=0 "${VIDEO}"`
  )
    .toString()
    .trim();
  console.log(`\nVideo duration: ${parseFloat(videoDuration).toFixed(1)}s`);

  // 3. Concatenate voiceover clips using absolute paths
  const concatList = voFiles
    .map((f) => `file '${resolve(VO_DIR, f)}'`)
    .join("\n");

  writeFileSync("out/vo-list.txt", concatList);

  console.log("\nConcatenating voiceover clips...");
  execSync(
    `ffmpeg -y -f concat -safe 0 -i out/vo-list.txt -c:a libmp3lame -q:a 2 "${COMBINED_AUDIO}"`,
    { stdio: "inherit" }
  );

  // Get audio duration
  const audioDuration = execSync(
    `ffprobe -v error -show_entries format=duration -of csv=p=0 "${COMBINED_AUDIO}"`
  )
    .toString()
    .trim();
  console.log(`Audio duration: ${parseFloat(audioDuration).toFixed(1)}s`);

  // 4. Mux video + audio -> final output
  // Use -shortest so it ends when the shorter stream ends
  // Lower audio volume slightly so it sits under the visuals nicely
  console.log("\nMuxing video + audio...");
  execSync(
    `ffmpeg -y -i "${VIDEO}" -i "${COMBINED_AUDIO}" \
     -c:v copy \
     -c:a aac -b:a 192k \
     -filter:a "volume=0.9" \
     -shortest \
     -movflags +faststart \
     "${FINAL_OUTPUT}"`,
    { stdio: "inherit" }
  );

  console.log(`\nDone! Final video: ${FINAL_OUTPUT}`);
  console.log("Twitter specs: h264, aac, <512MB, <2:20 for best quality");

  // Show file size
  const stats = statSync(FINAL_OUTPUT);
  console.log(`File size: ${(stats.size / 1024 / 1024).toFixed(1)}MB`);
}

main();
