/**
 * Command: alpha fleet
 *
 * Manage remote GCP training instances — deploy, train, monitor, resume.
 */
import { spawn, execSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";

// ── Types ──────────────────────────────────────────────────────────────────

interface FleetInstance {
  host: string;
  zone: string;
  machine: string;
  gpu: string;
  role: string;
  setupDone?: boolean;
}

interface FleetConfig {
  sshUser: string;
  sshKey: string;
  deployDir: string;
  instances: Record<string, FleetInstance>;
}

interface ExecResult {
  stdout: string;
  stderr: string;
  code: number;
}

// ── Config ─────────────────────────────────────────────────────────────────

function fleetJsonPath(): string {
  // Walk up from this file to repo root — works compiled or dev
  let dir = process.cwd();
  for (let i = 0; i < 10; i++) {
    if (existsSync(join(dir, "fleet.json"))) return join(dir, "fleet.json");
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return join(process.cwd(), "fleet.json");
}

function loadFleet(): FleetConfig {
  const p = fleetJsonPath();
  if (!existsSync(p)) {
    console.error(`fleet.json not found (looked in ${p})`);
    process.exit(1);
  }
  return JSON.parse(readFileSync(p, "utf8"));
}

function saveFleet(config: FleetConfig): void {
  writeFileSync(fleetJsonPath(), JSON.stringify(config, null, 2) + "\n");
}

function getInstance(config: FleetConfig, name: string): FleetInstance {
  const inst = config.instances[name];
  if (!inst) {
    const available = Object.keys(config.instances).join(", ");
    console.error(`Unknown instance "${name}". Available: ${available}`);
    process.exit(1);
  }
  return inst;
}

// ── SSH helpers ────────────────────────────────────────────────────────────

function expandPath(p: string): string {
  if (p.startsWith("~/")) return join(process.env.HOME || "/root", p.slice(2));
  return p;
}

function sshOpts(config: FleetConfig): string[] {
  return [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
    "-i", expandPath(config.sshKey),
  ];
}

function sshTarget(config: FleetConfig, host: string): string {
  return `${config.sshUser}@${host}`;
}

function ssh(config: FleetConfig, host: string, cmd: string, opts?: { stream?: boolean }): Promise<ExecResult> {
  return new Promise((resolve, reject) => {
    const proc = spawn("ssh", [...sshOpts(config), sshTarget(config, host), cmd], {
      stdio: opts?.stream ? ["ignore", "inherit", "inherit"] : ["ignore", "pipe", "pipe"],
    });
    let stdout = "", stderr = "";
    if (!opts?.stream) {
      proc.stdout!.on("data", (d: Buffer) => stdout += d);
      proc.stderr!.on("data", (d: Buffer) => stderr += d);
    }
    proc.on("error", reject);
    proc.on("close", (code) => resolve({ stdout, stderr, code: code ?? 1 }));
  });
}

function scp(config: FleetConfig, local: string, host: string, remote: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn("scp", [...sshOpts(config), local, `${sshTarget(config, host)}:${remote}`], {
      stdio: ["ignore", "inherit", "inherit"],
    });
    proc.on("error", reject);
    proc.on("close", (code) => code === 0 ? resolve() : reject(new Error(`scp exited ${code}`)));
  });
}

function rsyncFrom(config: FleetConfig, host: string, remote: string, local: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn("rsync", [
      "-avz", "--progress",
      "-e", `ssh ${sshOpts(config).join(" ")}`,
      `${sshTarget(config, host)}:${remote}`,
      local,
    ], { stdio: ["ignore", "inherit", "inherit"] });
    proc.on("error", reject);
    proc.on("close", (code) => code === 0 ? resolve() : reject(new Error(`rsync exited ${code}`)));
  });
}

function sshInteractive(config: FleetConfig, host: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const proc = spawn("ssh", [...sshOpts(config), sshTarget(config, host)], {
      stdio: "inherit",
    });
    proc.on("error", reject);
    proc.on("close", (code) => resolve(code ?? 1));
  });
}

// Pattern that matches real training processes but not pgrep/bash -c wrappers
const TRAIN_PGREP = `ps aux | grep -E '[a]lpha train|[n]ode.*train' | grep -v 'bash -c'`;

// ── Formatting ─────────────────────────────────────────────────────────────

function pad(s: string, n: number): string {
  return s.length >= n ? s.slice(0, n) : s + " ".repeat(n - s.length);
}

function dim(s: string): string { return `\x1b[2m${s}\x1b[0m`; }
function bold(s: string): string { return `\x1b[1m${s}\x1b[0m`; }
function green(s: string): string { return `\x1b[32m${s}\x1b[0m`; }
function red(s: string): string { return `\x1b[31m${s}\x1b[0m`; }
function yellow(s: string): string { return `\x1b[33m${s}\x1b[0m`; }
function cyan(s: string): string { return `\x1b[36m${s}\x1b[0m`; }

// Detect whether instance has full repo or compiled binary.
// Prefer repo (node) since bun-compiled binary may have Vulkan init issues.
async function trainPrefix(config: FleetConfig, host: string): Promise<string> {
  const dir = config.deployDir;
  const { code: hasRepo } = await ssh(config, host, `test -f ${dir}/apps/cli/dist/main.js`);
  if (hasRepo === 0) return `node apps/cli/dist/main.js`;
  const { code: hasBin } = await ssh(config, host, `test -x ${dir}/alpha`);
  if (hasBin === 0) return `./alpha`;
  return "";
}

// ── Subcommands ────────────────────────────────────────────────────────────

async function fleetDashboard(config: FleetConfig): Promise<void> {
  const names = Object.keys(config.instances);
  console.log(bold("\n  Fleet Dashboard\n"));

  const results = await Promise.allSettled(names.map(async (name) => {
    const inst = config.instances[name];
    // Gather status in one SSH call
    const { stdout, code } = await ssh(config, inst.host, [
      "uptime -p 2>/dev/null || echo 'unknown'",
      "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'no-gpu'",
      `${TRAIN_PGREP} || echo 'no-training'`,
      "df -h / --output=used,size,pcent 2>/dev/null | tail -1",
    ].join(" && echo '---' && "));
    return { name, inst, stdout, code };
  }));

  for (const r of results) {
    if (r.status === "rejected") continue;
    const { name, inst, stdout, code } = r.value;

    if (code !== 0) {
      console.log(`  ${pad(name, 12)} ${red("OFFLINE")}  ${dim(inst.host)}  ${dim(inst.role)}`);
      continue;
    }

    const sections = stdout.split("---\n").map(s => s.trim());
    const uptime = sections[0] || "?";
    const gpuLine = sections[1] || "no-gpu";
    const training = sections[2] || "no-training";
    const disk = sections[3]?.trim() || "?";

    let gpuStr = dim("no GPU");
    if (gpuLine !== "no-gpu") {
      const [util, used, total, temp] = gpuLine.split(",").map(s => s.trim());
      gpuStr = `${inst.gpu} ${cyan(util + "%")} ${used}/${total}MB ${dim(temp + "°C")}`;
    }

    const trainingStr = training === "no-training"
      ? dim("idle")
      : green("training") + dim(` (${training.split("\n").length} proc)`);

    console.log(`  ${bold(pad(name, 12))} ${green("UP")}  ${gpuStr}  ${trainingStr}  ${dim("disk " + disk)}`);
    console.log(`  ${" ".repeat(12)} ${dim(inst.host + "  " + inst.machine + "  " + uptime)}`);
  }
  console.log();
}

async function fleetStatus(config: FleetConfig, name: string): Promise<void> {
  const inst = getInstance(config, name);
  console.log(bold(`\n  ${name}`) + dim(` (${inst.host})\n`));

  const { stdout, code } = await ssh(config, inst.host, [
    "echo '=== GPU ===' && nvidia-smi 2>/dev/null || echo 'No GPU found'",
    `echo '=== Training ===' && ${TRAIN_PGREP} || echo 'No training running'`,
    "echo '=== Disk ===' && df -h / 2>/dev/null",
    `echo '=== Runs ===' && ls -dt ~/alpha/runs/*/ 2>/dev/null | head -5 || echo 'No runs'`,
  ].join(" && "));

  if (code !== 0) {
    console.log(red("  Could not reach instance."));
    return;
  }

  console.log(stdout);
}

async function fleetDeploy(config: FleetConfig, name: string, kv: Record<string, string>): Promise<void> {
  const names = kv["all"] === "true" ? Object.keys(config.instances) : [name];
  if (names.length === 1 && !names[0]) {
    console.error("Usage: alpha fleet deploy <name> [--all]");
    process.exit(1);
  }

  // Step 1: Build locally
  console.log(bold("\n  Building binary...\n"));
  try {
    execSync("npm run bun:compile", { stdio: "inherit", cwd: process.cwd() });
  } catch {
    console.error(red("\n  Build failed. Fix errors and retry."));
    process.exit(1);
  }

  const bunOut = join(process.cwd(), ".bun-out");
  const binaryPath = join(bunOut, "alpha");
  const nativeSrc = join(process.cwd(), "packages", "helios", "native", "helios_vk.c");
  const envPath = join(process.cwd(), ".env.local");

  if (!existsSync(binaryPath)) {
    console.error(red(`  Binary not found at ${binaryPath}`));
    process.exit(1);
  }

  for (const n of names) {
    const inst = getInstance(config, n);
    const dir = config.deployDir;
    console.log(bold(`\n  Deploying to ${n}`) + dim(` (${inst.host})\n`));

    // Step 2: Create remote dir
    console.log("  Creating remote directory...");
    await ssh(config, inst.host, `mkdir -p ${dir}`);

    // Step 3: Upload binary
    console.log("  Uploading binary...");
    await scp(config, binaryPath, inst.host, `${dir}/alpha`);
    await ssh(config, inst.host, `chmod +x ${dir}/alpha`);

    // Step 4: Upload helios_vk.c and compile
    console.log("  Uploading helios_vk.c...");
    await scp(config, nativeSrc, inst.host, `${dir}/helios_vk.c`);

    console.log("  Compiling helios_vk.node on instance...");
    const nodeHeaders = "$(node -p 'require(\"path\").resolve(process.execPath, \"..\", \"..\", \"include\", \"node\")')";
    const gccCmd = `gcc -shared -fPIC -O2 -Wall -I${nodeHeaders} -o ${dir}/helios_vk.node ${dir}/helios_vk.c -ldl`;
    const { code: gccCode, stderr: gccErr } = await ssh(config, inst.host, gccCmd, { stream: true });
    if (gccCode !== 0) {
      console.error(red(`\n  gcc failed on ${n}. Run 'alpha fleet setup ${n}' first.`));
      continue;
    }

    // Step 5: Upload .env.local if exists
    if (existsSync(envPath)) {
      console.log("  Uploading .env.local...");
      await scp(config, envPath, inst.host, `${dir}/.env.local`);
    }

    // Step 6: Verify
    const { stdout } = await ssh(config, inst.host, `ls -la ${dir}/alpha ${dir}/helios_vk.node`);
    console.log(green("\n  Deploy successful:"));
    console.log(dim("  " + stdout.trim().split("\n").join("\n  ")));
  }
  console.log();
}

async function fleetSetup(config: FleetConfig, name: string): Promise<void> {
  const inst = getInstance(config, name);
  console.log(bold(`\n  Setting up ${name}`) + dim(` (${inst.host})\n`));

  const script = `
set -e

# System packages
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential gcc git curl wget unzip

# Vulkan runtime + ICD
sudo apt-get install -y -qq libvulkan1 vulkan-tools mesa-vulkan-drivers
if [ -d /usr/share/nvidia ]; then
  echo "NVIDIA drivers detected"
  sudo apt-get install -y -qq nvidia-vulkan-common 2>/dev/null || true
fi

# Node.js (v22 LTS)
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y -qq nodejs
fi
echo "node: $(node --version)"

# Xvfb for headless Vulkan
sudo apt-get install -y -qq xvfb
if ! pgrep -x Xvfb >/dev/null; then
  Xvfb :99 -screen 0 1024x768x24 &
  echo "Started Xvfb on :99"
fi

# Verify Vulkan
DISPLAY=:99 vulkaninfo --summary 2>/dev/null | head -20 || echo "vulkaninfo not available (may still work)"

echo ""
echo "Setup complete!"
`.trim();

  const { code } = await ssh(config, inst.host, script, { stream: true });
  if (code !== 0) {
    console.error(red("\n  Setup failed. Check output above."));
    process.exit(1);
  }

  // Mark setup done
  config.instances[name].setupDone = true;
  saveFleet(config);
  console.log(green(`\n  ${name} setup complete. setupDone=true saved.\n`));
}

async function fleetTrain(config: FleetConfig, name: string, trainArgs: string[]): Promise<void> {
  const inst = getInstance(config, name);
  const dir = config.deployDir;
  const kv = parseTrainKV(trainArgs);

  // Detect train command
  const prefix = await trainPrefix(config, inst.host);
  if (!prefix) {
    console.error(red(`  No alpha binary or repo found on ${name}. Run: alpha fleet deploy ${name}`));
    process.exit(1);
  }

  // Check for existing training
  if (kv["force"] !== "true") {
    const { stdout: procs } = await ssh(config, inst.host, `${TRAIN_PGREP} || true`);
    if (procs.trim() && procs.trim() !== "") {
      const lines = procs.trim().split("\n").filter(l => l.trim());
      if (lines.length > 0) {
        console.error(yellow(`  Training already running on ${name}:`));
        console.error(dim("  " + lines.join("\n  ")));
        console.error(yellow("  Use --force to start anyway, or 'alpha fleet stop' first."));
        process.exit(1);
      }
    }
  }

  // Ensure Xvfb running
  await ssh(config, inst.host, "pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1024x768x24 &>/dev/null &)");

  // Build the training command
  const flagStr = trainArgs.filter(a => a.startsWith("--") && !a.startsWith("--force")).join(" ");
  const trainCmd = `cd ${dir} && export DISPLAY=:99; source .env.local 2>/dev/null; nohup ${prefix} train ${flagStr} > train.log 2>&1 & echo $!`;

  console.log(bold(`\n  Starting training on ${name}`) + dim(` (${inst.host})\n`));
  console.log(dim(`  ${prefix} train ${flagStr}\n`));

  const { stdout, code } = await ssh(config, inst.host, trainCmd);
  const pid = stdout.trim().split("\n").pop()?.trim();

  if (code !== 0) {
    console.error(red("  Failed to start training."));
    process.exit(1);
  }

  console.log(green(`  Training started!`));
  console.log(`  PID: ${bold(pid || "?")}`);
  console.log(`  Log: ${dir}/train.log`);
  console.log(dim(`\n  View logs: alpha fleet logs ${name}`));
  console.log(dim(`  Stop:      alpha fleet stop ${name}\n`));
}

async function fleetResume(config: FleetConfig, name: string, kv: Record<string, string>, extraArgs: string[]): Promise<void> {
  const inst = getInstance(config, name);
  const dir = config.deployDir;

  // Find run directory
  let runDir: string;
  if (kv["run"]) {
    runDir = `${dir}/runs/${kv["run"]}`;
  } else {
    const { stdout, code } = await ssh(config, inst.host, `ls -dt ${dir}/runs/*/ 2>/dev/null | head -1`);
    if (code !== 0 || !stdout.trim()) {
      console.error(red(`  No runs found on ${name}. Start fresh with: alpha fleet train ${name} ...`));
      process.exit(1);
    }
    runDir = stdout.trim();
  }

  // Verify run dir exists
  const { code: dirCheck } = await ssh(config, inst.host, `test -d ${runDir}`);
  if (dirCheck !== 0) {
    console.error(red(`  Run directory not found: ${runDir}`));
    process.exit(1);
  }

  // Find latest checkpoint
  const { stdout: ckptOut } = await ssh(config, inst.host,
    `ls ${runDir}/checkpoint-*.json 2>/dev/null | sed 's/.*checkpoint-//' | sed 's/.json//' | sort -n | tail -1`
  );
  const ckptStep = ckptOut.trim();
  if (!ckptStep) {
    console.error(red(`  No checkpoints found in ${runDir}`));
    console.error(dim(`  Start fresh: alpha fleet train ${name} ...`));
    process.exit(1);
  }
  const ckptPath = `${runDir}/checkpoint-${ckptStep}.json`;

  // Read config.json for original training args
  const { stdout: configJson, code: cfgCode } = await ssh(config, inst.host, `cat ${runDir}/config.json 2>/dev/null`);
  let originalFlags = "";
  if (cfgCode === 0 && configJson.trim()) {
    try {
      const cfg = JSON.parse(configJson);
      // Reconstruct flags from saved config
      // config.json uses modelConfig/trainConfig (not model/train)
      const flagMap: Record<string, string> = {};
      const mc = cfg.modelConfig || cfg.model;
      if (mc) {
        if (mc.vocabSize) flagMap["vocabSize"] = mc.vocabSize;
        if (mc.blockSize) flagMap["block"] = mc.blockSize;
        if (mc.nLayer) flagMap["layers"] = mc.nLayer;
        if (mc.nEmbd) flagMap["dim"] = mc.nEmbd;
        if (mc.nHead) flagMap["heads"] = mc.nHead;
        if (mc.dropout !== undefined) flagMap["dropout"] = mc.dropout;
        if (mc.ffnActivation) flagMap["activation"] = mc.ffnActivation;
        if (mc.ffnDim) flagMap["ffnDim"] = mc.ffnDim;
      }
      const tc = cfg.trainConfig || cfg.train;
      if (tc) {
        if (tc.iters) flagMap["iters"] = tc.iters;
        if (tc.batchSize) flagMap["batch"] = tc.batchSize;
        if (tc.lr) flagMap["lr"] = tc.lr;
        if (tc.lrMin) flagMap["lrMin"] = tc.lrMin;
        if (tc.warmupIters) flagMap["warmupIters"] = tc.warmupIters;
        if (tc.gradClip) flagMap["gradClip"] = tc.gradClip;
        if (tc.evalInterval) flagMap["evalInterval"] = tc.evalInterval;
        if (tc.seed) flagMap["seed"] = tc.seed;
        if (tc.weightDecay) flagMap["weightDecay"] = tc.weightDecay;
        if (tc.optimizer) flagMap["optim"] = tc.optimizer;
        if (tc.gradAccumSteps) flagMap["accumSteps"] = tc.gradAccumSteps;
        if (tc.sampleInterval) flagMap["sampleInterval"] = tc.sampleInterval;
        if (tc.spikeThreshold) flagMap["spikeThreshold"] = tc.spikeThreshold;
        if (tc.backend) flagMap["backend"] = tc.backend;
        if (tc.tokenizer) flagMap["tokenizer"] = tc.tokenizer;
        if (tc.symbio) flagMap["symbio"] = "true";
      }
      if (cfg.data) flagMap["data"] = cfg.data;
      if (cfg.domain) flagMap["domain"] = cfg.domain;
      if (cfg.checkpoint) flagMap["checkpoint"] = "true";
      if (cfg.fp16) flagMap["fp16"] = "true";

      // User overrides from extraArgs take precedence
      const overrides = parseTrainKV(extraArgs);
      for (const [k, v] of Object.entries(overrides)) {
        if (k !== "run" && k !== "force") flagMap[k] = v;
      }

      originalFlags = Object.entries(flagMap).map(([k, v]) => `--${k}=${v}`).join(" ");
    } catch {
      console.log(yellow("  Could not parse config.json, using only provided flags."));
      originalFlags = extraArgs.filter(a => a.startsWith("--") && !a.startsWith("--run")).join(" ");
    }
  } else {
    originalFlags = extraArgs.filter(a => a.startsWith("--") && !a.startsWith("--run")).join(" ");
  }

  console.log(bold(`\n  Resuming training on ${name}`));
  console.log(`  Run:        ${dim(runDir)}`);
  console.log(`  Checkpoint: ${dim(`step ${ckptStep}`)}`);

  // Check for existing training
  if (kv["force"] !== "true") {
    const { stdout: procs } = await ssh(config, inst.host, `${TRAIN_PGREP} || true`);
    const lines = procs.trim().split("\n").filter(l => l.trim());
    if (lines.length > 0 && procs.trim()) {
      console.error(yellow("\n  Training already running. Use --force to start anyway."));
      process.exit(1);
    }
  }

  // Detect train command
  const prefix = await trainPrefix(config, inst.host);
  if (!prefix) {
    console.error(red(`  No alpha binary or repo found on ${name}. Run: alpha fleet deploy ${name}`));
    process.exit(1);
  }

  // Ensure Xvfb
  await ssh(config, inst.host, "pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1024x768x24 &>/dev/null &)");

  const trainCmd = `cd ${dir} && export DISPLAY=:99; source .env.local 2>/dev/null; nohup ${prefix} train --resume=${ckptPath} --runDir=${runDir} ${originalFlags} > train.log 2>&1 & echo $!`;

  console.log(dim(`\n  ${prefix} train --resume=${ckptPath} --runDir=${runDir} ${originalFlags}\n`));

  const { stdout, code } = await ssh(config, inst.host, trainCmd);
  const pid = stdout.trim().split("\n").pop()?.trim();

  if (code !== 0) {
    console.error(red("  Failed to start training."));
    process.exit(1);
  }

  console.log(green(`  Resumed!`));
  console.log(`  PID: ${bold(pid || "?")}`);
  console.log(dim(`\n  View logs: alpha fleet logs ${name}\n`));
}

async function fleetStop(config: FleetConfig, name: string): Promise<void> {
  const inst = getInstance(config, name);

  const { stdout: procs } = await ssh(config, inst.host, `${TRAIN_PGREP} || true`);
  const lines = procs.trim().split("\n").filter(l => l.trim());

  if (lines.length === 0 || !procs.trim()) {
    console.log(dim(`\n  No training running on ${name}.\n`));
    return;
  }

  console.log(bold(`\n  Stopping training on ${name}\n`));
  for (const line of lines) {
    // ps aux format: USER PID ...
    const pid = line.trim().split(/\s+/)[1];
    if (!pid || !/^\d+$/.test(pid)) continue;
    console.log(`  Sending SIGTERM to PID ${pid}...`);
    await ssh(config, inst.host, `kill -TERM ${pid} 2>/dev/null || true`);
  }

  // Wait a moment and check
  await new Promise(r => setTimeout(r, 2000));
  const { stdout: check } = await ssh(config, inst.host, `${TRAIN_PGREP} || true`);
  if (check.trim()) {
    console.log(yellow("  Process still running. May need a moment to save checkpoint."));
  } else {
    console.log(green("  Training stopped."));
  }
  console.log();
}

async function fleetLogs(config: FleetConfig, name: string, kv: Record<string, string>): Promise<void> {
  const inst = getInstance(config, name);
  const dir = config.deployDir;
  const follow = kv["f"] === "true" || kv["follow"] === "true";

  let logPath: string;
  if (kv["run"]) {
    logPath = `${dir}/runs/${kv["run"]}/train.log`;
  } else {
    // Find the most recently modified train.log (root or in runs/)
    const { stdout: latestLog } = await ssh(config, inst.host,
      `ls -t ${dir}/train.log ${dir}/runs/*/train.log 2>/dev/null | head -1`
    );
    logPath = latestLog.trim() || `${dir}/train.log`;
  }

  if (follow) {
    console.log(dim(`  Tailing ${logPath} on ${name} (Ctrl+C to stop)\n`));
    await ssh(config, inst.host, `tail -f ${logPath}`, { stream: true });
  } else {
    const lines = kv["n"] || "50";
    const { stdout, code } = await ssh(config, inst.host, `tail -${lines} ${logPath}`);
    if (code !== 0) {
      console.error(red(`  Log not found: ${logPath}`));
      process.exit(1);
    }
    console.log(stdout);
  }
}

async function fleetSsh(config: FleetConfig, name: string): Promise<void> {
  const inst = getInstance(config, name);
  console.log(dim(`  Connecting to ${name} (${inst.host})...\n`));
  await sshInteractive(config, inst.host);
}

async function fleetRun(config: FleetConfig, name: string, cmd: string[]): Promise<void> {
  const inst = getInstance(config, name);
  const cmdStr = cmd.join(" ");
  console.log(dim(`  ${name}> ${cmdStr}\n`));
  const { stdout, stderr, code } = await ssh(config, inst.host, cmdStr);
  if (stdout) process.stdout.write(stdout);
  if (stderr) process.stderr.write(stderr);
  if (code !== 0) process.exit(code);
}

async function fleetSyncKeys(config: FleetConfig, name: string): Promise<void> {
  const inst = getInstance(config, name);
  const pubKeyPath = expandPath("~/.ssh/id_ed25519.pub");

  if (!existsSync(pubKeyPath)) {
    console.error(red(`  Public key not found: ${pubKeyPath}`));
    process.exit(1);
  }

  const pubKey = readFileSync(pubKeyPath, "utf8").trim();
  console.log(bold(`\n  Syncing SSH key to ${name}\n`));

  await ssh(config, inst.host, `mkdir -p ~/.ssh && echo '${pubKey}' >> ~/.ssh/authorized_keys && sort -u -o ~/.ssh/authorized_keys ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`);
  console.log(green("  Key synced.\n"));
}

async function fleetDownload(config: FleetConfig, name: string, kv: Record<string, string>): Promise<void> {
  const inst = getInstance(config, name);
  const dir = config.deployDir;

  const run = kv["run"];
  if (!run) {
    console.error("Usage: alpha fleet download <name> --run=<run-name>");
    process.exit(1);
  }

  const remotePath = `${dir}/runs/${run}/`;
  const localPath = join(process.cwd(), "runs", `${name}-${run}`);

  // Verify run exists
  const { code } = await ssh(config, inst.host, `test -d ${remotePath}`);
  if (code !== 0) {
    console.error(red(`  Run not found: ${remotePath}`));
    const { stdout } = await ssh(config, inst.host, `ls ${dir}/runs/ 2>/dev/null`);
    if (stdout.trim()) {
      console.log(dim("  Available runs: " + stdout.trim().split("\n").join(", ")));
    }
    process.exit(1);
  }

  console.log(bold(`\n  Downloading ${run} from ${name}\n`));
  console.log(`  Remote: ${dim(remotePath)}`);
  console.log(`  Local:  ${dim(localPath)}\n`);

  execSync(`mkdir -p "${localPath}"`, { stdio: "ignore" });
  await rsyncFrom(config, inst.host, remotePath, localPath + "/");

  console.log(green(`\n  Download complete: ${localPath}\n`));
}

// ── Helpers ────────────────────────────────────────────────────────────────

function parseTrainKV(args: string[]): Record<string, string> {
  const kv: Record<string, string> = {};
  for (const arg of args) {
    if (arg.startsWith("--")) {
      const eq = arg.indexOf("=");
      if (eq > 0) {
        kv[arg.slice(2, eq)] = arg.slice(eq + 1);
      } else {
        kv[arg.slice(2)] = "true";
      }
    }
  }
  return kv;
}

// ── Entry point ────────────────────────────────────────────────────────────

const FLEET_USAGE = `
alpha fleet — manage remote training instances

Commands:
  fleet                        Dashboard: all instances + status
  fleet status [name]          GPU, processes, disk, latest run
  fleet deploy <name> [--all]  Build bun binary + ship + compile .node
  fleet setup <name>           Install node, gcc, vulkan on fresh instance
  fleet train <name> [--flags] Start detached training via nohup
  fleet resume <name> [--run=X]  Find latest checkpoint, resume training
  fleet stop <name>            Kill training process
  fleet logs <name> [-f]       Tail training log
  fleet ssh <name>             Interactive SSH
  fleet run <name> -- <cmd>    Run arbitrary command
  fleet sync-keys <name>       Copy SSH pubkey to instance
  fleet download <name> --run=X  Download run directory locally
`.trim();

export async function fleetCmd(args: string[]): Promise<void> {
  const config = loadFleet();
  const sub = args[0];

  if (!sub || sub === "--help" || sub === "-h") {
    // No subcommand = dashboard (unless --help)
    if (!sub) {
      await fleetDashboard(config);
    } else {
      console.log(FLEET_USAGE);
    }
    return;
  }

  const name = args[1] || "";
  const rest = args.slice(2);
  const kv = parseTrainKV(rest);

  switch (sub) {
    case "status":
      if (!name) {
        await fleetDashboard(config);
      } else {
        await fleetStatus(config, name);
      }
      break;

    case "deploy":
      await fleetDeploy(config, name, { ...kv, ...parseTrainKV([args[1] || ""].filter(a => a.startsWith("--"))) });
      break;

    case "setup":
      if (!name) { console.error("Usage: alpha fleet setup <name>"); process.exit(1); }
      await fleetSetup(config, name);
      break;

    case "train":
      if (!name) { console.error("Usage: alpha fleet train <name> [--flags]"); process.exit(1); }
      await fleetTrain(config, name, rest);
      break;

    case "resume":
      if (!name) { console.error("Usage: alpha fleet resume <name> [--run=X]"); process.exit(1); }
      await fleetResume(config, name, kv, rest);
      break;

    case "stop":
      if (!name) { console.error("Usage: alpha fleet stop <name>"); process.exit(1); }
      await fleetStop(config, name);
      break;

    case "logs":
      if (!name) { console.error("Usage: alpha fleet logs <name> [-f]"); process.exit(1); }
      // Parse -f from args (not --f)
      const logsKV = { ...kv };
      if (rest.includes("-f")) logsKV["f"] = "true";
      await fleetLogs(config, name, logsKV);
      break;

    case "ssh":
      if (!name) { console.error("Usage: alpha fleet ssh <name>"); process.exit(1); }
      await fleetSsh(config, name);
      break;

    case "run": {
      if (!name) { console.error("Usage: alpha fleet run <name> -- <cmd>"); process.exit(1); }
      // Everything after "--" is the command
      const dashIdx = args.indexOf("--");
      const cmd = dashIdx >= 0 ? args.slice(dashIdx + 1) : rest;
      if (cmd.length === 0) { console.error("Usage: alpha fleet run <name> -- <cmd>"); process.exit(1); }
      await fleetRun(config, name, cmd);
      break;
    }

    case "sync-keys":
      if (!name) { console.error("Usage: alpha fleet sync-keys <name>"); process.exit(1); }
      await fleetSyncKeys(config, name);
      break;

    case "download":
      if (!name) { console.error("Usage: alpha fleet download <name> --run=X"); process.exit(1); }
      await fleetDownload(config, name, kv);
      break;

    default:
      console.error(`Unknown fleet command: ${sub}`);
      console.log(FLEET_USAGE);
      process.exit(1);
  }
}
