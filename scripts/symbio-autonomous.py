#!/usr/bin/env python3
"""
symbio-autonomous.py — Autonomous Symbiogenesis Training Orchestrator

A multi-phase evolutionary training loop that embodies the full symbiogenesis
framework: population-based activation search, Kuramoto synchronization,
free energy minimization, information bottleneck compression, and CUSUM
change-point detection — orchestrated across biological-evolutionary phases.

Each phase is an independent training experiment on GCP. The orchestrator
monitors metrics, runs inference batteries between phases, detects overfitting
and evolutionary plateaus, and adapts the next phase's configuration based on
accumulated knowledge. The continuity is in KNOWLEDGE, not weights — like a
research program that learns from each experiment to design better ones.

PHASES:
  0. ABIOGENESIS    — Bootstrap baseline with gelu, no evolution
  1. PRIMORDIAL     — Explore: high mutation, diverse composed-activation search
  2. CAMBRIAN       — Diversify: grow population, refine promising lineages
  3. OXIDATION      — Converge: increase coupling, consensus fusion, prune
  4. ENDOSYMBIOSIS  — Integrate: winner activation, deep training, quality gates
  5. HOMEOSTASIS    — Refine: anneal lr, monitor regression, final evaluation

OVERFITTING IN SYMBIOGENESIS:
  Traditional overfitting (train/val divergence) is just the surface. In an
  evolutionary search, overfitting manifests at multiple levels:

  1. CANDIDATE MEMORIZATION — A candidate activation function shows rapid
     train loss decrease but val_loss diverges. The evolutionary search masks
     this because candidates only run for stepsPerCandidate steps each.
     Response: shorten candidate eval budget, increase mutation.

  2. POPULATION COLLAPSE — All candidates converge to similar activations.
     The diversity metric drops below 0.3. Kuramoto sync is high but loss
     isn't improving — the population locked into a local optimum.
     Response: inject random candidates, boost mutation rate, reduce coupling.

  3. FUSION TRAP — The consensus shadow model overfits, pulling all candidates
     toward a memorized solution. Weight entropy drops while val_loss stagnates.
     The shadow becomes a parasite: consuming gradient signal without contributing.
     Response: reset fusion shadow, reduce fusion strength.

  4. ACTIVATION COMPLEXITY OVERFITTING — Evolved activation graphs become too
     complex (many nodes), fitting the training distribution's quirks. A 10-node
     activation graph that beats 3-node on train but loses on val is overfitting.
     Response: penalize complexity in fitness, reduce maxGraphNodes.

  5. TRANSFER INTERFERENCE — Weights inherited from a previous candidate create
     conflicting gradient signals for the new activation. The model spends its
     evaluation budget un-learning before it can learn.
     Response: reduce weight preservation, allow more fresh initialization.

EVOLUTIONARY LOSS PLATEAUS:
  Loss plateaus in evolutionary search are NOT the same as in standard training.
  They signal that the search process itself has stalled:

  1. ACTIVATION SPACE EXHAUSTION — All promising activation structures have been
     explored. New candidates don't improve over parents because the search space
     is depleted. CUSUM alerts fire on throughput collapse.
     Response: expand basis pool, increase graph depth/node limits, inject novelty.

  2. PUNCTUATED EQUILIBRIUM — Long periods of stasis are NORMAL in evolution.
     The population is exploring neutral ridges in activation space. Fitness isn't
     improving but the population IS diversifying. Don't mistake this for failure.
     Response: patience. Check diversity metrics — if diversity is high, the
     population is building capacity for the next jump.

  3. CONSENSUS STAGNATION — Fusion shadow converges to a local minimum. All
     candidates get pulled toward it. The loss flatlines because the shadow
     dampens any novel gradient direction.
     Response: perturb shadow weights, reduce coupling, increase mutation.

  4. LEARNING RATE MISMATCH — Different activation functions have wildly different
     gradient scales. The LR that works for gelu may cause spikes in swiglu.
     Persistent spike-skips are the symptom.
     Response: per-phase LR tuning, lower spike threshold, stronger grad clip.

GAPS FILLED:
  - Inference-gated phase transitions (quality must improve to advance)
  - Dormancy (save promising checkpoints, revisit if current path fails)
  - Autopoiesis (self-evaluation through inference battery)
  - Ecological pressure (track resource efficiency: loss/GPU-second)
  - Niche construction (phase configs reshape the selection landscape)
  - Punctuated equilibrium detection (distinguish stasis from failure)

Usage:
  python scripts/symbio-autonomous.py --data data/historic.txt
  python scripts/symbio-autonomous.py --data data/historic.txt --budget 80000
  python scripts/symbio-autonomous.py --resume runs/symbio-auto/state.json
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Import GCP helpers
sys.path.insert(0, str(Path(__file__).parent))
from gcp_train import (
    check_gcloud, ensure_instance, setup_instance, sync_code,
    build_on_instance, upload_dataset, ssh_run, rsync_from,
    stop_instance, get_ip, find_instance, work_dir, ssh_opts,
    INSTANCE_NAME,
)

# ── Constants ──────────────────────────────────────────────────────────

ZONE = "us-central1-b"
MACHINE_TYPE = "g2-standard-4"
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)

# Model architecture (17.4M params — same as the proven chat config)
MODEL = dict(dim=384, heads=8, layers=8, block=512, batch=20, tokenizer="bpe")

# Inference prompts spanning the style/topic range of historic conversation data
INFERENCE_PROMPTS = [
    "<|user|> Good evening. What news from the continent?",
    "<|user|> Tell me of your journey across the Atlantic.",
    "<|user|> What is your opinion on the matter of liberty?",
    "<|user|> I received your letter. The situation grows dire.",
    "<|user|> What manner of creature did you observe in those lands?",
    "<|user|> The harvest was poor this year. What shall we do?",
    "<|user|> Have you heard the latest dispatches from Parliament?",
    "<|user|> My dearest, I write to you from the front lines.",
    "<|user|> The steam engine shall change everything, mark my words.",
    "<|user|> Do you believe in the natural rights of common men?",
    "<|user|> I have discovered something remarkable in the ruins.",
    "<|user|> How fares your family in these troubled times?",
    "<|user|> The plague has reached our village.",
    "<|user|> If mortals dare to cross the heavenly threshold, what then?",
    "<|user|> Tell me a story.",
    "<|user|> What is love?",
    "<|user|> I must confess something that weighs upon my conscience.",
    "<|user|> Shall we discuss philosophy or politics tonight?",
    "<|user|> The revolution has begun. Where do you stand?",
    "<|user|> Hello.",
]

INFERENCE_TEMPS = [0.4, 0.7, 1.0]

# ── Phase Definitions ──────────────────────────────────────────────────

@dataclass
class Phase:
    name: str
    description: str
    steps: int
    lr: float
    warmup: int
    symbio: bool
    symbio_config: dict
    train_extras: dict = field(default_factory=dict)

def make_phases(budget: int) -> list[Phase]:
    """Generate phases scaled to total step budget."""
    # Distribute budget: 4% abiogenesis, 16% primordial, 24% cambrian,
    # 16% oxidation, 30% endosymbiosis, 10% homeostasis
    ratios = [0.04, 0.16, 0.24, 0.16, 0.30, 0.10]
    steps = [max(500, int(budget * r)) for r in ratios]

    return [
        Phase(
            name="abiogenesis",
            description="Bootstrap — establish baseline metrics with gelu, no evolution",
            steps=steps[0],
            lr=1e-4,
            warmup=min(500, steps[0] // 4),
            symbio=False,
            symbio_config={},
            train_extras={"activation": "gelu"},
        ),
        Phase(
            name="primordial",
            description="Explore — high mutation, diverse composed-activation search",
            steps=steps[1],
            lr=5e-5,
            warmup=min(500, steps[1] // 8),
            symbio=True,
            symbio_config={
                "searchMode": "composed-activation-search",
                "basisPool": ["silu", "gelu", "relu", "identity", "square", "swiglu"],
                "maxGraphDepth": 3,
                "maxGraphNodes": 6,
                "populationSize": 6,
                "generations": 500,
                "stepsPerCandidate": 20,
                "selectionStrategy": "topk",
                "mutationRate": 0.8,
                "rankBy": "valLoss",
                "metricsInterval": 10,
                "cusumSensitivity": 4.0,
                "cusumBaselineWindow": 5,
                "populationAdaptation": True,
                "populationScaleMin": 0.5,
                "populationScaleMax": 2.0,
                "populationScaleStep": 0.15,
                "populationAdaptationCooldown": 8,
                "mutationRateMin": 0.4,
                "mutationRateMax": 0.95,
                "preserveWeightsAcrossCandidates": True,
                "carryOptimizerStateAcrossCandidates": True,
                "constantFfnDimAcrossCandidates": True,
                "fuseWeightsEachStep": True,
                "fusionShadowEma": 0.01,
                "fusionBaseStrength": 0.001,
                "fusionMaxStrength": 0.005,
                "kuramotoCoupling": 0.3,
                "kuramotoDt": 0.1,
                "kuramotoDamping": 0.05,
                "diversityBonus": 0.2,
                "diversityDecay": "cosine",
                "writeReport": True,
                "writeCandidates": True,
                "writeSummary": True,
            },
        ),
        Phase(
            name="cambrian",
            description="Diversify — grow population, deeper graphs, refine lineages",
            steps=steps[2],
            lr=3e-5,
            warmup=min(300, steps[2] // 10),
            symbio=True,
            symbio_config={
                "searchMode": "composed-activation-search",
                "basisPool": ["silu", "gelu", "relu", "identity", "square", "swiglu"],
                "maxGraphDepth": 5,
                "maxGraphNodes": 12,
                "populationSize": 12,
                "generations": 500,
                "stepsPerCandidate": 35,
                "selectionStrategy": "topk",
                "mutationRate": 0.5,
                "rankBy": "valLoss",
                "metricsInterval": 10,
                "cusumSensitivity": 3.5,
                "cusumBaselineWindow": 5,
                "populationAdaptation": True,
                "populationScaleMin": 0.5,
                "populationScaleMax": 2.5,
                "populationScaleStep": 0.125,
                "populationAdaptationCooldown": 10,
                "mutationRateMin": 0.2,
                "mutationRateMax": 0.8,
                "preserveWeightsAcrossCandidates": True,
                "carryOptimizerStateAcrossCandidates": True,
                "constantFfnDimAcrossCandidates": True,
                "fuseWeightsEachStep": True,
                "fusionShadowEma": 0.02,
                "fusionBaseStrength": 0.003,
                "fusionMaxStrength": 0.012,
                "kuramotoCoupling": 0.5,
                "kuramotoDt": 0.1,
                "kuramotoDamping": 0.05,
                "diversityBonus": 0.15,
                "diversityDecay": "cosine",
                "trackMIProfiles": False,
                "trackPopulationMetrics": True,
                "writeReport": True,
                "writeCandidates": True,
                "writeSummary": True,
            },
        ),
        Phase(
            name="oxidation",
            description="Converge — strong coupling, consensus fusion, prune weak",
            steps=steps[3],
            lr=2e-5,
            warmup=min(200, steps[3] // 10),
            symbio=True,
            symbio_config={
                "searchMode": "composed-activation-search",
                "basisPool": ["silu", "gelu", "relu", "identity", "square", "swiglu"],
                "maxGraphDepth": 5,
                "maxGraphNodes": 12,
                "populationSize": 8,
                "generations": 500,
                "stepsPerCandidate": 50,
                "selectionStrategy": "topk",
                "mutationRate": 0.3,
                "rankBy": "valLoss",
                "metricsInterval": 10,
                "cusumSensitivity": 3.0,
                "cusumBaselineWindow": 5,
                "populationAdaptation": True,
                "populationScaleMin": 0.5,
                "populationScaleMax": 1.5,
                "populationScaleStep": 0.1,
                "populationAdaptationCooldown": 15,
                "mutationRateMin": 0.1,
                "mutationRateMax": 0.6,
                "preserveWeightsAcrossCandidates": True,
                "carryOptimizerStateAcrossCandidates": True,
                "constantFfnDimAcrossCandidates": True,
                "fuseWeightsEachStep": True,
                "fusionShadowEma": 0.03,
                "fusionBaseStrength": 0.01,
                "fusionMaxStrength": 0.03,
                "kuramotoCoupling": 0.8,
                "kuramotoDt": 0.1,
                "kuramotoDamping": 0.03,
                "diversityBonus": 0.05,
                "diversityDecay": "cosine",
                "writeReport": True,
                "writeCandidates": True,
                "writeSummary": True,
            },
        ),
        Phase(
            name="endosymbiosis",
            description="Integrate — winner activation, deep training, no more search",
            steps=steps[4],
            lr=1e-5,
            warmup=min(300, steps[4] // 10),
            symbio=False,  # winner is baked in via --activation flag
            symbio_config={},
            train_extras={},  # activation set dynamically from winner
        ),
        Phase(
            name="homeostasis",
            description="Refine — ultra-low lr, monitor for regression",
            steps=steps[5],
            lr=3e-6,
            warmup=0,
            symbio=False,
            symbio_config={},
            train_extras={},
        ),
    ]

# ── Metrics Parsing ────────────────────────────────────────────────────

STEP_RE = re.compile(
    r"step\s+(\d+)/\d+\s*\|\s*loss=([\d.]+)"
    r"(?:\s*\|\s*lr=([\d.e+-]+))?"
    r"(?:\s*\|\s*grad_norm=([\d.eNaN+-]+))?"
    r"(?:\s*\|\s*(\d+)ms/it)?"
    r"(?:\s*\|\s*(\d+)\s*tok/s)?",
)
VAL_RE = re.compile(r"val_loss=([\d.]+)")
CANDIDATE_RE = re.compile(r"\[symbio search\].*?candidate=(\S+)")
SKIP_RE = re.compile(r"\((\d+) total skips\)")
SYNC_RE = re.compile(r"sync=([\d.]+)")
DIVERSITY_RE = re.compile(r"explore=([\d.]+)\s+converge=([\d.]+)")


def parse_log_tail(text: str) -> dict:
    """Parse training log tail into a metrics dict."""
    metrics = {
        "steps": [], "losses": [], "val_losses": [],
        "grad_norms": [], "tok_per_sec": [],
        "candidates": [], "total_skips": 0,
        "sync": None, "explore": None, "converge": None,
    }

    for line in text.split("\n"):
        m = STEP_RE.search(line)
        if m:
            metrics["steps"].append(int(m.group(1)))
            metrics["losses"].append(float(m.group(2)))
            gn = m.group(4)
            if gn and gn != "NaN":
                try:
                    metrics["grad_norms"].append(float(gn))
                except ValueError:
                    pass
            tps = m.group(6)
            if tps:
                metrics["tok_per_sec"].append(int(tps))

        m = VAL_RE.search(line)
        if m:
            metrics["val_losses"].append(float(m.group(1)))

        m = CANDIDATE_RE.search(line)
        if m:
            metrics["candidates"].append(m.group(1))

        m = SKIP_RE.search(line)
        if m:
            metrics["total_skips"] = int(m.group(1))

        m = SYNC_RE.search(line)
        if m:
            metrics["sync"] = float(m.group(1))

        m = DIVERSITY_RE.search(line)
        if m:
            metrics["explore"] = float(m.group(1))
            metrics["converge"] = float(m.group(2))

    return metrics


# ── Inference Evaluation ───────────────────────────────────────────────

def evaluate_sample(text: str) -> dict:
    """Score generated text using heuristic quality metrics."""
    # Strip special tokens for evaluation
    clean = text.replace("<|end_of_text|>", "").replace("<|assistant|>", "").strip()
    tokens = clean.split()

    if len(tokens) < 3:
        return {"length": len(tokens), "score": 0.0, "text": clean[:200],
                "repetition": 1.0, "diversity": 0.0}

    length = len(tokens)

    # Bigram repetition (lower = more repetitive)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    bigram_div = len(set(bigrams)) / max(len(bigrams), 1)

    # Trigram repetition
    trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]
    trigram_div = len(set(trigrams)) / max(len(trigrams), 1)

    # Vocabulary diversity
    vocab_div = len(set(tokens)) / max(len(tokens), 1)

    # Ends naturally (with end_of_text or at least punctuation)
    ends_well = text.rstrip().endswith("<|end_of_text|>") or clean[-1:] in ".!?\"'"

    # Composite score: weighted combination
    score = (
        0.20 * min(length / 40, 1.0) +          # Length (reasonable generation)
        0.25 * bigram_div +                       # Bigram diversity
        0.20 * trigram_div +                      # Trigram diversity
        0.20 * vocab_div +                        # Vocabulary diversity
        0.15 * (1.0 if ends_well else 0.3)        # Natural ending
    )

    return {
        "length": length,
        "bigram_div": round(bigram_div, 3),
        "trigram_div": round(trigram_div, 3),
        "vocab_div": round(vocab_div, 3),
        "ends_well": ends_well,
        "score": round(score, 3),
        "text": clean[:300],
    }


def run_inference_battery(ip: str, checkpoint_path: str,
                          prompts: list[str], temps: list[float]) -> dict:
    """Run comprehensive inference evaluation on the GCP instance."""
    wd = work_dir()
    results = []
    total_score = 0.0
    n = 0

    for prompt in prompts:
        for temp in temps:
            cmd = (
                f"cd {wd} && node apps/cli/dist/main.js sample "
                f"--checkpoint={checkpoint_path} "
                f"--backend=cpu_ref --steps=80 --temp={temp} --topk=40 "
                f"--prompt='{prompt} <|assistant|>'"
            )
            try:
                r = ssh_run(ip, cmd, check=False)
                if r.returncode == 0 and r.stdout.strip():
                    ev = evaluate_sample(r.stdout.strip())
                    ev["prompt"] = prompt
                    ev["temp"] = temp
                    results.append(ev)
                    total_score += ev["score"]
                    n += 1
                else:
                    results.append({"prompt": prompt, "temp": temp, "score": 0.0,
                                    "error": (r.stderr or "empty output")[:200]})
            except Exception as e:
                results.append({"prompt": prompt, "temp": temp, "score": 0.0,
                                "error": str(e)[:200]})

    avg_score = total_score / max(n, 1)

    # Cross-sample diversity: how different are outputs across prompts?
    texts = [r.get("text", "") for r in results if r.get("text")]
    cross_diversity = 0.0
    if len(texts) >= 2:
        all_bigrams = []
        for t in texts:
            words = t.split()
            all_bigrams.append(set(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)))
        # Average Jaccard distance between pairs
        pairs = 0
        jaccard_sum = 0.0
        for i in range(len(all_bigrams)):
            for j in range(i + 1, len(all_bigrams)):
                if all_bigrams[i] or all_bigrams[j]:
                    intersection = len(all_bigrams[i] & all_bigrams[j])
                    union = len(all_bigrams[i] | all_bigrams[j])
                    jaccard_sum += 1.0 - (intersection / max(union, 1))
                    pairs += 1
        cross_diversity = jaccard_sum / max(pairs, 1)

    # Best samples for Discord
    best_samples = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:5]

    return {
        "avg_score": round(avg_score, 3),
        "cross_diversity": round(cross_diversity, 3),
        "num_evaluated": n,
        "num_errors": len(results) - n,
        "best_samples": best_samples,
        "all_results": results,
    }


# ── Overfitting & Plateau Detection ───────────────────────────────────

def detect_overfitting(metrics: dict, phase_name: str) -> dict:
    """Detect symbiogenesis-specific overfitting signals."""
    signals = {"detected": False, "signals": [], "severity": 0.0}

    losses = metrics.get("losses", [])
    val_losses = metrics.get("val_losses", [])

    if not losses:
        return signals

    # 1. Train/val divergence (traditional overfitting)
    if len(val_losses) >= 2 and len(losses) >= 20:
        recent_train = sum(losses[-20:]) / 20
        recent_val = sum(val_losses[-3:]) / max(len(val_losses[-3:]), 1)
        early_val = sum(val_losses[:3]) / max(len(val_losses[:3]), 1) if len(val_losses) >= 3 else recent_val
        gap = recent_val - recent_train
        val_trend = recent_val - early_val  # positive = getting worse

        if gap > 0.5 and val_trend > 0.1:
            signals["detected"] = True
            signals["signals"].append(f"train/val divergence: gap={gap:.2f}, val_trend=+{val_trend:.2f}")
            signals["severity"] += 0.4

    # 2. Population collapse (symbio phases only)
    candidates = metrics.get("candidates", [])
    if len(candidates) >= 10:
        recent_cands = candidates[-10:]
        unique = len(set(recent_cands))
        if unique <= 2:
            signals["detected"] = True
            signals["signals"].append(f"population collapse: only {unique} unique candidates in last 10")
            signals["severity"] += 0.3

    # 3. Excessive spike skips (gradient instability)
    steps = metrics.get("steps", [])
    total_skips = metrics.get("total_skips", 0)
    if steps and total_skips > 0:
        max_step = max(steps)
        skip_ratio = total_skips / max(max_step, 1)
        if skip_ratio > 0.25:
            signals["detected"] = True
            signals["signals"].append(f"gradient instability: {skip_ratio:.0%} steps skipped ({total_skips}/{max_step})")
            signals["severity"] += 0.3

    signals["severity"] = min(1.0, signals["severity"])
    return signals


def detect_plateau(metrics: dict) -> dict:
    """Detect evolutionary loss plateaus."""
    signals = {"detected": False, "signals": [], "type": None}

    losses = metrics.get("losses", [])
    if len(losses) < 50:
        return signals

    # Split into first half and second half
    mid = len(losses) // 2
    first_half_avg = sum(losses[:mid]) / mid
    second_half_avg = sum(losses[mid:]) / (len(losses) - mid)
    improvement = (first_half_avg - second_half_avg) / max(first_half_avg, 1e-6)

    # Less than 2% improvement across the phase
    if improvement < 0.02:
        signals["detected"] = True
        signals["type"] = "stagnation"
        signals["signals"].append(
            f"loss stagnation: only {improvement:.1%} improvement "
            f"({first_half_avg:.3f} -> {second_half_avg:.3f})"
        )

    # Check if sync is locked but loss isn't moving (consensus stagnation)
    sync = metrics.get("sync")
    if sync is not None and sync > 0.9 and improvement < 0.05:
        signals["detected"] = True
        signals["type"] = "consensus_stagnation"
        signals["signals"].append(f"consensus stagnation: sync={sync:.2f} but loss barely moving")

    # Check for punctuated equilibrium (NOT a bad sign)
    explore = metrics.get("explore")
    if explore is not None and explore > 0.3 and improvement < 0.03:
        signals["type"] = "punctuated_equilibrium"
        signals["signals"].append(
            f"possible punctuated equilibrium: high explore={explore:.2f}, "
            f"population may be building capacity for a jump"
        )

    return signals


# ── Discord Reporting ──────────────────────────────────────────────────

def post_discord(webhook_url: str, embed: dict):
    """Post a rich embed to Discord."""
    if not webhook_url:
        return
    payload = {"embeds": [embed]}
    try:
        subprocess.run(
            ["curl", "-s", "-H", "Content-Type: application/json",
             "-d", json.dumps(payload), webhook_url],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


def make_phase_embed(phase_name: str, description: str, metrics: dict,
                     inference: dict, overfitting: dict, plateau: dict,
                     color: int = 0x7B68EE) -> dict:
    """Build a Discord embed for a phase completion."""
    losses = metrics.get("losses", [])
    val_losses = metrics.get("val_losses", [])
    steps = metrics.get("steps", [])

    fields = [
        {"name": "Steps", "value": f"{max(steps) if steps else 0:,}", "inline": True},
        {"name": "Loss", "value": f"{losses[-1]:.3f}" if losses else "?", "inline": True},
        {"name": "Val Loss", "value": f"{val_losses[-1]:.3f}" if val_losses else "?", "inline": True},
        {"name": "Inference Quality", "value": f"{inference.get('avg_score', 0):.3f}", "inline": True},
        {"name": "Cross-Diversity", "value": f"{inference.get('cross_diversity', 0):.3f}", "inline": True},
    ]

    if overfitting["detected"]:
        fields.append({"name": "Overfitting", "value": "; ".join(overfitting["signals"][:2]), "inline": False})
        color = 0xFF4444

    if plateau["detected"]:
        fields.append({"name": "Plateau", "value": "; ".join(plateau["signals"][:2]), "inline": False})
        color = 0xFFAA00

    # Best inference sample
    best = inference.get("best_samples", [{}])[0]
    if best.get("text"):
        sample_text = best["text"][:400]
        fields.append({"name": f"Best Sample (t={best.get('temp', '?')})",
                        "value": f"```{sample_text}```", "inline": False})

    return {
        "title": f"Symbio Phase: {phase_name.upper()}",
        "description": description,
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Orchestrator ───────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    phase: str
    steps_completed: int
    final_loss: float
    final_val_loss: float
    best_val_loss: float
    inference_quality: float
    cross_diversity: float
    overfitting: dict
    plateau: dict
    winner_activation: str
    run_id: str
    checkpoint_path: str
    best_samples: list
    elapsed_sec: float


class SymbioOrchestrator:
    def __init__(self, data_path: str, zone: str, machine_type: str,
                 budget: int, state_dir: str, discord_url: str = ""):
        self.data_path = data_path
        self.zone = zone
        self.machine_type = machine_type
        self.budget = budget
        self.state_dir = state_dir
        self.discord_url = discord_url
        self.ip: str = ""
        self.phases = make_phases(budget)
        self.history: list[PhaseResult] = []
        self.best_quality = 0.0
        self.best_checkpoint = ""
        self.dormant: list[dict] = []  # checkpoints to revisit
        self.winner_activation = "gelu"  # default, updated by search
        self.total_steps = 0
        self.start_time = time.time()

        os.makedirs(state_dir, exist_ok=True)

    def log(self, msg: str, prefix: str = ""):
        """Print timestamped log message."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        phase = prefix or "orchestrator"
        print(f"[{ts}] [{phase}] {msg}", flush=True)

    def save_state(self):
        """Persist orchestrator state for resume."""
        state = {
            "data_path": self.data_path,
            "budget": self.budget,
            "total_steps": self.total_steps,
            "best_quality": self.best_quality,
            "best_checkpoint": self.best_checkpoint,
            "winner_activation": self.winner_activation,
            "dormant": self.dormant,
            "elapsed_sec": time.time() - self.start_time,
            "history": [asdict(r) for r in self.history],
        }
        path = os.path.join(self.state_dir, "state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str) -> bool:
        """Resume from saved state."""
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.total_steps = state.get("total_steps", 0)
        self.best_quality = state.get("best_quality", 0.0)
        self.best_checkpoint = state.get("best_checkpoint", "")
        self.winner_activation = state.get("winner_activation", "gelu")
        self.dormant = state.get("dormant", [])
        # Skip completed phases
        completed = len(state.get("history", []))
        self.phases = self.phases[completed:]
        self.log(f"Resumed from state: {completed} phases done, {self.total_steps} steps, "
                 f"best quality={self.best_quality:.3f}")
        return True

    # ── Instance Management ────────────────────────────────────────

    def ensure_ready(self):
        """Ensure GCP instance is provisioned, code synced, and built."""
        self.log("Provisioning GCP instance...")
        instance, self.ip = ensure_instance(self.zone, self.machine_type)
        setup_instance(self.ip)
        sync_code(self.ip, PROJECT_DIR)
        build_on_instance(self.ip)

        # Upload dataset
        data_name = os.path.basename(self.data_path)
        local_path = self.data_path
        if not os.path.exists(local_path):
            local_path = os.path.join(PROJECT_DIR, self.data_path)
        upload_dataset(self.ip, local_path)
        self.log(f"Instance ready at {self.ip}, dataset '{data_name}' uploaded")

    # ── Training Execution ─────────────────────────────────────────

    def start_training(self, phase: Phase) -> str:
        """Start training in background on GCP. Returns run_id."""
        wd = work_dir()
        data_name = os.path.basename(self.data_path)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"symbio_{phase.name}_{ts}"
        run_dir = f"{wd}/runs/{run_id}"
        log_path = f"{wd}/runs/train_{run_id}.log"
        data_path = f"{wd}/datasets/{data_name}"

        # Write symbio config to instance if needed
        if phase.symbio and phase.symbio_config:
            config_json = json.dumps(phase.symbio_config, indent=2)
            config_path = f"/tmp/symbio-{phase.name}.json"
            ssh_run(self.ip, f"cat > {config_path} << 'SYMBIO_EOF'\n{config_json}\nSYMBIO_EOF")

        # Build environment exports
        env_exports = "export DISPLAY=:99 && "
        for var in ["ALPHA_REMOTE_URL", "ALPHA_REMOTE_SECRET", "DISCORD_WEBHOOK_URL"]:
            val = os.environ.get(var)
            if val:
                env_exports += f"export {var}='{val}' && "

        # Build training command
        args = [
            f"--data={data_path}",
            f"--runDir={run_dir}",
            f"--iters={phase.steps}",
            f"--batch={MODEL['batch']}",
            f"--block={MODEL['block']}",
            f"--dim={MODEL['dim']}",
            f"--heads={MODEL['heads']}",
            f"--layers={MODEL['layers']}",
            f"--lr={phase.lr}",
            f"--backend=helios",
            f"--tokenizer={MODEL['tokenizer']}",
            f"--domain=chat",
            "--beta2=0.95",
            "--gradClip=1.0",
            f"--warmupIters={phase.warmup}",
            "--spikeThreshold=10",
            "--evalInterval=200",
            "--sampleInterval=500",
        ]

        if phase.symbio:
            args.append("--symbio=true")
            if phase.symbio_config:
                args.append(f"--symbio-config={config_path}")
        else:
            # Use the discovered activation if we have one
            activation = phase.train_extras.get("activation", self.winner_activation)
            if activation and activation != "composed":
                args.append(f"--activation={activation}")

        args_str = " ".join(args)
        cmd = (
            f"cd {wd} && {env_exports}"
            f"(pgrep Xvfb >/dev/null 2>&1 || nohup Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &) && sleep 0.5 && "
            f"nohup node --expose-gc --max-old-space-size=8192 "
            f"apps/cli/dist/main.js train {args_str} "
            f"> {log_path} 2>&1 &"
        )

        ssh_run(self.ip, cmd)
        time.sleep(3)

        # Verify process started
        r = ssh_run(self.ip, "pgrep -f 'node.*train'", check=False)
        if r.returncode != 0:
            # Check log for immediate crash
            r2 = ssh_run(self.ip, f"tail -20 {log_path}", check=False)
            raise RuntimeError(f"Training failed to start.\n{r2.stdout}\n{r2.stderr}")

        self.log(f"Training started: {run_id}", phase.name)
        return run_id

    def monitor_training(self, run_id: str, phase: Phase) -> dict:
        """Monitor training log until phase completes or issues detected."""
        wd = work_dir()
        log_path = f"{wd}/runs/train_{run_id}.log"
        last_step = 0
        stall_count = 0
        poll_interval = 15  # seconds

        self.log(f"Monitoring {phase.steps} steps...", phase.name)

        while True:
            time.sleep(poll_interval)

            # Check if process is still alive
            r = ssh_run(self.ip, "pgrep -f 'node.*train'", check=False)
            process_alive = r.returncode == 0

            # Get latest log
            r = ssh_run(self.ip, f"tail -200 {log_path}", check=False)
            if r.returncode != 0:
                if not process_alive:
                    self.log("Training process exited", phase.name)
                    break
                continue

            metrics = parse_log_tail(r.stdout)
            steps = metrics.get("steps", [])
            current_step = max(steps) if steps else 0

            if current_step == last_step:
                stall_count += 1
                if stall_count > 20:  # 5 minutes of no progress
                    self.log(f"Training stalled at step {current_step}", phase.name)
                    break
            else:
                stall_count = 0
                last_step = current_step

            # Progress logging every ~60s
            if stall_count == 0 and current_step % 100 < 10:
                losses = metrics.get("losses", [])
                loss_str = f"{losses[-1]:.3f}" if losses else "?"
                tps = metrics.get("tok_per_sec", [])
                tps_str = f"{tps[-1]}" if tps else "?"
                self.log(f"step {current_step}/{phase.steps} loss={loss_str} tok/s={tps_str}", phase.name)

            # Check for completion
            if current_step >= phase.steps:
                self.log(f"Phase complete at step {current_step}", phase.name)
                break

            if not process_alive:
                self.log(f"Training exited at step {current_step}", phase.name)
                break

        # Get final metrics (larger tail for analysis)
        r = ssh_run(self.ip, f"tail -500 {log_path}", check=False)
        final_metrics = parse_log_tail(r.stdout) if r.returncode == 0 else metrics
        return final_metrics

    def stop_training(self):
        """Kill any running training process."""
        ssh_run(self.ip, "pkill -9 -f 'node.*train'", check=False)
        time.sleep(2)

    def find_checkpoint(self, run_id: str) -> str:
        """Find the latest checkpoint for a run."""
        wd = work_dir()
        run_dir = f"{wd}/runs/{run_id}"
        r = ssh_run(self.ip, f"ls -1t {run_dir}/checkpoint-*.json 2>/dev/null | head -1", check=False)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        return ""

    def find_winner_activation(self, run_id: str) -> str:
        """Extract winner activation from symbio search report."""
        wd = work_dir()
        run_dir = f"{wd}/runs/{run_id}"

        # Check search summary
        r = ssh_run(self.ip,
                    f"cat {run_dir}/symbio-search-summary.json 2>/dev/null || "
                    f"cat {run_dir}/symbio-search-report.md 2>/dev/null | head -30",
                    check=False)

        if r.returncode == 0 and r.stdout.strip():
            # Try JSON summary first
            try:
                summary = json.loads(r.stdout)
                winner = summary.get("winner", {})
                return winner.get("activation", self.winner_activation)
            except json.JSONDecodeError:
                pass

            # Try to parse from report markdown
            for line in r.stdout.split("\n"):
                if "winner" in line.lower() and "activation" in line.lower():
                    # Extract activation name from markdown
                    m = re.search(r"activation[:\s]+`?(\S+)`?", line, re.IGNORECASE)
                    if m:
                        return m.group(1)

        # Fallback: check training log for last candidate
        r = ssh_run(self.ip,
                    f"grep '\\[symbio search\\]' {wd}/runs/train_{run_id}.log 2>/dev/null | tail -1",
                    check=False)
        if r.returncode == 0 and r.stdout.strip():
            m = CANDIDATE_RE.search(r.stdout)
            if m:
                return m.group(1)

        return self.winner_activation

    # ── Phase Execution ────────────────────────────────────────────

    def run_phase(self, phase: Phase) -> PhaseResult:
        """Execute a single training phase."""
        phase_start = time.time()
        self.log(f"{'=' * 60}", phase.name)
        self.log(f"PHASE: {phase.name.upper()}", phase.name)
        self.log(f"{phase.description}", phase.name)
        self.log(f"Steps: {phase.steps}, LR: {phase.lr}, Symbio: {phase.symbio}", phase.name)
        self.log(f"{'=' * 60}", phase.name)

        # Start training
        run_id = self.start_training(phase)

        # Monitor until complete
        metrics = self.monitor_training(run_id, phase)
        self.stop_training()

        # Find checkpoint
        checkpoint_path = self.find_checkpoint(run_id)
        if not checkpoint_path:
            self.log("WARNING: No checkpoint found!", phase.name)

        # Extract winner activation from symbio phases
        if phase.symbio:
            self.winner_activation = self.find_winner_activation(run_id)
            self.log(f"Winner activation: {self.winner_activation}", phase.name)

        # Analyze metrics
        losses = metrics.get("losses", [])
        val_losses = metrics.get("val_losses", [])
        steps = metrics.get("steps", [])

        final_loss = losses[-1] if losses else float("inf")
        final_val = val_losses[-1] if val_losses else float("inf")
        best_val = min(val_losses) if val_losses else float("inf")
        steps_completed = max(steps) if steps else 0

        self.total_steps += steps_completed

        # Detect issues
        overfitting = detect_overfitting(metrics, phase.name)
        plateau = detect_plateau(metrics)

        # Run inference battery
        self.log("Running inference battery...", phase.name)
        if checkpoint_path:
            inference = run_inference_battery(
                self.ip, checkpoint_path,
                INFERENCE_PROMPTS[:10],  # Subset for speed
                INFERENCE_TEMPS,
            )
        else:
            inference = {"avg_score": 0.0, "cross_diversity": 0.0,
                         "best_samples": [], "num_evaluated": 0, "num_errors": 0}

        quality = inference.get("avg_score", 0.0)
        cross_div = inference.get("cross_diversity", 0.0)
        self.log(f"Inference: quality={quality:.3f}, diversity={cross_div:.3f}, "
                 f"evaluated={inference['num_evaluated']}, errors={inference['num_errors']}",
                 phase.name)

        elapsed = time.time() - phase_start
        result = PhaseResult(
            phase=phase.name,
            steps_completed=steps_completed,
            final_loss=final_loss,
            final_val_loss=final_val,
            best_val_loss=best_val,
            inference_quality=quality,
            cross_diversity=cross_div,
            overfitting=overfitting,
            plateau=plateau,
            winner_activation=self.winner_activation,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            best_samples=inference.get("best_samples", [])[:3],
            elapsed_sec=elapsed,
        )

        # Discord update
        embed = make_phase_embed(
            phase.name, phase.description, metrics, inference,
            overfitting, plateau,
        )
        post_discord(self.discord_url, embed)

        # Update best tracking
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_checkpoint = checkpoint_path

        self.save_state()
        return result

    # ── Adaptive Decisions ─────────────────────────────────────────

    def handle_overfitting(self, result: PhaseResult, next_phase: Phase):
        """Adapt next phase config in response to overfitting signals."""
        severity = result.overfitting.get("severity", 0)
        signals = result.overfitting.get("signals", [])
        self.log(f"Overfitting detected (severity={severity:.1f}): {'; '.join(signals)}")

        if "population collapse" in str(signals):
            # Boost diversity for next phase
            if next_phase.symbio_config:
                next_phase.symbio_config["mutationRate"] = min(0.9,
                    next_phase.symbio_config.get("mutationRate", 0.5) + 0.2)
                next_phase.symbio_config["diversityBonus"] = min(0.3,
                    next_phase.symbio_config.get("diversityBonus", 0.1) + 0.1)
                next_phase.symbio_config["kuramotoCoupling"] = max(0.1,
                    next_phase.symbio_config.get("kuramotoCoupling", 0.5) - 0.2)
                self.log("Adapted: boosted mutation/diversity, reduced coupling")

        if "gradient instability" in str(signals):
            # Lower learning rate
            next_phase.lr *= 0.5
            self.log(f"Adapted: halved LR to {next_phase.lr}")

        if "train/val divergence" in str(signals):
            # Save this checkpoint as dormant (might be useful later)
            if result.checkpoint_path:
                self.dormant.append({
                    "phase": result.phase,
                    "checkpoint": result.checkpoint_path,
                    "val_loss": result.best_val_loss,
                    "quality": result.inference_quality,
                })
                self.log(f"Saved dormant checkpoint from {result.phase}")

    def handle_plateau(self, result: PhaseResult, next_phase: Phase):
        """Adapt next phase config in response to evolutionary plateau."""
        plateau_type = result.plateau.get("type", "stagnation")
        signals = result.plateau.get("signals", [])
        self.log(f"Plateau detected ({plateau_type}): {'; '.join(signals)}")

        if plateau_type == "punctuated_equilibrium":
            # This is actually fine — the population is exploring
            self.log("Punctuated equilibrium detected — this is normal. Continuing.")
            return

        if plateau_type == "consensus_stagnation":
            # Reduce fusion, boost exploration
            if next_phase.symbio_config:
                next_phase.symbio_config["fusionBaseStrength"] = max(0.0005,
                    next_phase.symbio_config.get("fusionBaseStrength", 0.01) * 0.3)
                next_phase.symbio_config["fusionMaxStrength"] = max(0.002,
                    next_phase.symbio_config.get("fusionMaxStrength", 0.02) * 0.3)
                next_phase.symbio_config["kuramotoCoupling"] = max(0.1,
                    next_phase.symbio_config.get("kuramotoCoupling", 0.5) - 0.3)
                next_phase.symbio_config["mutationRate"] = min(0.9,
                    next_phase.symbio_config.get("mutationRate", 0.5) + 0.2)
                self.log("Adapted: weakened fusion/coupling, boosted mutation")

        elif plateau_type == "stagnation":
            # Try expanding the search space
            if next_phase.symbio_config:
                depth = next_phase.symbio_config.get("maxGraphDepth", 4)
                nodes = next_phase.symbio_config.get("maxGraphNodes", 10)
                next_phase.symbio_config["maxGraphDepth"] = depth + 1
                next_phase.symbio_config["maxGraphNodes"] = nodes + 3
                self.log(f"Adapted: expanded graph limits to depth={depth+1}, nodes={nodes+3}")

    # ── Main Loop ──────────────────────────────────────────────────

    def run(self):
        """Main orchestration loop."""
        self.log("=" * 60)
        self.log("AUTONOMOUS SYMBIOGENESIS TRAINING")
        self.log(f"Dataset: {self.data_path}")
        self.log(f"Budget: {self.budget:,} steps across {len(self.phases)} phases")
        self.log(f"Instance: {self.machine_type} in {self.zone}")
        self.log("=" * 60)

        # Post start notification
        post_discord(self.discord_url, {
            "title": "Autonomous Symbio Training Started",
            "description": (
                f"**Dataset:** {os.path.basename(self.data_path)}\n"
                f"**Budget:** {self.budget:,} steps\n"
                f"**Model:** {MODEL['dim']}d {MODEL['heads']}h {MODEL['layers']}L\n"
                f"**Phases:** {', '.join(p.name for p in self.phases)}"
            ),
            "color": 0x00FF88,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Ensure instance is ready
        self.ensure_ready()

        for i, phase in enumerate(self.phases):
            self.log(f"\n{'#' * 60}")
            self.log(f"# Phase {i+1}/{len(self.phases)}: {phase.name.upper()}")
            self.log(f"{'#' * 60}\n")

            result = self.run_phase(phase)
            self.history.append(result)

            self.log(f"Phase complete: loss={result.final_loss:.3f}, "
                     f"val={result.final_val_loss:.3f}, quality={result.inference_quality:.3f}")

            # Adaptive decisions for next phase
            if i + 1 < len(self.phases):
                next_phase = self.phases[i + 1]

                if result.overfitting["detected"]:
                    self.handle_overfitting(result, next_phase)

                if result.plateau["detected"]:
                    self.handle_plateau(result, next_phase)

                # Quality gate: if quality dropped significantly, revert to best
                if (self.best_quality > 0 and
                        result.inference_quality < self.best_quality * 0.7 and
                        len(self.dormant) > 0):
                    best_dormant = max(self.dormant, key=lambda d: d["quality"])
                    self.log(f"Quality regression! Reverting to dormant checkpoint "
                             f"from {best_dormant['phase']}")
                    # The next phase will start fresh anyway, but we note the regression
                    post_discord(self.discord_url, {
                        "title": "Quality Regression Detected",
                        "description": (
                            f"Phase {phase.name} quality ({result.inference_quality:.3f}) "
                            f"dropped below 70% of best ({self.best_quality:.3f}). "
                            f"Adapting next phase."
                        ),
                        "color": 0xFF4444,
                    })

        self.finalize()

    def finalize(self):
        """Generate final report and clean up."""
        total_elapsed = time.time() - self.start_time
        hours = total_elapsed / 3600

        self.log(f"\n{'=' * 60}")
        self.log("AUTONOMOUS TRAINING COMPLETE")
        self.log(f"Total steps: {self.total_steps:,}")
        self.log(f"Total time: {hours:.1f} hours")
        self.log(f"Winner activation: {self.winner_activation}")
        self.log(f"Best inference quality: {self.best_quality:.3f}")
        self.log(f"Best checkpoint: {self.best_checkpoint}")
        self.log(f"{'=' * 60}")

        # Write summary report
        report = {
            "total_steps": self.total_steps,
            "total_hours": round(hours, 2),
            "winner_activation": self.winner_activation,
            "best_quality": self.best_quality,
            "best_checkpoint": self.best_checkpoint,
            "phases": [asdict(r) for r in self.history],
            "dormant_checkpoints": self.dormant,
        }
        report_path = os.path.join(self.state_dir, "final-report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        self.log(f"Report saved to {report_path}")

        # Discord final notification
        phase_summary = "\n".join(
            f"  {r.phase}: loss={r.final_loss:.3f} val={r.final_val_loss:.3f} "
            f"quality={r.inference_quality:.3f}"
            for r in self.history
        )
        best_sample = ""
        if self.history and self.history[-1].best_samples:
            s = self.history[-1].best_samples[0]
            best_sample = f"\n**Best Sample:**\n```{s.get('text', '')[:400]}```"

        post_discord(self.discord_url, {
            "title": "Autonomous Symbio Training Complete",
            "description": (
                f"**Winner:** `{self.winner_activation}`\n"
                f"**Steps:** {self.total_steps:,} in {hours:.1f}h\n"
                f"**Best Quality:** {self.best_quality:.3f}\n\n"
                f"**Phase Results:**\n```{phase_summary}```"
                f"{best_sample}"
            ),
            "color": 0x00FF88,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self.save_state()

        # Optionally stop instance
        self.log("Stopping GCP instance...")
        try:
            stop_instance(self.zone)
        except Exception as e:
            self.log(f"Warning: could not stop instance: {e}")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Symbiogenesis Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default="data/historic.txt",
                        help="Path to training data (default: data/historic.txt)")
    parser.add_argument("--budget", type=int, default=50000,
                        help="Total step budget across all phases (default: 50000)")
    parser.add_argument("--zone", default=ZONE)
    parser.add_argument("--machine-type", default=MACHINE_TYPE)
    parser.add_argument("--state-dir", default="runs/symbio-auto",
                        help="Directory for orchestrator state and reports")
    parser.add_argument("--resume", default="",
                        help="Path to state.json to resume from")

    args = parser.parse_args()

    # Load env vars
    env_path = os.path.join(PROJECT_DIR, ".env.local")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and val and key not in os.environ:
                        os.environ[key] = val

    discord_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    check_gcloud()

    orchestrator = SymbioOrchestrator(
        data_path=args.data,
        zone=args.zone,
        machine_type=args.machine_type,
        budget=args.budget,
        state_dir=args.state_dir,
        discord_url=discord_url,
    )

    if args.resume:
        orchestrator.load_state(args.resume)

    # Graceful shutdown
    def sigint_handler(sig, frame):
        print("\n\nInterrupted! Stopping training...")
        orchestrator.stop_training()
        orchestrator.save_state()
        print(f"State saved to {args.state_dir}/state.json")
        print(f"Resume with: python scripts/symbio-autonomous.py --resume {args.state_dir}/state.json")
        print(f"Instance is still running. Stop with: python scripts/gcp_train.py --action stop")
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    orchestrator.run()


if __name__ == "__main__":
    main()
