# /symbio-train — Autonomous Symbiogenesis Training

Run `scripts/symbio-autonomous.py` to execute multi-phase evolutionary training
on GCP. This skill orchestrates the full symbiogenesis lifecycle: population-based
activation search, Kuramoto synchronization, free energy minimization, and
inference-gated phase transitions.

## Quick Start

```bash
# Default: historic.txt, 50k step budget
source .env.local
python scripts/symbio-autonomous.py --data data/historic.txt

# Larger budget
python scripts/symbio-autonomous.py --data data/historic.txt --budget 100000

# Resume interrupted run
python scripts/symbio-autonomous.py --resume runs/symbio-auto/state.json
```

## What It Does

The orchestrator runs 6 biological phases, each an independent training experiment:

| Phase | Steps | Purpose | Key Config |
|-------|-------|---------|------------|
| **Abiogenesis** | 4% | Baseline with gelu, no evolution | High LR, establish reference |
| **Primordial** | 16% | Explore: high mutation, wide search | Pop=6, mut=0.8, coupling=0.3 |
| **Cambrian** | 24% | Diversify: grow population, refine | Pop=12, mut=0.5, deeper graphs |
| **Oxidation** | 16% | Converge: strong coupling, prune | Pop=8, mut=0.3, coupling=0.8 |
| **Endosymbiosis** | 30% | Deep training with winner activation | No search, cosine decay |
| **Homeostasis** | 10% | Ultra-low LR, monitor regression | Final refinement |

Between phases: comprehensive inference battery (20 prompts x 3 temperatures),
overfitting detection, plateau detection, and adaptive config adjustment.

## Understanding Overfitting in Symbiogenesis

Standard overfitting (train/val gap) is just one signal. In evolutionary search,
overfitting occurs at 5 levels:

1. **Candidate Memorization** — Single candidate overfits within its short eval window.
   Masked by the evolutionary cycle. Detect via per-candidate train/val ratio.

2. **Population Collapse** — All candidates converge to the same activation.
   Diversity drops below 0.3. The search becomes a random walk around one point.

3. **Fusion Trap** — The consensus shadow model memorizes, pulling all candidates
   toward it. Weight entropy drops while val_loss stagnates. The shadow is parasitic.

4. **Activation Complexity Overfitting** — Evolved graph is too complex (>6 nodes),
   fitting training quirks. Complex candidates beat simple ones on train but lose on val.

5. **Transfer Interference** — Inherited weights from a previous candidate conflict with
   the new activation's gradient direction. Model wastes eval budget un-learning.

The orchestrator detects these and adapts: boosting mutation, reducing coupling,
expanding search space, or saving dormant checkpoints for later.

## Understanding Evolutionary Loss Plateaus

Loss plateaus in evolutionary search mean the search itself has stalled, not
just the model:

1. **Activation Space Exhaustion** — All promising structures explored. CUSUM fires
   on throughput collapse. Response: expand basis pool, increase graph limits.

2. **Punctuated Equilibrium** — Long stasis is NORMAL. If diversity is high, the
   population is building capacity for a jump. Don't mistake this for failure.

3. **Consensus Stagnation** — High Kuramoto sync + flat loss = the fusion shadow
   trapped everyone in a local minimum. Response: perturb shadow, reduce coupling.

4. **Learning Rate Mismatch** — Different activations have different gradient scales.
   Persistent spike-skips are the symptom. Response: lower LR, stronger grad clip.

## Monitoring a Run

Check state: `cat runs/symbio-auto/state.json`
Check GCP: `python scripts/gcp_train.py --action status`
SSH in: `python scripts/gcp_train.py --action ssh`
Tail log: `gcloud compute ssh alpha-train --zone=us-central1-b --command="tail -f ~/alpha/runs/train_symbio_*.log"`

## When to Intervene

- **35%+ spike skips**: Gradient instability is too severe. Kill and restart with lower LR.
- **Quality regression**: If inference quality drops >30% from best, something broke.
  The orchestrator will flag this but you may want to inspect manually.
- **Phase stuck >2x expected time**: Training may have crashed silently. Check the log.
- **Instance cost**: At ~$0.70/hr for L4, a 50k-step run is roughly 8-12 hours ($6-8).

## Gaps This Fills

Beyond the existing symbio TypeScript implementation, this orchestrator adds:

- **Multi-phase lifecycle** with adaptive phase transitions
- **Inference-gated progression** — quality must improve to advance
- **Dormancy** — promising checkpoints saved for later if current path regresses
- **Autopoiesis** — self-evaluation through inference battery
- **Ecological pressure** — track loss improvement per GPU-second
- **Niche construction** — phase configs reshape the selection landscape
- **Punctuated equilibrium detection** — distinguish stasis from failure

## Files

- `scripts/symbio-autonomous.py` — Main orchestrator
- `runs/symbio-auto/state.json` — Persistent state (for resume)
- `runs/symbio-auto/final-report.json` — Post-training summary
- Configs generated dynamically per phase (written to instance at `/tmp/symbio-*.json`)
