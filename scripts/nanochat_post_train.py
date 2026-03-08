"""
Full Post-Training Pipeline: Protocol Adaptation -> SFT -> RL -> Self-Play
Implements the complete pipeline from docs/nanochat-post-training-pipeline.md

Run as:
    python -m scripts.post_train --depth=8 --device-batch-size=16
    python -m scripts.post_train --depth=12 --device-batch-size=8

Phases:
    1. Protocol Adaptation (midtraining) - full-sequence loss, teach format
    2. SFT with NEFTune (two-stage curriculum) - assistant-only loss
    3. RL with GRPO - simplified policy gradient on verifiable rewards
    4. Self-Play Refinement - iterative improvement loop
"""

import gc
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import math
import time
import random
import argparse
import itertools
import torch
import torch.nn.functional as F
import torch.distributed as dist
import wandb

from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb,
    get_base_dir, autodetect_device_type, get_peak_flops,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
)
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(description="Full post-training pipeline")
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--model-step", type=int, default=None)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--skip-midtrain", action="store_true")
parser.add_argument("--skip-sft", action="store_true")
parser.add_argument("--skip-rl", action="store_true")
parser.add_argument("--skip-selfplay", action="store_true")
parser.add_argument("--neftune-alpha", type=float, default=5.0)
parser.add_argument("--rl-num-samples", type=int, default=16)
parser.add_argument("--rl-max-tokens", type=int, default=256)
parser.add_argument("--rl-temperature", type=float, default=1.0)
parser.add_argument("--rl-examples-per-step", type=int, default=4)
parser.add_argument("--rl-num-steps", type=int, default=200)
parser.add_argument("--selfplay-rounds", type=int, default=3)
parser.add_argument("--selfplay-num-samples", type=int, default=8)
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--eval-every", type=int, default=-1)
parser.add_argument("--midtrain-steps", type=int, default=-1, help="midtrain steps (-1=auto: 10*depth)")
parser.add_argument("--sft-s1-steps", type=int, default=-1, help="SFT stage 1 steps (-1=auto)")
parser.add_argument("--sft-s2-steps", type=int, default=-1, help="SFT stage 2 steps (-1=auto)")
args = parser.parse_args()

# =============================================================================
# INIT
# =============================================================================

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
base_dir = get_base_dir()

print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-posttrain", name=args.run, config=vars(args))


# =============================================================================
# NEFTUNE: Noise-Augmented Embedding Hook
# =============================================================================

class NEFTuneHook:
    """Add uniform noise to embeddings during training.
    noise ~ Uniform(-alpha/sqrt(L*d), alpha/sqrt(L*d))
    Regularization that improves instruction-following quality.
    """
    def __init__(self, model, alpha=5.0):
        self.alpha = alpha
        self.handle = None
        self._model = model
        self.enabled = False

    def _hook(self, module, input, output):
        if self.enabled and self._model.training:
            dims = output.shape[1] * output.shape[2]
            mag = self.alpha / math.sqrt(dims)
            noise = torch.zeros_like(output).uniform_(-mag, mag)
            return output + noise
        return output

    def attach(self):
        self.handle = self._model.transformer.wte.register_forward_hook(self._hook)
        print0(f"NEFTune attached (alpha={self.alpha})")

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# =============================================================================
# DATA GENERATORS
# =============================================================================

def make_midtrain_data(tokenizer, max_seq_len, device_batch_size):
    """Phase 1: Protocol Adaptation — full sequence loss (no masking).
    Teaches conversation format, special tokens, MC protocol, tool-use.
    """
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")
    tasks = [
        SmolTalk(split="train"),
        CustomJSON(filepath=identity_path),
        *[MMLU(subset="auxiliary_train", split="train") for _ in range(3)],
        *[GSM8K(subset="main", split="train") for _ in range(4)],
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ]
    dataset = TaskMixture(tasks)
    print0(f"  Midtrain mixture: {len(dataset):,} rows")

    row_capacity = max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    cursor = ddp_rank
    epoch = 1

    while True:
        rows = []
        for _ in range(device_batch_size):
            row = []
            while len(row) < row_capacity:
                conv = dataset[cursor % len(dataset)]
                ids, mask = tokenizer.render_conversation(conv)
                remaining = row_capacity - len(row)
                if len(ids) <= remaining:
                    row.extend(ids)
                else:
                    row.extend(ids[:remaining])
                cursor += ddp_world_size
                if cursor >= len(dataset) * ddp_world_size:
                    cursor = cursor % (len(dataset) * ddp_world_size)
                    epoch += 1
            rows.append(row[:row_capacity])

        batch = torch.tensor(rows, dtype=torch.long)
        inputs = batch[:, :-1].to(device=device, dtype=torch.int32).contiguous()
        targets = batch[:, 1:].to(device=device, dtype=torch.int64).contiguous()
        yield inputs, targets, epoch


def make_sft_data(tokenizer, max_seq_len, device_batch_size, stage=1):
    """Phase 2: SFT — assistant-only loss masking with best-fit packing.
    Stage 1: Reasoning-heavy (more GSM8K/MMLU, explicit CoT)
    Stage 2: General quality (more SmolTalk, implicit reasoning)
    """
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")

    if stage == 1:
        tasks = [
            SmolTalk(split="train"),
            CustomJSON(filepath=identity_path),
            CustomJSON(filepath=identity_path),
            *[MMLU(subset="auxiliary_train", split="train") for _ in range(4)],
            *[GSM8K(subset="main", split="train") for _ in range(6)],
            SimpleSpelling(size=200000, split="train"),
            SpellingBee(size=80000, split="train"),
        ]
    else:
        tasks = [
            SmolTalk(split="train"),
            SmolTalk(split="train"),
            CustomJSON(filepath=identity_path),
            CustomJSON(filepath=identity_path),
            *[MMLU(subset="auxiliary_train", split="train") for _ in range(2)],
            *[GSM8K(subset="main", split="train") for _ in range(2)],
            SimpleSpelling(size=100000, split="train"),
        ]

    dataset = TaskMixture(tasks)
    print0(f"  SFT Stage {stage} mixture: {len(dataset):,} rows")

    row_capacity = max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    cursor = ddp_rank
    epoch = 1
    conv_buffer = []

    def refill():
        nonlocal cursor, epoch
        while len(conv_buffer) < 100:
            conv = dataset[cursor % len(dataset)]
            ids, mask = tokenizer.render_conversation(conv)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= len(dataset) * ddp_world_size:
                cursor = cursor % (len(dataset) * ddp_world_size)
                epoch += 1

    while True:
        rows = []
        mask_rows = []
        row_lengths = []

        for _ in range(device_batch_size):
            row = []
            mask_row = []
            content_len = 0
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < 100:
                    refill()
                remaining = row_capacity - len(row)
                best_idx, best_len = -1, 0
                for i, (conv, _) in enumerate(conv_buffer):
                    if len(conv) <= remaining and len(conv) > best_len:
                        best_idx, best_len = i, len(conv)
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            if not padded:
                content_len = row_capacity
            row_lengths.append(content_len)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        batch = torch.tensor(rows, dtype=torch.long)
        inputs = batch[:, :-1].to(device=device, dtype=torch.int32).contiguous()
        targets = batch[:, 1:].to(device=device, dtype=torch.int64).contiguous()

        mt = torch.tensor(mask_rows, dtype=torch.int8)[:, 1:].to(device=device)
        targets[mt == 0] = -1
        for i, cl in enumerate(row_lengths):
            if cl < row_capacity:
                targets[i, cl - 1:] = -1

        yield inputs, targets, epoch


# =============================================================================
# EVALUATION
# =============================================================================

def run_full_eval(orig_model, tokenizer, step, phase_name, batch_size):
    """Run ChatCORE evaluation."""
    from scripts.chat_eval import run_chat_eval

    orig_model.eval()
    engine = Engine(orig_model, tokenizer)

    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    categorical = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
    baselines = {
        'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
        'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
    }

    results = {}
    for task_name in all_tasks:
        limit = None if task_name in categorical else 24
        acc = run_chat_eval(
            task_name, orig_model, tokenizer, engine,
            batch_size=batch_size, max_problems=limit)
        results[task_name] = acc
        print0(f"    {task_name}: {100 * acc:.2f}%")

    def centered_mean(tasks):
        return sum(
            (results[t] - baselines[t]) / (1 - baselines[t])
            for t in tasks
        ) / len(tasks)

    chatcore = centered_mean(all_tasks)
    chatcore_cat = centered_mean(categorical)
    print0(f"  [{phase_name}] ChatCORE: {chatcore:.4f} | cat: {chatcore_cat:.4f}")

    wandb_run.log({
        "step": step, "phase": phase_name,
        "chatcore": chatcore, "chatcore_cat": chatcore_cat,
        **{f"{phase_name}/{t}": v for t, v in results.items()},
    })
    return chatcore, results


# =============================================================================
# GENERIC TRAINING LOOP
# =============================================================================

def train_phase(model, orig_model, optimizer, data_gen, tokenizer, token_bytes,
                phase_name, max_seq_len, neftune_hook=None, scaler=None,
                max_steps=1000, eval_every=-1, warmdown_ratio=0.5,
                init_lr_frac=0.8, final_lr_frac=0.0, save_source="sft"):
    """Generic training loop for midtraining and SFT phases.
    Always uses explicit max_steps for reliable LR scheduling."""

    total_batch_size = args.device_batch_size * max_seq_len * ddp_world_size
    step = 0
    total_time = 0
    smooth_loss = 0
    ema_beta = 0.9
    best_chatcore = -1

    if neftune_hook:
        neftune_hook.enable()

    print0(f"\n  Starting {phase_name} ({max_steps} steps)...")

    for inputs, targets, epoch in data_gen:
        if step >= max_steps:
            break

        # LR schedule: constant then linear warmdown
        progress = step / max_steps
        if progress > 1.0 - warmdown_ratio:
            decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
            lrm = (1 - decay) + decay * final_lr_frac
        else:
            lrm = 1.0

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm * init_lr_frac

        # Muon momentum warmup
        frac = min(step / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        for group in optimizer.param_groups:
            if group.get('kind') == 'muon':
                group["momentum"] = momentum

        model.train()
        synchronize()
        t0 = time.time()

        if scaler:
            loss = model(inputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

        model.zero_grad(set_to_none=True)
        synchronize()
        dt = time.time() - t0

        step += 1
        train_loss = loss.item()
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased = smooth_loss / (1 - ema_beta ** step)
        tok_per_sec = int(total_batch_size / dt)
        if step > 10:
            total_time += dt

        if step % 25 == 0:
            eta_min = (max_steps - step) * dt / 60 if dt > 0 else 0
            print0(f"  [{phase_name}] step {step:05d}/{max_steps} | loss: {debiased:.4f} | "
                   f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/s: {tok_per_sec:,} | eta: {eta_min:.1f}m")

        if step % 50 == 0:
            wandb_run.log({
                "step": step, "phase": phase_name,
                f"{phase_name}/loss": debiased, f"{phase_name}/lrm": lrm,
            })

        if eval_every > 0 and step > 0 and step % eval_every == 0:
            if neftune_hook:
                neftune_hook.disable()
            chatcore, _ = run_full_eval(orig_model, tokenizer, step, phase_name, args.device_batch_size)
            if chatcore > best_chatcore:
                best_chatcore = chatcore
            if neftune_hook:
                neftune_hook.enable()

        if step == 1:
            gc.collect(); gc.freeze(); gc.disable()
        elif step % 2000 == 0:
            gc.collect()

    if neftune_hook:
        neftune_hook.disable()

    print0(f"  [{phase_name}] Done: {step} steps, {total_time / 60:.1f}m")

    depth = orig_model.config.n_layer
    tag = args.model_tag or f"d{depth}"
    ckpt_dir = os.path.join(base_dir, f"post_{save_source}_checkpoints", tag)
    save_checkpoint(ckpt_dir, step, orig_model.state_dict(), optimizer.state_dict(), {
        "step": step, "phase": phase_name,
        "model_config": {
            "sequence_len": max_seq_len,
            "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": depth,
            "n_head": orig_model.config.n_head,
            "n_kv_head": orig_model.config.n_kv_head,
            "n_embd": orig_model.config.n_embd,
            "window_pattern": orig_model.config.window_pattern,
        },
    }, rank=ddp_rank)
    print0(f"  Saved to {ckpt_dir}")

    chatcore, results = run_full_eval(orig_model, tokenizer, step,
                                       f"{phase_name}_final", args.device_batch_size)
    gc.enable(); gc.collect()
    return step, chatcore, results


# =============================================================================
# PHASE 3: RL WITH GRPO
# =============================================================================

def run_rl_phase(model, orig_model, tokenizer, max_seq_len):
    """Simplified GRPO: on-policy, no KL, no PPO, no value function.

    For each problem:
        1. Generate K completions
        2. Score with binary reward
        3. advantage = reward - mean(rewards)
        4. L = -sum(log p(t) * advantage) / valid_tokens
        5. Gradient step
    """
    print0(f"\n  Starting RL (GRPO)...")

    optimizer = orig_model.setup_optimizer(
        unembedding_lr=0.004 * 0.05, embedding_lr=0.2 * 0.05,
        matrix_lr=0.02 * 0.05, weight_decay=0.0,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    train_task = GSM8K(subset="main", split="train")
    engine = Engine(orig_model, tokenizer)
    num_steps = args.rl_num_steps
    device_bs = min(args.device_batch_size, args.rl_num_samples)

    print0(f"  RL: {num_steps} steps, {args.rl_num_samples} samples/problem, "
           f"{args.rl_examples_per_step} problems/step, {len(train_task)} problems")

    rewards_history = []
    pidxs = list(range(len(train_task)))
    random.shuffle(pidxs)
    pcursor = 0
    seed = 42

    for step in range(num_steps):
        lrm = 1.0 - step / num_steps
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        step_rewards = []
        model.train()
        optimizer.zero_grad()

        for _ in range(args.rl_examples_per_step):
            pidx = pidxs[pcursor % len(pidxs)]
            pcursor += 1
            if pcursor >= len(pidxs):
                random.shuffle(pidxs)
                pcursor = 0

            example = train_task.get_example(pidx)
            conversation = example["messages"]
            prompt_conv = {"messages": [m for m in conversation if m["role"] == "user"]}
            prompt_ids, _ = tokenizer.render_conversation(prompt_conv)
            ast_id = tokenizer.encode_special("<|assistant_start|>")
            prompt_ids.append(ast_id)
            prompt_len = len(prompt_ids)

            # Generate rollouts (generate_batch expects list of ints)
            orig_model.eval()
            all_seqs = []
            for _ in range(args.rl_num_samples // device_bs):
                seed += 1
                seqs, _ = engine.generate_batch(
                    prompt_ids, num_samples=device_bs,
                    max_tokens=args.rl_max_tokens,
                    temperature=args.rl_temperature, top_k=50, seed=seed)
                all_seqs.extend(seqs)

            # Score (generate_batch returns full seqs including prompt)
            rewards = []
            for seq in all_seqs:
                completion_ids = seq[prompt_len:]
                text = tokenizer.decode(completion_ids)
                r = float(train_task.reward(example, text))
                rewards.append(r)

            rtensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            step_rewards.extend(rewards)
            advantages = rtensor - rtensor.mean()

            if advantages.abs().max() < 1e-8:
                continue

            # Policy gradient
            model.train()
            for seq, adv in zip(all_seqs, advantages):
                if abs(adv.item()) < 1e-8:
                    continue
                # seq already includes prompt from generate_batch
                if len(seq) < 2:
                    continue

                st = torch.tensor([seq], dtype=torch.long, device=device)
                inp = st[:, :-1].to(dtype=torch.int32)
                tgt = st[:, 1:].to(dtype=torch.int64)
                # Mask prompt tokens (only train on completion)
                tgt[:, :prompt_len - 1] = -1

                per_token_loss = model(inp, tgt, loss_reduction='none')
                # CE returns (B*T,), reshape to (B, T) for masking
                logp = -per_token_loss.view(inp.shape)

                valid = (tgt >= 0).float()
                pg = (logp * valid * adv).sum()
                nv = valid.sum().clamp(min=1)
                pg = pg / (nv * args.rl_num_samples * args.rl_examples_per_step)
                (-pg).backward()

        optimizer.step()
        optimizer.zero_grad()

        mr = sum(step_rewards) / max(len(step_rewards), 1)
        rewards_history.append(mr)

        if step % 10 == 0:
            ra = sum(rewards_history[-20:]) / max(len(rewards_history[-20:]), 1)
            print0(f"  [RL] step {step:04d}/{num_steps} | reward: {mr:.3f} | avg: {ra:.3f}")
            wandb_run.log({"step": step, "rl/reward": mr, "rl/avg": ra})

        if step > 0 and step % 50 == 0:
            run_full_eval(orig_model, tokenizer, step, "rl", args.device_batch_size)

    # Save
    depth = orig_model.config.n_layer
    tag = args.model_tag or f"d{depth}"
    ckpt_dir = os.path.join(base_dir, "post_rl_checkpoints", tag)
    save_checkpoint(ckpt_dir, num_steps, orig_model.state_dict(), None, {
        "step": num_steps, "phase": "rl",
        "model_config": {
            "sequence_len": max_seq_len, "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": depth, "n_head": orig_model.config.n_head,
            "n_kv_head": orig_model.config.n_kv_head, "n_embd": orig_model.config.n_embd,
            "window_pattern": orig_model.config.window_pattern,
        },
    })
    print0(f"  Saved RL checkpoint to {ckpt_dir}")
    chatcore, results = run_full_eval(orig_model, tokenizer, num_steps,
                                       "rl_final", args.device_batch_size)
    gc.collect()
    return chatcore, results


# =============================================================================
# PHASE 4: SELF-PLAY REFINEMENT
# =============================================================================

def run_selfplay_phase(model, orig_model, tokenizer, max_seq_len):
    """Self-play: generate -> evaluate -> train on successes."""
    print0(f"\n  Starting Self-Play ({args.selfplay_rounds} rounds)...")

    train_task = GSM8K(subset="main", split="train")
    engine = Engine(orig_model, tokenizer)
    device_bs = min(args.device_batch_size, args.selfplay_num_samples)
    seed = 1000
    chatcore = -1

    for rnd in range(args.selfplay_rounds):
        print0(f"\n  [Self-Play] Round {rnd + 1}/{args.selfplay_rounds}")

        # Generate and collect successes
        successes = []
        total, correct = 0, 0
        num_problems = min(200, len(train_task))
        indices = random.sample(range(len(train_task)), num_problems)

        orig_model.eval()
        for pidx in indices:
            example = train_task.get_example(pidx)
            conv = example["messages"]
            prompt_conv = {"messages": [m for m in conv if m["role"] == "user"]}
            pids, _ = tokenizer.render_conversation(prompt_conv)
            ast_id = tokenizer.encode_special("<|assistant_start|>")
            pids.append(ast_id)
            plen = len(pids)

            for _ in range(args.selfplay_num_samples // device_bs):
                seed += 1
                seqs, _ = engine.generate_batch(
                    pids, num_samples=device_bs,
                    max_tokens=args.rl_max_tokens, temperature=0.8, top_k=50, seed=seed)
                for seq in seqs:
                    total += 1
                    completion_ids = seq[plen:]
                    text = tokenizer.decode(completion_ids)
                    if float(train_task.reward(example, text)) > 0.5:
                        correct += 1
                        successes.append({
                            "messages": [conv[0], {"role": "assistant", "content": text}]
                        })

        sr = correct / max(total, 1)
        print0(f"  [Self-Play] {correct}/{total} correct ({100 * sr:.1f}%), {len(successes)} successes")
        wandb_run.log({"selfplay_round": rnd + 1, "selfplay/solve_rate": sr})

        if len(successes) < 10:
            print0(f"  [Self-Play] Too few successes, stopping early")
            break

        # Train on successes
        opt = orig_model.setup_optimizer(
            unembedding_lr=0.004 * 0.1, embedding_lr=0.2 * 0.1,
            matrix_lr=0.02 * 0.1, weight_decay=0.0)
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

        compiled = torch.compile(orig_model, dynamic=False)
        bos = tokenizer.get_bos_token_id()
        random.shuffle(successes)
        nsteps = max(1, len(successes) // args.device_batch_size)
        row_cap = max_seq_len + 1

        compiled.train()
        for ts in range(nsteps):
            batch_convs = successes[ts * args.device_batch_size:(ts + 1) * args.device_batch_size]
            if not batch_convs:
                break

            rows, mrows = [], []
            for c in batch_convs:
                ids, mask = tokenizer.render_conversation(c)
                if len(ids) > row_cap:
                    ids, mask = ids[:row_cap], mask[:row_cap]
                elif len(ids) < row_cap:
                    pad = row_cap - len(ids)
                    ids.extend([bos] * pad)
                    mask.extend([0] * pad)
                rows.append(ids)
                mrows.append(mask)

            while len(rows) < args.device_batch_size:
                rows.append([bos] * row_cap)
                mrows.append([0] * row_cap)

            bt = torch.tensor(rows, dtype=torch.long)
            inp = bt[:, :-1].to(device=device, dtype=torch.int32).contiguous()
            tgt = bt[:, 1:].to(device=device, dtype=torch.int64).contiguous()
            mt = torch.tensor(mrows, dtype=torch.int8)[:, 1:].to(device=device)
            tgt[mt == 0] = -1

            lrm = 1.0 - ts / max(nsteps, 1)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lrm

            loss = compiled(inp, tgt)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            if ts % 25 == 0:
                print0(f"    [SP Train] step {ts}/{nsteps} | loss: {loss.item():.4f}")

        chatcore, _ = run_full_eval(orig_model, tokenizer, rnd + 1,
                                     f"selfplay_r{rnd + 1}", args.device_batch_size)
        print0(f"  [Self-Play] Round {rnd + 1} ChatCORE: {chatcore:.4f}")

    # Save
    depth = orig_model.config.n_layer
    tag = args.model_tag or f"d{depth}"
    ckpt_dir = os.path.join(base_dir, "post_selfplay_checkpoints", tag)
    save_checkpoint(ckpt_dir, args.selfplay_rounds, orig_model.state_dict(), None, {
        "phase": "selfplay", "rounds": args.selfplay_rounds,
        "model_config": {
            "sequence_len": max_seq_len, "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": depth, "n_head": orig_model.config.n_head,
            "n_kv_head": orig_model.config.n_kv_head, "n_embd": orig_model.config.n_embd,
            "window_pattern": orig_model.config.window_pattern,
        },
    })
    print0(f"  Saved self-play checkpoint to {ckpt_dir}")
    return chatcore


# =============================================================================
# MAIN
# =============================================================================

def main():
    print0("\n" + "=" * 70)
    print0("  FULL POST-TRAINING PIPELINE")
    print0("  Protocol Adaptation -> SFT (2-stage + NEFTune) -> RL -> Self-Play")
    print0("=" * 70)

    # Load base model
    print0(f"\nLoading base model (depth={args.depth})...")
    orig_model, tokenizer, meta = load_model(
        "base", device, phase="train",
        model_tag=args.model_tag, step=args.model_step)
    max_seq_len = meta.get("max_seq_len", 2048)
    depth = orig_model.config.n_layer
    nparams = sum(p.numel() for p in orig_model.parameters())
    print0(f"  depth={depth}, dim={orig_model.config.n_embd}, params={nparams / 1e6:.1f}M")

    token_bytes = get_token_bytes(device=device)
    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

    # Download identity data
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")
    if not os.path.exists(identity_path):
        print0("Downloading identity conversations...")
        import subprocess
        subprocess.run([
            "curl", "-sL", "-o", identity_path,
            "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
        ], check=True)

    # Auto-compute step budgets based on depth (scale with model size)
    # nanochat's built-in SFT does ~one epoch which is ~2000 steps for d12
    midtrain_steps = args.midtrain_steps if args.midtrain_steps > 0 else max(200, depth * 40)
    sft_s1_steps = args.sft_s1_steps if args.sft_s1_steps > 0 else max(500, depth * 150)
    sft_s2_steps = args.sft_s2_steps if args.sft_s2_steps > 0 else max(300, depth * 80)
    print0(f"  Step budgets: midtrain={midtrain_steps}, sft_s1={sft_s1_steps}, sft_s2={sft_s2_steps}")
    print0(f"  RL steps={args.rl_num_steps}, selfplay rounds={args.selfplay_rounds}")

    # Baseline eval
    print0("\n--- Baseline Evaluation ---")
    baseline_cc, _ = run_full_eval(orig_model, tokenizer, 0, "baseline", args.device_batch_size)

    results_summary = {"baseline": baseline_cc}

    # ── PHASE 1: Protocol Adaptation ─────────────────────────────────
    if not args.skip_midtrain:
        print0("\n" + "-" * 60)
        print0("  PHASE 1: Protocol Adaptation (Midtraining)")
        print0(f"  Full-sequence loss, {midtrain_steps} steps. Teaching format.")
        print0("-" * 60)

        compiled = torch.compile(orig_model, dynamic=False)
        opt = orig_model.setup_optimizer(
            unembedding_lr=0.004, embedding_lr=0.3,
            matrix_lr=0.02, weight_decay=0.0)

        opt_data = load_optimizer_state(
            "base", device, rank=ddp_rank,
            model_tag=args.model_tag, step=args.model_step)
        if opt_data is not None:
            lrs = [g["lr"] for g in opt.param_groups]
            opt.load_state_dict(opt_data)
            del opt_data
            for g, lr in zip(opt.param_groups, lrs):
                g["lr"] = lr
            print0("  Warm-started optimizer momentum")

        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

        dg = make_midtrain_data(tokenizer, max_seq_len, args.device_batch_size)
        _, mid_cc, _ = train_phase(
            compiled, orig_model, opt, dg, tokenizer, token_bytes,
            "midtrain", max_seq_len, max_steps=midtrain_steps,
            eval_every=args.eval_every,
            warmdown_ratio=0.5, init_lr_frac=0.8, save_source="midtrain", scaler=scaler)
        results_summary["midtrain"] = mid_cc
        del compiled, opt; gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # ── PHASE 2: SFT (2-Stage + NEFTune) ─────────────────────────────
    if not args.skip_sft:
        print0("\n" + "-" * 60)
        print0("  PHASE 2: SFT (2-Stage Curriculum + NEFTune)")
        print0("  Stage 1: Reasoning-enhanced | Stage 2: General quality")
        print0("-" * 60)

        neftune = NEFTuneHook(orig_model, alpha=args.neftune_alpha)
        neftune.attach()

        stage_steps = {1: sft_s1_steps, 2: sft_s2_steps}
        for stage in [1, 2]:
            print0(f"\n  --- SFT Stage {stage} ({stage_steps[stage]} steps) ---")
            compiled = torch.compile(orig_model, dynamic=False)
            lr_scale = 1.0 if stage == 1 else 0.5
            opt = orig_model.setup_optimizer(
                unembedding_lr=0.004 * lr_scale, embedding_lr=0.3 * lr_scale,
                matrix_lr=0.02 * lr_scale, weight_decay=0.0)
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]

            dg = make_sft_data(tokenizer, max_seq_len, args.device_batch_size, stage)
            _, sft_cc, _ = train_phase(
                compiled, orig_model, opt, dg, tokenizer, token_bytes,
                f"sft_s{stage}", max_seq_len, neftune_hook=neftune,
                max_steps=stage_steps[stage], eval_every=args.eval_every,
                warmdown_ratio=0.5 if stage == 1 else 0.3,
                init_lr_frac=0.8 if stage == 1 else 0.5,
                save_source=f"sft_s{stage}", scaler=scaler)
            results_summary[f"sft_s{stage}"] = sft_cc
            del compiled, opt; gc.collect()
            if device_type == "cuda":
                torch.cuda.empty_cache()

        neftune.remove()

    # ── PHASE 3: RL (GRPO) ───────────────────────────────────────────
    if not args.skip_rl:
        print0("\n" + "-" * 60)
        print0("  PHASE 3: RL with Simplified GRPO")
        print0("  On-policy, no KL, no PPO clipping, binary rewards.")
        print0("-" * 60)

        compiled = torch.compile(orig_model, dynamic=False)
        rl_cc, _ = run_rl_phase(compiled, orig_model, tokenizer, max_seq_len)
        results_summary["rl"] = rl_cc
        del compiled; gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # ── PHASE 4: Self-Play ───────────────────────────────────────────
    if not args.skip_selfplay:
        print0("\n" + "-" * 60)
        print0("  PHASE 4: Self-Play Refinement")
        print0("  Generate -> Evaluate -> Train on successes.")
        print0("-" * 60)

        compiled = torch.compile(orig_model, dynamic=False)
        sp_cc = run_selfplay_phase(compiled, orig_model, tokenizer, max_seq_len)
        results_summary["selfplay"] = sp_cc
        del compiled; gc.collect()

    # ── FINAL REPORT ──────────────────────────────────────────────────
    print0("\n" + "=" * 70)
    print0("  POST-TRAINING COMPLETE")
    print0("=" * 70)
    for phase, score in results_summary.items():
        print0(f"  {phase:20s} ChatCORE: {score:.4f}")
    print0("")
    print0("  python -m scripts.chat_cli")
    print0("  python -m scripts.chat_web")
    print0("=" * 70)

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
