#!/usr/bin/env python3
"""
Comprehensive CUDA reference benchmark for Alpha training operations.

Tests every operation type used in the 300M GPT training pipeline at exact
training shapes. Outputs JSON for comparison with Helios.

Usage:
    python3 scripts/bench-ops-cuda.py --iters=30 --warmup=8
"""

from __future__ import annotations

import argparse
import json
import time
import math
import sys


def bench(fn, warmup, iters, sync):
    """Run fn with warmup, return median time in ms."""
    for _ in range(warmup):
        fn()
    sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"PyTorch import failed: {exc}"}))
        return

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "CUDA not available"}))
        return

    device = torch.device("cuda")
    sync = torch.cuda.synchronize
    # Enable TF32 for fair comparison — Helios coop matrix does f16*f16→f32,
    # TF32 is NVIDIA's equivalent mixed-precision path for FP32 workloads.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    W, I = args.warmup, args.iters

    # Training config: 300M model, batch=1, block=512, dim=1024, heads=16, ffn=2752, vocab=64000
    B, T, D, H, Dh = 1, 512, 1024, 16, 64
    FFN = 2752
    V = 64000
    BT = B * T  # 512
    BH = B * H  # 16

    results = {
        "ok": True,
        "framework": "pytorch_cuda",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "iters": I,
        "warmup": W,
        "ops": {},
    }

    def record(name, ms, flops=0, bytes_rw=0, note=""):
        entry = {"ms": round(ms, 4)}
        if flops > 0:
            entry["tflops"] = round((flops / (ms / 1000)) / 1e12, 3)
        if bytes_rw > 0:
            entry["gbps"] = round((bytes_rw / (ms / 1000)) / 1e9, 1)
        if note:
            entry["note"] = note
        results["ops"][name] = entry

    # ── 1. MATMUL (training shapes, transposed: A @ B^T) ─────────────────────

    matmul_shapes = [
        ("matmul_qkv",       BT, 3072, 1024, "QKV projection"),
        ("matmul_attn_out",  BT, 1024, 1024, "attention output"),
        ("matmul_swiglu_g",  BT, 2752, 1024, "SwiGLU gate"),
        ("matmul_swiglu_u",  BT, 2752, 1024, "SwiGLU up"),
        ("matmul_swiglu_p",  BT, 1024, 2752, "SwiGLU proj"),
        ("matmul_lm_head",   BT, V,    1024, "LM head"),
    ]

    for name, M, N, K, note in matmul_shapes:
        a = torch.randn(M, K, device=device)
        w = torch.randn(N, K, device=device)  # stored transposed
        ms = bench(lambda: a @ w.T, W, I, sync)
        record(name, ms, flops=2 * M * N * K, note=note)
        del a, w

    # Backward matmuls (weight gradients: A^T @ dOut)
    bwd_shapes = [
        ("matmul_bwd_qkv",      1024, 3072, BT, "QKV weight grad"),
        ("matmul_bwd_attn_out",  1024, 1024, BT, "attn out weight grad"),
        ("matmul_bwd_swiglu_g",  1024, 2752, BT, "SwiGLU gate weight grad"),
        ("matmul_bwd_lm_head",   1024, V,    BT, "LM head weight grad"),
    ]

    for name, M, N, K, note in bwd_shapes:
        a = torch.randn(K, M, device=device)  # A^T means original is [K, M]
        b = torch.randn(K, N, device=device)  # dOut is [K, N]
        ms = bench(lambda: a.T @ b, W, I, sync)
        record(name, ms, flops=2 * M * N * K, note=note)
        del a, b

    # Square matmuls for reference
    for sz in [1024, 2048, 3072, 4096]:
        a = torch.randn(sz, sz, device=device)
        b = torch.randn(sz, sz, device=device)
        ms = bench(lambda: a @ b, W, I, sync)
        record(f"matmul_{sz}sq", ms, flops=2 * sz**3, note=f"{sz}x{sz}x{sz}")
        del a, b

    # ── 2. ELEMENT-WISE OPS ──────────────────────────────────────────────────

    x1024 = torch.randn(BT, D, device=device)
    x2752 = torch.randn(BT, FFN, device=device)
    y1024 = torch.randn(BT, D, device=device)

    sz1024 = BT * D
    sz2752 = BT * FFN

    ms = bench(lambda: x1024 + y1024, W, I, sync)
    record("add_512x1024", ms, bytes_rw=sz1024 * 4 * 3, note="a+b (read a,b write c)")

    ms = bench(lambda: x1024 * y1024, W, I, sync)
    record("mul_512x1024", ms, bytes_rw=sz1024 * 4 * 3, note="a*b")

    ms = bench(lambda: F.gelu(x1024, approximate="tanh"), W, I, sync)
    record("gelu_512x1024", ms, bytes_rw=sz1024 * 4 * 2, note="GELU activation")

    ms = bench(lambda: F.silu(x2752), W, I, sync)
    record("silu_512x2752", ms, bytes_rw=sz2752 * 4 * 2, note="SiLU activation")

    ms = bench(lambda: x1024 * 0.125, W, I, sync)
    record("scale_512x1024", ms, bytes_rw=sz1024 * 4 * 2, note="scalar multiply")

    ms = bench(lambda: torch.neg(x1024), W, I, sync)
    record("neg_512x1024", ms, bytes_rw=sz1024 * 4 * 2, note="negate")

    ms = bench(lambda: torch.exp(x1024), W, I, sync)
    record("exp_512x1024", ms, bytes_rw=sz1024 * 4 * 2, note="exp")

    # GELU backward (fused)
    gelu_x = x1024.clone().requires_grad_(True)
    gelu_out = F.gelu(gelu_x, approximate="tanh")
    gelu_grad = torch.randn_like(gelu_out)
    def gelu_bwd():
        if gelu_x.grad is not None:
            gelu_x.grad.zero_()
        gelu_out.backward(gelu_grad, retain_graph=True)
    ms = bench(gelu_bwd, W, I, sync)
    record("gelu_bwd_512x1024", ms, bytes_rw=sz1024 * 4 * 3, note="GELU backward (fused)")
    del gelu_x, gelu_out, gelu_grad

    del x1024, x2752, y1024

    # Large tensor element-wise (memory bandwidth test)
    big = torch.randn(4096, 4096, device=device)
    big2 = torch.randn(4096, 4096, device=device)
    sz_big = 4096 * 4096
    ms = bench(lambda: big + big2, W, I, sync)
    record("add_4096x4096", ms, bytes_rw=sz_big * 4 * 3, note="large add bandwidth test")
    ms = bench(lambda: big * 2.0, W, I, sync)
    record("scale_4096x4096", ms, bytes_rw=sz_big * 4 * 2, note="large scale bandwidth test")
    del big, big2

    # LM head weight-sized element-wise (sustained bandwidth on 256MB)
    lm_w1 = torch.randn(1024, V, device=device)
    lm_w2 = torch.randn(1024, V, device=device)
    lm_sz = 1024 * V
    ms = bench(lambda: lm_w1 + lm_w2, W, I, sync)
    record("add_lm_head_1024x64000", ms, bytes_rw=lm_sz * 4 * 3, note="LM head-sized add (256MB)")
    del lm_w1, lm_w2

    # ── 3. LAYERNORM ─────────────────────────────────────────────────────────

    ln_x = torch.randn(BT, D, device=device)
    ln_w = torch.randn(D, device=device)
    ln_b = torch.randn(D, device=device)

    ms = bench(lambda: F.layer_norm(ln_x, (D,), ln_w, ln_b, eps=1e-5), W, I, sync)
    record("layernorm_512x1024", ms, bytes_rw=BT * D * 4 * 3, note="LayerNorm fwd")

    # LayerNorm backward
    ln_x_req = ln_x.clone().requires_grad_(True)
    ln_out = F.layer_norm(ln_x_req, (D,), ln_w, ln_b, eps=1e-5)
    grad_out = torch.randn_like(ln_out)

    def ln_bwd():
        if ln_x_req.grad is not None:
            ln_x_req.grad.zero_()
        ln_out.backward(grad_out, retain_graph=True)

    ms = bench(ln_bwd, W, I, sync)
    record("layernorm_bwd_512x1024", ms, note="LayerNorm backward")
    del ln_x, ln_w, ln_b, ln_x_req, ln_out, grad_out

    # ── 4. SOFTMAX ───────────────────────────────────────────────────────────

    attn_scores = torch.randn(BH, T, T, device=device)
    ms = bench(lambda: F.softmax(attn_scores, dim=-1), W, I, sync)
    record("softmax_attn_16x512x512", ms,
           bytes_rw=BH * T * T * 4 * 2, note="attention softmax")

    logits_sm = torch.randn(BT, V, device=device)
    ms = bench(lambda: F.softmax(logits_sm, dim=-1), W, I, sync)
    record("softmax_logits_512x64000", ms,
           bytes_rw=BT * V * 4 * 2, note="output softmax")
    del attn_scores, logits_sm

    # ── 5. CROSS-ENTROPY ─────────────────────────────────────────────────────

    ce_logits = torch.randn(BT, V, device=device, requires_grad=True)
    ce_targets = torch.randint(0, V, (BT,), device=device)

    ms = bench(lambda: F.cross_entropy(ce_logits, ce_targets), W, I, sync)
    record("cross_entropy_fwd_512x64000", ms, note="CE loss forward")

    ce_loss = F.cross_entropy(ce_logits, ce_targets)

    def ce_bwd():
        if ce_logits.grad is not None:
            ce_logits.grad.zero_()
        ce_loss.backward(retain_graph=True)

    ms = bench(ce_bwd, W, I, sync)
    record("cross_entropy_bwd_512x64000", ms, note="CE loss backward")
    del ce_logits, ce_targets, ce_loss

    # ── 6. FLASH ATTENTION ───────────────────────────────────────────────────

    q = torch.randn(B, H, T, Dh, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, Dh, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, Dh, device=device, dtype=torch.float16)

    try:
        ms = bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            W, I, sync,
        )
        record("flash_attn_fwd_b1_h16_t512_d64", ms,
               flops=2 * B * H * T * T * Dh * 2, note="SDPA fwd (f16)")
    except Exception as exc:
        record("flash_attn_fwd_b1_h16_t512_d64", 0, note=f"FAILED: {exc}")

    # Flash attention backward
    try:
        q2 = torch.randn(B, H, T, Dh, device=device, dtype=torch.float16, requires_grad=True)
        k2 = torch.randn(B, H, T, Dh, device=device, dtype=torch.float16, requires_grad=True)
        v2 = torch.randn(B, H, T, Dh, device=device, dtype=torch.float16, requires_grad=True)
        out = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        dO = torch.randn_like(out)
        def flash_bwd():
            if q2.grad is not None:
                q2.grad.zero_()
                k2.grad.zero_()
                v2.grad.zero_()
            out.backward(dO, retain_graph=True)
        ms = bench(flash_bwd, W, I, sync)
        record("flash_attn_bwd_b1_h16_t512_d64", ms,
               flops=2 * B * H * T * T * Dh * 4, note="SDPA bwd (f16)")
        del q2, k2, v2, out, dO
    except Exception as exc:
        record("flash_attn_bwd_b1_h16_t512_d64", 0, note=f"FAILED: {exc}")

    del q, k, v

    # ── 7. EMBEDDING ─────────────────────────────────────────────────────────

    emb_w = torch.randn(V, D, device=device)
    indices = torch.randint(0, V, (BT,), device=device)

    ms = bench(lambda: F.embedding(indices, emb_w), W, I, sync)
    record("embedding_fwd_64000x1024", ms,
           bytes_rw=BT * D * 4, note="embedding lookup")

    # Embedding backward (scatter-add gradients)
    emb_w_req = emb_w.clone().requires_grad_(True)
    emb_out = F.embedding(indices, emb_w_req)
    grad_emb = torch.randn_like(emb_out)

    def emb_bwd():
        if emb_w_req.grad is not None:
            emb_w_req.grad.zero_()
        emb_out.backward(grad_emb, retain_graph=True)

    ms = bench(emb_bwd, W, I, sync)
    record("embedding_bwd_64000x1024", ms,
           bytes_rw=BT * D * 4 + V * D * 4, note="embedding backward (scatter-add)")
    del emb_w, emb_w_req, emb_out, grad_emb, indices

    # ── 8. ADAMW STEP ────────────────────────────────────────────────────────

    # Simulate AdamW for one layer's worth of params (~8.5M)
    p_size = 1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024  # ~8.5M
    param = torch.randn(p_size, device=device)
    grad = torch.randn(p_size, device=device)
    m = torch.zeros(p_size, device=device)
    v_state = torch.zeros(p_size, device=device)

    def adamw_step():
        nonlocal m, v_state
        m.mul_(0.9).add_(grad, alpha=0.1)
        v_state.mul_(0.999).addcmul_(grad, grad, value=0.001)
        m_hat = m / (1 - 0.9)
        v_hat = v_state / (1 - 0.999)
        param.addcdiv_(m_hat, v_hat.sqrt().add_(1e-8), value=-3e-4)
        param.mul_(1 - 3e-4 * 0.1)  # weight decay

    ms = bench(adamw_step, W, I, sync)
    record("adamw_step_8.5M", ms,
           bytes_rw=p_size * 4 * 7, note="AdamW for 1 layer (~8.5M params)")
    del param, grad, m, v_state

    # ── 8b. GRADIENT ACCUMULATION ─────────────────────────────────────────
    lm_grad = torch.randn(1024, V, device=device)
    lm_acc = torch.randn(1024, V, device=device)
    lm_size = 1024 * V

    ms = bench(lambda: lm_acc.add_(lm_grad), W, I, sync)
    record("grad_accum_lm_head", ms,
           bytes_rw=lm_size * 4 * 3, note="gradient accumulation (add_inplace)")

    ms = bench(lambda: lm_acc.mul_(0.5), W, I, sync)
    record("grad_scale_lm_head", ms,
           bytes_rw=lm_size * 4 * 2, note="gradient clipping scale (scale_inplace)")
    del lm_grad, lm_acc

    # ── 8b2. DROPOUT MASK GENERATION ────────────────────────────────────
    drop_shape = (BT, D)
    drop_p = torch.full(drop_shape, 0.9, device=device)  # keep_prob = 1-0.1 = 0.9
    ms = bench(lambda: torch.bernoulli(drop_p) / 0.9, W, I, sync)
    record("dropout_mask_512x1024", ms,
           bytes_rw=BT * D * 4, note="GPU-side dropout mask")
    del drop_p

    # ── 8c. GRADIENT NORM (sum of squares) ────────────────────────────────
    p_size_norm = 1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024
    grad_norm_t = torch.randn(p_size_norm, device=device)
    ms = bench(lambda: (grad_norm_t * grad_norm_t).sum(), W, I, sync)
    record("grad_norm_8.5M", ms,
           bytes_rw=p_size_norm * 4, note="sum of squares for gradient norm")
    del grad_norm_t

    # ── 9. FUSED OPS ────────────────────────────────────────────────────────

    # Residual + dropout + add
    residual = torch.randn(BT, D, device=device)
    projected = torch.randn(BT, D, device=device)
    mask = (torch.rand(BT, D, device=device) > 0.1).float() / 0.9

    ms = bench(lambda: residual + projected * mask, W, I, sync)
    record("residual_dropout_add_512x1024", ms,
           bytes_rw=BT * D * 4 * 4, note="residual + projected * mask")
    del residual, projected, mask

    # ── 10. KERNEL LAUNCH OVERHEAD ───────────────────────────────────────────

    small = torch.randn(64, 64, device=device)
    small2 = torch.randn(64, 64, device=device)

    # 200 sequential small adds
    def many_small_adds():
        r = small
        for _ in range(200):
            r = r + small2
        return r

    sync()
    t0 = time.perf_counter()
    for _ in range(I):
        many_small_adds()
    sync()
    total = (time.perf_counter() - t0) * 1000
    per_add = total / (I * 200)
    record("launch_overhead_200_adds", total / I,
           note=f"200 sequential 64x64 adds, {per_add:.3f}ms/add")

    del small, small2

    # ── 11. TRANSPOSE ────────────────────────────────────────────────────────

    t_in = torch.randn(BH, T, Dh, device=device)
    ms = bench(lambda: t_in.transpose(1, 2).contiguous(), W, I, sync)
    record("transpose_16x512x64", ms,
           bytes_rw=BH * T * Dh * 4 * 2, note="transpose + contiguous")
    del t_in

    # ── 12. SLICE ────────────────────────────────────────────────────────────

    qkv = torch.randn(BT, 3 * D, device=device)
    ms = bench(lambda: (qkv[:, :D], qkv[:, D:2*D], qkv[:, 2*D:]), W, I, sync)
    record("slice_qkv_512x3072", ms, note="3-way slice for Q,K,V")
    del qkv

    # ── 13. END-TO-END FORWARD PASS (single layer, no autograd) ─────────

    with torch.no_grad():
        ln_w1 = torch.randn(D, device=device)
        ln_b1 = torch.randn(D, device=device)
        wqkv = torch.randn(3 * D, D, device=device)
        wo = torch.randn(D, D, device=device)
        ln_w2 = torch.randn(D, device=device)
        ln_b2 = torch.randn(D, device=device)
        w_gate = torch.randn(FFN, D, device=device)
        w_up = torch.randn(FFN, D, device=device)
        w_proj = torch.randn(D, FFN, device=device)
        x_in = torch.randn(BT, D, device=device)

        def one_layer_fwd():
            # Attention sub-layer
            h = F.layer_norm(x_in, (D,), ln_w1, ln_b1)
            qkv_out = h @ wqkv.T
            q, kk, vv = qkv_out.split(D, dim=-1)
            q = q.view(B, T, H, Dh).transpose(1, 2).contiguous().half()
            kk = kk.view(B, T, H, Dh).transpose(1, 2).contiguous().half()
            vv = vv.view(B, T, H, Dh).transpose(1, 2).contiguous().half()
            attn = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            attn = attn.float().transpose(1, 2).contiguous().view(BT, D)
            out = attn @ wo.T
            x = x_in + out
            # MLP sub-layer
            h = F.layer_norm(x, (D,), ln_w2, ln_b2)
            gate = F.silu(h @ w_gate.T)
            up = h @ w_up.T
            x = x + (gate * up) @ w_proj.T
            return x

        ms = bench(one_layer_fwd, W, I, sync)
        record("single_layer_fwd", ms, note="full transformer layer (attn+mlp) no autograd")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
