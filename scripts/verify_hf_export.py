#!/usr/bin/env python3
"""
G3 golden-token test: prove Alpha's own cpu_ref forward == the exported
safetensors loaded in HuggingFace `transformers`, with ZERO custom code.

This validates RoPE (rotate_half convention), RMSNorm eps, SwiGLU, tied
embeddings, and the safetensors weight orientation / wqkv split IN ONE SHOT —
the from-scratch equivalent of "reproduce a pretrained checkpoint's logits",
needing no foreign weights.

Setup (one-time, into the box uv venv; torch-cpu ~200MB):
    cd /mnt/donto-data/alpha-corpora
    uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv transformers safetensors numpy

Inputs:
    --export-dir     the HF model repo written by `alpha export-hf`
    --alpha-logits   JSON written by `alpha logits --json --out=...`
                     ({config, prompts:[{prompt, tokens, logits:[[..]], top1}]})

PASS (exit 0): top-1 agreement 100% of positions AND max abs logit diff < --tol
               (default 1e-3, fp32 both sides).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Alpha ↔ transformers golden-token verifier")
    ap.add_argument("--export-dir", required=True, help="HF model dir from `alpha export-hf`")
    ap.add_argument("--alpha-logits", required=True, help="JSON from `alpha logits --json --out=`")
    ap.add_argument("--tol", type=float, default=1e-3, help="max abs logit diff threshold")
    ap.add_argument("--json-out", help="optional machine-readable parity report")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print(f"[verify] missing deps ({e}).", file=sys.stderr)
        print("[verify] install: uv pip install --python .venv torch "
              "--index-url https://download.pytorch.org/whl/cpu ; "
              "uv pip install --python .venv transformers safetensors numpy", file=sys.stderr)
        return 2

    torch.manual_seed(0)

    with open(args.alpha_logits) as f:
        dump = json.load(f)
    prompts = dump["prompts"]
    acfg = dump.get("config", {})
    print(f"[verify] alpha config: {json.dumps(acfg)}")

    # Zero custom code: plain from_pretrained on the exported dir.
    print(f"[verify] loading {args.export_dir} in transformers (fp32, no trust_remote_code) ...")
    try:
        # transformers >= 5 renamed torch_dtype → dtype.
        model = AutoModelForCausalLM.from_pretrained(args.export_dir, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.export_dir, torch_dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.export_dir)

    # Sanity: the arch really is stock LlamaForCausalLM.
    arch = model.config.architectures[0] if model.config.architectures else type(model).__name__
    print(f"[verify] loaded {type(model).__name__} (config.architectures={model.config.architectures}) "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    assert "Llama" in arch, f"expected a Llama arch, got {arch}"

    total_pos = 0
    total_top1_match = 0
    global_max_abs = 0.0
    tok_mismatches = 0
    per_prompt = []

    for pi, p in enumerate(prompts):
        toks = p["tokens"]
        if not toks:
            continue
        alpha_logits = np.asarray(p["logits"], dtype=np.float64)  # [T, V]
        T, V = alpha_logits.shape

        # Feed the SAME token ids Alpha used → isolates the MODEL export from any
        # tokenizer discrepancy. (Tokenizer parity checked separately below.)
        input_ids = torch.tensor([toks], dtype=torch.long)
        with torch.no_grad():
            hf_logits = model(input_ids).logits[0].to(torch.float64).cpu().numpy()  # [T, V]

        assert hf_logits.shape == alpha_logits.shape, \
            f"shape {hf_logits.shape} != {alpha_logits.shape}"

        abs_diff = np.abs(hf_logits - alpha_logits)
        max_abs = float(abs_diff.max())
        alpha_top1 = alpha_logits.argmax(axis=-1)
        hf_top1 = hf_logits.argmax(axis=-1)
        top1_match = int((alpha_top1 == hf_top1).sum())

        # Tokenizer parity: does the exported HF tokenizer reproduce Alpha's ids?
        hf_ids = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        tok_ok = (hf_ids == toks)
        if not tok_ok:
            tok_mismatches += 1

        total_pos += T
        total_top1_match += top1_match
        global_max_abs = max(global_max_abs, max_abs)
        per_prompt.append((pi, T, top1_match, max_abs, tok_ok))
        print(f"[verify] prompt {pi}: T={T} top1={top1_match}/{T} "
              f"max|Δlogit|={max_abs:.3e} tokenizer_parity={'ok' if tok_ok else 'MISMATCH'}")

    if total_pos == 0:
        print("[verify] no non-empty prompts", file=sys.stderr)
        return 2

    top1_agree = total_top1_match / total_pos
    print("\n[verify] ==== SUMMARY ====")
    print(f"[verify] positions            : {total_pos}")
    print(f"[verify] top-1 agreement      : {total_top1_match}/{total_pos} = {top1_agree*100:.2f}%")
    print(f"[verify] max abs logit diff   : {global_max_abs:.3e}  (threshold {args.tol:.1e})")
    print(f"[verify] tokenizer parity     : {len(prompts)-tok_mismatches}/{len(prompts)} prompts exact")

    passed = (total_top1_match == total_pos) and (global_max_abs < args.tol) and tok_mismatches == 0
    print(f"[verify] RESULT               : {'PASS ✅' if passed else 'FAIL ❌'}")
    if args.json_out:
        export_dir = Path(args.export_dir).resolve()
        alpha_logits = Path(args.alpha_logits).resolve()
        report = {
            "schema": "alpha-hf-export-parity-v1",
            "status": "PASS" if passed else "FAIL",
            "export": {
                "path": str(export_dir),
                "model_sha256": sha256_file(export_dir / "model.safetensors"),
                "tokenizer_sha256": sha256_file(export_dir / "tokenizer.json"),
                "config_sha256": sha256_file(export_dir / "config.json"),
            },
            "alpha_logits": {
                "path": str(alpha_logits),
                "sha256": sha256_file(alpha_logits),
                "prompts": len(prompts),
            },
            "requirements": {
                "top1_exact": True,
                "tokenizer_exact": True,
                "max_abs_logit_tolerance": args.tol,
            },
            "observed": {
                "positions": total_pos,
                "top1_matches": total_top1_match,
                "tokenizer_matches": len(prompts) - tok_mismatches,
                "max_abs_logit_delta": global_max_abs,
            },
        }
        output_path = Path(args.json_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    if not passed:
        if total_top1_match != total_pos:
            print("[verify]   → top-1 disagreement: check RoPE convention / weight orientation", file=sys.stderr)
        if global_max_abs >= args.tol:
            print("[verify]   → logit drift: check rms_norm_eps, rope_theta, head split", file=sys.stderr)
        if tok_mismatches:
            print("[verify]   → tokenizer drift: exported tokenizer does not reproduce Alpha token ids", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
