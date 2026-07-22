#!/usr/bin/env python3
"""
Cross-check the exported HF tokenizer.json against Alpha's own ByteBpeTokenizer.

Loads the exported `tokenizer.json` with the HuggingFace `tokenizers` library and
asserts it produces IDENTICAL ids to Alpha's encode() on a large document sample.
This proves the exported artifact is a faithful, zero-custom-code drop-in.

Inputs are produced by the TS driver:
    npx tsx scripts/export_and_dump_for_verify.ts --out=<dir> --docs=10000 --vocab=12288
which writes  <dir>/tokenizer.json  and  <dir>/samples.jsonl  (one {"text","ids"} per line).

If <dir>/samples.jsonl is missing, this script runs that driver for you.

Setup on the box (venv already exists):
    . /mnt/donto-data/alpha-corpora/.venv/bin/activate
    uv pip install tokenizers
    python3 scripts/verify_tokenizer_export.py --dir=/tmp/alpha-tok-verify

Exit code 0 = all ids matched; 1 = mismatches or setup error.
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/alpha-tok-verify",
                    help="dir containing tokenizer.json + samples.jsonl")
    ap.add_argument("--docs", type=int, default=10000, help="doc count if driver must run")
    ap.add_argument("--vocab", type=int, default=12288, help="vocab size if driver must run")
    ap.add_argument("--max-report", type=int, default=10, help="max mismatches to print")
    args = ap.parse_args()

    tok_path = os.path.join(args.dir, "tokenizer.json")
    samples_path = os.path.join(args.dir, "samples.jsonl")

    if not (os.path.exists(tok_path) and os.path.exists(samples_path)):
        print(f"[verify] inputs missing in {args.dir}; running TS driver...")
        cmd = ["npx", "tsx", "scripts/export_and_dump_for_verify.ts",
               f"--out={args.dir}", f"--docs={args.docs}", f"--vocab={args.vocab}"]
        subprocess.run(cmd, cwd=REPO, check=True)

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("[verify] ERROR: the 'tokenizers' package is not installed.\n"
              "  . /mnt/donto-data/alpha-corpora/.venv/bin/activate && uv pip install tokenizers",
              file=sys.stderr)
        return 1

    tok = Tokenizer.from_file(tok_path)
    print(f"[verify] loaded {tok_path} (vocab_size={tok.get_vocab_size()})")

    total = 0
    mismatches = 0
    first_divergences = []
    with open(samples_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text, ours = rec["text"], rec["ids"]
            total += 1
            hf_ids = tok.encode(text, add_special_tokens=True).ids
            if hf_ids != ours:
                mismatches += 1
                if len(first_divergences) < args.max_report:
                    # locate first divergent position
                    j = 0
                    while j < min(len(ours), len(hf_ids)) and ours[j] == hf_ids[j]:
                        j += 1
                    first_divergences.append({
                        "text_head": text[:80],
                        "our_len": len(ours),
                        "hf_len": len(hf_ids),
                        "first_diff_pos": j,
                        "ours_ctx": ours[max(0, j - 3):j + 4],
                        "hf_ctx": hf_ids[max(0, j - 3):j + 4],
                    })

    print(f"[verify] compared {total} docs: {total - mismatches} match, {mismatches} mismatch")
    if mismatches:
        for d in first_divergences:
            print("  MISMATCH:", json.dumps(d, ensure_ascii=False))
        print(f"[verify] FAILED ({mismatches}/{total} docs differ)")
        return 1
    print(f"[verify] PASS: all {total} docs produce identical ids "
          f"({100.0:.2f}% agreement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
