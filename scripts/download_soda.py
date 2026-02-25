#!/usr/bin/env python3
"""Download SODA dataset from Hugging Face and format as chat dataset.

Streams data to disk to avoid OOM. Outputs one conversation per line with
<|user|> and <|assistant|> turn markers, matching the dailydialog.txt format.
"""

import os
import sys

def main():
    from datasets import load_dataset

    output_soda = "/home/ajax/repos/models/alpha/data/soda.txt"
    output_chat = "/home/ajax/repos/models/alpha/data/chat.txt"
    dailydialog = "/home/ajax/repos/models/alpha/data/dailydialog.txt"

    # Download and process SODA train split (streaming to avoid loading all into RAM)
    print("Loading SODA train split...")
    ds = load_dataset("allenai/soda", split="train", trust_remote_code=True)

    total_convos = 0
    total_turns = 0
    skipped = 0

    print(f"Processing {len(ds)} SODA dialogues...")
    with open(output_soda, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            dialogue = example.get("dialogue", [])
            if not dialogue or len(dialogue) < 2:
                skipped += 1
                continue

            # Build conversation with alternating user/assistant markers
            parts = []
            for j, utterance in enumerate(dialogue):
                utterance = utterance.strip()
                if not utterance:
                    continue
                if j % 2 == 0:
                    parts.append(f"<|user|> {utterance}")
                else:
                    parts.append(f"<|assistant|> {utterance}")
                total_turns += 1

            if len(parts) >= 2:
                line = " ".join(parts)
                f.write(line + "\n")
                total_convos += 1

            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1} examples ({total_convos} conversations so far)")

    print(f"SODA: {total_convos} conversations, {total_turns} turns, {skipped} skipped")

    # Count dailydialog stats
    dd_convos = 0
    dd_turns = 0
    with open(dailydialog, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dd_convos += 1
            dd_turns += line.count("<|user|>") + line.count("<|assistant|>")

    print(f"DailyDialog: {dd_convos} conversations, {dd_turns} turns")

    # Concatenate: dailydialog first, then soda
    print("Concatenating into chat.txt...")
    with open(output_chat, "w", encoding="utf-8") as out:
        with open(dailydialog, "r", encoding="utf-8") as f:
            for line in f:
                out.write(line)
        with open(output_soda, "r", encoding="utf-8") as f:
            for line in f:
                out.write(line)

    # Final stats
    chat_size = os.path.getsize(output_chat)
    soda_size = os.path.getsize(output_soda)

    total_all_convos = dd_convos + total_convos
    total_all_turns = dd_turns + total_turns

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"SODA conversations:       {total_convos:>10,}")
    print(f"SODA turns:               {total_turns:>10,}")
    print(f"SODA file size:           {soda_size:>10,} bytes ({soda_size / 1024 / 1024:.1f} MB)")
    print(f"DailyDialog conversations:{dd_convos:>10,}")
    print(f"DailyDialog turns:        {dd_turns:>10,}")
    print(f"---")
    print(f"Combined conversations:   {total_all_convos:>10,}")
    print(f"Combined turns:           {total_all_turns:>10,}")
    print(f"Combined file size:       {chat_size:>10,} bytes ({chat_size / 1024 / 1024:.1f} MB)")
    print(f"Output: {output_chat}")
    print("=" * 60)

    # Clean up intermediate soda.txt
    os.remove(output_soda)
    print(f"Cleaned up intermediate file: {output_soda}")


if __name__ == "__main__":
    main()
