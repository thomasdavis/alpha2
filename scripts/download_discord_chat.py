#!/usr/bin/env python3
"""
Download aaronmoo12/Discord-Dialogues from HuggingFace and convert to chat training format.

Output format (one conversation per line):
<|user|> message text <|assistant|> response text <|user|> follow up...

Filters:
- Only multi-turn conversations (3+ turns)
- Basic NSFW/slur filter
- URL removal
- Whitespace normalization
- Up to 200,000 conversations
"""

import re
import os
import sys
from datasets import load_dataset

TARGET_COUNT = 200_000
MIN_TURNS = 3
OUTPUT_PATH = "/home/ajax/repos/models/alpha/data/discord_chat.txt"

# Basic profanity/slur filter - common slurs and NSFW terms
BLOCKED_WORDS = {
    "nigger", "nigga", "faggot", "fag", "retard", "retarded",
    "kike", "chink", "spic", "wetback", "tranny", "shemale",
    "coon", "darkie", "gook", "beaner", "towelhead", "raghead",
    "dyke", "homo",
    "nsfw", "hentai", "porn", "pornhub", "xvideos", "xxx",
    "dick pic", "nude", "nudes", "sexting",
}

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r'\s+')


def contains_blocked(text):
    """Check if text contains any blocked words."""
    lower = text.lower()
    for word in BLOCKED_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', lower):
            return True
    return False


def clean_text(text):
    """Remove URLs, normalize whitespace."""
    text = URL_PATTERN.sub('', text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    return text.strip()


def parse_chatml(content):
    """
    Parse ChatML format into list of (role, message) tuples.
    
    ChatML format:
    <|im_start|>user
    message<|im_end|>
    <|im_start|>assistant
    response<|im_end|>
    """
    turns = []
    parts = content.split("<|im_start|>")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = part.replace("<|im_end|>", "").strip()
        lines = part.split("\n", 1)
        if len(lines) < 2:
            continue
        role = lines[0].strip().lower()
        message = lines[1].strip()
        if role in ("user", "assistant") and message:
            turns.append((role, message))
    return turns


def convert_to_our_format(turns):
    """Convert turns to our format: <|user|> msg <|assistant|> msg ..."""
    parts = []
    for role, message in turns:
        cleaned = clean_text(message)
        if not cleaned:
            return ""
        parts.append(f"<|{role}|> {cleaned}")
    return " ".join(parts)


def main():
    print("Loading aaronmoo12/Discord-Dialogues from HuggingFace...")
    print("(This may take a while for the initial download)")
    sys.stdout.flush()
    
    ds = load_dataset("aaronmoo12/Discord-Dialogues", split="train", streaming=True)
    
    conversations = []
    total_seen = 0
    skipped_short = 0
    skipped_nsfw = 0
    skipped_empty = 0
    
    for example in ds:
        total_seen += 1
        
        if total_seen % 50_000 == 0:
            print(f"  Processed {total_seen:,} examples, kept {len(conversations):,} so far...")
            sys.stdout.flush()
        
        # Get the conversation text - try common field names
        text = ""
        for key in ["text", "content", "conversation", "dialogue"]:
            val = example.get(key, "")
            if val and "<|im_start|>" in str(val):
                text = str(val)
                break
        
        # If no ChatML found, try first string field
        if not text:
            for key in example:
                if isinstance(example[key], str) and len(example[key]) > 50:
                    text = example[key]
                    break
        
        if not text:
            skipped_empty += 1
            continue
        
        # Parse ChatML
        turns = parse_chatml(text)
        
        # Filter: minimum turns
        if len(turns) < MIN_TURNS:
            skipped_short += 1
            continue
        
        # Filter: NSFW/profanity
        full_text = " ".join(msg for _, msg in turns)
        if contains_blocked(full_text):
            skipped_nsfw += 1
            continue
        
        # Convert to our format
        line = convert_to_our_format(turns)
        if not line:
            skipped_empty += 1
            continue
        
        conversations.append(line)
        
        if len(conversations) >= TARGET_COUNT:
            print(f"  Reached target of {TARGET_COUNT:,} conversations.")
            break
    
    print(f"\nProcessing complete:")
    print(f"  Total examples seen: {total_seen:,}")
    print(f"  Kept: {len(conversations):,}")
    print(f"  Skipped (< {MIN_TURNS} turns): {skipped_short:,}")
    print(f"  Skipped (NSFW/slurs): {skipped_nsfw:,}")
    print(f"  Skipped (empty/parse error): {skipped_empty:,}")
    
    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv + "\n")
    
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"  Conversations: {len(conversations):,}")
    
    # Calculate average turns
    total_turns = 0
    for conv in conversations:
        total_turns += conv.count("<|user|>") + conv.count("<|assistant|>")
    avg_turns = total_turns / len(conversations) if conversations else 0
    print(f"  Average turns per conversation: {avg_turns:.1f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
