# Checkpoint catalog and preservation contract

## Canonical native recovery points

| Native file in archive | Step | Meaning | SHA-256 |
|---|---:|---|---|
| checkpoints/base-pretrain-step-61036.alph | 61,036 pretrain | completed 1,000,013,824-token base | 08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049 |
| checkpoints/sft-best-retained-step-29000.alph | 29,000 SFT | best held-out loss among surviving full SFT checkpoints, 1.6412250 | 03eaac3e7be06e8fb5720415a334b36d7ef5019fcff72ca9227636b84011a7f3 |
| checkpoints/sft-terminal-step-30322.alph | 30,322 SFT | exact one-epoch terminal continuation state, held-out 1.6439665 | 6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8 |

Each ALPH file contains model parameters, AdamW tensors and optimizer step, RNG state, tokenizer
artifacts, model configuration, and training step. These are the only files in the release that can
continue native Alpha training without reconstructing optimizer state.

## The unavailable 28,500 point

Step 28,500 recorded the sharper validation value 1.3334526, but it was an evaluation-only half
checkpoint. No native checkpoint was written. Never describe it as recoverable or best-checkpoint
weights.

## Selection guidance

- Use the base step 61,036 checkpoint for a clean repaired SFT recipe. This is the preferred scientific
  choice because it avoids inheriting the unshuffled SFT trajectory.
- Use SFT step 30,322 only when the experiment explicitly studies continuation from the terminal state.
- Use SFT step 29,000 only for a declared branch from the slightly better surviving full validation
  point.
- Never choose solely from teacher-forced validation loss. A future selector must include sealed
  generation behavior and answer-start metrics.

## Storage locations

Local hardlink bundle:

    /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/

Public immutable archive:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints
    revision 7198d1a1f094ffe88d06399ea99fecbd78fa8b66

The local files are hard links to canonical mounted-drive artifacts. They protect against path-level
cleanup without duplicating roughly 2.1 GB of blocks. Deleting either hardlink name does not delete the
other immediately, but no archive path should be removed without a separately authorized retention
review.

## Verification

Verify the entire local bundle:

    cd /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730
    nice -n 19 ionice -c 3 sha256sum -c MANIFEST.sha256

Verify one selected checkpoint after transfer:

    sha256sum checkpoints/sft-terminal-step-30322.alph

Compare that exact value with the table above before a launcher can see the file.

## Public inference export is not a recovery checkpoint

The standard model at ajaxdavis/alpha-60m-chat contains model.safetensors and tokenizer/config files.
Its weights SHA-256 is:

    6bb349085512c45fe5cf732209a82a5c5196d2d7a12f0aea16bdb042546dca92

It intentionally omits optimizer and RNG state and must not be presented as a lossless training resume.
It is useful for standard Transformers inference and independent parity checks only.

## Preservation rules

- Keep MANIFEST.sha256 with every local or remote bundle.
- Keep contracts, metrics, audits, and failed evaluations beside the weights.
- Never overwrite a checkpoint filename with replayed or converted bytes.
- Record transfers by exact byte count and SHA-256 before pruning any source copy.
- Preserve abandoned metric tails under distinct evidence names; do not splice them into canonical
  trajectories.
- Do not upload secret files, local webhook configuration, SSH keys, or RunPod credentials.
