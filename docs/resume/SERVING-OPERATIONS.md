# Public serving operations

The public service exists to make the failed-quality artifact inspectable. It is not a claim that the
model is useful and it must never substitute another model.

## Architecture

    Hugging Face static Space
        -> https://donto.org/alpha-60m
        -> Caddy reverse proxy
        -> 127.0.0.1:7860
        -> alpha2-hf-backend.service
        -> native Alpha CPU inference
        -> selected native corrective checkpoint step 1,200

Space:

    https://huggingface.co/spaces/ajaxdavis/alpha-60m-chat
    revision d87e0950baf0a16ccd2859c2cee6314602ba2881

Model:

    https://huggingface.co/ajaxdavis/alpha-60m-chat
    revision ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0

Checkpoint source:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints
    revision ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60

Repair v2 did not pass selection and was deliberately not deployed. Its recovery-only revision
`c1117378c0bc8b81b408be09c000f80ea9f027d7` adds rejected optimizer-bearing branch states to the training
archive; it does not change the model, Space, backend checkpoint, or quality claim in this document.

## Runtime source

- Application source: apps/hf/src/
- Unit source: apps/hf/alpha2-hf-backend.service
- Optional paid Docker Space: apps/hf/Dockerfile.space
- Public static page builder: scripts/build_hf_static_space.ts
- Static Space publisher: scripts/publish_hf_space.py
- Runtime source commit: `e55cb23d894ff5b7eeb818428ffe9bc0ea76490c`

The installed backend bundle is /home/ajax/bin/alpha-60m-space-server.mjs. Its recorded SHA-256 is:

    c2bd8a24387584cf0eae11082adef235e62a7d12b901c749e5ddd23b18b642f4

The adjacent commit sidecar identifies the runtime source commit, not later documentation changes.

## Read-only health checks

    systemctl is-enabled alpha2-hf-backend.service
    systemctl is-active alpha2-hf-backend.service
    systemctl show alpha2-hf-backend.service \
      -p ActiveState -p SubState -p NRestarts -p MemoryCurrent -p MemoryPeak
    curl -fsS https://donto.org/alpha-60m/health
    curl -fsS https://donto.org/alpha-60m/evidence
    curl -fsS https://donto.org/alpha-60m/v1/models
    curl -fsSI https://ajaxdavis-alpha-60m-chat.static.hf.space/index.html

Health must report:

- model ajaxdavis/alpha-60m-chat;
- 57,688,576 parameters;
- checkpoint step 1,200;
- quality_gate FAIL.

The evidence endpoint must additionally report 55/100 structural, 30/100 empty, 31/100 loops, QA 0/200, and
checkpoint SHA `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`.

Do not use generation output as a routine service heartbeat. Health and metadata endpoints avoid
confusing operational checks with new model evaluation.

## Restart procedure

If the backend is unhealthy:

1. Inspect bounded logs:

       journalctl -u alpha2-hf-backend.service -n 200 --no-pager

2. Verify the unit still points to the selected step-1,200 checkpoint.
3. Verify the checkpoint hash before restart.
4. Validate Caddy configuration if routing changed.
5. Restart only the backend:

       sudo systemctl restart alpha2-hf-backend.service

6. Recheck ActiveState, NRestarts, memory, public health, CORS, and static page HTTP status.

Do not restart unrelated donto services as a first response.

## Resource and security contract

- loopback-only application listener;
- Caddy is the public boundary;
- service runs at nice 19 and idle I/O priority;
- CPU quota is 100 percent;
- memory maximum is 3 GB;
- checkpoint is read-only;
- no webhook, HF token, RunPod credential, or SSH key is required at inference time;
- public responses carry X-Alpha-Quality-Gate: FAIL;
- CORS is intentionally open for the static Hugging Face origin;
- maximum context is 512 tokens and maximum completion is 256 tokens.

## Expected empty response

An HTTP 200 response with empty assistant content, finish_reason stop, and alpha.empty_eos=true means
the model selected EOS immediately. It is not a network failure.

The UI must continue displaying an explicit empty-response message. Do not:

- replace empty output with canned text;
- retry until a nonempty sample appears;
- silently raise temperature;
- call another model;
- suppress EOS for a minimum length.

## Rebuild and publication boundary

Code changes require the normal TypeScript build and focused tests. A Space update must create a new
immutable Hub revision and update evidence. Do not rewrite the archived revision in documentation.

The current account lacked paid Docker Space entitlement, so the public Space is static and calls the
OVH backend. If entitlement later exists, the Dockerfile remains a reproducible option, but migration
must preserve exact checkpoint identity, API behavior, quality warnings, and browser verification.

## 2026-07-31 cutover proof

Canonical runtime and browser evidence:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/

The public API returned “It's going well, thank you. How about you?” with `finish_reason=stop` and
`X-Alpha-Quality-Gate: FAIL`. The static Space was exercised in a real browser at desktop and 390-pixel
viewports. The final narrow view had no horizontal overflow, one main landmark, the shortened mobile status
“Quality gate · fail,” and no browser errors. Screenshots are under `public/browser/`.
