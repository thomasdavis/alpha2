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
        -> terminal ALPH checkpoint step 30,322

Space:

    https://huggingface.co/spaces/ajaxdavis/alpha-60m-chat
    revision be0bd0428631d1585b13ddf9e93a8ed2d9254606

Model:

    https://huggingface.co/ajaxdavis/alpha-60m-chat
    revision b481f46924b7a4777a029de1ffb44c06cc925d4c

Checkpoint source:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints
    revision 7198d1a1f094ffe88d06399ea99fecbd78fa8b66

## Runtime source

- Application source: apps/hf/src/
- Unit source: apps/hf/alpha2-hf-backend.service
- Optional paid Docker Space: apps/hf/Dockerfile.space
- Public static page builder: scripts/build_hf_static_space.ts
- Static Space publisher: scripts/publish_hf_space.py
- Runtime source commit: 5bd723db49b15df1b80a279a016c68727270bacc

The installed backend bundle is /home/ajax/bin/alpha-60m-space-server.mjs. Its recorded SHA-256 is:

    f36416475ee8ab3f321e38064275b914c0aee4267c8cdda262aef5315913208c

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
- checkpoint step 30,322;
- quality_gate FAIL.

Do not use generation output as a routine service heartbeat. Health and metadata endpoints avoid
confusing operational checks with new model evaluation.

## Restart procedure

If the backend is unhealthy:

1. Inspect bounded logs:

       journalctl -u alpha2-hf-backend.service -n 200 --no-pager

2. Verify the unit still points to the archived terminal checkpoint.
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
- maximum context is 1,024 tokens and maximum completion is 256 tokens.

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
