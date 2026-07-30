# ── Build stage ────────────────────────────────────────────────────────────
FROM node:22-slim AS build
WORKDIR /app

COPY package.json package-lock.json turbo.json tsconfig.base.json ./
COPY packages/core/ packages/core/
COPY packages/inference/ packages/inference/
COPY packages/tokenizers/ packages/tokenizers/
COPY apps/hf/ apps/hf/

RUN npm ci
RUN npx turbo build --filter=@alpha/hf...

# Download the exact native terminal checkpoint from the public recovery archive.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages --no-cache-dir huggingface_hub
ARG CHECKPOINT_REVISION=main
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ajaxdavis/alpha-60m-training-checkpoints', filename='checkpoints/sft-terminal-step-30322.alph', revision='${CHECKPOINT_REVISION}', local_dir='/app/model-files')"
RUN echo "6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8  /app/model-files/checkpoints/sft-terminal-step-30322.alph" | sha256sum -c -

# ── Production stage ──────────────────────────────────────────────────────
FROM node:22-slim
WORKDIR /app

COPY --from=build --chown=node /app/apps/hf/dist/index.js ./server.js
COPY --from=build --chown=node /app/model-files/checkpoints/sft-terminal-step-30322.alph ./checkpoint.alph

ENV CHECKPOINT_PATH=/app/checkpoint.alph
ENV PORT=7860

USER node
EXPOSE 7860

CMD ["node", "server.js"]
