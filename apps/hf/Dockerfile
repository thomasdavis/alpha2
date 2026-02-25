# ── Build stage ────────────────────────────────────────────────────────────
FROM node:22-slim AS build
WORKDIR /app

COPY package.json package-lock.json turbo.json tsconfig.base.json ./
COPY packages/core/ packages/core/
COPY packages/tokenizers/ packages/tokenizers/
COPY apps/hf/ apps/hf/

RUN npm ci
RUN npx turbo build --filter=@alpha/hf...

# Download checkpoint from HF model repo (huggingface_hub handles Xet/LFS)
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages huggingface_hub
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ajaxdavis/alpha-v0-historic', local_dir='/app/model-files')"

# ── Production stage ──────────────────────────────────────────────────────
FROM node:22-slim
WORKDIR /app

COPY --from=build --chown=node /app/apps/hf/dist/index.js ./server.js
COPY --from=build --chown=node /app/model-files/checkpoint-5000.json ./checkpoint.json

ENV CHECKPOINT_PATH=/app/checkpoint.json
ENV PORT=7860

USER node
EXPOSE 7860

CMD ["node", "server.js"]
