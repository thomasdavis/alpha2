# ── Build stage ────────────────────────────────────────────────────────────
FROM node:22-slim AS build
ARG CACHE_BUST=2
WORKDIR /app

COPY package.json package-lock.json turbo.json tsconfig.base.json ./
COPY apps/web/ apps/web/
COPY packages/core/ packages/core/
COPY packages/db/ packages/db/
COPY packages/tensor/ packages/tensor/
COPY packages/autograd/ packages/autograd/
COPY packages/tokenizers/ packages/tokenizers/
COPY packages/model/ packages/model/
COPY packages/train/ packages/train/
COPY packages/helios/ packages/helios/

RUN npm ci
RUN npx turbo build --filter=@alpha/web...

# Download checkpoint from HF model repo using huggingface_hub (handles Xet/LFS correctly)
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages huggingface_hub
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ajaxdavis/alpha-v0-historic', local_dir='/app/model-files')"
RUN ls -lh /app/model-files/

# ── Production stage ──────────────────────────────────────────────────────
FROM node:22-slim

WORKDIR /app

COPY --from=build --chown=node /app/apps/web/.next/standalone ./
COPY --from=build --chown=node /app/apps/web/.next/static apps/web/.next/static

# Checkpoint baked into image
RUN mkdir -p /app/outputs/20260224064526_4kwo && chown -R node:node /app/outputs
COPY --from=build --chown=node /app/model-files/checkpoint-5000.json /app/outputs/20260224064526_4kwo/checkpoint-5000.json
COPY --from=build --chown=node /app/model-files/config.json /app/outputs/20260224064526_4kwo/config.json

ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV OUTPUTS_DIR=/app/outputs

USER node
EXPOSE 3000

CMD ["node", "apps/web/server.js"]
