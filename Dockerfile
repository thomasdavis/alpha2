# ── Build stage ────────────────────────────────────────────────────────────
FROM node:22-slim AS build
WORKDIR /app

ARG RAILWAY_GIT_COMMIT_SHA=""
ARG COMMIT_SHA=""

RUN apt-get update && apt-get install -y gcc libnode-dev && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json turbo.json tsconfig.base.json ./
COPY apps/ apps/
COPY packages/ packages/

RUN npm ci
RUN npx turbo build

# Bake commit hash into build-info.json for the /api/version endpoint
RUN echo "{\"sha\":\"$(echo ${RAILWAY_GIT_COMMIT_SHA:-${COMMIT_SHA:-unknown}} | cut -c1-7)\",\"message\":\"\"}" > apps/server/dist/build-info.json

# ── Production stage ──────────────────────────────────────────────────────
FROM node:22-slim
WORKDIR /app

COPY package.json package-lock.json ./
COPY apps/ apps/
COPY packages/ packages/

RUN npm ci --omit=dev

COPY --from=build /app/apps/server/dist/ apps/server/dist/
COPY --from=build /app/packages/core/dist/ packages/core/dist/
COPY --from=build /app/packages/tensor/dist/ packages/tensor/dist/
COPY --from=build /app/packages/autograd/dist/ packages/autograd/dist/
COPY --from=build /app/packages/tokenizers/dist/ packages/tokenizers/dist/
COPY --from=build /app/packages/model/dist/ packages/model/dist/
COPY --from=build /app/packages/train/dist/ packages/train/dist/
COPY --from=build /app/packages/helios/dist/ packages/helios/dist/
COPY --from=build /app/packages/helios/native/helios_vk.node packages/helios/native/helios_vk.node
COPY --from=build /app/packages/db/dist/ packages/db/dist/
COPY public/ public/

ENV PORT=3000
EXPOSE 3000

CMD ["node", "--max-old-space-size=512", "apps/server/dist/server.js"]
