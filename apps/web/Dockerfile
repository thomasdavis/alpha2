# ── Build stage ────────────────────────────────────────────────────────────
FROM node:22-slim AS build
WORKDIR /app

ARG RAILWAY_GIT_COMMIT_SHA=""
ARG COMMIT_SHA=""

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

# Bake commit hash into build-info.json for the /api/version endpoint
RUN echo "{\"sha\":\"$(echo ${RAILWAY_GIT_COMMIT_SHA:-${COMMIT_SHA:-unknown}} | cut -c1-7)\",\"message\":\"\"}" > apps/web/build-info.json

# ── Production stage ──────────────────────────────────────────────────────
FROM node:22-slim
WORKDIR /app

COPY --from=build /app/apps/web/.next/standalone ./
COPY --from=build /app/apps/web/.next/static apps/web/.next/static
COPY --from=build /app/apps/web/build-info.json apps/web/build-info.json

ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV OUTPUTS_DIR=/app/outputs
EXPOSE 3000

CMD ["node", "apps/web/server.js"]
