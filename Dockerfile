# ── Build stage ────────────────────────────────────────────────────────────
FROM node:22-slim AS build
WORKDIR /app

COPY package.json package-lock.json turbo.json tsconfig.base.json ./
COPY apps/ apps/
COPY packages/ packages/

RUN npm ci
RUN npx turbo build

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
COPY --from=build /app/packages/db/dist/ packages/db/dist/
COPY public/ public/

ENV PORT=3000
EXPOSE 3000

CMD ["node", "--max-old-space-size=512", "apps/server/dist/server.js"]
