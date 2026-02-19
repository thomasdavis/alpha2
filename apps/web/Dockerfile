# ── Build stage ────────────────────────────────────────────────────────────
FROM node:22-slim AS build
WORKDIR /app

COPY package.json package-lock.json turbo.json tsconfig.base.json ./
COPY apps/web/ apps/web/
COPY packages/core/ packages/core/
COPY packages/db/ packages/db/

RUN npm ci
RUN npx turbo build --filter=@alpha/web...

# ── Production stage ──────────────────────────────────────────────────────
FROM node:22-slim
WORKDIR /app

COPY --from=build /app/apps/web/.next/standalone ./
COPY --from=build /app/apps/web/.next/static apps/web/.next/static
COPY --from=build /app/apps/web/public apps/web/public

ENV PORT=3000
ENV HOSTNAME=0.0.0.0
EXPOSE 3000

CMD ["node", "apps/web/server.js"]
