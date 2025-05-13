# ── Stage 1: build ───────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci --omit=dev           # reproducible, no extra dev junk
COPY frontend .
RUN npm run build               # → .next, public, etc.

# ── Stage 2: runtime ─────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000
COPY --from=builder /app ./
EXPOSE 3000
CMD ["npm","start"]
