# syntax=docker/dockerfile:1
FROM node:22.12.0-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY web ./
RUN npm run build

FROM rust:1.88-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY proto proto
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p zai-server \
    && cp target/release/zai-server /usr/local/bin/zai-server

FROM debian:bookworm-slim
RUN useradd --system --uid 10001 zai && mkdir -p /data && chown zai /data
COPY --from=builder /usr/local/bin/zai-server /usr/local/bin/zai-server
COPY --from=web /web/dist /app/web
ENV ZAI_DATA_DIR=/data ZAI_WEB_DIR=/app/web ZAI_HTTP_ADDR=0.0.0.0:8080 ZAI_GRPC_ADDR=0.0.0.0:50051
USER zai
EXPOSE 8080 50051
ENTRYPOINT ["zai-server"]
