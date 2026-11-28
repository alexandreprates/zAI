# Toolchain image for builds, tests and lint. The repository is bind-mounted at /workspace.
FROM node:22.12.0-bookworm-slim AS node

FROM rust:1.88-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

RUN rustup component add clippy rustfmt

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYO3_PYTHON=/opt/venv/bin/python \
    CARGO_TARGET_DIR=/cache/target \
    UV_CACHE_DIR=/cache/uv \
    npm_config_cache=/cache/npm

RUN uv python install 3.12 \
    && uv venv --python 3.12 /opt/venv \
    && uv pip install "maturin>=1.9,<2" "pytest>=8.3" "ruff>=0.12" \
    # Generates the gRPC stubs of python/zai_agent/proto (make proto); its protobuf major must
    # match the runtime pin in python/pyproject.toml.
    && uv pip install "grpcio-tools==1.85.4" \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"

WORKDIR /workspace
