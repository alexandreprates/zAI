# syntax=docker/dockerfile:1
# TORCH_INDEX selects the PyTorch build: "cpu" for the local pilot (any arch),
# "cu126" for the Tesla P100 server (CUDA 12.6 wheels still ship sm_60 kernels).
ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.8.0

FROM rust:1.88-bookworm AS wheel
COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install "maturin>=1.9,<2"
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    /opt/venv/bin/maturin build --release --locked \
        -m crates/zai-py/Cargo.toml -i /opt/venv/bin/python --compatibility off -o /wheels

FROM python:3.12-slim-bookworm
ARG TORCH_INDEX
ARG TORCH_VERSION
COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/
ENV UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 ZAI_CONFIG_DIR=/app/configs/env
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}"
COPY --from=wheel /wheels /wheels
COPY python /app/python
COPY configs /app/configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /wheels/*.whl /app/python "tensorboard>=2.17"
WORKDIR /app
CMD ["python", "-m", "zai_rl.check_gpu"]
