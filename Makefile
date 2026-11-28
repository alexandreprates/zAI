# All targets run inside containers; only Docker (Compose v2) is needed on the host.
# Targets prefixed with `ci-` run inside the dev container.

# Build and run images for the host's native architecture, even if the shell sets
# DOCKER_DEFAULT_PLATFORM (emulated amd64 would distort simulation benchmarks).
# compose.gpu.yaml still pins the trainer to linux/amd64 for the GPU server.
unexport DOCKER_DEFAULT_PLATFORM

DEV := docker compose run --rm -T dev
GPU := docker compose -f compose.yaml -f compose.gpu.yaml

.PHONY: test lint fmt proto shell build up train-league train-league-gpu arena-gpu probe-gpu distill-gpu export-model export-model-gpu train-human train-human-gpu tensorboard web-install web-dev check-gpu ci-test ci-lint ci-proto ci-proto-check ci-fmt ci-py-develop ci-web-install

test: ## Run Rust and Python tests
	$(DEV) make ci-test

lint: ## Run rustfmt check, clippy, ruff, the gRPC stub drift check and the TypeScript type check
	$(DEV) make ci-lint

proto: ## Regenerate the Python gRPC stubs (python/zai_agent/proto) from proto/zai/v1/agent.proto
	$(DEV) make ci-proto

fmt: ## Format Rust and Python code
	$(DEV) make ci-fmt

shell: ## Open a shell in the dev container
	docker compose run --rm dev bash

build: ## Build the server (with the web viewer) and (CPU) trainer images
	docker compose --profile train build server trainer

up: ## Run the server with the viewer on http://localhost:8080
	docker compose up --build server

train-human: ## Train the human (M4) in the trainer container; ARGS="--total-steps N --run-id ID"
	docker compose --profile train run --rm --build trainer python -m zai_rl.train --config /app/configs/train/human_s0.yaml $(ARGS)

train-human-gpu: ## On the GPU server: human PPO with CUDA (configs/train/human_s0_gpu.yaml); see docs/GPU_SERVER.md
	$(GPU) --profile train run --rm --build trainer python -m zai_rl.train --config /app/configs/train/human_s0_gpu.yaml $(ARGS)

train-zombie: ## Train the zombies (M5) in the trainer container; ARGS="--total-steps N --run-id ID"
	docker compose --profile train run --rm --build trainer python -m zai_rl.train_zombie --config /app/configs/train/zombie_s0.yaml $(ARGS)

train-zombie-gpu: ## On the GPU server: zombie MAPPO with CUDA (configs/train/zombie_s0_gpu.yaml)
	$(GPU) --profile train run --rm --build trainer python -m zai_rl.train_zombie --config /app/configs/train/zombie_s0_gpu.yaml $(ARGS)

export-replays: ## Record a human checkpoint's episodes as viewer replays; ARGS="--checkpoint /checkpoints/human/<run>/latest.pt --episodes 10"
	docker compose --profile train run --rm --build trainer python -m arena.export_replays $(ARGS)

export-replays-gpu: ## On the GPU server: same as export-replays, reusing the CUDA trainer image (inference runs on CPU)
	$(GPU) --profile train run --rm --build trainer python -m arena.export_replays $(ARGS)

train-league: ## League co-training of both sides (M6) in the trainer container; ARGS="--total-steps N --run-id ID"
	docker compose --profile train run --rm --build trainer python -m zai_rl.train_league --config /app/configs/train/league_cpu.yaml $(ARGS)

train-league-gpu: ## On the GPU server: league co-training (configs/train/league_s0small_gpu.yaml)
	$(GPU) --profile train run --rm --build trainer python -m zai_rl.train_league --config /app/configs/train/league_s0small_gpu.yaml $(ARGS)

arena-gpu: ## On the GPU server: league arena (cross-play Elo, tactics) next to train-league-gpu; ARGS="--run-id ID"
	$(GPU) --profile train run --rm --build trainer python -m arena.league $(ARGS)

probe-gpu: ## On the GPU server: exploitability probe of a league; ARGS="--run-id ID --side both --steps 10000000"
	$(GPU) --profile train run --rm --build trainer python -m arena.probe $(ARGS)

distill-gpu: ## On the GPU server: distill a checkpoint into the current observation layout; ARGS="--config /app/configs/train/distill/<name>.yaml --run-id ID"
	$(GPU) --profile train run --rm --build trainer python -m zai_rl.distill $(ARGS)

export-model: ## Export checkpoints as ONNX models for live viewer matches; ARGS="--checkpoint /checkpoints/<side>/<run>/latest.pt"
	docker compose --profile train run --rm --build trainer python -m arena.export_model $(ARGS)

export-model-gpu: ## On the GPU server: same as export-model, reusing the CUDA trainer image
	$(GPU) --profile train run --rm --build trainer python -m arena.export_model $(ARGS)

tensorboard: ## TensorBoard on http://localhost:6006 over the runs volume
	docker compose --profile tensorboard up tensorboard

web-install: ## Install the viewer's npm dependencies (into a volume)
	$(DEV) make ci-web-install

web-dev: ## Vite dev server on http://localhost:5173, proxying to a server on the host's port 8080
	docker compose run --rm -p 5173:5173 -e ZAI_SERVER=http://host.docker.internal:8080 dev bash -c "make ci-web-install && cd web && npm run dev"

check-gpu: ## On the GPU server: verify the CUDA build has sm_60 kernels and sees the P100
	$(GPU) run --rm --build trainer python -m zai_rl.check_gpu --require-gpu

ci-py-develop:
	uv pip install -e python
	maturin develop --uv --release -m crates/zai-py/Cargo.toml

ci-test: ci-py-develop
	cargo test --workspace --exclude zai-py
	pytest python

ci-web-install:
	cd web && npm ci

# Stubs import package-relative (`zai_agent.proto`) through the -I mapping.
PROTOC_PY = python -m grpc_tools.protoc -Izai_agent/proto=../proto/zai/v1 ../proto/zai/v1/agent.proto

ci-proto:
	cd python && $(PROTOC_PY) --python_out=. --pyi_out=. --grpc_python_out=.

ci-proto-check:
	@tmp=$$(mktemp -d) && cd python && $(PROTOC_PY) --python_out=$$tmp --pyi_out=$$tmp --grpc_python_out=$$tmp \
		&& for f in agent_pb2.py agent_pb2.pyi agent_pb2_grpc.py; do \
			cmp -s $$tmp/zai_agent/proto/$$f zai_agent/proto/$$f \
				|| { echo "python/zai_agent/proto/$$f is stale: run make proto"; exit 1; }; \
		done

ci-lint: ci-web-install ci-proto-check
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	ruff check python
	ruff format --check python
	cd web && npm run typecheck

ci-fmt:
	cargo fmt --all
	ruff format python
	ruff check --fix python
