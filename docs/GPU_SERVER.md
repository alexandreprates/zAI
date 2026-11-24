# Running on the GPU server

Target: Linux host with an NVIDIA Tesla P100 (Pascal, `sm_60`), 24 CPU cores, 64 GB RAM.
Everything runs in containers; the host only needs Docker and the NVIDIA driver/toolkit.

## 1. Host prerequisites (once)

1. NVIDIA driver with CUDA 12.6 support (driver 560 or newer). Check with `nvidia-smi`.
2. Docker Engine with the Compose v2 plugin.
3. NVIDIA Container Toolkit, configured for Docker:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker && \
     sudo systemctl restart docker
   ```

4. Smoke test the GPU from a container:

   ```bash
   docker run --rm --gpus all \
     nvidia/cuda:12.6.3-base-ubuntu24.04 \
     nvidia-smi
   ```

## 2. Get the code and verify the trainer image

Copy the repository to the server (for example with `git clone` or `rsync -a --exclude target`),
then from the repository root:

```bash
make check-gpu
```

This builds the trainer image with the CUDA 12.6 PyTorch wheels (`compose.gpu.yaml`,
`TORCH_INDEX=cu126`) and fails unless the build ships `sm_60` kernels and sees the P100.
CUDA 13 and some newer PyTorch CUDA builds dropped Pascal; if this check fails, pin an older
`TORCH_VERSION` in `docker/trainer.Dockerfile` whose `cu126` wheels still include `sm_60`.

## 3. Start the server and train

```bash
docker compose up -d --build server
make train-human-gpu ARGS="--run-id human-s0-gpu1"
```

- Config: `configs/train/human_s0_gpu.yaml` (1024 envs, 20 simulation threads, 200 M steps).
- Checkpoints: `checkpoints` volume, `/checkpoints/human/<run_id>/` (`latest.pt` plus one per
  20 iterations). Resume with `ARGS="--resume /checkpoints/human/<run_id>/latest.pt"`.
- Long runs: start inside `tmux`/`screen`, or detach with
  `docker compose -f compose.yaml -f compose.gpu.yaml --profile train run -d trainer ...`.

## 4. Watch it

The viewer and TensorBoard listen on the server. From your machine, tunnel the ports:

```bash
ssh -N \
  -L 8080:localhost:8080 \
  -L 6006:localhost:6006 \
  user@gpu-server
```

Then open http://localhost:8080 ("Training runs") and, after `make tensorboard` on the
server, http://localhost:6006.

## 5. Watch the trained policy play

Trainers export the latest model of each run to ONNX on every checkpoint
(`<data volume>/models/<run_id>.onnx` + `.json`). The viewer lists those models under
"trained" in the Human / Zombies selects of "New match"; the server runs them live on the CPU
with `tract` (`crates/zai-server/src/policy.rs`). Picking a model switches "Map" to the env
config it was trained on (a warning shows when a model plays on another map), and
"Deterministic policies" takes the most likely action instead of sampling. Checkpoints from
before the automatic export are exported by hand:

```bash
make export-model-gpu ARGS="--checkpoint /checkpoints/human/human-s0-gpu1/latest.pt /checkpoints/zombie/zombie-s0small-gpu1/latest.pt"
```

A checkpoint's episodes can also be recorded offline as replays (CPU inference, a few seconds
per episode) and watched under "Replays" in the viewer:

```bash
docker compose up -d --build server
make export-replays-gpu ARGS="--checkpoint /checkpoints/human/human-s0-gpu1/latest.pt --episodes 10"
```

The replays are listed as `policy/scripted`. Defaults: the S0 env config, scripted zombies and
sampled actions; add `--deterministic` for argmax actions or `--zombies random`. Rebuild the
server first (as above) if it predates policy replays, or it skips those files.

## What to check in the first hour

- `perf/steps_per_s` (expected tens of thousands; the CPU pilot did ~2.2 k).
- `eval/survived` vs `eval/baseline_survived` (scripted human, ~80% on S0): the M4 gate.
- `train/entropy` falling slowly from ~2.8 and `train/explained_variance` above ~0.8.
