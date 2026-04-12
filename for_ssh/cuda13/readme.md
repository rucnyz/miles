## Build with CUDA 13 + SSH

在 `my/` 目录下执行（需要两步）：

```shell
cd my

# 1. 先构建 CUDA 13 base 镜像
docker compose build base

# 2. 再构建并启动（默认端口 2222）
docker compose up --build

# 或者自定义 SSH 端口
SSH_PORT=2224 docker compose up --build
```

构建流程：
1. 先构建 `slime-cuda13-base` 镜像（基于 `docker/Dockerfile`，启用 CUDA 13）
2. 再基于它构建最终镜像（添加 SSH 配置）

```shell
docker build -t slime_ssh .

docker run --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --runtime=nvidia \
  --name slime_yuzhou \
  -v /home/yineng/shared_model:/root/.cache \
  --privileged \
  -p 2224:22 \
  -d \
  slime_ssh
```

```shell
--model-loader-extra-config='{"enable_multithread_load": "true","num_threads": 64}'
```

sglang debug
```shell
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' CUDA_COREDUMP_FILE="/home/sglang/bbuf/dump/cuda_coredump_%h.%p.%t" python3 -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --model-loader-extra-config='{"enable_multithread_load": "true","num_threads": 64}' --enable-piecewise-cuda-graph
```