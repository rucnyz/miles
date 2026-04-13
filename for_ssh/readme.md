## Build with SSH

```shell
cd for_ssh

# 构建并启动（默认端口 2222）
docker compose up --build

# 或者自定义 SSH 端口
SSH_PORT=2224 docker compose up --build

# 如果 opensage 代码不在默认位置 (../../opensage-adk-dev)
OPENSAGE_DIR=/path/to/opensage-adk-dev SSH_PORT=2224 docker compose up --build
```

构建时自动完成：
- OpenSage 依赖安装（google-adk, litellm 等）
- SWE-Gym 数据下载并转换为 Miles 格式（`/root/swe.jsonl`）

运行时自动挂载：
- Miles 代码 → `/root/miles`
- OpenSage 代码 → `/root/opensage`（PYTHONPATH 已配置）

## Training

### Step 1: 下载模型权重

```shell
docker exec -it miles_yuzhou bash

hf download zai-org/GLM-4.7-Flash --local-dir /root/GLM-4.7-Flash
```

### Step 2: 转换模型权重为 Megatron 格式（一次性）

```shell
cd /root/miles
source scripts/models/glm4.7-flash.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-4.7-Flash \
    --save /root/GLM-4.7-Flash_torch_dist
```

### Step 3: 启动训练

```shell
cd /root/miles/examples/experimental/opensage

# Debug 模式（快速验证流水线）
python run.py --config configs/debug.yaml

# 完整训练
python run.py --config configs/default.yaml

# 自定义参数（CLI 覆盖 YAML）
python run.py --config configs/default.yaml --prompt-data /root/my_data.jsonl --num-gpus 4
```

查看训练日志：

```shell
ray job logs <JOB_ID> --follow
```

### 准备自定义数据（可选）

```shell
cd /root/miles/examples/experimental/swe-agent-v2

# 处理本地数据
python download_and_process_data.py --input /data/tb.jsonl --output /root/tb.jsonl \
  --agent-name terminus-2 --prompt-key instruction

# 合并为混合数据集
cat /root/swe.jsonl /root/tb.jsonl > /root/mixed.jsonl
```

## Deprecated
```shell
docker build -t miles_ssh .

docker run --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --runtime=nvidia \
  --name miles_yuzhou \
  -v /home/yineng/shared_model:/root/.cache \
  --privileged \
  -p 2224:22 \
  -d \
  miles_ssh
```

```shell
--model-loader-extra-config='{"enable_multithread_load": "true","num_threads": 64}'
```

sglang debug
```shell
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' CUDA_COREDUMP_FILE="/home/sglang/bbuf/dump/cuda_coredump_%h.%p.%t" python3 -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --model-loader-extra-config='{"enable_multithread_load": "true","num_threads": 64}' --enable-piecewise-cuda-graph
```
