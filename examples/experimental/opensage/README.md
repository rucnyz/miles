# OpenSage: Agent-Environment RL Training with ADK

Integrates [OpenSage](https://github.com/opensage-agent/opensage-adk) agents with Miles' GRPO training pipeline. Uses **TITO (Token In Token Out)** for exact token-level training signals — Miles handles all token tracking externally, so OpenSage only needs to run the agent and return a reward.

## Architecture

```
Miles container (GPU)                          OpenSage (in-process)
┌──────────────────────────────────┐
│                                  │
│  Ray job → train.py              │
│    ├─ MegatronTrainRayActor (×N) │
│    ├─ SGLangEngine (×N, 1/GPU)   │
│    └─ RolloutManager             │
│                                  │
│  agentic_tool_call.generate      │           ┌────────────────────────────┐
│    1. Create TITO session        │           │                            │
│    2. Call opensage_agent_       │           │  opensage_agent_function   │
│       function.run() ───────────────────────►│    3. Create LiteLlm       │
│    7. Collect TITO records       │           │       (base_url → Miles)   │
│    8. Build training samples     │           │    4. Evaluation._generate │
│                                  │           │       _one(task)           │
│  Miles Session Server (:30000)   │           │       └─ Agent loop        │
│    /sessions/{id}/v1/chat/       │           │          ├─ LLM calls ─┐   │
│      completions                 │           │          ├─ Tool calls │   │
│    - Proxies to SGLang engines   │◄───────────────────────────────────┘   │
│    - Records tokens (TITO)       │           │    5. Compute reward       │
│                                  │           │    6. Return {reward,      │
│  SGLang engines (1 per GPU)      │           │       exit_status,         │
│    /v1/chat/completions          │           │       agent_metrics}       │
│                                  │           │                            │
│  Megatron (training, GRPO)       │           └────────────────────────────┘
└──────────────────────────────────┘
```

## Files

| File | Description |
| --- | --- |
| `opensage_agent_function.py` | Agent function + reward_func — bridges Miles to OpenSage |
| `run.py` | Training launcher — reads YAML config, sets up env, launches training |
| `configs/default.yaml` | Full training parameters |
| `configs/debug.yaml` | Debug mode (small batch) |
| `example.py` | Smoke test with mock session server |

## Usage

### Run training

```bash
cd /root/miles/examples/experimental/opensage

# SeCodePLT benchmark
python run.py --config configs/default.yaml \
  --benchmark secodeplt --agent vul_agent_static_tools

# Harbor benchmark (auto-downloads from harbor registry)
python run.py --config configs/default.yaml \
  --benchmark harbor --agent harbor_agent --dataset-path swebench

# Harbor benchmark (local task directory)
python run.py --config configs/default.yaml \
  --benchmark harbor --agent harbor_agent --dataset-path /data/my_tasks

# Debug mode
python run.py --config configs/debug.yaml \
  --benchmark harbor --agent harbor_agent --dataset-path swebench
```

### CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--config` | (required) | YAML config file for training parameters |
| `--benchmark` | `secodeplt` | OpenSage benchmark name |
| `--agent` | `vul_agent_static_tools` | OpenSage agent name |
| `--dataset-path` | (empty) | Dataset path — local dir or harbor registry name |
| `--model-name` | `model` | Model name passed to the agent |
| `--num-gpus` | `8` | Number of GPUs |

### Supported benchmarks

| Benchmark | `--benchmark` | `--dataset-path` | Description |
| --- | --- | --- | --- |
| SeCodePLT | `secodeplt` | (not needed) | Vulnerability detection, auto-downloads from HuggingFace |
| SWE-Bench Pro | `swe_bench_pro` | (not needed) | Software engineering, auto-downloads from HuggingFace |
| CyberGym | `cybergym` | (not needed) | Cybersecurity challenges |
| Harbor tasks | `harbor` | `swebench`, `compilebench`, etc. | Any of 60+ Harbor benchmarks |

Harbor tasks are auto-downloaded from the harbor registry to `~/.cache/harbor/tasks/`. You can also point `--dataset-path` at a local directory of pre-generated Harbor task directories.

## How It Works

1. **Session creation**: `agentic_tool_call.generate` creates a TITO session on Miles Session Server
2. **Dispatch to OpenSage**: calls `opensage_agent_function.run()` with the session's `base_url`
3. **LiteLlm setup**: `MilesAdapter` creates a `LiteLlm(base_url=session_server)` — standard OpenAI-compatible client
4. **Agent execution**: `Evaluation._generate_one(task)` runs the ADK agent with tool calls in sandboxed Docker containers
5. **LLM calls proxied**: every `LiteLlm` call goes through Miles Session Server → SGLang, tokens recorded automatically
6. **Reward**: computed by the benchmark's `reward_func()` after agent finishes
7. **TITO collection**: Miles collects all session records, builds `Sample` objects with token IDs, logprobs, and loss masks
8. **Training**: GRPO policy update using Megatron, weights synced back to SGLang engines

## Testing

```bash
# Smoke test (no GPU required)
pip install fastapi uvicorn httpx google-adk litellm
python example.py
```
