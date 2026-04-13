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
│    2. Call opensage_agent_        │           │  opensage_agent_function   │
│       function.run() ───────────────────────►│    3. Create LiteLlm       │
│    7. Collect TITO records       │           │       (base_url → Miles)   │
│    8. Build training samples     │           │    4. Evaluation._generate │
│                                  │           │       _one(task)           │
│  Miles Session Server (:30000)   │           │       └─ Agent loop        │
│    /sessions/{id}/v1/chat/       │           │          ├─ LLM calls ─┐   │
│      completions                 │           │          ├─ Tool calls  │   │
│    - Proxies to SGLang engines   │◄──────────────────────────────────┘   │
│    - Records tokens (TITO)       │           │    5. Compute reward       │
│                                  │           │    6. Return {reward,      │
│  SGLang engines (1 per GPU)      │           │       exit_status,         │
│    /v1/chat/completions          │           │       agent_metrics}       │
│                                  │           │                            │
│  Megatron (training, GRPO)       │           └────────────────────────────┘
└──────────────────────────────────┘
```

### Key difference from swe-agent-v2 (Harbor)

| | swe-agent-v2 | opensage |
|---|---|---|
| Agent runtime | Separate container (agent_env) | In-process (same container as Miles) |
| Agent server | server.py (FastAPI, port 11000) | Direct Python call (no HTTP) |
| Token tracking | Miles TITO (external) | Miles TITO (external) |
| Sandbox | Harbor Docker containers | OpenSage Docker sandboxes |
| Grading | Harbor test.sh + verifier | OpenSage Evaluation.reward_func() |

## Files

| File | Description |
| --- | --- |
| `opensage_agent_function.py` | Custom agent function — bridges Miles to OpenSage evaluation |
| `generate.py` | Reward function, agent metrics aggregation, `RolloutFn` |
| `run.py` | Training launcher — handles Ray lifecycle, model loading, and job submission |
| `example.py` | Smoke test with mock session server |

On the OpenSage side:

| File | Description |
| --- | --- |
| `adapters/miles.py` | MilesAdapter — creates LiteLlm, runs agent, returns reward |
| `client.py` | `miles_generate()` method on RLSession |

## Setup

### Prerequisites

- Miles container with GPU support (nvidia-container-toolkit)
- OpenSage installed (`pip install -e /path/to/opensage-adk-dev`)
- Model weights downloaded (e.g. `zai-org/GLM-4.7-Flash`)
- Model converted to Megatron format (see swe-agent-v2 Step 6)

### Step 1: Prepare data

```bash
# Inside miles container
cd /root/miles/examples/experimental/swe-agent-v2
python download_and_process_data.py --input SWE-Gym/SWE-Gym --output /root/swe.jsonl
```

### Step 2: Run training

```bash
cd /root/miles/examples/experimental/opensage

# Debug mode (quick pipeline verification)
python run.py --mode debug_rollout_only \
  --prompt-data /root/swe.jsonl \
  --num-gpus-per-node 8

# Full training
python run.py \
  --prompt-data /root/swe.jsonl \
  --num-gpus-per-node 8
```

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `OPENSAGE_AGENT_NAME` | `vul_agent_static_tools` | OpenSage agent name |
| `OPENSAGE_BENCHMARK_NAME` | `secodeplt` | Benchmark for reward computation |
| `AGENT_MODEL_NAME` | `model` | Model name passed to the agent |
| `MILES_HOST_IP` | `$(hostname)` | IP/hostname for inter-process communication |

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
