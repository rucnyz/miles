"""OpenSage + Miles training launcher.

Usage:
    python run.py --config configs/default.yaml
    python run.py --config configs/debug.yaml
    python run.py --config configs/default.yaml --prompt-data /root/my_data.jsonl
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MILES_ROOT = SCRIPT_DIR.parents[2]  # examples/experimental/opensage -> miles root


def parse_yaml_to_args(config_path: str) -> list[str]:
    """Convert YAML config to CLI args list."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not cfg:
        return []
    args = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        else:
            args.extend([f"--{k}", str(v)])
    return args


def cleanup():
    """Kill stale sglang/ray processes to free GPUs."""
    pid, ppid = os.getpid(), os.getppid()
    for target in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run(
            f"pgrep -f '{target}' | grep -v '^{pid}$' | grep -v '^{ppid}$' "
            f"| xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="YAML config file")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--megatron-path", default="/root/Megatron-LM")
    parser.add_argument("--agent", default=os.getenv("OPENSAGE_AGENT_NAME", "vul_agent_static_tools"))
    parser.add_argument("--benchmark", default=os.getenv("OPENSAGE_BENCHMARK_NAME", "secodeplt"))
    parser.add_argument("--model-name", default=os.getenv("AGENT_MODEL_NAME", "model"))
    parser.add_argument("--dataset-path", default=os.getenv("OPENSAGE_DATASET_PATH", ""),
                        help="Dataset path for Evaluation (e.g. 'swebench' for harbor auto-download)")
    parser.add_argument("--hf-checkpoint", default="/root/GLM-4.7-Flash",
                        help="Local model path or HuggingFace model name")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip model checkpoint conversion")
    parser.add_argument("--skip-cleanup", action="store_true")
    # Extra args after -- are appended to training CLI (override YAML)
    args, extra = parser.parse_known_args()

    if not args.skip_cleanup:
        cleanup()

    # Auto-convert HF checkpoint to Megatron format (skips if already done)
    if not args.skip_prepare:
        import miles.utils.external_utils.command_utils as U
        U.convert_checkpoint(
            model_name=Path(args.hf_checkpoint).name,
            megatron_model_type="glm4.7-flash",
            num_gpus_per_node=args.num_gpus,
            hf_checkpoint=args.hf_checkpoint,
            megatron_path=args.megatron_path,
        )

    # Build training CLI args from YAML + overrides
    train_args = parse_yaml_to_args(args.config)
    train_args.extend(extra)

    # OpenSage-specific agent args (always appended)
    train_args.extend([
        "--custom-generate-function-path", "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path", "opensage_agent_function.run",
        "--custom-rm-path", "opensage_agent_function.reward_func",
        "--tito-model", "glm47",
        "--chat-template-path", "autofix",
        "--use-session-server",
        "--session-server-port", "30000",
        "--generate-multi-samples",
        "--dynamic-sampling-filter-path", "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted",
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(args.num_gpus),
        "--rollout-num-gpus", str(args.num_gpus),
    ])

    # Environment
    env = {
        **os.environ,
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{MILES_ROOT}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENSAGE_AGENT_NAME": args.agent,
        "OPENSAGE_BENCHMARK_NAME": args.benchmark,
        "OPENSAGE_DATASET_PATH": args.dataset_path,
        "AGENT_MODEL_NAME": args.model_name,
        "MILES_HOST_IP": os.getenv("MILES_HOST_IP", socket.gethostname()),
    }

    # Launch via miles train CLI
    cmd = [
        sys.executable, "-m", "miles.cli.train",
        "--megatron-model-type", "glm4.7-flash",
        *train_args,
    ]

    print(f"Launching: {' '.join(cmd[:6])} ... ({len(train_args)} args)")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
