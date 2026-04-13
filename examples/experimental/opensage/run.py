"""OpenSage + Miles training launcher.

Usage:
    python run.py --config configs/default.yaml
    python run.py --config configs/debug.yaml
    python run.py --config configs/default.yaml --dataset-path swebench --benchmark harbor
"""

import os
import socket
from pathlib import Path

import yaml

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_yaml_to_args(config_path: str) -> str:
    """Convert YAML config to CLI args string."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not cfg:
        return ""
    parts = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


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
    parser.add_argument("--hf-checkpoint", default="zai-org/GLM-4.7-Flash",
                        help="HuggingFace model name or local path")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip model checkpoint conversion")
    args, extra_args = parser.parse_known_args()

    # Auto-convert HF checkpoint to Megatron format (skips if already done)
    if not args.skip_prepare:
        U.convert_checkpoint(
            model_name=Path(args.hf_checkpoint).name,
            megatron_model_type="glm4.7-flash",
            num_gpus_per_node=args.num_gpus,
            hf_checkpoint=args.hf_checkpoint,
            megatron_path=args.megatron_path,
        )

    # Build training args: YAML config + extra CLI overrides
    train_args = parse_yaml_to_args(args.config)
    if extra_args:
        train_args += " " + " ".join(extra_args)

    # OpenSage-specific agent args
    train_args += (
        " --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate"
        " --custom-agent-function-path opensage_agent_function.run"
        " --custom-rm-path opensage_agent_function.reward_func"
        " --tito-model glm47"
        " --chat-template-path autofix"
        " --use-session-server"
        " --session-server-port 30000"
        " --generate-multi-samples"
        " --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted"
        f" --actor-num-nodes 1"
        f" --actor-num-gpus-per-node {args.num_gpus}"
        f" --rollout-num-gpus {args.num_gpus}"
    )

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{U.repo_base_dir}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENSAGE_AGENT_NAME": args.agent,
        "OPENSAGE_BENCHMARK_NAME": args.benchmark,
        "OPENSAGE_DATASET_PATH": args.dataset_path,
        "AGENT_MODEL_NAME": args.model_name,
        "MILES_HOST_IP": os.getenv("MILES_HOST_IP", socket.gethostname()),
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus,
        megatron_model_type="glm4.7-flash",
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    main()
