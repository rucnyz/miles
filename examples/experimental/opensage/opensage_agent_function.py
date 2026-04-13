"""
Custom agent function and reward for Miles <-> OpenSage integration.

Usage in run.py:
    --custom-agent-function-path opensage_agent_function.run
    --custom-rm-path opensage_agent_function.reward_func

Environment variables:
    OPENSAGE_AGENT_NAME:     Agent name (default: vul_agent_static_tools)
    OPENSAGE_BENCHMARK_NAME: Benchmark name (default: secodeplt)
    AGENT_MODEL_NAME:        Model name for the agent (default: model)
"""

import asyncio
import logging
import os
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by OpenSage evaluation during generate()."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


# -- Agent function --

_client = None
_client_lock = asyncio.Lock()


async def _get_client():
    global _client
    if _client is not None:
        return _client

    async with _client_lock:
        if _client is not None:
            return _client

        import opensage

        agent_name = os.getenv("OPENSAGE_AGENT_NAME", "vul_agent_static_tools")
        benchmark_name = os.getenv("OPENSAGE_BENCHMARK_NAME", "secodeplt")
        model_name = os.getenv("AGENT_MODEL_NAME", "model")

        logger.info(
            f"Creating OpenSage client: agent={agent_name}, "
            f"benchmark={benchmark_name}, model={model_name}"
        )
        _client = opensage.create(
            agent_name=agent_name,
            benchmark_name=benchmark_name,
            model_name=model_name,
        )
        return _client


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run an OpenSage agent on a single task instance.

    Called by Miles' agentic_tool_call.generate(). The base_url points to
    Miles' session server which proxies LLM calls to sglang while recording
    tokens for TITO training.
    """
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}
    model_name = os.getenv("AGENT_MODEL_NAME", "model")

    try:
        client = await _get_client()

        with client.init_session() as session:
            result = await session.miles_generate(
                base_url=base_url,
                prompt=prompt,
                metadata=metadata,
                sampling_params=request_kwargs,
                model_name=model_name,
            )

        return result

    except Exception as e:
        logger.error(f"OpenSage agent call failed: {e}", exc_info=True)
        return {
            "reward": 0.0,
            "exit_status": f"Error: {type(e).__name__}",
            "agent_metrics": {},
            "eval_report": {"error": str(e)},
        }
