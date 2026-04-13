"""
Custom agent function for Miles agentic_tool_call.generate.

Dispatches to an OpenSage agent and returns reward + metrics.
OpenSage runs the agent using Miles' session server as the LLM endpoint,
so Miles automatically records all tokens via TITO.

Usage:
    --custom-agent-function-path examples.experimental.opensage.opensage_agent_function.run

Environment variables:
    OPENSAGE_AGENT_NAME:     Agent name (default: vul_agent_static_tools)
    OPENSAGE_BENCHMARK_NAME: Benchmark name (default: secodeplt)
    AGENT_MODEL_NAME:        Model name for the agent (default: model)
"""

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized client (one per process)
_client = None
_client_lock = asyncio.Lock()


async def _get_client():
    """Get or create the OpenSage client (singleton per process)."""
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
    Miles' session server, which proxies LLM calls to sglang while recording
    tokens for TITO training.

    Args:
        base_url: Miles session server endpoint
        prompt: Task prompt (string or chat messages)
        request_kwargs: Sampling parameters
        metadata: Task metadata from Miles sample
        **kwargs: Additional arguments (ignored)

    Returns:
        dict with {reward, exit_status, agent_metrics, eval_report}
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
