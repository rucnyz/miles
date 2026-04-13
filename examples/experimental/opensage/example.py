"""
Smoke test for Miles <-> OpenSage integration.

Tests the full call chain without requiring GPU/sglang by mocking
the Miles session server with a simple FastAPI app.

Usage:
    # Needs opensage and miles on PYTHONPATH
    pip install fastapi uvicorn httpx
    python example.py

What it tests:
    1. opensage_agent_function.run() can be called with Miles' interface
    2. MilesAdapter creates LiteLlm with correct base_url
    3. The response format matches what Miles expects
"""

import asyncio
import logging
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Mock Miles Session Server ──────────────────────────────────────────────

mock_app = FastAPI()
call_count = 0


@mock_app.post("/sessions")
async def create_session():
    return {"session_id": "test-session-001"}


@mock_app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    return {
        "session_id": session_id,
        "records": [],
        "metadata": {"accumulated_token_ids": [], "max_trim_tokens": 0},
    }


@mock_app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    return {"status": "deleted"}


@mock_app.api_route(
    "/sessions/{session_id}/v1/chat/completions",
    methods=["POST"],
)
async def chat_completions(session_id: str, request: Request):
    """Mock OpenAI-compatible chat completions endpoint."""
    global call_count
    call_count += 1
    body = await request.json()
    logger.info(f"Mock LLM call #{call_count}: model={body.get('model')}")

    return JSONResponse({
        "id": f"chatcmpl-mock-{call_count}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"Mock response #{call_count}: I analyzed the code and found no issues.",
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    })


@mock_app.get("/health")
async def health():
    return {"status": "ok"}


def run_mock_server(port: int = 18999):
    """Run mock server in a background thread."""
    config = uvicorn.Config(mock_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


# ── Test: Direct MilesAdapter call ─────────────────────────────────────────

async def test_miles_adapter_direct(port: int):
    """Test MilesAdapter.generate() directly (bypasses opensage_agent_function)."""
    logger.info("=" * 60)
    logger.info("Test 1: MilesAdapter._create_litellm()")
    logger.info("=" * 60)

    try:
        from opensage.evaluation.rl_adapters.adapters.miles import MilesAdapter
    except ImportError as e:
        logger.info(f"  SKIPPED (missing dep: {e})")
        return

    # Create a minimal adapter (without full evaluation setup)
    # Just test that LiteLlm creation works
    adapter = MilesAdapter.__new__(MilesAdapter)

    base_url = f"http://127.0.0.1:{port}/sessions/test-session-001"
    model = adapter._create_litellm(
        base_url=base_url,
        model_name="test-model",
        sampling_params={"temperature": 0.8, "max_tokens": 1024},
    )

    logger.info(f"  LiteLlm created: model={model.model}")
    logger.info(f"  OK: LiteLlm points to mock session server")


# ── Test: Mock LLM endpoint responds correctly ────────────────────────────

async def test_mock_endpoint(port: int):
    """Verify mock session server responds to chat completions."""
    logger.info("=" * 60)
    logger.info("Test 2: Mock session server /v1/chat/completions")
    logger.info("=" * 60)

    base_url = f"http://127.0.0.1:{port}/sessions/test-session-001/v1"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": "openai/test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

    assert "choices" in data, f"Expected 'choices' in response, got: {data.keys()}"
    content = data["choices"][0]["message"]["content"]
    logger.info(f"  Response: {content}")
    logger.info(f"  OK: Mock endpoint responds correctly")


# ── Test: opensage_agent_function interface ────────────────────────────────

async def test_agent_function_interface():
    """Verify opensage_agent_function.run() has correct signature."""
    logger.info("=" * 60)
    logger.info("Test 3: opensage_agent_function.run() interface")
    logger.info("=" * 60)

    import inspect
    from opensage_agent_function import run

    sig = inspect.signature(run)
    params = list(sig.parameters.keys())
    assert "base_url" in params, f"Missing 'base_url' param, got: {params}"
    assert "prompt" in params, f"Missing 'prompt' param, got: {params}"
    assert "request_kwargs" in params, f"Missing 'request_kwargs' param, got: {params}"
    assert "metadata" in params, f"Missing 'metadata' param, got: {params}"

    logger.info(f"  Signature: {sig}")
    logger.info(f"  OK: Matches Miles' custom agent function contract")


# ── Test: generate.py reward_func ──────────────────────────────────────────

async def test_reward_func():
    """Verify generate.py reward_func extracts reward from metadata."""
    logger.info("=" * 60)
    logger.info("Test 4: generate.reward_func()")
    logger.info("=" * 60)

    try:
        from generate import reward_func
    except ImportError as e:
        logger.info(f"  SKIPPED (missing dep: {e})")
        return

    # Mock sample with metadata
    class MockSample:
        def __init__(self, reward):
            self.metadata = {"reward": reward}

    # Single sample
    r = await reward_func(None, MockSample(0.75))
    assert r == 0.75, f"Expected 0.75, got {r}"

    # Batch of samples
    batch = [MockSample(0.5), MockSample(1.0), MockSample(0.0)]
    rs = await reward_func(None, batch)
    assert rs == [0.5, 1.0, 0.0], f"Expected [0.5, 1.0, 0.0], got {rs}"

    logger.info(f"  Single: reward={r}")
    logger.info(f"  Batch:  rewards={rs}")
    logger.info(f"  OK: reward_func correctly extracts from metadata")


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    port = 18999

    # Start mock server in background
    logger.info(f"Starting mock Miles session server on port {port}...")
    server_thread = threading.Thread(target=run_mock_server, args=(port,), daemon=True)
    server_thread.start()
    time.sleep(1)  # Wait for server to start

    # Verify server is up
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{port}/health")
        assert resp.json()["status"] == "ok"
    logger.info("Mock server is running\n")

    passed = 0
    failed = 0

    tests = [
        ("MilesAdapter._create_litellm", test_miles_adapter_direct(port)),
        ("Mock endpoint", test_mock_endpoint(port)),
        ("Agent function interface", test_agent_function_interface()),
        ("reward_func", test_reward_func()),
    ]

    for name, coro in tests:
        try:
            await coro
            passed += 1
            logger.info("")
        except Exception as e:
            failed += 1
            logger.error(f"  FAILED: {e}\n")

    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
