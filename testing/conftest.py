"""Shared test setup. Tests never call a real model by accident: the pipeline's default router is the ONNX
router here (the app's default, the LLM router, needs an API key and costs money). A test that wants another
router sets CUSMA_ROUTER itself."""
import pytest


@pytest.fixture(autouse=True)
def _offline_default_router(monkeypatch):
    from agents import pipeline
    monkeypatch.setenv("CUSMA_ROUTER", "onnx")
    pipeline._router_cache.clear()
    yield
    pipeline._router_cache.clear()
