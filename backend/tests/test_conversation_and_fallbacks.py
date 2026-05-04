from types import SimpleNamespace
from uuid import uuid4

from app.services.memory import MemoryService
from app.services.orchestration import MultiAgentOrchestrator


class DummySettings:
    memory_provider = "mem0"
    mem0_api_key = None
    mem0_base_url = None
    zep_api_key = None
    zep_base_url = None


class DummySession:
    pass


def test_memory_provider_falls_back_to_local_note_when_mem0_unconfigured():
    service = MemoryService(DummySettings(), DummySession())
    assert service.provider.name == "mem0"
    assert service.provider.available is False
    assert "local memory store remains active" in (service.provider.note or "")


def test_orchestrator_appends_fallback_trace_entry():
    orchestrator = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    state: dict = {"conversation_id": str(uuid4())}

    orchestrator._append_trace(state, agent="tool_use", status="fallback", summary="Fell back to retrieval")

    assert state["agent_trace"][0]["agent"] == "tool_use"
    assert state["agent_trace"][0]["status"] == "fallback"
