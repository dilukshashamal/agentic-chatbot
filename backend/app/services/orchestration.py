from __future__ import annotations

import ast
import json
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, TypedDict
from uuid import UUID

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import ConversationRecord, DocumentRecord
from app.models.schemas import AgentTrace, ChatRequest, ChatResponse, Citation, MemoryAction, MemoryHit
from app.services.conversations import ConversationService
from app.services.documents import RetrievedChunk, trim_excerpt
from app.services.exports import ExportService
from app.services.grounding import GroundingAssessment, assess_grounding_support
from app.services.memory import MemoryService
from app.services.metrics import observe_query
from app.services.model_management import ModelManagementService, RuntimeModelProfile
from app.services.retrieval import PgVectorRetriever

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful degradation
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - graceful degradation
    BeautifulSoup = None

try:  # pragma: no cover - optional runtime dependency while requirements are updating
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback for environments without langgraph
    END = "__end__"
    START = "__start__"
    MemorySaver = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


class AgentState(TypedDict, total=False):
    conversation_id: str
    document_id: str | None
    question: str
    top_k: int
    include_sources: bool
    export_formats: list[str]
    user_preferences: dict[str, str]
    memory_summary: str
    conversation_history: list[dict[str, str]]
    short_term_memory: list[str]
    semantic_memories: list[dict[str, Any]]
    memory_actions: list[dict[str, Any]]
    knowledge_graph_topics: list[str]
    custom_instructions: str
    runtime_profile: dict[str, Any]
    experiment_run_id: str | None
    route: str
    route_reason: str
    reasoning_mode: str
    pending_agents: list[str]
    retrieved_chunks: list[RetrievedChunk]
    document_result: dict[str, Any]
    analytical_result: dict[str, Any]
    tool_result: dict[str, Any]
    citation_result: dict[str, Any]
    final_answer: str
    grounded: bool
    confidence: float
    citations: list[Citation]
    system_notes: list[str]
    agent_trace: list[dict[str, Any]]
    export_artifacts: list[dict[str, Any]]
    response: ChatResponse


class MultiAgentOrchestrator:
    def __init__(self, settings: Settings, session: Session) -> None:
        self.settings = settings
        self.session = session
        self.retriever = PgVectorRetriever(settings, session)
        self.conversations = ConversationService(session)
        self.exports = ExportService(settings)
        self.memory = MemoryService(settings, session)
        self.model_management = ModelManagementService(settings, session)
        self._llm_cache: dict[str, ChatGoogleGenerativeAI] = {}
        self._graph = None
        self._checkpointer = MemorySaver() if LANGGRAPH_AVAILABLE and MemorySaver is not None else None

    def _get_llm(self, model_name: str | None = None) -> ChatGoogleGenerativeAI:
        chosen_model = model_name or self.settings.orchestration_model
        if chosen_model not in self._llm_cache:
            if not self.settings.google_api_key:
                raise RuntimeError("GOOGLE_API_KEY is required to query Gemini.")
            self._llm_cache[chosen_model] = ChatGoogleGenerativeAI(
                model=chosen_model,
                temperature=self.settings.temperature,
                google_api_key=self.settings.google_api_key,
            )
        return self._llm_cache[chosen_model]

    @staticmethod
    def _extract_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _cleanup_json(raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        return cleaned.strip()

    def _invoke_json(self, system_prompt: str, human_prompt: str) -> dict[str, Any]:
        raw = self._get_llm().invoke(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        return json.loads(self._cleanup_json(self._extract_message_text(raw.content)))

    def _runtime_profile_from_state(self, state: AgentState) -> RuntimeModelProfile:
        data = state.get("runtime_profile", {})
        return RuntimeModelProfile(
            chat_model=str(data.get("chat_model", self.settings.chat_model)),
            embedding_model=str(data.get("embedding_model", self.settings.embedding_model)),
            retrieval_config_version=str(data.get("retrieval_config_version", self.settings.retrieval_config_version)),
            prompt_template_version=str(data.get("prompt_template_version", self.settings.prompt_template_version)),
            assignment_bucket=str(data.get("assignment_bucket", "control")),
            shadow_enabled=bool(data.get("shadow_enabled", False)),
            shadow_profile=dict(data.get("shadow_profile", {})),
        )

    def _invoke_json_with_state(self, state: AgentState, system_prompt: str, human_prompt: str) -> dict[str, Any]:
        runtime_profile = self._runtime_profile_from_state(state)
        raw = self._get_llm(runtime_profile.chat_model).invoke(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        return json.loads(self._cleanup_json(self._extract_message_text(raw.content)))

    def _ready_document_count(self) -> int:
        return (
            self.session.scalar(
                select(func.count()).select_from(DocumentRecord).where(DocumentRecord.status == "ready")
            )
            or 0
        )

    def _validate_document_scope(self, payload: ChatRequest) -> None:
        if payload.document_id is not None:
            document = self.session.get(DocumentRecord, payload.document_id)
            if document is None:
                raise RuntimeError("The selected document does not exist.")
            if document.status != "ready":
                raise RuntimeError("The selected document is not ready yet.")
            return

        if self._ready_document_count() == 0:
            raise RuntimeError("There are no ready documents to search yet.")

    def _history_payload(self, conversation_id: str) -> list[dict[str, str]]:
        turns = self.conversations.recent_turns(
            UUID(conversation_id),
            limit=self.settings.conversation_history_window,
        )
        return [
            {
                "query": turn.query,
                "answer": turn.answer,
                "route": turn.route or "document_grounded",
            }
            for turn in turns
        ]

    def _append_trace(
        self,
        state: AgentState,
        agent: str,
        status: str,
        summary: str,
        retries: int = 0,
    ) -> None:
        state.setdefault("agent_trace", []).append(
            {
                "agent": agent,
                "status": status,
                "summary": summary,
                "retries": retries,
            }
        )

    def _checkpoint_payload(self, state: AgentState) -> dict[str, Any]:
        return {
            "conversation_id": state.get("conversation_id"),
            "question": state.get("question"),
            "route": state.get("route"),
            "reasoning_mode": state.get("reasoning_mode"),
            "short_term_memory": state.get("short_term_memory", []),
            "semantic_memories": state.get("semantic_memories", []),
            "memory_actions": state.get("memory_actions", []),
            "knowledge_graph_topics": state.get("knowledge_graph_topics", []),
            "pending_agents": state.get("pending_agents", []),
            "system_notes": state.get("system_notes", []),
            "agent_trace": state.get("agent_trace", []),
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "score": chunk.score,
                }
                for chunk in state.get("retrieved_chunks", [])
            ],
            "document_result": state.get("document_result", {}),
            "analytical_result": state.get("analytical_result", {}),
            "tool_result": state.get("tool_result", {}),
            "citation_result": state.get("citation_result", {}),
            "final_answer": state.get("final_answer"),
        }

    def _save_checkpoint(self, conversation_id: str, node_name: str, status: str, state: AgentState) -> None:
        self.conversations.add_checkpoint(
            conversation_id=UUID(conversation_id),
            node_name=node_name,
            status=status,
            state_payload=self._checkpoint_payload(state),
        )

    def _handle_memory_command(
        self,
        conversation: ConversationRecord,
        payload: ChatRequest,
        state: AgentState,
    ) -> ChatResponse | None:
        command = self.memory.parse_command(payload.query)
        if command is None:
            return None

        if command.action == "remember" and command.target:
            actions = self.memory.remember(conversation.id, command.target)
            answer = f"I'll remember that: {command.target}"
        elif command.action == "forget_all":
            actions = self.memory.forget(conversation.id, forget_all=True)
            answer = "I cleared the stored conversation memory for this session."
        elif command.action in {"forget", "forget_last"}:
            actions = self.memory.forget(conversation.id, target=command.target)
            answer = "I removed the matching memory I had stored." if actions else "I couldn't find a matching stored memory to remove."
        else:
            actions = [MemoryAction(action="ignored", target="memory", detail="That memory command was not understood.")]
            answer = "I couldn't understand that memory instruction."

        conversation = self.conversations.get_conversation(conversation.id) or conversation
        short_term = self.memory.load_short_term_context(conversation)
        response = ChatResponse(
            conversation_id=conversation.id,
            answer=answer,
            grounded=True,
            confidence=0.99,
            answer_mode="grounded",
            citations=[],
            retrieved_chunks=0,
            system_notes=["Memory command processed without document retrieval."],
            route="memory_command",
            memory_summary=conversation.memory_summary,
            short_term_memory=short_term["short_term_memory"],
            long_term_memory=self.memory.semantic_search(conversation.id, command.target or payload.query),
            memory_actions=actions,
            knowledge_graph_topics=self.memory.top_topics(conversation.id),
            agent_trace=[AgentTrace(agent="memory", status="completed", summary="Processed explicit memory command.", retries=0)],
            exports=[],
        )
        return response

    @staticmethod
    def _retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        return round(sum(chunk.score for chunk in chunks) / len(chunks), 2)

    def _grounding_assessment(self, question: str, chunks: list[RetrievedChunk]) -> GroundingAssessment:
        return assess_grounding_support(
            question,
            chunks,
            min_retrieval_score=self.settings.min_retrieval_score,
        )

    @staticmethod
    def _unsupported_answer_text() -> str:
        return "I could not find enough reliable support in the uploaded documents to answer that."

    def _select_citations(
        self,
        requested_chunk_ids: list[str],
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        retrieved_map = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        citations: list[Citation] = []
        for chunk_id in requested_chunk_ids:
            chunk = retrieved_map.get(chunk_id)
            if chunk is None:
                continue
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    page=chunk.page_number,
                    score=chunk.score,
                    excerpt=trim_excerpt(chunk.content),
                )
            )
        return citations

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            page_label = chunk.page_number if chunk.page_number is not None else "Unknown"
            blocks.append(
                f"[{chunk.chunk_id}] Source {chunk.source} | Page {page_label} | score={chunk.score}\n{chunk.content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _next_agent(state: AgentState) -> str:
        queue = state.get("pending_agents", [])
        if not queue:
            return "citation"
        return queue[0]

    def _pop_agent(self, state: AgentState, agent_name: str) -> None:
        state["pending_agents"] = [agent for agent in state.get("pending_agents", []) if agent != agent_name]

    def _run_with_retry(self, agent_name: str, state: AgentState, fn) -> AgentState:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_agent_retries + 1):
            try:
                result = fn(state)
                self._append_trace(result, agent_name, "completed", f"{agent_name} completed.", retries=attempt)
                self._save_checkpoint(result["conversation_id"], agent_name, "completed", result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.max_agent_retries:
                    state.setdefault("system_notes", []).append(
                        f"{agent_name} failed and fallback mode was used: {exc}"
                    )
                    self._append_trace(state, agent_name, "fallback", str(exc), retries=attempt)
                    self._save_checkpoint(state["conversation_id"], agent_name, "fallback", state)
                    return state
                time.sleep(0.3 * (2**attempt))
        raise RuntimeError(str(last_error))  # pragma: no cover - unreachable loop guard

    @staticmethod
    def _heuristic_route(question: str) -> dict[str, Any]:
        lowered = question.lower()
        agents = ["document"]
        route = "document_grounded"
        reasoning_mode = "direct"
        if any(token in lowered for token in ["compare", "difference", "trend", "across", "versus"]):
            agents = ["document", "analytical"]
            route = "cross_document_analysis"
        if any(token in lowered for token in ["calculate", "sum", "average", "chart", "plot", "latest", "current"]):
            if "tool" not in agents:
                agents.append("tool")
            reasoning_mode = "react"
            route = "tool_augmented_analysis"
        return {
            "route": route,
            "reasoning_mode": reasoning_mode,
            "agents": agents,
            "route_reason": "Fallback heuristic routing was used.",
        }

    def _memory_agent(self, state: AgentState) -> AgentState:
        conversation = self.conversations.get_conversation(UUID(state["conversation_id"]))
        if conversation is None:
            state["memory_summary"] = "New conversation."
            state["short_term_memory"] = []
            state["semantic_memories"] = []
            state["knowledge_graph_topics"] = []
            self._append_trace(state, "memory", "completed", "Started a new conversation memory.")
            self._save_checkpoint(state["conversation_id"], "memory", "completed", state)
            return state

        short_term = self.memory.load_short_term_context(conversation)
        semantic_hits = self.memory.semantic_search(conversation.id, state["question"])
        preferences = {
            **(conversation.user_preferences or {}),
            **state.get("user_preferences", {}),
        }
        memory_parts: list[str] = []
        if short_term["short_term_memory"]:
            memory_parts.append("Recent turns: " + " | ".join(short_term["short_term_memory"][-3:]))
        if semantic_hits:
            memory_parts.append(
                "Relevant long-term memory: "
                + " | ".join(hit.content for hit in semantic_hits[:3])
            )
        if preferences:
            pref_summary = ", ".join(f"{key}={value}" for key, value in preferences.items())
            memory_parts.append(f"Preferences: {pref_summary}")
        if short_term.get("custom_instructions"):
            memory_parts.append(f"Custom instructions: {short_term['custom_instructions']}")

        state["memory_summary"] = " | ".join(memory_parts) if memory_parts else "New conversation."
        state["short_term_memory"] = short_term["short_term_memory"]
        state["semantic_memories"] = [hit.model_dump(mode="json") for hit in semantic_hits]
        state["knowledge_graph_topics"] = self.memory.top_topics(conversation.id)
        state["custom_instructions"] = short_term.get("custom_instructions") or ""
        self._append_trace(state, "memory", "completed", "Loaded short-term, long-term, and graph memory context.")
        self._save_checkpoint(state["conversation_id"], "memory", "completed", state)
        return state

    def _router_agent(self, state: AgentState) -> AgentState:
        system_prompt = (
            "You are the supervisor for a multi-agent document intelligence system. "
            "Route the query to specialist agents. Return JSON with keys: route, reasoning_mode, "
            "agents, route_reason. Allowed agent names: document, analytical, tool. "
            "Use reasoning_mode=react only when tools or multi-step reasoning are needed."
        )
        human_prompt = (
            f"Question: {state['question']}\n"
            f"Conversation memory: {state.get('memory_summary', 'None')}\n"
            f"Knowledge graph topics: {', '.join(state.get('knowledge_graph_topics', [])) or 'None'}\n"
            f"Ready documents: {self._ready_document_count()}\n"
            "Focus on grounded document answers first, then add analysis or tools only if justified."
        )
        try:
            routed = self._invoke_json_with_state(state, system_prompt, human_prompt)
        except Exception:
            routed = self._heuristic_route(state["question"])

        allowed_agents = [agent for agent in routed.get("agents", ["document"]) if agent in {"document", "analytical", "tool"}]
        if "document" not in allowed_agents:
            allowed_agents.insert(0, "document")

        state["route"] = routed.get("route", "document_grounded")
        state["reasoning_mode"] = routed.get("reasoning_mode", "direct")
        state["route_reason"] = routed.get("route_reason", "The router chose the best specialist path for the question.")
        state["pending_agents"] = allowed_agents
        self._append_trace(state, "router", "completed", state["route_reason"])
        self._save_checkpoint(state["conversation_id"], "router", "completed", state)
        return state

    def _document_agent(self, state: AgentState) -> AgentState:
        top_k = min(state.get("top_k", 4), self.settings.max_context_chunks)
        document_id = UUID(state["document_id"]) if state.get("document_id") else None
        runtime_profile = self._runtime_profile_from_state(state)
        retrieved_chunks = self.retriever.retrieve(
            query=state["question"],
            top_k=top_k,
            document_id=document_id,
            embedding_model=runtime_profile.embedding_model,
        )
        state["retrieved_chunks"] = retrieved_chunks
        self._pop_agent(state, "document")

        if not retrieved_chunks:
            state["document_result"] = {
                "summary": "",
                "entities": [],
                "key_insights": [],
                "grounded": False,
                "notes": ["No relevant chunks were retrieved."],
                "cited_chunk_ids": [],
            }
            return state

        support = self._grounding_assessment(state["question"], retrieved_chunks)
        if not support.supported:
            state["document_result"] = {
                "summary": "",
                "entities": [],
                "key_insights": [],
                "grounded": False,
                "notes": [support.reason],
                "cited_chunk_ids": [],
            }
            return state

        system_prompt = (
            "You are a document understanding agent. Analyze retrieved PDF chunks only. "
            "Return JSON with keys: summary, entities, key_insights, grounded, notes, cited_chunk_ids. "
            "Keep the summary concise and factual."
        )
        human_prompt = (
            f"Question: {state['question']}\n"
            f"Allowed chunk ids: {', '.join(chunk.chunk_id for chunk in retrieved_chunks)}\n"
            f"Retrieved context:\n{self._build_context(retrieved_chunks)}"
        )
        try:
            result = self._invoke_json_with_state(state, system_prompt, human_prompt)
        except Exception:
            result = {
                "summary": trim_excerpt(retrieved_chunks[0].content, 400),
                "entities": [],
                "key_insights": [trim_excerpt(chunk.content, 180) for chunk in retrieved_chunks[:2]],
                "grounded": retrieved_chunks[0].score >= self.settings.min_retrieval_score,
                "notes": ["Fallback summary was generated from retrieved text snippets."],
                "cited_chunk_ids": [chunk.chunk_id for chunk in retrieved_chunks[:2]],
            }
        state["document_result"] = result
        return state

    def _analytical_agent(self, state: AgentState) -> AgentState:
        self._pop_agent(state, "analytical")
        chunks = state.get("retrieved_chunks", [])
        if len({chunk.source for chunk in chunks}) < 2:
            state["analytical_result"] = {
                "analysis": "",
                "comparisons": [],
                "trends": [],
                "notes": ["Not enough document diversity was retrieved for cross-document analysis."],
                "cited_chunk_ids": [chunk.chunk_id for chunk in chunks[:2]],
            }
            return state

        system_prompt = (
            "You are an analytical agent. Compare themes, patterns, and differences across the supplied documents only. "
            "Return JSON with keys: analysis, comparisons, trends, notes, cited_chunk_ids."
        )
        human_prompt = (
            f"Question: {state['question']}\n"
            f"Retrieved context:\n{self._build_context(chunks)}"
        )
        try:
            state["analytical_result"] = self._invoke_json_with_state(state, system_prompt, human_prompt)
        except Exception:
            grouped_sources = sorted({chunk.source for chunk in chunks})
            state["analytical_result"] = {
                "analysis": f"Relevant evidence was found across {len(grouped_sources)} documents: {', '.join(grouped_sources[:4])}.",
                "comparisons": [],
                "trends": [],
                "notes": ["Fallback analytical summary was used."],
                "cited_chunk_ids": [chunk.chunk_id for chunk in chunks[:3]],
            }
        return state

    @staticmethod
    def _safe_math(expression: str) -> float:
        allowed_names = {name: getattr(math, name) for name in ["ceil", "floor", "sqrt", "log", "sin", "cos", "tan"]}
        allowed_names.update({"abs": abs, "round": round})
        parsed = ast.parse(expression, mode="eval")
        for node in ast.walk(parsed):
            if not isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Constant,
                    ast.Name,
                    ast.Load,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.Mod,
                    ast.USub,
                    ast.UAdd,
                    ast.Call,
                ),
            ):
                raise ValueError("Unsupported calculator expression.")
        return float(eval(compile(parsed, "<calc>", "eval"), {"__builtins__": {}}, allowed_names))

    def _web_search(self, query: str) -> list[dict[str, str]]:
        if not self.settings.web_search_enabled or BeautifulSoup is None:
            return []
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[dict[str, str]] = []
        for item in soup.select(".result")[:5]:
            title_node = item.select_one(".result__title")
            snippet_node = item.select_one(".result__snippet")
            link_node = item.select_one(".result__url")
            title = title_node.get_text(" ", strip=True) if title_node else "Untitled"
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            link = link_node.get_text(" ", strip=True) if link_node else ""
            results.append({"title": title, "snippet": snippet, "link": link})
        return results

    @staticmethod
    def _run_python_transform(code: str, data: dict[str, Any]) -> Any:
        lowered = code.lower()
        forbidden_tokens = ["import", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess"]
        if any(token in lowered for token in forbidden_tokens):
            raise ValueError("The requested code transformation used a blocked operation.")
        namespace = {
            "__builtins__": {},
            "data": data,
            "result": None,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "round": round,
            "list": list,
            "dict": dict,
        }
        exec(code, namespace, namespace)
        return namespace.get("result")

    def _generate_chart(self, conversation_id: str, chunks: list[RetrievedChunk]) -> str:
        if not MATPLOTLIB_AVAILABLE or plt is None:
            raise RuntimeError("Chart generation requires matplotlib to be installed.")
        output_path = self.settings.export_dir / f"{conversation_id}-chart.png"
        labels = [chunk.source[:18] for chunk in chunks[:5]]
        values = [chunk.score for chunk in chunks[:5]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color="#2563eb")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Relevance score")
        ax.set_title("Top retrieved evidence")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return str(output_path)

    def _tool_agent(self, state: AgentState) -> AgentState:
        self._pop_agent(state, "tool")
        tools_available = []
        if self.settings.web_search_enabled:
            tools_available.append("web_search")
        if self.settings.calculator_enabled:
            tools_available.append("calculator")
        if self.settings.code_interpreter_enabled:
            tools_available.append("code_interpreter")
        if self.settings.chart_generation_enabled and MATPLOTLIB_AVAILABLE:
            tools_available.append("chart_generator")
        tools_available.extend(["report_generator", "email_generator"])

        observations: list[dict[str, Any]] = []
        final_answer = ""
        for _step in range(self.settings.supervisor_max_steps):
            system_prompt = (
                "You are a tool-use agent using a ReAct style loop. "
                "Choose the single best next action. Return JSON with keys: action, action_input, summary, final_answer. "
                f"Allowed actions: {', '.join(tools_available)}, finish. "
                "Use finish when you have enough evidence."
            )
            human_prompt = (
                f"Question: {state['question']}\n"
                f"Current observations: {json.dumps(observations)}\n"
                f"Retrieved document evidence: {json.dumps([{'chunk_id': c.chunk_id, 'source': c.source, 'score': c.score} for c in state.get('retrieved_chunks', [])[:4]])}"
            )
            try:
                decision = self._invoke_json_with_state(state, system_prompt, human_prompt)
            except Exception:
                lowered = state["question"].lower()
                if any(token in lowered for token in ["latest", "today", "current", "recent"]):
                    decision = {"action": "web_search", "action_input": state["question"], "summary": "Checked live web results."}
                elif any(char.isdigit() for char in lowered) and any(op in lowered for op in ["+", "-", "*", "/", "average"]):
                    decision = {"action": "calculator", "action_input": state["question"], "summary": "Ran a calculation."}
                elif any(token in lowered for token in ["chart", "plot", "visual"]):
                    decision = {"action": "chart_generator", "action_input": "retrieval_scores", "summary": "Generated a chart."}
                else:
                    decision = {"action": "finish", "action_input": "", "summary": "No extra tool use was needed.", "final_answer": ""}

            action = decision.get("action", "finish")
            action_input = decision.get("action_input", "")
            if action == "finish":
                final_answer = decision.get("final_answer", "") or final_answer
                break

            if action == "web_search":
                try:
                    results = self._web_search(str(action_input))
                    observations.append({"tool": action, "result": results})
                except Exception as exc:
                    observations.append({"tool": action, "error": str(exc)})
            elif action == "calculator":
                expression = re.sub(r"[^0-9+\-*/().,^% ]", " ", str(action_input)).replace("^", "**")
                try:
                    result = self._safe_math(expression)
                    observations.append({"tool": action, "result": result, "expression": expression})
                except Exception as exc:
                    observations.append({"tool": action, "error": str(exc), "expression": expression})
            elif action == "code_interpreter":
                data = {
                    "retrieved_chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "score": chunk.score,
                            "page_number": chunk.page_number,
                        }
                        for chunk in state.get("retrieved_chunks", [])
                    ],
                    "document_result": state.get("document_result", {}),
                    "analytical_result": state.get("analytical_result", {}),
                }
                try:
                    result = self._run_python_transform(str(action_input), data)
                    observations.append({"tool": action, "result": result})
                except Exception as exc:
                    observations.append({"tool": action, "error": str(exc)})
            elif action == "chart_generator":
                try:
                    chart_path = self._generate_chart(state["conversation_id"], state.get("retrieved_chunks", []))
                    observations.append({"tool": action, "result": chart_path})
                except Exception as exc:
                    observations.append({"tool": action, "error": str(exc)})
            elif action == "report_generator":
                report = (
                    f"Question: {state['question']}\n"
                    f"Document summary: {state.get('document_result', {}).get('summary', '')}\n"
                    f"Analysis: {state.get('analytical_result', {}).get('analysis', '')}"
                )
                observations.append({"tool": action, "result": report})
            elif action == "email_generator":
                email = (
                    "Subject: Document insight summary\n\n"
                    f"Here is a concise update on your request:\n\n{state.get('document_result', {}).get('summary', '')}"
                )
                observations.append({"tool": action, "result": email})
            else:
                observations.append({"tool": action, "error": "Unsupported tool action."})

        state["tool_result"] = {
            "observations": observations,
            "final_answer": final_answer,
        }
        return state

    def _citation_agent(self, state: AgentState) -> AgentState:
        chunks = state.get("retrieved_chunks", [])
        draft_answer = state.get("final_answer") or state.get("document_result", {}).get("summary", "")
        if not chunks:
            state["citation_result"] = {
                "grounded": False,
                "confidence": 0.0,
                "cited_chunk_ids": [],
                "notes": ["No evidence was available for citation validation."],
            }
            self._save_checkpoint(state["conversation_id"], "citation", "completed", state)
            return state

        support = self._grounding_assessment(state["question"], chunks)
        if not support.supported:
            state["citation_result"] = {
                "grounded": False,
                "confidence": min(self._retrieval_confidence(chunks[:2]), 0.35),
                "cited_chunk_ids": [],
                "notes": [support.reason],
            }
            self._save_checkpoint(state["conversation_id"], "citation", "completed", state)
            return state

        system_prompt = (
            "You are a citation agent. Validate whether the draft answer is grounded in the retrieved document evidence. "
            "Return JSON with keys: grounded, confidence, cited_chunk_ids, notes."
        )
        human_prompt = (
            f"Question: {state['question']}\n"
            f"Draft answer: {draft_answer}\n"
            f"Allowed chunk ids: {', '.join(chunk.chunk_id for chunk in chunks)}\n"
            f"Retrieved context:\n{self._build_context(chunks)}"
        )
        try:
            citation_result = self._invoke_json_with_state(state, system_prompt, human_prompt)
        except Exception:
            citation_result = {
                "grounded": chunks[0].score >= self.settings.min_retrieval_score,
                "confidence": self._retrieval_confidence(chunks[:2]),
                "cited_chunk_ids": [chunk.chunk_id for chunk in chunks[:2]],
                "notes": ["Fallback citation validation was used."],
            }
        state["citation_result"] = citation_result
        self._save_checkpoint(state["conversation_id"], "citation", "completed", state)
        return state

    def _finalize(self, state: AgentState) -> AgentState:
        chunks = state.get("retrieved_chunks", [])
        doc_result = state.get("document_result", {})
        analytical_result = state.get("analytical_result", {})
        tool_result = state.get("tool_result", {})
        support = self._grounding_assessment(state["question"], chunks)

        if not support.supported:
            response = ChatResponse(
                conversation_id=UUID(state["conversation_id"]),
                answer=self._unsupported_answer_text(),
                grounded=False,
                confidence=min(self._retrieval_confidence(chunks[:2]), 0.35),
                answer_mode="insufficient_context",
                citations=[],
                retrieved_chunks=len(chunks),
                system_notes=list(
                    dict.fromkeys(
                        [
                            state.get("route_reason", ""),
                            support.reason,
                            *([self.memory.provider.note] if self.memory.provider.note else []),
                            *doc_result.get("notes", []),
                            *analytical_result.get("notes", []),
                        ]
                    )
                ),
                route=state.get("route", "document_grounded"),
                memory_summary=state.get("memory_summary"),
                short_term_memory=state.get("short_term_memory", []),
                long_term_memory=[MemoryHit(**item) for item in state.get("semantic_memories", [])],
                memory_actions=[MemoryAction(**item) for item in state.get("memory_actions", [])],
                knowledge_graph_topics=state.get("knowledge_graph_topics", []),
                agent_trace=[AgentTrace(**item) for item in state.get("agent_trace", [])],
                exports=[],
            )

            if state.get("export_formats"):
                response = response.model_copy(
                    update={
                        "exports": self._create_exports(
                            response,
                            [str(item) for item in state.get("export_formats", [])],
                        )
                    }
                )

            state["response"] = response
            state["final_answer"] = response.answer
            state["grounded"] = response.grounded
            state["confidence"] = response.confidence
            state["citations"] = response.citations
            state["export_artifacts"] = [artifact.model_dump(mode="json") for artifact in response.exports]
            self._save_checkpoint(state["conversation_id"], "finalize", "completed", state)
            return state

        citations = self._select_citations(state.get("citation_result", {}).get("cited_chunk_ids", []), chunks)

        system_prompt = (
            "You are the final response agent. Create a polished answer using only the supplied evidence and tool outputs. "
            "If confidence is low or grounding is weak, be transparent. Return JSON with keys: answer, grounded, confidence, system_notes."
        )
        human_prompt = (
            f"Question: {state['question']}\n"
            f"Conversation memory: {state.get('memory_summary', 'None')}\n"
            f"Document summary: {json.dumps(doc_result)}\n"
            f"Analytical result: {json.dumps(analytical_result)}\n"
            f"Tool result: {json.dumps(tool_result)}\n"
            f"Citation result: {json.dumps(state.get('citation_result', {}))}"
        )
        try:
            final_result = self._invoke_json_with_state(state, system_prompt, human_prompt)
        except Exception:
            grounded = bool(state.get("citation_result", {}).get("grounded", False) and citations)
            final_result = {
                "answer": tool_result.get("final_answer")
                or analytical_result.get("analysis")
                or doc_result.get("summary")
                or self._unsupported_answer_text(),
                "grounded": grounded,
                "confidence": min(
                    float(state.get("citation_result", {}).get("confidence", 0.0)),
                    self._retrieval_confidence(chunks[:2]),
                ),
                "system_notes": doc_result.get("notes", []) + analytical_result.get("notes", []),
            }

        grounded = bool(final_result.get("grounded")) and bool(citations)
        response = ChatResponse(
            conversation_id=UUID(state["conversation_id"]),
            answer=str(final_result.get("answer", "")).strip()
            or self._unsupported_answer_text(),
            grounded=grounded,
            confidence=max(0.0, min(float(final_result.get("confidence", 0.0)), 1.0)),
            answer_mode="grounded" if grounded else "insufficient_context",
            citations=citations if state.get("include_sources", True) else [],
            retrieved_chunks=len(chunks),
            system_notes=list(
                dict.fromkeys(
                    [
                        state.get("route_reason", ""),
                        *([self.memory.provider.note] if self.memory.provider.note else []),
                        *state.get("citation_result", {}).get("notes", []),
                        *final_result.get("system_notes", []),
                    ]
                )
            ),
            route=state.get("route", "document_grounded"),
            memory_summary=state.get("memory_summary"),
            short_term_memory=state.get("short_term_memory", []),
            long_term_memory=[MemoryHit(**item) for item in state.get("semantic_memories", [])],
            memory_actions=[MemoryAction(**item) for item in state.get("memory_actions", [])],
            knowledge_graph_topics=state.get("knowledge_graph_topics", []),
            agent_trace=[AgentTrace(**item) for item in state.get("agent_trace", [])],
            exports=[],
        )

        if state.get("export_formats"):
            response = response.model_copy(
                update={
                    "exports": self._create_exports(
                        response,
                        [str(item) for item in state.get("export_formats", [])],
                    )
                }
            )

        state["response"] = response
        state["final_answer"] = response.answer
        state["grounded"] = response.grounded
        state["confidence"] = response.confidence
        state["citations"] = response.citations
        state["export_artifacts"] = [artifact.model_dump(mode="json") for artifact in response.exports]
        self._save_checkpoint(state["conversation_id"], "finalize", "completed", state)
        return state

    def _create_exports(self, response: ChatResponse, formats: list[str]):
        return self.exports.export_chat_response(
            conversation_id=response.conversation_id,
            response=response,
            formats=formats,
        )

    def _log_query_experiment(
        self,
        *,
        response: ChatResponse,
        payload: ChatRequest,
        retrieved_chunks: list[RetrievedChunk],
        runtime_profile: RuntimeModelProfile,
        latency_ms: float,
    ) -> None:
        retrieval_metrics = self.model_management.retrieval_metrics(
            retrieved_chunks=retrieved_chunks,
            cited_chunk_ids={citation.chunk_id for citation in response.citations},
            top_k=max(payload.top_k, 1),
            grounding_threshold=self.settings.min_retrieval_score,
        )
        llm_metrics = self.model_management.llm_metrics(
            grounded=response.grounded,
            confidence=response.confidence,
            latency_ms=latency_ms,
        )
        costs = self.model_management.estimate_query_cost(
            query_text=payload.query,
            answer_text=response.answer,
            retrieved_chunks=len(retrieved_chunks),
            query_type=response.route,
        )
        observe_query(
            route=response.route,
            bucket=runtime_profile.assignment_bucket,
            answer_mode=response.answer_mode,
            latency_seconds=max(latency_ms / 1000.0, 0.0),
            cost_usd=costs.get("estimated_total_cost_usd", 0.0),
            chat_model=runtime_profile.chat_model,
        )
        experiment = self.model_management.log_query_experiment(
            conversation_id=response.conversation_id,
            experiment_name=f"query-{response.route}",
            query_type=response.route,
            runtime_profile=runtime_profile,
            parameters_json={
                "top_k": payload.top_k,
                "include_sources": payload.include_sources,
                "document_id": str(payload.document_id) if payload.document_id else None,
                "pipeline_version": self.settings.pipeline_version,
            },
            metrics_json={
                **retrieval_metrics,
                **llm_metrics,
                "grounded": float(response.grounded),
                "confidence": response.confidence,
            },
            costs_json=costs,
            latency_ms=latency_ms,
            metadata_json={
                "route": response.route,
                "answer_mode": response.answer_mode,
                "retrieved_chunks": len(retrieved_chunks),
            },
        )
        if runtime_profile.shadow_enabled:
            self.model_management.log_shadow_evaluation(
                experiment_run_id=experiment.id,
                conversation_id=response.conversation_id,
                runtime_profile=runtime_profile,
                latency_ms=latency_ms,
                metrics_json={
                    "grounded": float(response.grounded),
                    "confidence": response.confidence,
                    "proxy_ndcg": retrieval_metrics.get("ndcg", 0.0),
                },
                metadata_json={
                    "mode": "metadata_only",
                    "note": "Shadow candidate was logged without affecting the user response.",
                },
            )

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE or StateGraph is None:
            return None
        if self._graph is not None:
            return self._graph

        graph = StateGraph(AgentState)
        graph.add_node("memory", lambda state: self._memory_agent(state))
        graph.add_node("router", lambda state: self._router_agent(state))
        graph.add_node("document", lambda state: self._run_with_retry("document", state, self._document_agent))
        graph.add_node("analytical", lambda state: self._run_with_retry("analytical", state, self._analytical_agent))
        graph.add_node("tool", lambda state: self._run_with_retry("tool", state, self._tool_agent))
        graph.add_node("citation", lambda state: self._run_with_retry("citation", state, self._citation_agent))
        graph.add_node("finalize", lambda state: self._run_with_retry("finalize", state, self._finalize))

        graph.add_edge(START, "memory")
        graph.add_edge("memory", "router")
        graph.add_conditional_edges(
            "router",
            self._next_agent,
            {
                "document": "document",
                "analytical": "analytical",
                "tool": "tool",
                "citation": "citation",
            },
        )
        for node_name in ["document", "analytical", "tool"]:
            graph.add_conditional_edges(
                node_name,
                self._next_agent,
                {
                    "document": "document",
                    "analytical": "analytical",
                    "tool": "tool",
                    "citation": "citation",
                },
            )
        graph.add_edge("citation", "finalize")
        graph.add_edge("finalize", END)

        self._graph = graph.compile(checkpointer=self._checkpointer)
        return self._graph

    def _run_without_langgraph(self, state: AgentState) -> AgentState:
        state = self._memory_agent(state)
        state = self._router_agent(state)
        while state.get("pending_agents"):
            next_agent = self._next_agent(state)
            if next_agent == "document":
                state = self._run_with_retry("document", state, self._document_agent)
            elif next_agent == "analytical":
                state = self._run_with_retry("analytical", state, self._analytical_agent)
            elif next_agent == "tool":
                state = self._run_with_retry("tool", state, self._tool_agent)
            else:
                break
        state = self._run_with_retry("citation", state, self._citation_agent)
        state = self._run_with_retry("finalize", state, self._finalize)
        return state

    def _persist_response(
        self,
        response: ChatResponse,
        payload: ChatRequest,
        retrieved_chunks: list[RetrievedChunk] | None = None,
    ) -> ChatResponse:
        turn = self.conversations.add_turn(
            conversation_id=response.conversation_id,
            query=payload.query,
            answer=response.answer,
            route=response.route,
            grounded=response.grounded,
            confidence=response.confidence,
            response_payload=response.model_dump(mode="json"),
        )
        self.conversations.add_checkpoint(
            conversation_id=response.conversation_id,
            turn_id=turn.id,
            node_name="turn_persisted",
            status="completed",
            state_payload=response.model_dump(mode="json"),
        )
        conversation = self.conversations.get_conversation(response.conversation_id)
        if conversation is None:
            return response

        memory_update = self.memory.update_after_response(
            conversation=conversation,
            query=payload.query,
            response_text=response.answer,
            route=response.route,
            retrieved_chunks=retrieved_chunks or [],
            user_preferences=payload.user_preferences,
        )
        active_document_id = memory_update.get("active_document_id")
        self.conversations.update_memory(
            conversation_id=response.conversation_id,
            memory_summary=memory_update.get("memory_summary") or response.memory_summary,
            user_preferences=payload.user_preferences,
            active_document_id=active_document_id,
            query_refinement_history=memory_update.get("query_refinement_history"),
            interaction_patterns=memory_update.get("interaction_patterns"),
            custom_instructions=conversation.custom_instructions,
        )
        return response.model_copy(
            update={
                "memory_summary": memory_update.get("memory_summary") or response.memory_summary,
                "knowledge_graph_topics": self.memory.top_topics(response.conversation_id),
            }
        )

    def _simple_retrieval_response(
        self,
        payload: ChatRequest,
        conversation_id: str,
        error_message: str,
        runtime_profile: RuntimeModelProfile | None = None,
    ) -> ChatResponse:
        active_profile = runtime_profile or RuntimeModelProfile(
            chat_model=self.settings.chat_model,
            embedding_model=self.settings.embedding_model,
            retrieval_config_version=self.settings.retrieval_config_version,
            prompt_template_version=self.settings.prompt_template_version,
            assignment_bucket="control",
            shadow_enabled=False,
            shadow_profile={},
        )
        retrieved_chunks = self.retriever.retrieve(
            query=payload.query,
            top_k=min(payload.top_k, self.settings.max_context_chunks),
            document_id=payload.document_id,
            embedding_model=active_profile.embedding_model,
        )
        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page=chunk.page_number,
                score=chunk.score,
                excerpt=trim_excerpt(chunk.content),
            )
            for chunk in retrieved_chunks[:2]
        ]
        support = self._grounding_assessment(payload.query, retrieved_chunks)
        grounded = support.supported
        answer = trim_excerpt(retrieved_chunks[0].content, 400) if grounded and retrieved_chunks else self._unsupported_answer_text()
        conversation = self.conversations.get_conversation(UUID(conversation_id))
        memory_hits = self.memory.semantic_search(UUID(conversation_id), payload.query)
        short_term = self.memory.load_short_term_context(conversation) if conversation is not None else {"short_term_memory": []}
        return ChatResponse(
            conversation_id=UUID(conversation_id),
            answer=answer,
            grounded=grounded,
            confidence=self._retrieval_confidence(retrieved_chunks[:2]) if grounded else min(self._retrieval_confidence(retrieved_chunks[:2]), 0.35),
            answer_mode="grounded" if grounded else "insufficient_context",
            citations=citations if grounded and payload.include_sources else [],
            retrieved_chunks=len(retrieved_chunks),
            system_notes=[
                f"The orchestrator fell back to simpler retrieval: {error_message}",
                *( [] if grounded else [support.reason]),
                *([self.memory.provider.note] if self.memory.provider.note else []),
            ],
            route="fallback_retrieval",
            memory_summary=conversation.memory_summary if conversation is not None else None,
            short_term_memory=short_term["short_term_memory"],
            long_term_memory=memory_hits,
            memory_actions=[],
            knowledge_graph_topics=self.memory.top_topics(UUID(conversation_id)),
            agent_trace=[
                AgentTrace(
                    agent="supervisor",
                    status="fallback",
                    summary="A simpler retrieval fallback was used after orchestration failed.",
                    retries=self.settings.max_agent_retries,
                )
            ],
            exports=[],
        )

    def answer_question(self, payload: ChatRequest) -> ChatResponse:
        started_at = time.perf_counter()
        conversation = self.conversations.get_or_create_conversation(payload.conversation_id, payload.user_preferences)
        runtime_profile = self.model_management.runtime_profile(str(conversation.id))
        command_response = self._handle_memory_command(conversation, payload, {})
        if command_response is not None:
            persisted = self._persist_response(command_response, payload, [])
            self._log_query_experiment(
                response=persisted,
                payload=payload,
                retrieved_chunks=[],
                runtime_profile=runtime_profile,
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
            )
            return persisted

        self._validate_document_scope(payload)
        state: AgentState = {
            "conversation_id": str(conversation.id),
            "document_id": str(payload.document_id) if payload.document_id else None,
            "question": payload.query,
            "top_k": payload.top_k,
            "include_sources": payload.include_sources,
            "export_formats": payload.export_formats,
            "user_preferences": payload.user_preferences,
            "conversation_history": self._history_payload(str(conversation.id)),
            "memory_summary": conversation.memory_summary or "",
            "short_term_memory": [],
            "semantic_memories": [],
            "memory_actions": [],
            "knowledge_graph_topics": [],
            "custom_instructions": conversation.custom_instructions or "",
            "runtime_profile": {
                "chat_model": runtime_profile.chat_model,
                "embedding_model": runtime_profile.embedding_model,
                "retrieval_config_version": runtime_profile.retrieval_config_version,
                "prompt_template_version": runtime_profile.prompt_template_version,
                "assignment_bucket": runtime_profile.assignment_bucket,
                "shadow_enabled": runtime_profile.shadow_enabled,
                "shadow_profile": runtime_profile.shadow_profile,
            },
            "experiment_run_id": None,
            "pending_agents": [],
            "system_notes": [],
            "agent_trace": [],
        }
        self._save_checkpoint(str(conversation.id), "start", "started", state)

        retrieved_chunks: list[RetrievedChunk] = []
        try:
            graph = self._build_graph()
            if graph is not None:
                final_state = graph.invoke(
                    state,
                    config={"configurable": {"thread_id": str(conversation.id)}},
                )
            else:
                final_state = self._run_without_langgraph(state)
            response = final_state["response"]
            retrieved_chunks = final_state.get("retrieved_chunks", [])
        except Exception as exc:
            response = self._simple_retrieval_response(payload, str(conversation.id), str(exc), runtime_profile)

        persisted = self._persist_response(response, payload, retrieved_chunks)
        self._log_query_experiment(
            response=persisted,
            payload=payload,
            retrieved_chunks=retrieved_chunks,
            runtime_profile=runtime_profile,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )
        return persisted
