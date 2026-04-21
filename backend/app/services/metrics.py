from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Iterator

try:  # pragma: no cover - optional dependency
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful degradation
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = None
    Histogram = None
    generate_latest = None
    PROMETHEUS_AVAILABLE = False


UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _counter(name: str, documentation: str, labelnames: tuple[str, ...]):
    if not PROMETHEUS_AVAILABLE or Counter is None:
        return None
    return Counter(name, documentation, labelnames=labelnames)


def _histogram(name: str, documentation: str, labelnames: tuple[str, ...]):
    if not PROMETHEUS_AVAILABLE or Histogram is None:
        return None
    return Histogram(name, documentation, labelnames=labelnames)


HTTP_REQUESTS_TOTAL = _counter(
    "rag_http_requests_total",
    "Total HTTP requests served by the backend.",
    ("method", "path", "status"),
)
HTTP_REQUEST_DURATION_SECONDS = _histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "path"),
)
QUERY_RUNS_TOTAL = _counter(
    "rag_query_runs_total",
    "Total RAG query executions.",
    ("route", "bucket", "answer_mode"),
)
QUERY_LATENCY_SECONDS = _histogram(
    "rag_query_latency_seconds",
    "Latency for user query processing.",
    ("route", "bucket"),
)
QUERY_COST_USD_TOTAL = _counter(
    "rag_query_cost_usd_total",
    "Estimated query cost aggregated in USD.",
    ("route", "bucket", "chat_model"),
)
EXPERIMENT_RUNS_TOTAL = _counter(
    "rag_experiment_runs_total",
    "Total model-management experiment runs.",
    ("experiment_type", "status", "bucket"),
)
INDEX_RUNS_TOTAL = _counter(
    "rag_index_runs_total",
    "Total indexing runs.",
    ("status", "embedding_model"),
)
INDEX_LATENCY_SECONDS = _histogram(
    "rag_index_latency_seconds",
    "Index build latency in seconds.",
    ("embedding_model",),
)
SHADOW_EVALUATIONS_TOTAL = _counter(
    "rag_shadow_evaluations_total",
    "Total shadow evaluations logged.",
    ("status", "bucket"),
)
PROVIDER_PUBLISH_TOTAL = _counter(
    "rag_provider_publish_total",
    "Total external provider publish attempts.",
    ("provider", "event_type", "status"),
)
AGENT_RUNS_TOTAL = _counter(
    "rag_agent_runs_total",
    "Total multi-agent step executions.",
    ("agent", "route", "status"),
)
AGENT_DURATION_SECONDS = _histogram(
    "rag_agent_duration_seconds",
    "Latency for multi-agent step execution.",
    ("agent", "route", "status"),
)
AGENT_RETRIES_TOTAL = _counter(
    "rag_agent_retries_total",
    "Total retry attempts used by each multi-agent step.",
    ("agent", "route"),
)


def normalize_path(path: str) -> str:
    normalized = UUID_PATTERN.sub("{id}", path)
    normalized = re.sub(r"/rollback/[^/]+/[^/]+", "/rollback/{model_kind}/{semantic_version}", normalized)
    return normalized


def observe_http_request(method: str, path: str, status: int, duration_seconds: float) -> None:
    normalized = normalize_path(path)
    if HTTP_REQUESTS_TOTAL is not None:
        HTTP_REQUESTS_TOTAL.labels(method=method, path=normalized, status=str(status)).inc()
    if HTTP_REQUEST_DURATION_SECONDS is not None:
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=normalized).observe(duration_seconds)


def observe_query(route: str, bucket: str, answer_mode: str, latency_seconds: float, cost_usd: float, chat_model: str) -> None:
    if QUERY_RUNS_TOTAL is not None:
        QUERY_RUNS_TOTAL.labels(route=route, bucket=bucket, answer_mode=answer_mode).inc()
    if QUERY_LATENCY_SECONDS is not None:
        QUERY_LATENCY_SECONDS.labels(route=route, bucket=bucket).observe(latency_seconds)
    if QUERY_COST_USD_TOTAL is not None:
        QUERY_COST_USD_TOTAL.labels(route=route, bucket=bucket, chat_model=chat_model).inc(max(cost_usd, 0.0))


def observe_experiment(experiment_type: str, status: str, bucket: str) -> None:
    if EXPERIMENT_RUNS_TOTAL is not None:
        EXPERIMENT_RUNS_TOTAL.labels(experiment_type=experiment_type, status=status, bucket=bucket).inc()


def observe_index(status: str, embedding_model: str, latency_seconds: float) -> None:
    if INDEX_RUNS_TOTAL is not None:
        INDEX_RUNS_TOTAL.labels(status=status, embedding_model=embedding_model).inc()
    if INDEX_LATENCY_SECONDS is not None:
        INDEX_LATENCY_SECONDS.labels(embedding_model=embedding_model).observe(latency_seconds)


def observe_shadow(status: str, bucket: str) -> None:
    if SHADOW_EVALUATIONS_TOTAL is not None:
        SHADOW_EVALUATIONS_TOTAL.labels(status=status, bucket=bucket).inc()


def observe_provider_publish(provider: str, event_type: str, success: bool) -> None:
    if PROVIDER_PUBLISH_TOTAL is not None:
        PROVIDER_PUBLISH_TOTAL.labels(
            provider=provider,
            event_type=event_type,
            status="success" if success else "failure",
        ).inc()


def observe_agent_execution(agent: str, route: str, status: str, duration_seconds: float, retries: int = 0) -> None:
    if AGENT_RUNS_TOTAL is not None:
        AGENT_RUNS_TOTAL.labels(agent=agent, route=route, status=status).inc()
    if AGENT_DURATION_SECONDS is not None:
        AGENT_DURATION_SECONDS.labels(agent=agent, route=route, status=status).observe(max(duration_seconds, 0.0))
    if retries > 0 and AGENT_RETRIES_TOTAL is not None:
        AGENT_RETRIES_TOTAL.labels(agent=agent, route=route).inc(retries)


def render_metrics() -> bytes:
    if not PROMETHEUS_AVAILABLE or generate_latest is None:
        return b"# prometheus_client is not installed\n"
    return generate_latest()


@contextmanager
def measure_time() -> Iterator[callable]:
    started = time.perf_counter()

    def elapsed_seconds() -> float:
        return max(time.perf_counter() - started, 0.0)

    yield elapsed_seconds
