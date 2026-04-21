from __future__ import annotations

from dataclasses import dataclass
import re

from app.services.documents import RetrievedChunk, normalize_text, tokenize

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)
STRONG_SIMILARITY_THRESHOLD = 0.82
@dataclass(frozen=True)
class GroundingAssessment:
    supported: bool
    reason: str
    query_terms: list[str]
    matched_terms: list[str]
    required_match_count: int
    max_chunk_match_count: int
    max_vector_score: float
    max_retrieval_score: float


def _canonical_token(token: str) -> str:
    lowered = token.lower()
    if lowered.endswith("'s") and len(lowered) > 2:
        return lowered[:-2]
    return lowered


def _looks_like_math_query(question: str) -> bool:
    cleaned = normalize_text(question)
    return bool(cleaned) and bool(re.search(r"\d", cleaned)) and bool(re.search(r"[+\-*/=]", cleaned))


def _informative_query_terms(question: str) -> list[str]:
    tokens = [_canonical_token(token) for token in tokenize(normalize_text(question))]
    filtered = [
        token
        for token in tokens
        if token not in STOPWORDS
        and not token.isdigit()
        and sum(char.isalpha() for char in token) >= 2
    ]
    if filtered:
        return list(dict.fromkeys(filtered))
    return []


def _matched_terms(query_terms: list[str], text: str) -> set[str]:
    if not query_terms:
        return set()
    doc_tokens = {_canonical_token(token) for token in tokenize(normalize_text(text))}
    return {term for term in query_terms if term in doc_tokens}


def assess_grounding_support(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    min_retrieval_score: float,
    strong_similarity_threshold: float = STRONG_SIMILARITY_THRESHOLD,
) -> GroundingAssessment:
    if not chunks:
        return GroundingAssessment(
            supported=False,
            reason="No document evidence was retrieved for this question.",
            query_terms=[],
            matched_terms=[],
            required_match_count=0,
            max_chunk_match_count=0,
            max_vector_score=0.0,
            max_retrieval_score=0.0,
        )

    query_terms = _informative_query_terms(question)
    max_retrieval_score = max(chunk.score for chunk in chunks)
    max_vector_score = max(chunk.vector_score for chunk in chunks)

    if max_retrieval_score < min_retrieval_score:
        return GroundingAssessment(
            supported=False,
            reason="Retrieved evidence scored below the grounding threshold.",
            query_terms=query_terms,
            matched_terms=[],
            required_match_count=0,
            max_chunk_match_count=0,
            max_vector_score=round(max_vector_score, 4),
            max_retrieval_score=round(max_retrieval_score, 4),
        )

    if not query_terms:
        reason = (
            "This looks like a calculation or out-of-scope question, and I could not verify it from the uploaded documents."
            if _looks_like_math_query(question)
            else "The question did not contain enough document-groundable terms to verify against the uploaded knowledge base."
        )
        return GroundingAssessment(
            supported=False,
            reason=reason,
            query_terms=[],
            matched_terms=[],
            required_match_count=0,
            max_chunk_match_count=0,
            max_vector_score=round(max_vector_score, 4),
            max_retrieval_score=round(max_retrieval_score, 4),
        )

    required_match_count = 1 if len(query_terms) == 1 else min(2, len(query_terms))
    matched_terms: set[str] = set()
    max_chunk_match_count = 0
    for chunk in chunks[:3]:
        chunk_matches = _matched_terms(query_terms, chunk.content)
        matched_terms.update(chunk_matches)
        max_chunk_match_count = max(max_chunk_match_count, len(chunk_matches))

    matched_term_list = sorted(matched_terms)
    if len(matched_terms) >= required_match_count or max_chunk_match_count >= required_match_count:
        return GroundingAssessment(
            supported=True,
            reason="Retrieved evidence contains direct support for the question terms.",
            query_terms=query_terms,
            matched_terms=matched_term_list,
            required_match_count=required_match_count,
            max_chunk_match_count=max_chunk_match_count,
            max_vector_score=round(max_vector_score, 4),
            max_retrieval_score=round(max_retrieval_score, 4),
        )

    if matched_term_list and max_vector_score >= strong_similarity_threshold:
        return GroundingAssessment(
            supported=True,
            reason="Retrieved evidence passed the strong semantic-similarity fallback.",
            query_terms=query_terms,
            matched_terms=matched_term_list,
            required_match_count=required_match_count,
            max_chunk_match_count=max_chunk_match_count,
            max_vector_score=round(max_vector_score, 4),
            max_retrieval_score=round(max_retrieval_score, 4),
        )

    preview_terms = ", ".join(query_terms[:4]) or "the question"
    return GroundingAssessment(
        supported=False,
        reason=(
            "Retrieved chunks did not contain enough direct support for the question terms "
            f"({preview_terms})."
        ),
        query_terms=query_terms,
        matched_terms=matched_term_list,
        required_match_count=required_match_count,
        max_chunk_match_count=max_chunk_match_count,
        max_vector_score=round(max_vector_score, 4),
        max_retrieval_score=round(max_retrieval_score, 4),
    )
