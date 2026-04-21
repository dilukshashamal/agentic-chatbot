from uuid import uuid4

from app.services.documents import RetrievedChunk
from app.services.grounding import assess_grounding_support


def _chunk(
    content: str,
    *,
    score: float = 0.52,
    vector_score: float = 0.74,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc-test-chunk-0001",
        document_id=uuid4(),
        content=content,
        page_number=1,
        source="sample.pdf",
        score=score,
        vector_score=vector_score,
        bm25_score=0.0,
        overlap_score=0.0,
    )


def test_grounding_rejects_unrelated_question_even_with_moderate_similarity() -> None:
    assessment = assess_grounding_support(
        "Who is the CEO of Google?",
        [
            _chunk(
                "Introduction to Artificial Intelligence. Artificial intelligence refers to the simulation of human intelligence in machines."
            )
        ],
        min_retrieval_score=0.18,
    )

    assert assessment.supported is False
    assert "question terms" in assessment.reason


def test_grounding_accepts_direct_term_support() -> None:
    assessment = assess_grounding_support(
        "What is a DNA sequence?",
        [
            _chunk(
                "A DNA sequence is made of nucleotides such as adenine, cytosine, guanine, and thymine."
            )
        ],
        min_retrieval_score=0.18,
    )

    assert assessment.supported is True
    assert "dna" in assessment.matched_terms


def test_grounding_can_fall_back_to_strong_semantic_similarity() -> None:
    assessment = assess_grounding_support(
        "Who is the CEO of Google?",
        [
            _chunk(
                "Google's leadership page lists Sundar Pichai as the chief executive officer.",
                vector_score=0.91,
            )
        ],
        min_retrieval_score=0.18,
    )

    assert assessment.supported is True
    assert "semantic-similarity fallback" in assessment.reason


def test_grounding_rejects_math_query_without_documentable_terms() -> None:
    assessment = assess_grounding_support(
        "what is 2+1?",
        [
            _chunk(
                "ni Group G1 G2 G3 G4 G5 G6 A T T C C G G T C G G A C A C G A T G A T G A C A T T C 2.4."
            )
        ],
        min_retrieval_score=0.18,
    )

    assert assessment.supported is False
    assert "calculation or out-of-scope question" in assessment.reason
