from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import DocumentRecord

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")
GID_TOKEN_PATTERN = re.compile(r"/gid\d+")
LONG_HEX_PATTERN = re.compile(r"\b[a-f0-9]{24,}\b", re.IGNORECASE)
UNDERSCORE_RUN_PATTERN = re.compile(r"_{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    page_number: int | None
    source: str
    score: float
    vector_score: float
    bm25_score: float
    overlap_score: float


def normalize_text(text: str) -> str:
    cleaned = GID_TOKEN_PATTERN.sub(" ", text)
    cleaned = LONG_HEX_PATTERN.sub(" ", cleaned)
    cleaned = UNDERSCORE_RUN_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace("[CrossRef]", " ").replace("[PubMed]", " ")
    cleaned = cleaned.replace("|", " ")
    return " ".join(cleaned.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def text_quality_score(text: str) -> float:
    cleaned = normalize_text(text)
    tokens = tokenize(cleaned)
    if not tokens:
        return 0.0

    alpha_tokens = sum(1 for token in tokens if any(char.isalpha() for char in token))
    digit_tokens = sum(1 for token in tokens if token.isdigit())
    long_tokens = sum(1 for token in tokens if len(token) >= 18)

    score = alpha_tokens / len(tokens)
    score -= min(digit_tokens / max(len(tokens), 1), 0.35) * 0.35
    score -= min(long_tokens / max(len(tokens), 1), 0.4) * 0.45

    if len(cleaned) < 60:
        score *= 0.8
    if re.search(r"[.!?]", cleaned):
        score += 0.08

    return max(0.0, min(score, 1.0))


def trim_excerpt(text: str, limit: int = 280) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return "Preview unavailable."

    candidates = [
        segment.strip()
        for segment in SENTENCE_SPLIT_PATTERN.split(cleaned)
        if len(segment.strip()) >= 48
    ]

    if candidates:
        cleaned = max(candidates, key=text_quality_score)

    if text_quality_score(cleaned) < 0.28:
        return "Preview hidden because this page contains low-quality extracted text."

    if len(cleaned) <= limit:
        return cleaned

    return cleaned[: limit - 3].rstrip() + "..."


def sanitize_filename(filename: str) -> str:
    candidate = Path(filename).name.strip() or "document.pdf"
    sanitized = FILENAME_SANITIZER.sub("-", candidate).strip(".-")
    if not sanitized.lower().endswith(".pdf"):
        sanitized = f"{sanitized or 'document'}.pdf"
    return sanitized


class DocumentService:
    def __init__(self, settings: Settings, session: Session) -> None:
        self.settings = settings
        self.session = session

    def list_documents(self) -> list[DocumentRecord]:
        statement = select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
        return list(self.session.scalars(statement))

    def get_document(self, document_id: UUID) -> DocumentRecord | None:
        return self.session.get(DocumentRecord, document_id)

    def create_uploaded_document(self, upload: UploadFile) -> DocumentRecord:
        filename = sanitize_filename(upload.filename or "document.pdf")
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF uploads are supported.")

        document = DocumentRecord(
            file_name=filename,
            storage_path="",
            status="processing",
        )
        self.session.add(document)
        self.session.flush()

        storage_path = self._build_storage_path(document.id, filename)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        with storage_path.open("wb") as target:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, target)

        document.storage_path = str(storage_path)
        self.session.commit()
        self.session.refresh(document)
        return document

    def mark_processing(self, document: DocumentRecord) -> None:
        document.status = "processing"
        document.error_message = None
        self.session.add(document)
        self.session.commit()

    def mark_failed(self, document: DocumentRecord, error_message: str) -> None:
        document.status = "failed"
        document.error_message = error_message
        self.session.add(document)
        self.session.commit()

    def _build_storage_path(self, document_id: UUID, filename: str) -> Path:
        return self.settings.upload_dir / str(document_id) / filename
