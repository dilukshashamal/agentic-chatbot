from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session, new_session
from app.models.schemas import BuildIndexResponse, DocumentSummary, DocumentUploadResponse
from app.services.documents import DocumentService
from app.services.rag import RAGService

router = APIRouter(prefix="/documents", tags=["documents"])


def _index_document_in_background(document_id: UUID) -> None:
    settings = get_settings()
    session = new_session()
    try:
        RAGService(settings, session).rebuild_index(document_id)
    except Exception as exc:
        session.rollback()
        document_service = DocumentService(settings, session)
        document = document_service.get_document(document_id)
        if document is not None:
            document_service.mark_failed(document, str(exc))
    finally:
        session.close()


@router.get("", response_model=list[DocumentSummary])
def list_documents(session: Session = Depends(get_db_session)) -> list[DocumentSummary]:
    documents = DocumentService(get_settings(), session).list_documents()
    return [DocumentSummary.model_validate(document) for document in documents]


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: Session = Depends(get_db_session),
) -> DocumentUploadResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    try:
        document = DocumentService(get_settings(), session).create_uploaded_document(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - transport-level fallback
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(_index_document_in_background, document.id)
    return DocumentUploadResponse(
        message="Upload accepted. Indexing started in the background.",
        document=DocumentSummary.model_validate(document),
    )


@router.post("/{document_id}/reindex", response_model=BuildIndexResponse)
async def reindex_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> BuildIndexResponse:
    document_service = DocumentService(settings, session)
    document = document_service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    document_service.mark_processing(document)
    background_tasks.add_task(_index_document_in_background, document.id)
    return BuildIndexResponse(
        message="Reindex requested.",
        document_id=document.id,
        page_count=document.page_count or 0,
        chunk_count=document.chunk_count or 0,
    )
