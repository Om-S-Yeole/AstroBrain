from typing import Optional

from pydantic import BaseModel


class PDFBatch(BaseModel):
    text: str
    page_start: int
    page_end: int


class MetaDataDict(BaseModel):
    text: str
    source: str
    page_start: int
    page_end: int
    # extra: dict


class ChunkDict(BaseModel):
    chunk: str
    metadata: MetaDataDict


class VectorPayload(BaseModel):
    id: str
    values: list[float]
    metadata: MetaDataDict


class RetrievedChunk(BaseModel):
    id: str
    score: Optional[float] = None
    text: str
    metadata: Optional[MetaDataDict] = None
