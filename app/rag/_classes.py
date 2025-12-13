from pydantic import BaseModel


class PDFBatch(BaseModel):
    text: str
    page_start: int
    page_end: int


class MetaDataDict(BaseModel):
    source: str
    page_start: int
    page_end: int
    extra: dict


class ChunkDict(BaseModel):
    chunk: str
    metadata: MetaDataDict


class VectorPayload(BaseModel):
    id: str
    values: list[float]
    metadata: MetaDataDict
