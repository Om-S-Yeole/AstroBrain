from typing import TypedDict


class PDFBatch(TypedDict):
    text: str
    page_start: int
    page_end: int


class MetaDataDict(TypedDict):
    source: str
    page_start: int
    page_end: int
    extra: dict


class ChunkDict(TypedDict):
    chunk: str
    metadata: MetaDataDict
