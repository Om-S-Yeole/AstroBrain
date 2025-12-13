import os
import warnings
from pathlib import Path
from typing import Iterator, Tuple

from pypdf import PdfReader

from app.rag import (
    ChunkDict,
    MetaDataDict,
    PDFBatch,
    chunk_text,
    prepare_metadata,
    preprocess_batch,
)


def stream_pdf_pages(path: str, show_warn: bool = True) -> Iterator[Tuple[int, str]]:
    """
    Stream individual pages from a PDF file as (page_number, text) tuples.

    Parameters
    ----------
    path : str
        Path to the PDF file.
    show_warn : bool, optional
        Whether to show warnings for extraction errors. Default is True.

    Yields
    ------
    tuple of (int, str)
        A tuple containing the page number (0-indexed) and extracted text.

    Raises
    ------
    TypeError
        If path is not a string.
    FileNotFoundError
        If the PDF file does not exist.

    Examples
    --------
    >>> for page_no, text in stream_pdf_pages("document.pdf"):
    ...     print(f"Page {page_no}: {len(text)} characters")
    Page 0: 1234 characters
    Page 1: 987 characters
    """
    if not isinstance(path, str):
        raise TypeError(f"Expected type of path is str. Got {type(path)}.")

    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    reader = PdfReader(pdf_path)

    for i, page in enumerate(reader.pages):
        page_no = i

        try:
            text = page.extract_text(extraction_mode="plain") or ""
        except Exception as e:
            if show_warn:
                warnings.warn(f"Unable to extract text from page {page_no}: {e}")
            text = ""

        yield page_no, text


def stream_pdf_batches(
    path: str, batch_size: int = 5, show_warn: bool = True
) -> Iterator[PDFBatch]:
    """
    Stream batches of consecutive PDF pages as combined text.

    Groups consecutive pages into batches of specified size, combining their
    text content into a single string per batch.

    Parameters
    ----------
    path : str
        Path to the PDF file.
    batch_size : int, optional
        Number of pages to include in each batch. Default is 5.
    show_warn : bool, optional
        Whether to show warnings for extraction errors. Default is True.

    Yields
    ------
    PDFBatch
        A batch dictionary with 'text', 'page_start', and 'page_end' keys.

    Raises
    ------
    TypeError
        If batch_size is not an integer.
    ValueError
        If batch_size is less than 1.
    FileNotFoundError
        If the PDF file does not exist.

    Examples
    --------
    >>> for batch in stream_pdf_batches("document.pdf", batch_size=3):
    ...     print(f"Pages {batch['page_start']}-{batch['page_end']}")
    Pages 0-2
    Pages 3-5
    """
    if not isinstance(batch_size, int):
        raise TypeError(f"Expected type of batch_size is int. Got {type(batch_size)}.")
    if batch_size < 1:
        raise ValueError(
            f"Batch size must be greater than or equal to 1. Got {batch_size}."
        )

    page_it = stream_pdf_pages(path, show_warn)

    while True:
        batch_text = []
        page_start = None
        page_end = None

        for _ in range(batch_size):
            try:
                page_no, text = next(page_it)
            except StopIteration:
                break

            if page_start is None:
                page_start = page_no
            page_end = page_no
            batch_text.append(text)

        if page_start is None:
            return

        yield PDFBatch(
            text="".join(batch_text), page_start=page_start, page_end=page_end
        )


def ingest_pdf_to_batches_and_metadata(
    path: str, source_name: str, batch_size: int = 5, show_warn: bool = True
) -> Iterator[Tuple[PDFBatch, MetaDataDict]]:
    """
    Ingest PDF pages into preprocessed batches with associated metadata.

    Streams batches of PDF pages, preprocesses the text, and generates
    metadata for each batch.

    Parameters
    ----------
    path : str
        Path to the PDF file.
    source_name : str
        Identifier name for the source document.
    batch_size : int, optional
        Number of pages per batch. Default is 5.
    show_warn : bool, optional
        Whether to show warnings for extraction errors. Default is True.

    Yields
    ------
    tuple of (PDFBatch, MetaDataDict)
        A tuple containing the preprocessed batch and its metadata.

    Raises
    ------
    TypeError
        If source_name is not a string.
    FileNotFoundError
        If the PDF file does not exist.

    Examples
    --------
    >>> for batch, meta in ingest_pdf_to_batches_and_metadata("doc.pdf", "my_doc"):
    ...     print(f"Source: {meta['source']}, Pages: {meta['page_start']}-{meta['page_end']}")
    Source: my_doc, Pages: 0-4
    """
    if not isinstance(source_name, str):
        raise TypeError(
            f"Expected type of source_name is str. Got {type(source_name)}."
        )

    batch_it: Iterator[PDFBatch] = stream_pdf_batches(path, batch_size, show_warn)

    for _, raw_batch in enumerate(batch_it):
        batch = preprocess_batch(raw_batch)
        metadata = prepare_metadata(
            batch.text, source_name, batch["page_start"], batch["page_end"], extra=None
        )
        yield batch, metadata


def ingest_and_chunk_pdf(
    path: str,
    source_name: str,
    batch_size: int = 5,
    show_warn: bool = True,
    chunk_size: int = 500,
    overlap: int = 100,
) -> Iterator[ChunkDict]:
    """
    Ingest a PDF file and yield preprocessed, chunked text with metadata.

    This function combines PDF ingestion, preprocessing, and text chunking
    into a single pipeline, yielding individual chunks with their metadata.

    Parameters
    ----------
    path : str
        Path to the PDF file.
    source_name : str
        Identifier name for the source document.
    batch_size : int, optional
        Number of pages to batch together before chunking. Default is 5.
    show_warn : bool, optional
        Whether to show warnings for extraction errors. Default is True.
    chunk_size : int, optional
        Maximum size of each text chunk in characters. Default is 500.
    overlap : int, optional
        Number of overlapping characters between chunks. Default is 100.

    Yields
    ------
    ChunkDict
        A dictionary with 'chunk' (text) and 'metadata' keys.

    Raises
    ------
    FileNotFoundError
        If the PDF file does not exist.
    TypeError
        If source_name is not a string.

    Examples
    --------
    >>> for chunk_dict in ingest_and_chunk_pdf("doc.pdf", "my_document"):
    ...     print(f"Chunk: {chunk_dict['chunk'][:50]}...")
    ...     print(f"Source: {chunk_dict['metadata']['source']}")
    Chunk: This is the beginning of the first chunk...
    Source: my_document
    """
    batch_it: Iterator[Tuple[PDFBatch, MetaDataDict]] = (
        ingest_pdf_to_batches_and_metadata(path, source_name, batch_size, show_warn)
    )

    for batch_content in batch_it:
        pdf_batch, metadatadict = batch_content

        chunks: list[str] = chunk_text(pdf_batch["text"], chunk_size, overlap)

        for chunk in chunks:
            yield ChunkDict(
                chunk=chunk, metadata=MetaDataDict(**metadatadict, text=chunk)
            )


def ingest_directory(
    dir_path: str,
    batch_size: int = 5,
    show_warn: bool = True,
    chunk_size: int = 500,
    overlap: int = 100,
) -> Iterator[ChunkDict]:
    """
    Recursively ingest all PDF files from a directory and yield text chunks.

    Traverses a directory recursively, processing all PDF files found and
    yielding their preprocessed, chunked content with metadata. Hidden files,
    symlinks, and non-PDF files are skipped.

    Parameters
    ----------
    dir_path : str
        Path to the directory to ingest.
    batch_size : int, optional
        Number of pages to batch together before chunking. Default is 5.
    show_warn : bool, optional
        Whether to show warnings for extraction errors. Default is True.
    chunk_size : int, optional
        Maximum size of each text chunk in characters. Default is 500.
    overlap : int, optional
        Number of overlapping characters between chunks. Default is 100.

    Yields
    ------
    ChunkDict
        A dictionary with 'chunk' (text) and 'metadata' keys for each chunk
        from all PDF files in the directory tree.

    Raises
    ------
    TypeError
        If dir_path is not a string.
    FileNotFoundError
        If the directory does not exist.

    Notes
    -----
    - PDF filenames are normalized by converting to lowercase and replacing
      spaces, hyphens, and periods with underscores.
    - The function processes directories recursively.
    - Currently only supports PDF files; other file types are ignored.

    Examples
    --------
    >>> for chunk_dict in ingest_directory("./documents"):
    ...     print(f"Source: {chunk_dict['metadata']['source']}")
    ...     print(f"Chunk length: {len(chunk_dict['chunk'])}")
    Source: document_1
    Chunk length: 485
    Source: document_2
    Chunk length: 512
    """
    if not isinstance(dir_path, str):
        raise TypeError(f"Expected type of dir_path is str. Got {type(dir_path)}.")

    if not Path(dir_path).exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")

    with os.scandir(dir_path) as dir_it:
        for entry in dir_it:
            entry_name, entry_ext = os.path.splitext(entry.name)

            if entry.name.startswith("."):
                continue
            elif entry.is_symlink():
                continue
            elif entry.is_dir():
                yield from ingest_directory(
                    entry.path, batch_size, show_warn, chunk_size, overlap
                )
            elif (
                entry.is_file() and entry_ext.lower() == ".pdf"
            ):  # Currently only supports .pdf files
                entry_name = (
                    entry_name.lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace(".", "_")
                )

                yield from ingest_and_chunk_pdf(
                    entry.path, entry_name, batch_size, show_warn, chunk_size, overlap
                )
            else:
                continue
