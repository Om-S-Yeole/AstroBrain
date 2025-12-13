from app.rag import MetaDataDict


def prepare_metadata(
    text: str, source: str, page_start: int, page_end: int, extra=None
) -> MetaDataDict:
    """
    Create a metadata dictionary for document chunks with associated text.

    This function constructs a structured metadata object that associates text content
    with its source document, page range, and optional additional metadata. It's
    primarily used in the document ingestion pipeline to track chunk provenance.

    Parameters
    ----------
    text : str
        The text content of the document chunk.
    source : str
        The source identifier or filename of the document (e.g., "document.pdf").
    page_start : int
        The starting page number for the content (typically 0-indexed or 1-indexed).
    page_end : int
        The ending page number for the content (inclusive).
    extra : dict or None, optional
        Additional metadata to include, such as author, date, or custom fields.
        Default is None, which initializes as an empty dictionary.

    Returns
    -------
    MetaDataDict
        A metadata dictionary containing text, source, page range, and extra information.
        Structure: {'text': str, 'source': str, 'page_start': int, 'page_end': int, 'extra': dict}

    Raises
    ------
    TypeError
        If text or source is not a string, page_start or page_end are not integers,
        or extra is not a dict or None.

    Examples
    --------
    >>> prepare_metadata("Sample text", "document.pdf", 1, 5)
    MetaDataDict(text='Sample text', source='document.pdf', page_start=1, page_end=5, extra={})

    >>> prepare_metadata(
    ...     text="Chapter content",
    ...     source="book.pdf",
    ...     page_start=10,
    ...     page_end=15,
    ...     extra={"author": "John Doe", "chapter": 2}
    ... )
    MetaDataDict(text='Chapter content', source='book.pdf', page_start=10, page_end=15,
                 extra={'author': 'John Doe', 'chapter': 2})
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected type of text is str. Got {type(text)}")
    if not isinstance(source, str):
        raise TypeError(f"Expected type of source is str. Got {type(source)}.")
    if not isinstance(page_start, int):
        raise TypeError(f"Expected type of page_start is int. Got {type(page_start)}.")
    if not isinstance(page_end, int):
        raise TypeError(f"Expected type of page_end is int. Got {type(page_end)}.")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError(f"Expected type of extra is dict or None. Got {type(extra)}.")

    if extra is None:
        extra = {}

    return MetaDataDict(
        text=text, source=source, page_start=page_start, page_end=page_end, extra=extra
    )
