from app.rag import MetaDataDict


def prepare_metadata(
    source: str, page_start: int, page_end: int, extra=None
) -> MetaDataDict:
    """
    Create a metadata dictionary for document chunks.

    Parameters
    ----------
    source : str
        The source identifier or filename of the document.
    page_start : int
        The starting page number for the content.
    page_end : int
        The ending page number for the content.
    extra : dict or None, optional
        Additional metadata to include. Default is None.

    Returns
    -------
    MetaDataDict
        A metadata dictionary containing source, page range, and extra information.

    Raises
    ------
    TypeError
        If source is not a string, page_start or page_end are not integers,
        or extra is not a dict or None.

    Examples
    --------
    >>> prepare_metadata("document.pdf", 1, 5)
    MetaDataDict(source='document.pdf', page_start=1, page_end=5, extra={})

    >>> prepare_metadata("doc.pdf", 10, 15, extra={"author": "John"})
    MetaDataDict(source='doc.pdf', page_start=10, page_end=15, extra={'author': 'John'})
    """
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
        source=source, page_start=page_start, page_end=page_end, extra=extra
    )
