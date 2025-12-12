from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.

    This function splits text into smaller chunks with configurable size and overlap,
    using separators in order of priority: double newlines, single newlines, periods,
    and spaces.

    Parameters
    ----------
    text : str
        The text to split into chunks.
    chunk_size : int, optional
        Maximum size of each chunk in characters. Default is 500.
    overlap : int, optional
        Number of overlapping characters between consecutive chunks. Default is 100.

    Returns
    -------
    list of str
        List of text chunks, each with maximum length of chunk_size.

    Raises
    ------
    TypeError
        If text is not a string, or chunk_size or overlap are not integers.
    ValueError
        If overlap is greater than or equal to chunk_size.

    Examples
    --------
    >>> text = "This is a sample text. It will be split into chunks."
    >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
    >>> len(chunks)
    3

    >>> chunk_text("Short text", chunk_size=50, overlap=10)
    ['Short text']
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected type of text is str. Got {type(text)}.")
    if not isinstance(chunk_size, int):
        raise TypeError(f"Expected type of chunk_size is int. Got {type(chunk_size)}.")
    if not isinstance(overlap, int):
        raise TypeError(f"Expected type of overlap is int. Got {type(overlap)}.")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap must be smaller than chunk_size. Got overlap: {overlap} and chunk_size: {chunk_size}"
        )

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    return splitter.split_text(text)
