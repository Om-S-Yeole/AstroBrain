import re

from app.rag import PDFBatch


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing unwanted characters and formatting.

    This function performs comprehensive text cleaning including:
    - Removing control characters
    - Normalizing Unicode punctuation
    - Removing isolated page numbers and headers
    - Fixing hyphenated line breaks
    - Merging continuation lines
    - Normalizing units and spacing

    Parameters
    ----------
    text : str
        The raw text to clean.

    Returns
    -------
    str
        The cleaned and normalized text.

    Raises
    ------
    TypeError
        If text is not a string.

    Examples
    --------
    >>> clean_text("Word-\nbreak continues here")
    'Wordbreak continues here'

    >>> clean_text("km. deg. km / s")
    'km deg km/s'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected type of text is str, got {type(text)}")

    # 1. Remove control characters
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

    # 2. Normalize unicode punctuation
    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "-",
        "\u00ad": "",  # soft hyphen
        "\u2028": " ",  # unicode line separator
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 3. Remove isolated page numbers (but not numbers in text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE)

    # 4. Remove repeated headers/footers (simple heuristic)
    # Example: Remove lines in ALL CAPS under ~30 characters
    text = re.sub(r"^[A-Z\s]{5,40}$", "", text, flags=re.MULTILINE)

    # 5. Fix broken hyphenated line breaks (word- \n break → word)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 6. Merge lines that should be continued (line ends without punctuation)
    text = re.sub(r"(?<![\.\!\?\:])\n+(?!\n)", " ", text)

    # 7. Normalize units
    unit_patterns = {
        r"\bkm\.\b": "km",
        r"\bdeg\.\b": "deg",
        r"\bdegrees\b": "deg",
        r"km\s*/\s*s": "km/s",
    }
    for pattern, repl in unit_patterns.items():
        text = re.sub(pattern, repl, text)

    # 8. Remove extra spaces and collapse multiple newlines
    text = re.sub(r"[ \t]+", " ", text)  # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # max double newlines
    text = text.strip()

    return text


def normalize_text(text: str) -> str:
    """
    Normalize text by standardizing punctuation, units, and spacing.

    This function performs targeted normalization including:
    - Converting Unicode punctuation to ASCII equivalents
    - Standardizing scientific units (km, deg, km/s)
    - Normalizing spacing around punctuation
    - Converting specific uppercase units to lowercase

    Parameters
    ----------
    text : str
        The text to normalize.

    Returns
    -------
    str
        The normalized text.

    Raises
    ------
    TypeError
        If text is not a string.

    Examples
    --------
    >>> normalize_text("Distance: 100 km.")
    'Distance: 100 km'

    >>> normalize_text("Speed–fast")
    'Speed-fast'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected type of text is str, got {type(text)}")

    # -------------------------------------
    # 1. Normalize unicode punctuation
    # -------------------------------------
    unicode_map = {
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
        "…": "...",  # ellipsis
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "-",  # bullets to hyphen
        "·": "",  # mid dot often noise
        "\u00ad": "",  # soft hyphen
        "\u2028": " ",  # line separator
    }

    for key, val in unicode_map.items():
        text = text.replace(key, val)

    # -------------------------------------
    # 2. Normalize scientific units
    # -------------------------------------
    unit_patterns = {
        r"\bkm\.\b": "km",
        r"\bKm\b": "km",
        r"\bKM\b": "km",
        r"\bdeg\.\b": "deg",
        r"\bDegrees\b": "deg",
        r"\bdegrees\b": "deg",
        r"km\s*/\s*s": "km/s",
        r"KM/S": "km/s",
    }

    for pattern, replacement in unit_patterns.items():
        text = re.sub(pattern, replacement, text)

    # -------------------------------------
    # 3. Standardize spacing around punctuation
    # -------------------------------------
    text = re.sub(r"\s*-\s*", "-", text)  # hyphen spacing
    text = re.sub(r"\s*:\s*", ": ", text)  # colon spacing
    text = re.sub(r"\s*,\s*", ", ", text)  # comma spacing
    text = re.sub(r"\s*;\s*", "; ", text)  # semicolon spacing

    # -------------------------------------
    # 4. Normalize multiple spaces
    # -------------------------------------
    text = re.sub(r"[ \t]+", " ", text)

    # -------------------------------------
    # 5. Lowercase optional scientific formatting
    # -------------------------------------
    # You do NOT want to lowercase the whole text
    # because of formulas, abbreviations, and constants.
    # But some documents contain excessive capitalization.

    # Lowercase only certain units, symbols, and acronyms.
    lowercase_map = {
        "KM": "km",
        "DEG": "deg",
        "Km/s": "km/s",
    }

    for old, new in lowercase_map.items():
        text = text.replace(old, new)

    # -------------------------------------
    # 6. Final trim
    # -------------------------------------
    text = text.strip()

    return text


def preprocess_page(text: str) -> str:
    """
    Preprocess a single page of text by cleaning and normalizing.

    This is a convenience function that applies both cleaning and normalization
    operations in sequence.

    Parameters
    ----------
    text : str
        The raw page text to preprocess.

    Returns
    -------
    str
        The preprocessed text.

    Raises
    ------
    TypeError
        If text is not a string.

    See Also
    --------
    clean_text : Performs text cleaning operations.
    normalize_text : Performs text normalization operations.

    Examples
    --------
    >>> preprocess_page("Raw page text with   extra spaces")
    'Raw page text with extra spaces'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected type of text is str, got {type(text)}")

    return normalize_text(clean_text(text))


def preprocess_batch(batch: PDFBatch) -> PDFBatch:
    """
    Preprocess a batch of PDF pages by cleaning and normalizing the text.

    Parameters
    ----------
    batch : PDFBatch
        A batch dictionary containing 'text', 'page_start', and 'page_end' keys.

    Returns
    -------
    PDFBatch
        The batch with preprocessed text.

    Raises
    ------
    TypeError
        If batch is not a PDFBatch instance.

    See Also
    --------
    preprocess_page : Preprocesses individual page text.

    Examples
    --------
    >>> batch = PDFBatch(text="Raw text", page_start=1, page_end=5)
    >>> processed = preprocess_batch(batch)
    >>> processed['text']
    'Raw text'
    """
    if not isinstance(batch, dict):
        raise TypeError(
            f"Expected type of batch is PDFBatch. See ingest.py file for PDFBatch implementation. Got {type(batch)}."
        )

    batch["text"] = preprocess_page(batch["text"])

    return batch
