import re
import uuid
from typing import Literal, Tuple

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.rag import ChunkDict, MetaDataDict, VectorPayload


class Embedder:
    """
    A unified interface for text embedding using various providers.

    This class provides a consistent API for generating text embeddings using
    different embedding providers (OpenAI, Google Generative AI, or Ollama).
    It handles model initialization, text embedding, batch processing, and
    vector payload preparation for vector databases.

    Parameters
    ----------
    embedding_model : {'openai', 'ollama', 'google'}
        The embedding provider to use.
    dimensions : int, optional
        The dimensionality of the embedding vectors. Default is 1024.
        Note: For Ollama models, this may be overridden by the model's native dimensions.
    api_key : str or None, optional
        API key required for 'openai' and 'google' providers. Default is None.
    ollama_model : str or None, optional
        The Ollama model name, required when using 'ollama' provider. Default is None.

    Attributes
    ----------
    embedder_provider : str
        The selected embedding provider.
    model : Embeddings
        The initialized embedding model instance.
    model_name : str
        The specific model name being used.
    dimensions : int
        The actual dimensionality of the embedding vectors.

    Raises
    ------
    TypeError
        If embedding_model is not a string or dimensions is not an integer.
    ValueError
        If embedding_model is not one of the supported providers, dimensions is less than 2,
        api_key is missing for 'openai' or 'google', or ollama_model is missing for 'ollama'.

    Examples
    --------
    >>> embedder = Embedder(embedding_model="openai", dimensions=512, api_key="sk-...")
    >>> embedding = embedder.embed_text("Hello, world!")
    >>> len(embedding)
    512

    >>> embedder = Embedder(embedding_model="ollama", ollama_model="nomic-embed-text")
    >>> embeddings = embedder.embed_batch(["Text 1", "Text 2"])
    >>> len(embeddings)
    2
    """

    def __init__(
        self,
        embedding_model: Literal["openai", "ollama", "google"],
        dimensions: int = 1024,
        api_key: str | None = None,
        ollama_model: str | None = None,
    ):
        """
        Initialize the Embedder with the specified embedding provider and configuration.

        Parameters
        ----------
        embedding_model : {'openai', 'ollama', 'google'}
            The embedding provider to use.
        dimensions : int, optional
            The desired embedding vector dimensions. Default is 1024.
        api_key : str or None, optional
            API key for OpenAI or Google providers. Default is None.
        ollama_model : str or None, optional
            Model name for Ollama provider. Default is None.

        Raises
        ------
        TypeError
            If embedding_model is not a string or dimensions is not an integer.
        ValueError
            If embedding_model is not supported, dimensions is less than 2,
            or required authentication parameters are missing.
        """
        if not isinstance(embedding_model, str):
            raise TypeError(
                f"Expected type of embedding_model is str. Got {type(embedding_model)}."
            )
        if embedding_model not in ["openai", "ollama", "google"]:
            raise ValueError(
                f"Valid values for embedding_model are openai, ollama, or google. Got {embedding_model}."
            )
        if not isinstance(dimensions, int):
            raise TypeError(
                f"Expected type of dimensions is int. Got {type(dimensions)}."
            )
        if dimensions < 2:
            raise ValueError(
                f"Value of dimensions must be greater than 1. Got {dimensions}."
            )

        self.embedder_provider = embedding_model
        self.model, self.model_name, self.dimensions = self._init_embedding_model(
            embedding_model, dimensions, api_key, ollama_model
        )

    def _init_embedding_model(
        self,
        embedding_model: Literal["openai", "ollama", "google"],
        dimensions: int = 1024,
        api_key: str | None = None,
        ollama_model: str | None = None,
    ) -> Tuple[Embeddings, str, int]:
        """
        Initialize the specific embedding model based on the provider.

        This internal method handles the initialization of the appropriate
        embedding model instance based on the selected provider, configuring
        it with the necessary parameters.

        Parameters
        ----------
        embedding_model : {'openai', 'ollama', 'google'}
            The embedding provider to initialize.
        dimensions : int, optional
            The desired embedding vector dimensions. Default is 1024.
        api_key : str or None, optional
            API key for OpenAI or Google providers. Default is None.
        ollama_model : str or None, optional
            Model name for Ollama provider. Default is None.

        Returns
        -------
        tuple of (Embeddings, str, int)
            A tuple containing:
            - The initialized embedding model instance
            - The model name/type string
            - The actual dimensions of the embedding vectors

        Raises
        ------
        ValueError
            If required parameters (api_key or ollama_model) are missing for the selected provider.

        Notes
        -----
        For Ollama models, the actual dimensions are determined by the model itself
        and may differ from the requested dimensions parameter.
        """
        model = None
        model_type = None
        dim = None
        match embedding_model:
            case "openai":
                if api_key is None:
                    raise ValueError(
                        "Expected api_key when 'openai' is selected as embedding_model."
                    )
                model = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    dimensions=dimensions,
                    api_key=api_key,
                )
                model_type = "text-embedding-3-large"
                dim = dimensions
            case "google":
                if api_key is None:
                    raise ValueError(
                        "Expected api_key when 'google' is selected as embedding_model."
                    )
                model = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-001",
                    api_key=api_key,
                    task_type="SEMANTIC_SIMILARITY",
                    request_options={"output_dimensionality": dimensions},
                )
                model_type = "gemini-embedding-001"
                dim = dimensions
            case "ollama":
                if ollama_model is None:
                    raise ValueError(
                        "Expected argument ollama_model when 'ollama' is selected as embedding_model."
                    )
                model = OllamaEmbeddings(
                    model=ollama_model,
                    validate_model_on_init=True,
                    repeat_penalty=1.2,
                )
                temp_embed = model.embed_query("Hello world")
                # Dimensions may override when ollama is used.
                dim = len(temp_embed)
                model_type = ollama_model

        return model, model_type, dim

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text string.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list of float
            The embedding vector as a list of floating-point numbers.
            Returns an empty list if the text is empty or only whitespace.

        Raises
        ------
        TypeError
            If text is not a string.

        Examples
        --------
        >>> embedder = Embedder(embedding_model="openai", api_key="sk-...")
        >>> embedding = embedder.embed_text("Sample text")
        >>> len(embedding)
        1024
        >>> embedder.embed_text("   ")
        []
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected type of text is str. Got {type(text)}.")
        if not text.strip():
            return []
        return self.model.embed_query(text)

    def embed_batch(self, chunks: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for multiple text strings in batch.

        This method is more efficient than calling embed_text multiple times
        as it processes all texts in a single batch operation.

        Parameters
        ----------
        chunks : list of str
            A list of text strings to embed.

        Returns
        -------
        list of list of float
            A list of embedding vectors, one for each input text.

        Raises
        ------
        TypeError
            If chunks is not a list or if any element in chunks is not a string.

        Examples
        --------
        >>> embedder = Embedder(embedding_model="openai", api_key="sk-...")
        >>> texts = ["First text", "Second text", "Third text"]
        >>> embeddings = embedder.embed_batch(texts)
        >>> len(embeddings)
        3
        >>> all(len(emb) == embedder.dimensions for emb in embeddings)
        True
        """
        if not isinstance(chunks, list):
            raise TypeError(f"Expected type of chunks is list. Got {type(chunks)}.")
        if not all(isinstance(chunk, str) for chunk in chunks):
            raise TypeError(
                "Expected that chunks is list of str. Got non str element in chunks"
            )
        return self.model.embed_documents(chunks)

    def generate_id(self, source_name: str, page_start: int, page_end: int) -> str:
        """
        Generate a unique identifier for a text chunk based on its source and page range.

        The ID combines a normalized source name, page range, and a UUID to ensure
        uniqueness while maintaining traceability to the source document.

        Parameters
        ----------
        source_name : str
            The name or identifier of the source document.
        page_start : int
            The starting page number of the chunk.
        page_end : int
            The ending page number of the chunk.

        Returns
        -------
        str
            A unique identifier in the format: "{normalized_source}_{page_start}_{page_end}_{uuid}".

        Raises
        ------
        TypeError
            If source_name is not a string, or page_start/page_end are not integers.

        Notes
        -----
        The source_name is normalized by:
        - Converting to lowercase
        - Replacing non-alphanumeric characters with underscores
        - Stripping leading/trailing underscores

        Examples
        --------
        >>> embedder = Embedder(embedding_model="openai", api_key="sk-...")
        >>> id1 = embedder.generate_id("My Document.pdf", 1, 5)
        >>> id1.startswith("my_document_pdf_1_5_")
        True
        >>> id2 = embedder.generate_id("My Document.pdf", 1, 5)
        >>> id1 != id2  # Different UUIDs ensure uniqueness
        True
        """
        if not isinstance(source_name, str):
            raise TypeError(
                f"Expected type of source_name is str. Got {type(source_name)}."
            )
        if not isinstance(page_start, int):
            raise TypeError(
                f"Expected type of page_start is int. Got {type(page_start)}."
            )
        if not isinstance(page_end, int):
            raise TypeError(f"Expected type of page_end is int. Got {type(page_end)}.")

        source_name = re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_")

        id = f"{source_name}_{page_start}_{page_end}_{uuid.uuid4()}"

        return id

    def prepare_vector_payload(
        self, embedding: list[float], metadata: dict | MetaDataDict, vector_id: str
    ) -> VectorPayload:
        """
        Create a vector payload ready for insertion into a vector database.

        Combines an embedding vector, metadata, and unique ID into a structured
        payload format suitable for vector database operations.

        Parameters
        ----------
        embedding : list of float
            The embedding vector.
        metadata : MetaDataDict
            Dictionary containing metadata about the text chunk (source, page range, etc.).
        vector_id : str
            Unique identifier for this vector.

        Returns
        -------
        VectorPayload
            A dictionary with 'id', 'values', and 'metadata' keys formatted for vector database insertion.

        Raises
        ------
        TypeError
            If embedding is not a list of floats, metadata is not a dict, or vector_id is not a string.

        Examples
        --------
        >>> embedder = Embedder(embedding_model="openai", api_key="sk-...")
        >>> embedding = [0.1, 0.2, 0.3, ...]
        >>> metadata = {"source": "doc.pdf", "page_start": 1, "page_end": 5, "extra": {}}
        >>> vector_id = "doc_1_5_uuid"
        >>> payload = embedder.prepare_vector_payload(embedding, metadata, vector_id)
        >>> payload['id']
        'doc_1_5_uuid'
        """
        if not isinstance(embedding, list):
            raise TypeError(
                f"Expected type of embedding is list. Got {type(embedding)}."
            )
        if not all(isinstance(element, float) for element in embedding):
            raise TypeError(
                "Expected that embedding is list of float. Got non float element in list."
            )
        metadata = MetaDataDict.model_validate(metadata)
        if not isinstance(vector_id, str):
            raise TypeError(
                f"Expected type of vector_id is str. Got {type(vector_id)}."
            )

        return VectorPayload(id=vector_id, values=embedding, metadata=metadata)

    def embed_chunk(self, chunk_dict: ChunkDict) -> VectorPayload:
        """
        Process a complete chunk dictionary into a vector payload.

        This is a convenience method that combines embedding generation, ID creation,
        and payload preparation in a single operation.

        Parameters
        ----------
        chunk_dict : ChunkDict
            A dictionary containing 'chunk' (text) and 'metadata' keys.

        Returns
        -------
        VectorPayload
            A complete vector payload ready for vector database insertion.

        Raises
        ------
        TypeError
            If chunk_dict is not a dictionary with the expected structure.

        See Also
        --------
        embed_text : Generates embeddings for text.
        generate_id : Creates unique identifiers.
        prepare_vector_payload : Formats the final payload.

        Examples
        --------
        >>> embedder = Embedder(embedding_model="openai", api_key="sk-...")
        >>> chunk_dict = {
        ...     "chunk": "This is a text chunk",
        ...     "metadata": {"source": "doc.pdf", "page_start": 1, "page_end": 5, "extra": {}}
        ... }
        >>> payload = embedder.embed_chunk(chunk_dict)
        >>> 'id' in payload and 'values' in payload and 'metadata' in payload
        True
        """
        chunk_dict = ChunkDict.model_validate(chunk_dict)

        chunk, metadata = chunk_dict["chunk"], chunk_dict["metadata"]

        embed = self.embed_text(chunk)
        id = self.generate_id(
            metadata["source"], metadata["page_start"], metadata["page_end"]
        )
        return self.prepare_vector_payload(embed, metadata, id)
