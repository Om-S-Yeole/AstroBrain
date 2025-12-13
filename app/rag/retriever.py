from app.rag._classes import RetrievedChunk
from app.rag.vectorstore import VectorStore


class Retriever:
    """
    A semantic search retriever for querying vector databases.

    This class provides an interface for semantic similarity search using vector embeddings.
    It wraps a VectorStore instance and provides methods to embed queries and retrieve
    the most semantically similar documents.

    Parameters
    ----------
    vectorstore : VectorStore
        An initialized VectorStore instance containing the indexed documents.
    top_k : int, optional
        Default number of top results to retrieve for queries. Default is 10.

    Attributes
    ----------
    vectorstore : VectorStore
        The vector database instance used for storage and retrieval.
    default_top_k : int
        The default number of results to return in queries.

    Raises
    ------
    TypeError
        If vectorstore is not a VectorStore instance or top_k is not an integer.
    ValueError
        If top_k is less than 1.

    Examples
    --------
    >>> vectorstore = VectorStore(
    ...     api_key_pinecone="your-key",
    ...     index_name="my-index",
    ...     namespace="docs",
    ...     embedding_model="openai",
    ...     api_key_embedder="openai-key"
    ... )
    >>> retriever = Retriever(vectorstore, top_k=5)
    >>> results = retriever.retrieve("What is orbital mechanics?")
    >>> len(results)
    5
    """

    def __init__(self, vectorstore: VectorStore, top_k: int = 10):
        """
        Initialize the Retriever with a vector store and default retrieval parameters.

        Parameters
        ----------
        vectorstore : VectorStore
            An initialized VectorStore instance.
        top_k : int, optional
            Default number of results to retrieve. Default is 10.

        Raises
        ------
        TypeError
            If vectorstore is not a VectorStore instance or top_k is not an integer.
        ValueError
            If top_k is less than 1.
        """
        if not isinstance(vectorstore, VectorStore):
            raise TypeError(
                f"Expected type of vectorstore is VectorStore. Got {type(vectorstore)}"
            )
        if not isinstance(top_k, int):
            raise TypeError(f"Expected type of top_k is int. Got {type(top_k)}")
        if top_k < 1:
            raise ValueError(f"Expected top_k >= 1. Got {top_k}.")

        self.vectorstore = vectorstore
        self.default_top_k = top_k

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a text query into an embedding vector.

        This method uses the vectorstore's embedder to generate a vector
        representation of the query text for semantic similarity search.

        Parameters
        ----------
        query : str
            The search query text to embed.

        Returns
        -------
        list of float
            The embedding vector representing the query.

        Raises
        ------
        TypeError
            If query is not a string.
        ValueError
            If query is an empty string or contains only whitespace.

        Examples
        --------
        >>> retriever = Retriever(vectorstore)
        >>> embedding = retriever.embed_query("orbital mechanics")
        >>> len(embedding)
        1024
        >>> isinstance(embedding, list)
        True
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected type of query is str. Got {type(query)}")

        if not query.strip():
            raise ValueError("Got empty string as query.")

        return self.vectorstore.embedder.embed_text(query.strip())

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict | None = None,
        include_metadata: bool = True,
        include_scores: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most semantically similar documents for a given query.

        This method embeds the query and searches the vector database for the most
        similar documents based on cosine similarity. Results can be filtered by
        metadata and configured to include or exclude metadata and similarity scores.

        Parameters
        ----------
        query : str
            The search query text.
        top_k : int, optional
            Number of top results to retrieve. Default is 10.
        metadata_filter : dict or None, optional
            Pinecone metadata filter to narrow down search results.
            Example: {"source": "document.pdf"} or {"page_start": {"$gte": 10}}.
            Default is None (no filtering).
        include_metadata : bool, optional
            Whether to include metadata in the results. Default is True.
        include_scores : bool, optional
            Whether to include similarity scores in the results. Default is True.

        Returns
        -------
        list of RetrievedChunk
            List of retrieved document chunks, each containing:
            - id: Unique identifier
            - text: The chunk text content
            - score: Similarity score (if include_scores=True)
            - metadata: Associated metadata (if include_metadata=True)
            Returns an empty list if no matches are found.

        Raises
        ------
        TypeError
            If top_k is not an integer, metadata_filter is not a dict or None,
            or include_metadata/include_scores are not booleans.
        ValueError
            If top_k is less than 1 or query is empty.

        Examples
        --------
        >>> retriever = Retriever(vectorstore, top_k=5)
        >>> results = retriever.retrieve("What is orbital velocity?")
        >>> for chunk in results:
        ...     print(f"Score: {chunk.score:.3f}")
        ...     print(f"Text: {chunk.text[:100]}...")
        ...     print(f"Source: {chunk.metadata['source']}")
        Score: 0.892
        Text: Orbital velocity is the velocity needed to remain in orbit...
        Source: orbital_mechanics.pdf

        >>> # Retrieve with metadata filtering
        >>> filtered_results = retriever.retrieve(
        ...     "satellite orbits",
        ...     top_k=3,
        ...     metadata_filter={"source": "nasa_handbook.pdf"}
        ... )
        >>> len(filtered_results)
        3

        >>> # Retrieve without scores or metadata
        >>> minimal_results = retriever.retrieve(
        ...     "escape velocity",
        ...     include_metadata=False,
        ...     include_scores=False
        ... )
        >>> minimal_results[0].score is None
        True
        """
        if not isinstance(top_k, int):
            raise TypeError(f"Expected type of top_k is int. Got {type(top_k)}")
        if top_k < 1:
            raise ValueError(f"Expected top_k >= 1. Got {top_k}.")
        if not isinstance(metadata_filter, dict) and metadata_filter is not None:
            raise TypeError(
                f"Expected type of metadata_filter is dict or None. Got {type(metadata_filter)}"
            )
        if not isinstance(include_metadata, bool):
            raise TypeError(
                f"Expected type of include_metadata is bool. Got {type(include_metadata)}"
            )
        if not isinstance(include_scores, bool):
            raise TypeError(
                f"Expected type of include_scores is bool. Got {type(include_scores)}"
            )

        embed_query: list[float] = self.embed_query(query)

        query_results = self.vectorstore.index.query(
            namespace=self.vectorstore.namespace_name,
            vector=embed_query,
            top_k=top_k,
            filter=metadata_filter,
            include_metadata=True,
            include_values=False,
        )

        if query_results["matches"]:
            res = []
            for query_result in query_results["matches"]:
                res.append(
                    RetrievedChunk(
                        id=query_result["id"],
                        text=query_result["metadata"].get("text"),
                        score=query_result["score"] if include_scores else None,
                        metadata=query_result["metadata"] if include_metadata else None,
                    )
                )
            return res
        else:
            return []
