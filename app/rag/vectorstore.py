import warnings
from typing import Iterator, Literal

from pinecone import Pinecone, ServerlessSpec, UpsertResponse

from app.rag._classes import ChunkDict, VectorPayload
from app.rag.embedder import Embedder


class VectorStore:
    """
    A vector database interface for managing embeddings in Pinecone.

    This class provides a high-level interface for storing, retrieving, and managing
    vector embeddings in Pinecone. It handles index creation, embedding generation,
    batch operations, and namespace management.

    Parameters
    ----------
    api_key_pinecone : str
        API key for authenticating with Pinecone.
    index_name : str
        Name of the Pinecone index to use or create.
    namespace : str
        Namespace within the index for organizing vectors.
    embedding_model : {'openai', 'google', 'ollama', 'hugging_face'}
        The embedding provider to use.
    api_key_embedder : str or None, optional
        API key for the embedding provider (required for 'openai' and 'google'). Default is None.
    dimensions : int, optional
        Desired dimensionality of embedding vectors. Default is 1024.
        Note: May be overridden by the actual embedder dimensions.
    ollama_model : str or None, optional
        Model name for Ollama provider (required when using 'ollama'). Default is None.
    hugging_face_model : str or None, optional
        Model name for Hugging Face provider (required when using 'hugging_face'). Default is None.
    hf_device : {'cuda', 'cpu'}, optional
        Device to use for Hugging Face models. Default is 'cuda'.
    cloud : str, optional
        Cloud provider for Pinecone serverless ('aws', 'gcp', or 'azure'). Default is 'aws'.
    region : str, optional
        Cloud region for the Pinecone index. Default is 'us-east-1'.
    index_deletion_protection : bool, optional
        Whether to enable deletion protection for the index. Default is False.

    Attributes
    ----------
    index_name : str
        Name of the Pinecone index.
    namespace_name : str
        Namespace for vector operations.
    embedding_model : str
        The embedding provider being used.
    embedder : Embedder
        The initialized embedder instance.
    dimensions : int
        The actual dimensions of the embedding vectors.
    pinecone_client : Pinecone
        The Pinecone client instance.
    index : pinecone.Index
        The Pinecone index instance.

    Raises
    ------
    TypeError
        If any parameter has an incorrect type.
    ValueError
        If embedding_model is not a supported provider.
    ConnectionError
        If unable to connect to Pinecone client or index.
    RuntimeError
        If embedder dimensions don't match index dimensions.

    Warnings
    --------
    RuntimeWarning
        If requested dimensions differ from actual embedder dimensions.

    Examples
    --------
    >>> vectorstore = VectorStore(
    ...     api_key_pinecone="your-pinecone-key",
    ...     index_name="my-index",
    ...     namespace="documents",
    ...     embedding_model="openai",
    ...     api_key_embedder="your-openai-key",
    ...     dimensions=1024
    ... )
    Connected to index: my-index successfully.

    >>> payload = {"id": "doc1", "values": [...], "metadata": {...}}
    >>> vectorstore.upsert_vector(payload)
    <UpsertResponse: ...>
    """

    def __init__(
        self,
        api_key_pinecone: str,
        index_name: str,
        namespace: str,
        embedding_model: Literal["openai", "google", "ollama", "hugging_face"],
        api_key_embedder: str | None = None,
        dimensions: int = 1024,
        ollama_model: str | None = None,
        hugging_face_model: str | None = None,
        hf_device: Literal["cuda", "cpu"] = "cuda",
        cloud: str = "aws",
        region: str = "us-east-1",
        index_deletion_protection: bool = False,
    ):
        """
        Initialize the VectorStore with Pinecone and embedding configurations.

        Parameters
        ----------
        api_key_pinecone : str
            API key for Pinecone authentication.
        index_name : str
            Name of the Pinecone index.
        namespace : str
            Namespace for organizing vectors.
        embedding_model : {'openai', 'google', 'ollama', 'hugging_face'}
            Embedding provider to use.
        api_key_embedder : str or None, optional
            API key for the embedding provider. Default is None.
        dimensions : int, optional
            Desired embedding dimensions. Default is 1024.
        ollama_model : str or None, optional
            Ollama model name. Default is None.
        hugging_face_model : str or None, optional
            Hugging Face model name. Default is None.
        hf_device : {'cuda', 'cpu'}, optional
            Device for Hugging Face models. Default is 'cuda'.
        cloud : str, optional
            Cloud provider for Pinecone. Default is 'aws'.
        region : str, optional
            Cloud region. Default is 'us-east-1'.
        index_deletion_protection : bool, optional
            Enable deletion protection. Default is False.

        Raises
        ------
        TypeError
            If any parameter has incorrect type.
        ValueError
            If embedding_model is not supported.
        ConnectionError
            If unable to connect to Pinecone.
        RuntimeError
            If embedder and index dimensions don't match.
        """
        if not isinstance(api_key_pinecone, str):
            raise TypeError(
                f"Expected type of api_key_pinecone is str. Got {type(api_key_pinecone)}"
            )
        if not isinstance(index_name, str):
            raise TypeError(
                f"Expected type of index_name is str. Got {type(index_name)}"
            )
        if not isinstance(namespace, str):
            raise TypeError(f"Expected type of namespace is str. Got {type(namespace)}")
        if not isinstance(embedding_model, str):
            raise TypeError(
                f"Expected type of embedding_model is str. Got {type(embedding_model)}"
            )
        if embedding_model not in ["openai", "google", "ollama", "hugging_face"]:
            raise ValueError(
                f"Expected value of embedding_model is from ['openai', 'google', 'ollama', 'hugging_face']. Got {embedding_model}."
            )
        if not isinstance(api_key_embedder, str) and api_key_embedder is not None:
            raise TypeError(
                f"Expected type of api_key_embedder is str or None. Got {type(api_key_embedder)}"
            )
        if not isinstance(dimensions, int):
            raise TypeError(
                f"Expected type of dimensions is int. Got {type(dimensions)}"
            )
        if not isinstance(ollama_model, str) and ollama_model is not None:
            raise TypeError(
                f"Expected type of ollama_model is str or None. Got {type(ollama_model)}"
            )
        if not isinstance(hugging_face_model, str) and hugging_face_model is not None:
            raise TypeError(
                f"Expected type of hugging_face_model is str or None. Got {type(hugging_face_model)}"
            )
        if not isinstance(hf_device, str):
            raise TypeError(f"Expected type of hf_device is str. Got {type(hf_device)}")
        if hf_device not in ["cuda", "cpu"]:
            raise ValueError(
                f"Expected value of hf_device is from ['cuda', 'cpu']. Got {hf_device}."
            )
        if not isinstance(cloud, str):
            raise TypeError(f"Expected type of cloud is str. Got {type(cloud)}")
        if not isinstance(region, str):
            raise TypeError(f"Expected type of region is str. Got {type(region)}")
        if not isinstance(index_deletion_protection, bool):
            raise TypeError(
                f"Expected type of index_deletion_protection is bool. Got {type(index_deletion_protection)}"
            )

        self.index_name = index_name
        self.namespace_name = namespace
        self.embedding_model = embedding_model
        self.cloud = cloud
        self.region = region

        self.embedder = Embedder(
            embedding_model=self.embedding_model,
            dimensions=dimensions,
            api_key=api_key_embedder,
            ollama_model=ollama_model,
            hugging_face_model=hugging_face_model,
            hf_device=hf_device,
        )

        self.ollama_model = ollama_model
        self.hugging_face_model = hugging_face_model
        self.hf_device = hf_device
        self.embedder_dimensions = self.embedder.dimensions
        if dimensions != self.embedder_dimensions:
            warnings.warn(
                f"Dimensions provided are {dimensions}, but dimensions according to embedder chosen are {self.embedder_dimensions}. Using embedder dimensions",
                RuntimeWarning,
            )
        self.dimensions = self.embedder_dimensions
        self.index_deletion_protection = index_deletion_protection

        try:
            self.pinecone_client = Pinecone(api_key=api_key_pinecone)
        except Exception as e:
            raise ConnectionError(f"Unable to connect with Pinecone client. Error: {e}")

        if not self.pinecone_client.has_index(self.index_name):
            print(
                f"Index with name {self.index_name} do not exist. Creating this index..."
            )
            try:
                self.pinecone_client.create_index(
                    name=self.index_name,
                    vector_type="dense",
                    dimension=self.dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                    deletion_protection=(
                        "enabled" if self.index_deletion_protection else "disabled"
                    ),
                )
                print(f"Index with name {self.index_name} created successfully.")
            except Exception as e:
                raise Exception(
                    f"Failed to create index with index name: {self.index_name}. Error: {e}"
                )

        try:
            self.index = self.pinecone_client.Index(self.index_name)
            print(f"Connected to index: {self.index_name} successfully.")
        except Exception as e:
            raise ConnectionError(
                f"Unable to connect with Pinecone index: {self.index_name}. Error: {e}"
            )

        self.index_dimensions = self.index.describe_index_stats().dimension

        if self.dimensions != self.index_dimensions:
            raise RuntimeError(
                f"Dimensions of embedder are {self.dimensions} but index has dimensions {self.index_dimensions}."
            )
        else:
            self.dimensions = self.index_dimensions

    def upsert_vector(self, payload: dict | VectorPayload) -> UpsertResponse:
        """
        Insert or update a single vector in the Pinecone index.

        Parameters
        ----------
        payload : dict or VectorPayload
            A vector payload containing 'id', 'values', and 'metadata' keys.
            Can be a dictionary or VectorPayload instance.

        Returns
        -------
        UpsertResponse
            Response object from Pinecone containing upsert statistics.

        Raises
        ------
        ValueError
            If the vector dimensions don't match the embedder dimensions.
        ValidationError
            If the payload structure is invalid.

        Examples
        --------
        >>> payload = {
        ...     "id": "doc_1_5_uuid",
        ...     "values": [0.1, 0.2, ...],  # length must match dimensions
        ...     "metadata": {"source": "doc.pdf", "page_start": 1, "page_end": 5}
        ... }
        >>> response = vectorstore.upsert_vector(payload)
        >>> response.upserted_count
        1
        """
        payload = VectorPayload.model_validate(payload)
        if len(payload.values) != self.dimensions:
            raise ValueError(
                f"Dimensions of embedder and the dimensions of the upserting vector must match. Got dimensions, embedder: {self.dimensions} and upserting vector: {len(payload.values)}"
            )

        return self.index.upsert(
            vectors=[payload.model_dump()], namespace=self.namespace_name
        )

    def upsert_batch(self, batch: list[dict | VectorPayload]) -> UpsertResponse:
        """
        Insert or update multiple vectors in a single batch operation.

        This method is more efficient than calling upsert_vector multiple times
        as it sends all vectors in a single request to Pinecone.

        Parameters
        ----------
        batch : list of dict or VectorPayload
            List of vector payloads to upsert. Each payload must contain
            'id', 'values', and 'metadata' keys.

        Returns
        -------
        UpsertResponse
            Response object from Pinecone containing upsert statistics.

        Raises
        ------
        TypeError
            If batch is not a list.
        ValueError
            If batch is empty or if any vector's dimensions don't match embedder dimensions.
        ValidationError
            If any payload structure is invalid.

        Examples
        --------
        >>> batch = [
        ...     {"id": "doc1", "values": [...], "metadata": {...}},
        ...     {"id": "doc2", "values": [...], "metadata": {...}},
        ...     {"id": "doc3", "values": [...], "metadata": {...}}
        ... ]
        >>> response = vectorstore.upsert_batch(batch)
        >>> response.upserted_count
        3
        """
        if not isinstance(batch, list):
            raise TypeError(f"Expected type of batch is list. Got {type(batch)}")
        if not batch:
            raise ValueError("Batch must contain at least one vector")

        validated_batch = []
        for payload in batch:
            payload = VectorPayload.model_validate(payload)
            if len(payload.values) != self.dimensions:
                raise ValueError(
                    f"Dimensions of embedder and the dimensions of the upserting vector must match. Got dimensions, embedder: {self.dimensions} and upserting vector: {len(payload.values)}"
                )
            validated_batch.append(payload.model_dump())
        return self.index.upsert(namespace=self.namespace_name, vectors=validated_batch)

    def upsert_iterator(
        self,
        payload_it: Iterator[dict | VectorPayload],
        batch_size: int = 100,
        upsert_lesser_batch: bool = True,
        show_progress: bool = True,
        show_progress_after_batches: int = 20,
    ) -> int:
        """
        Upsert vectors from an iterator in batches.

        This method consumes an iterator of vector payloads and upserts them
        in configurable batch sizes. It's designed for streaming large datasets
        without loading everything into memory.

        Parameters
        ----------
        payload_it : Iterator of dict or VectorPayload
            Iterator yielding vector payloads to upsert.
        batch_size : int, optional
            Number of vectors to include in each batch. Default is 100.
        upsert_lesser_batch : bool, optional
            Whether to upsert the final batch if it's smaller than batch_size. Default is True.
        show_progress : bool, optional
            Whether to display progress updates. Default is True.
        show_progress_after_batches : int, optional
            Show progress after this many batches. Default is 20.

        Returns
        -------
        int
            Total number of batches successfully upserted.

        Raises
        ------
        TypeError
            If any parameter has incorrect type.
        ValueError
            If batch_size < 1 or show_progress_after_batches < 1.

        Notes
        -----
        The iterator is consumed lazily, so memory usage remains constant
        regardless of the total number of vectors.

        Examples
        --------
        >>> def vector_generator():
        ...     for i in range(1000):
        ...         yield {"id": f"vec_{i}", "values": [...], "metadata": {...}}
        ...
        >>> num_batches = vectorstore.upsert_iterator(
        ...     vector_generator(),
        ...     batch_size=50,
        ...     show_progress=True
        ... )
        20 batches upserted successfully.
        Iterator ended.
        >>> num_batches
        20
        """
        if not isinstance(batch_size, int):
            raise TypeError(
                f"Expected type of batch_size is int. Got {type(batch_size)}"
            )
        if batch_size < 1:
            raise ValueError(f"Expected batch_size >= 1. Got {batch_size}.")
        if not isinstance(upsert_lesser_batch, bool):
            raise TypeError(
                f"Expected type of upsert_lesser_batch is bool. Got {type(upsert_lesser_batch)}"
            )
        if not isinstance(show_progress_after_batches, int):
            raise TypeError(
                f"Expected type of show_progress_after_batches is int. Got {type(show_progress_after_batches)}"
            )
        if not isinstance(show_progress, bool):
            raise TypeError(
                f"Expected type of show_progress is bool. Got {type(show_progress)}"
            )
        if show_progress_after_batches < 1:
            raise ValueError(
                f"Expected show_progress_after_batches >= 1. Got {show_progress_after_batches}."
            )

        iterator_running: bool = True
        num_batches_inserted: int = 0

        while True:
            batch = []
            for _ in range(batch_size):
                try:
                    payload = next(payload_it)
                except StopIteration:
                    print("Iterator ended.")
                    iterator_running = False
                    break
                batch.append(payload)

            if len(batch) == batch_size:
                self.upsert_batch(batch)
                num_batches_inserted += 1
            elif upsert_lesser_batch:
                if batch:
                    self.upsert_batch(batch)
                    num_batches_inserted += 1
                else:
                    print("Empty batch can not be upserted")

            if show_progress:
                if num_batches_inserted % show_progress_after_batches == 0:
                    print(f"{num_batches_inserted} batches upserted successfully.")

            if not iterator_running:
                break

        return num_batches_inserted

    def upsert_from_chunkdict_iterator(
        self,
        chunk_dict_it: Iterator[ChunkDict],
        batch_size: int = 100,
        upsert_lesser_batch: bool = True,
        show_progress: bool = True,
        show_progress_after_batches: int = 20,
    ) -> int:
        """
        Upsert vectors from an iterator of ChunkDict objects in batches.

        This method consumes an iterator of ChunkDict objects, embeds each chunk,
        and upserts them in configurable batch sizes. It's a convenience method
        that combines embedding and upserting in a single operation.

        Parameters
        ----------
        chunk_dict_it : Iterator of ChunkDict
            Iterator yielding ChunkDict objects containing chunk text and metadata.
        batch_size : int, optional
            Number of vectors to include in each batch. Default is 100.
        upsert_lesser_batch : bool, optional
            Whether to upsert the final batch if it's smaller than batch_size. Default is True.
        show_progress : bool, optional
            Whether to display progress updates. Default is True.
        show_progress_after_batches : int, optional
            Show progress after this many batches. Default is 20.

        Returns
        -------
        int
            Total number of batches successfully upserted.

        Raises
        ------
        TypeError
            If any parameter has incorrect type.
        ValueError
            If batch_size < 1 or show_progress_after_batches < 1.

        Notes
        -----
        This method automatically handles embedding generation for each chunk
        using the vectorstore's embedder before upserting.

        Examples
        --------
        >>> from app.rag.ingest import ingest_and_chunk_pdf
        >>> chunks = ingest_and_chunk_pdf("document.pdf", "my_doc")
        >>> num_batches = vectorstore.upsert_from_chunkdict_iterator(
        ...     chunks,
        ...     batch_size=50
        ... )
        20 batches upserted successfully.
        >>> num_batches
        20
        """
        vector_paylod_it = self.embedder.embed_chunk_iterator(chunk_dict_it)
        return self.upsert_iterator(
            vector_paylod_it,
            batch_size,
            upsert_lesser_batch,
            show_progress,
            show_progress_after_batches,
        )

    def delete_namespace(self, namespace_name: str | None):
        """
        Delete all vectors in a specified namespace.

        This operation permanently removes all vectors from the namespace.
        Use with caution as this action cannot be undone.

        Parameters
        ----------
        namespace_name : str or None
            Name of the namespace to delete. If None, uses the instance's default namespace.

        Raises
        ------
        TypeError
            If namespace_name is not a string or None.

        Warnings
        --------
        This is a destructive operation that cannot be undone. All vectors
        in the namespace will be permanently deleted.

        Examples
        --------
        >>> vectorstore.delete_namespace("test_namespace")
        # All vectors in 'test_namespace' are deleted

        >>> vectorstore.delete_namespace(None)
        # Deletes vectors from the instance's default namespace
        """
        if not isinstance(namespace_name, str) and namespace_name is not None:
            raise TypeError(
                f"Expected type of namespace_name is str or None. Got {type(namespace_name)}"
            )

        if namespace_name is None:
            namespace_name = self.namespace_name

        self.index.delete_namespace(namespace_name)
