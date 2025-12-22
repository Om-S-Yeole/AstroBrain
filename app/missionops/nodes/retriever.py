from typing import List

from langchain_core.runnables import RunnableConfig

from app.missionops.state import MissionOpsState, RetrievedDoc
from app.rag import RetrievedChunk, Retriever, VectorStore


def retriever(state: MissionOpsState, config: RunnableConfig):
    """
    Retrieve aerospace knowledge documents from vector database.

    This node executes the search queries generated in previous steps against
    a vector database to retrieve relevant aerospace documentation. For each
    query, it retrieves the top-k most relevant chunks and extracts their
    text, source, and page information.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - data_query : list of str
            Search queries to execute against the vector database
        - retrieved_docs : list of dict
            Previously retrieved documents to append to
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["vectorstore"] : VectorStore
            The vector database instance for document retrieval
        - configurable["top_k"] : int
            Number of top results to retrieve per query

    Returns
    -------
    dict
        A dictionary containing:
        - retrieved_docs : list of dict
            Updated list of retrieved documents, each containing:
            - text : str
                Content of the retrieved chunk
            - source : str
                Source document identifier
            - page_no : int
                Approximate page number (average of start and end pages)

    Notes
    -----
    The retriever uses vector similarity search to find relevant chunks.
    Metadata filtering is not applied, and similarity scores are not included
    in the output. The page number is computed as the average of the chunk's
    start and end page numbers.
    """
    vectorstore: VectorStore = config["configurable"]["vectorstore"]
    top_k = config["configurable"]["top_k"]
    vector_retriever = Retriever(vectorstore, top_k)

    retrieved_docs: List[RetrievedDoc] = []

    for query in state["data_query"]:
        query_specific_vectors: List[RetrievedChunk] = vector_retriever.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=None,
            include_metadata=True,
            include_scores=False,
        )

        for retrieved_chunk in query_specific_vectors:
            retrieved_doc = {
                "text": retrieved_chunk.text,
                "source": retrieved_chunk.metadata.source,
                "page_no": (
                    retrieved_chunk.metadata.page_start
                    + retrieved_chunk.metadata.page_end
                )
                // 2,
            }
            retrieved_docs.append(retrieved_doc)

    return {"retrieved_docs": state["retrieved_docs"] + retrieved_docs}
