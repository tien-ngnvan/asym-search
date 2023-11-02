import uuid
import logging
from typing import TYPE_CHECKING, Optional, Dict, Callable, \
                    List, Any, Tuple, Iterable

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema import vectorstore
from langchain.utils import xor_args


logger = logging.getLogger()



if TYPE_CHECKING:
    import chromadb
    import chromadb.config
    #from chromadb.api.types import ID, OneOrMany, Where, WhereDocument


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    
    
DEFAULT_K = 5



class DPRChroma(vectorstore):
    """ `Chroma` vector store re-implement use for Dual Encoder architecture
    You must have the ``chromadb`` python package installed.

    Example:
        .. code-block:: python
        
    """

    def __init__(
        self, 
        collection_name: str = 'chromadb',
        embedding_function: Optional[Embeddings] = None, 
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.Client] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ):
        """Initialize with a Chroma client."""
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            raise ImportError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )

        if client is not None:
            self._client_settings = client_settings
            self._client = client
            self._persist_directory = persist_directory
        else:
            if client_settings:
                # If client_settings is provided with persist_directory specified,
                # then it is "in-memory and persisting to disk" mode.
                client_settings.persist_directory = (
                    persist_directory or client_settings.persist_directory
                )
                if client_settings.persist_directory is not None:
                    # Maintain backwards compatibility with chromadb < 0.4.0
                    major, minor, _ = chromadb.__version__.split(".")
                    if int(major) == 0 and int(minor) < 4:
                        client_settings.chroma_db_impl = "duckdb+parquet"

                _client_settings = client_settings
            elif persist_directory:
                # Maintain backwards compatibility with chromadb < 0.4.0
                major, minor, _ = chromadb.__version__.split(".")
                if int(major) == 0 and int(minor) < 4:
                    _client_settings = chromadb.config.Settings(
                        chroma_db_impl="duckdb+parquet",
                    )
                else:
                    _client_settings = chromadb.config.Settings(is_persistent=True)
                _client_settings.persist_directory = persist_directory
            else:
                _client_settings = chromadb.config.Settings()
            self._client_settings = _client_settings
            self._client = chromadb.Client(_client_settings)
            self._persist_directory = (
                _client_settings.persist_directory or persist_directory
            )
            
        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._passage_function
            if self._embedding_function is not None
            else None,
            metadata=collection_metadata,
        )
        self.override_relevance_score_fn = relevance_score_fn
        
    
    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Query the chroma collection."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings.encode_query,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )
        
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:
            embeddings = self._embedding_function.encode_context(texts)
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids
        
    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Chroma.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.
        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List of Documents most similar to the query vector.
        """
        results = self.__query_collection(
            query_embeddings=embedding,
            n_results=k,
            where=filter,
            where_document=where_document,
        )
        return _results_to_docs(results)
    
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return docs most similar to embedding vector and similarity score.

        Args:
            embedding (List[float]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        results = self.__query_collection(
            query_embeddings=embedding,
            n_results=k,
            where=filter,
            where_document=where_document,
        )
        return _results_to_docs_and_scores(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
            )
        else:
            query_embedding = self._embedding_function.encode_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
            )

        return _results_to_docs_and_scores(results)
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        distance = "ip"
        distance_key = "hnsw:space"
        metadata = self._collection.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]

        if distance == "cosine":
            return self._cosine_relevance_score_fn
        elif distance == "l2":
            return self._euclidean_relevance_score_fn
        elif distance == "ip":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {distance}."
                "Consider providing relevance_score_fn to Chroma constructor."
            )
            
    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        self._collection.delete(ids=ids)