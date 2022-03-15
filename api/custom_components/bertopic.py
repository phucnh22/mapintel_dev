from typing import List, Tuple, Union
import logging
import joblib
import numpy as np

import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from bertopic import BERTopic
from bertopic._utils import check_documents_type, check_embeddings_shape, check_is_fitted
from bertopic.backend._utils import select_backend

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class BERTopic2(BERTopic):
    def transform(self,
                  documents: Union[str, List[str]],
                  embeddings: np.ndarray = None) -> Tuple[List[int], np.ndarray]:
        """ After having fit a model, use transform to predict new instances
        Arguments:
            documents: A single document or a list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model.
        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution which is returned by default.
                           If `calculate_probabilities` in BERTopic is set to False, then the
                           probabilities are not calculated to speed up computation and
                           decrease memory usage.
        Usage:
        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic().fit(docs)
        topics, _ = topic_model.transform(docs)
        ```
        If you want to use your own embeddings:
        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer
        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)
        # Create topic model
        topic_model = BERTopic().fit(docs, embeddings)
        topics, _ = topic_model.transform(docs, embeddings)
        ```
        """
        check_is_fitted(self)
        check_embeddings_shape(embeddings, documents)

        if isinstance(documents, str):
            documents = [documents]

        if embeddings is None:
            embeddings = self._extract_embeddings(documents,
                                                  method="document",
                                                  verbose=self.verbose)

        umap_embeddings = self.umap_model.transform(embeddings)
        predictions, _ = hdbscan.approximate_predict(self.hdbscan_model, umap_embeddings)

        if self.calculate_probabilities:
            probabilities = hdbscan.membership_vector(self.hdbscan_model, umap_embeddings)
        else:
            probabilities = None

        if self.mapped_topics:
            predictions = self._map_predictions(predictions)
            probabilities = self._map_probabilities(probabilities)

        return embeddings, umap_embeddings, predictions, probabilities

    @classmethod
    def load(cls,
            path: str,
            embedding_model=None):
        """ Loads the model from the specified path
        Arguments:
            path: the location and name of the BERTopic file you want to load
            embedding_model: If the embedding_model was not saved to save space or to load
                            it in from the cloud, you can load it in by specifying it here.
        Usage:
        ```python
        BERTopic.load("my_model")
        ```
        or if you did not save the embedding model:
        ```python
        BERTopic.load("my_model", embedding_model="paraphrase-MiniLM-L6-v2")
        ```
        """
        with open(path, 'rb') as file:
            if embedding_model:
                topic_model = joblib.load(file)
                topic_model.embedding_model = embedding_model
                topic_model.embedding_model = select_backend(embedding_model)
            else:
                topic_model = joblib.load(file)
            return topic_model
