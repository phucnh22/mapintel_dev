import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from api.custom_components.custom_pipe import CustomPipeline
from api.config import PIPELINE_YAML_PATH, LOG_LEVEL, INDEXING_NU_PIPELINE_NAME

from fastapi import APIRouter
from pydantic import BaseModel
from api.controller.utils import RequestLimiter

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

router = APIRouter()


class Request_query(BaseModel):
    query: str


class Response_query(BaseModel):
    status: str
    query_text: Optional[str]
    query_umap: Optional[List[float]]


class Request_training(BaseModel):
    date_range: Optional[List[str]] = None
    update_documents: bool = True


class Response(BaseModel):
    status: str

class Response_topic_names(BaseModel):
    status: str
    topic_names: List[str]


PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_NU_PIPELINE_NAME)
logger.info(f"Loaded pipeline nodes: {PIPELINE.graph.nodes.keys()}")
concurrency_limiter = RequestLimiter(4)

# TODO make this generic for other pipelines with different naming
retriever = PIPELINE.get_node(name="Retriever")
document_store = retriever.document_store if retriever else None


@router.post("/umap-query", response_model=Response_query)
def umap_query(request: Request_query):
    """UMAP query endpoint.

    Loads the TopicRetriever with its trained Topic model. Uses the underlying trained UMAP model to call
    transform() on the embedding of the query string and returns the resulting 2-dimensional UMAP embedding.
    """
    with concurrency_limiter.run():
        # Get query UMAP embeddings
        logger.info("Obtaining the UMAP embedding of the query.")
        results = retriever.embed_queries_umap([request.query])

        return {'status': 'Success', 'query_text': request.query, 'query_umap': results[0].tolist()}


@router.get("/topic-names", response_model=Response_topic_names)
def topic_names():
    """Topic Names endpoint.
    
    Gets the unique topic names in the document store.
    """
    return {'status': 'Success', 'topic_names': retriever.get_topic_names()}
            

@router.post("/topic-training", response_model=Response)
def topic_training(request: Request_training):
    """ Topic model training endpoint.

    Trains the Retriever's topic model with the documents in the database and updates
    the instances in the database using the new model. This endpoint can be used to 
    update the topic model on a regular basis. Saves the trained model to disk.
    
    Note: this endpoint requires a considerable amount of allocated memory to be performed.
    """
    try:
        # Get documents from the document store within the specified data range
        result_generator = document_store.get_all_documents_generator(
            return_embedding=True,
            filters=[
                {
                    "range": {
                        "publishedat": {
                            "gte": request.date_range[0],
                            "lte": request.date_range[1]
                        }
                    }
                }
            ]
        )
        # Add procedures to the previous generator - avoid unnecessary fields in memory
        result = list(zip(*_encoded_results(result_generator)))
        docs, embeddings = result[0], np.array(result[1])

        # Training the topic model - passing both documents and embeddings
        retriever.train(docs, embeddings)
        
        # Updates the instances in the database using the new trained topic model
        if request.update_documents:
            document_store.update_embeddings(
                retriever, 
                update_existing_embeddings=True 
            )

        return {'status': 'Success'}
    except Exception as e:
        logger.error(e)
        return {'status': 'Fail'}


def _encoded_results(results):
    for hit in results:
        document = hit.text.replace("#SEPTAG#", " ")
        embedding = hit.embedding
        yield document, embedding