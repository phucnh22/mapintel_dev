import json
import logging
from typing import Dict, Union, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from api.controller.search import document_store

router = APIRouter()

logger = logging.getLogger(__name__)


# TODO: Change the haystack.schema.Label so we don't need to pass is_correct_document (redundant).
class ExtractiveQAFeedback(BaseModel):
    question: str = Field(..., description="The question input by the user, i.e., the query.")
    answer: str = Field(..., description="The answer string. Text of the document with document_id.")
    document_id: str = Field(..., description="The document in the query result for which feedback is given.")
    model_id: Optional[int] = Field(None, description="The model used for the query.")
    is_correct_answer: bool = Field(..., description="Whether the answer is relevant or not.")
    is_correct_document: bool = Field(..., description="Whether the document is relevant or not. Same as is_correct_answer.",)


class FilterRequest(BaseModel):
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None


@router.post("/feedback")
def user_feedback(feedback: ExtractiveQAFeedback):
    """Feedback endpoint.

    Writes the feedback labels of responses to a query into the document store. 
    """
    document_store.write_labels([{"origin": "user-feedback", **feedback.dict()}])


@router.post("/eval-feedback")
def eval_extractive_qa_feedback(filters: FilterRequest = None):
    """Eval Feedback endpoint.

    Return basic accuracy metrics based on the user feedback.
    Which ratio of documents was relevant?
    You can supply filters in the request to only use a certain subset of labels.

    **Example:**

        ```
            | curl --location --request POST 'http://127.0.0.1:8000/eval-doc-qa-feedback' \
            | --header 'Content-Type: application/json' \
            | --data-raw '{ "filters": {"document_id": ["XRR3xnEBCYVTkbTystOB"]} }'

    """

    if filters:
        filters = filters.filters
        filters["origin"] = ["user-feedback"]
    else:
        filters = {"origin": ["user-feedback"]}

    labels = document_store.get_all_labels(filters=filters)

    if len(labels) > 0:
        answer_feedback = [1 if l.is_correct_answer else 0 for l in labels]
        answer_accuracy = sum(answer_feedback) / len(answer_feedback)

        res = {"answer_accuracy": answer_accuracy, "n_feedback": len(labels)}
    else:
        res = {"answer_accuracy": None, "n_feedback": 0}
    return res
