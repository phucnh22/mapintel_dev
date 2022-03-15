import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import newsapi

from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from api.custom_components.custom_pipe import CustomPipeline
from api.custom_components.text_cleaner import documents_cleaner
from api.config import (
    PIPELINE_YAML_PATH, 
    INDEXING_NU_PIPELINE_NAME, 
    FILE_UPLOAD_PATH, 
    INDEXING_FU_PIPELINE_NAME
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Loading File Upload Indexing Pipeline
try:
    INDEXING_FU_PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_FU_PIPELINE_NAME)
except KeyError:
    INDEXING_FU_PIPELINE = None
    logger.info("File Upload Indexing Pipeline not found in the YAML configuration. File Upload API will not be available.")

# Create directory for uploading files
os.makedirs(FILE_UPLOAD_PATH, exist_ok=True)  

# Loading News Upload Indexing Pipeline
try:
    INDEXING_NU_PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_NU_PIPELINE_NAME)
except KeyError:
    INDEXING_NU_PIPELINE = None
    logger.info("News Upload Indexing Pipeline not found in the YAML configuration. News Upload API will not be available.")

# Getting NewsAPI key and establish connection to NewsAPI
load_dotenv("/run/secrets/dotenv-file")  # hard-coded: path to secret passed through docker-compose
NEWSAPIKEY = os.environ.get("NEWSAPIKEY")


class Response(BaseModel):
    status: str


@router.post("/file-upload", response_model=Response)
def file_upload(
    file: UploadFile = File(...),
    meta: Optional[str] = Form("null"),  # JSON serialized string
    remove_numeric_tables: Optional[bool] = Form(None),
    remove_whitespace: Optional[bool] = Form(None),
    remove_empty_lines: Optional[bool] = Form(None),
    remove_header_footer: Optional[bool] = Form(None),
    valid_languages: Optional[List[str]] = Form(None),
    split_by: Optional[str] = Form(None),
    split_length: Optional[int] = Form(None),
    split_overlap: Optional[int] = Form(None),
    split_respect_sentence_boundary: Optional[bool] = Form(False)  # Set to False temporarly because of unresolved issue https://github.com/deepset-ai/haystack/issues/1038
):
    """File Upload endpoint.

    Receives a document as input from any file type, extracts its text content, 
    preprocesses it, gets the corresponding embeddings and adds it to the document
    store.
    """
    if not INDEXING_FU_PIPELINE:
        raise HTTPException(status_code=501, detail="Indexing Pipeline is not configured.")
    try:
        file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        meta = json.loads(meta) or {}
        meta["name"] = file.filename

        INDEXING_FU_PIPELINE.run(
            file_path=file_path,
            meta=meta,
            remove_numeric_tables=remove_numeric_tables,
            remove_whitespace=remove_whitespace,
            remove_empty_lines=remove_empty_lines,
            remove_header_footer=remove_header_footer,
            valid_languages=valid_languages,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary
        )
    except:
        return {'status': 'Fail'}
    else:
        return {'status': 'Success'}
    finally:
        file.file.close()


@router.post("/news-upload", response_model=Response)
def news_upload():
    """News Upload endpoint.

    Gets the latest news from NewsAPI and respective metadata, cleans the documents 
    and runs them through the indexing pipeline to be stored in the database.
    """
    try:
        # Open NewsApi connection
        news_client = newsapi.NewsApiClient(api_key=NEWSAPIKEY)

        if not INDEXING_NU_PIPELINE:
            raise HTTPException(status_code=501, detail="Indexing Pipeline is not configured.")
        
        document_batch = []

        # Get top_headlines
        logger.info("Getting top_headlines results.")
        for categ in newsapi.const.categories:
            request = news_client.get_top_headlines(
                category=categ, 
                language="en", 
                country="us", 
                page_size=100
                )
            # Adding category field and updating source field to every document in articles
            for r in request["articles"]:
                r.update({"category": categ, "source": r["source"]["id"]})
            # getting source id
            document_batch += (request["articles"])

        # Get everything using source categories
        logger.info("Getting everything results.")
        for categ in newsapi.const.categories:
            # get sources with specified category
            sources = set(map(lambda x: x["id"], news_client.get_sources(category=categ, language="en")["sources"]))
            request = news_client.get_everything(
                sources=",".join(sources), 
                from_param=datetime.today() - timedelta(1),
                to=datetime.today(), 
                page_size=100
                )
            # Guaranteeing requests only come from requested sources
            request_sources = set(map(lambda x: x["source"]['id'], request["articles"]))
            diff_sources = request_sources.difference(sources)
            request["articles"] = list(filter(lambda x: x["source"]["id"] not in diff_sources, request["articles"]))
            # Adding category field and updating source field to every document in articles
            for r in request["articles"]:
                r.update({"category": categ, "source": r["source"]["id"]})
            document_batch += request["articles"]

        # Cleaning the documents
        dicts = documents_cleaner(document_batch)

        # Embeds the documents in dicts and writes them to the document store
        logger.info("Running indexing pipeline.")
        INDEXING_NU_PIPELINE.run(
            documents=dicts
        )
    except:
        return {'status': 'Fail'}
    else:
        return {'status': 'Success'}
