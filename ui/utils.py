import os
import json
import logging

import requests
import streamlit as st

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"
DOC_REQUEST_GENERATOR = "all-docs-generator"
UMAP_QUERY = "umap-query"
TOPIC_NAMES = "topic-names"
NUM_DOCS = "doc-count"

logger = logging.getLogger(__name__)


@st.cache(show_spinner=False)  # If the input parameters didn't change then streamlit doesn't execute the function again
def retrieve_doc(query, filters=None, top_k_reader=10, top_k_retriever=100):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    req = {"query": query, "filters": filters, "top_k_retriever": top_k_retriever, "top_k_reader": top_k_reader}
    response_raw = requests.post(url, json=req).json()

    # Format response
    result = []
    answers = response_raw["answers"]
    for i in range(len(answers)):
        answer = answers[i]["answer"]
        if answer:
            relevance = round(answers[i]["score"], 1)
            meta_source = answers[i]["meta"].get("source", None)
            meta_publishedat = answers[i]["meta"].get("publishedat", None)
            meta_topic = answers[i]["meta"].get("topic_label", None) 
            meta_url = answers[i]["meta"].get("url", None)
            meta_imageurl = answers[i]["meta"].get("urltoimage", None)
            meta_umapembeddings = answers[i]["meta"].get("umap_embeddings", None)
            document_id = answers[i]["document_id"]
            result.append(
                {
                    "answer": answer,
                    "relevance": relevance,
                    "source": meta_source,
                    "publishedat": meta_publishedat,
                    "topic": meta_topic,
                    "url": meta_url,
                    "image_url": meta_imageurl,
                    "umap_embeddings": meta_umapembeddings,
                    "document_id": document_id
                }
            )
    return result, response_raw


@st.cache(show_spinner=False)
def get_all_docs(filters=None, batch_size=None, sample_size=None):
    # Query Haystack API
    url = f"{API_ENDPOINT}/{DOC_REQUEST_GENERATOR}"
    req = {"filters": filters, "batch_size": batch_size}
    response_generator = requests.get(url, json=req, stream=True).iter_lines(delimiter=b"#SEP#")

    if sample_size is None:
        sample_size = float('inf')
    
    final_docs = []
    i = 0
    for line in response_generator:
        if i % batch_size == 0:
            if i == 0:
                logger.info(f"Begin generator.")
            else:
                logger.info(f"Iteration done: {i}.")
        # Filter out keep-alive new lines
        if line:
            doc = json.loads(line.decode('utf-8'))
            answer = doc["answer"]
            if answer:
                meta_source = doc["meta"].get("source", None)
                meta_publishedat = doc["meta"].get("publishedat", None)
                meta_topic = doc["meta"].get("topic_label", None) 
                meta_url = doc["meta"].get("url", None)
                meta_imageurl = doc["meta"].get("urltoimage", None)
                meta_umapembeddings = doc["meta"].get("umap_embeddings", None)
                meta_umapembeddingsx = None if meta_umapembeddings is None else meta_umapembeddings[0]
                meta_umapembeddingsy = None if meta_umapembeddings is None else meta_umapembeddings[1]
                document_id = doc["document_id"]
                final_docs.append(
                    {
                        "answer": answer,
                        "source": meta_source,
                        "publishedat": meta_publishedat,
                        "topic": meta_topic,
                        "url": meta_url,
                        "image_url": meta_imageurl,
                        "umap_embeddings_x": meta_umapembeddingsx,
                        "umap_embeddings_y": meta_umapembeddingsy,
                        "document_id": document_id
                    }
                )
                i += 1
        # Exit the loop when we reach sample_size
        if i == sample_size:
            logger.info(f"Final iteration number: {i}")
            break
    return final_docs        


@st.cache(show_spinner=False)
def umap_query(query):
    url = f"{API_ENDPOINT}/{UMAP_QUERY}"
    req = {
        "query": query
    }
    response_raw = requests.post(url, json=req).json()
    return response_raw


def topic_names():
    url = f"{API_ENDPOINT}/{TOPIC_NAMES}"
    response_raw = requests.get(url, json={}).json()
    return response_raw["topic_names"]


@st.cache(show_spinner=False)
def doc_count(filters=None):
    url = f"{API_ENDPOINT}/{NUM_DOCS}"
    req = {"filters": filters}
    response_raw = requests.get(url, json=req).json()
    return response_raw["num_documents"]


def feedback_doc(question, answer, document_id, model_id, is_correct_answer, is_correct_document):
    # Feedback Haystack API
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "question": question,
        "answer": answer,
        "document_id": document_id,
        "model_id": model_id,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document
    }
    response_raw = requests.post(url, json=req).json()
    return response_raw


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("file", file)]
    response_raw = requests.post(url, files=files).json()
    return response_raw
