# Mapintel API

This is a FastAPI application that sets the endpoints necessary for the Mapintel UI application.

## API modules
The API consists of 4 modules: Search, Topic, Upload, and Feedback. Each module is composed of a given number of endpoints. Below is a list of the API endpoints divided by the 4 modules.

### Search
- query: 
> Performs a query on the document store based on semantic search and approximate nearest neighbors. Also, applies boolean filters to the documents before performing semantic search.
- all-docs-generator: 
> Returns a Streaming Response consisting of a generator that iterates over the document store, given a set of boolean filters. The documents aren't iterated in any particular order.
- doc-count: 
> Gets the number of documents in the document store that satisfy a particular boolean filter.

### Topic
- umap-query: 
> Loads the TopicRetriever with its trained Topic model. Uses the underlying trained UMAP model to call transform() on the embedding of the query string and returns the resulting 2-dimensional UMAP embedding.
- topic-names: 
> Gets the unique topic names in the document store.
- topic-training: 
> Trains the Retriever's topic model with the documents in the database and updates the instances in the database using the new model. This endpoint can be used to update the topic model on a regular basis. Saves the trained model to disk.

### Upload
- file-upload: 
> Receives a document as input from any file type, extracts its text content, preprocesses it, gets the corresponding embeddings, and adds it to the document store.
- news-upload: 
> Gets the latest news from NewsAPI and respective metadata, cleans the documents, and runs them through the indexing pipeline to be stored in the database.

### Feedback
- feedback: 
> Writes the feedback labels of responses to a query into the document store.
- eval-feedback: 
> Return basic accuracy metrics based on the user feedback. Which ratio of documents was relevant? You can supply filters in the request to only use a certain subset of labels.

## Usage

### Option 1: Container

Just run
```
docker-compose --profile api up
``` 
in the root folder of the Mapintel repository. This will start two containers (Open Distro for Elasticsearch, FastAPI application).
You can check the API documentation and try its endpoints at `https://localhost:8000/docs`. 

You can also make HTTP requests:

```python
# Example using python and the requests library
import requests

# Make a request to the query endpoint
url = "http://localhost:8000/query"
req = {"query": "Brexit news", "filters": None, "top_k_retriever": 100, "top_k_reader": 10}
response_raw = requests.post(url, json=req).json()
```

### Option 2: Local

**Currently not implemented.**

**Requirements**: This expects a running Open Distro for Elasticsearch instance at `https://localhost:9200`. Also, all python and system dependencies must be satisfied.

### Option 3: Container + UI

If you want to interact with the API through the UI application, you can run
```
docker-compose --profile api --profile ui up
``` 
in the root folder of the Mapintel repository.