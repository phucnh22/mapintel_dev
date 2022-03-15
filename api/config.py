import os

PIPELINE_YAML_PATH = os.getenv("PIPELINE_YAML_PATH", "api/pipelines.yaml")
QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "query")
INDEXING_FU_PIPELINE_NAME = os.getenv("INDEXING_FU_PIPELINE_NAME", "indexing_file_uploads")
INDEXING_NU_PIPELINE_NAME = os.getenv("INDEXING_NU_PIPELINE_NAME", "indexing_news_uploads")


FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", "./file-upload")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")
