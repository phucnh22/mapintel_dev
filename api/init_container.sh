#!/bin/bash

# TODO: Save the models to disk to avoid downloading them everytime.
# Solve: OSError: Unable to load weights from pytorch checkpoint file. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True)
# # Download models if they aren't already
# if [ ! -d "data/external/ms-marco-MiniLM-L-6-v2" ]; then
#     GIT_LFS_SKIP_SMUDGE=1
#     git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 ./data/external/ms-marco-MiniLM-L-6-v2
# else
#     echo "ms-marco-MiniLM-L-6-v2 already exists."
# fi

# if [ ! -d "data/external/msmarco-distilbert-base-v4" ]; then
#     GIT_LFS_SKIP_SMUDGE=1
#     git clone https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v4 ./data/external/msmarco-distilbert-base-v4
# else
#     echo "msmarco-distilbert-base-v4 already exists."
# fi

# Load backups (wait 20 sec for ODFE node initialization)
sleep 20 && python3 api/load_backups.py

# Launch application
gunicorn api.application:app -b 0.0.0.0 -k uvicorn.workers.UvicornWorker --workers 1 --timeout 180