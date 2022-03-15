from contextlib import contextmanager
from threading import Semaphore

from fastapi import HTTPException


class RequestLimiter:
    """Limit concurrent requests for a given endpoint.
    
    In the case of question answering on very large documents, 
    the requests can take several seconds. With the default 
    FastAPI/Uvicorn/Gunicorn deployment, the requests get processed 
    concurrently on a GPU, slowing down all the requests. To provide 
    consistent user experience, the API can respond with a server-busy
    error code if the in-process requests exceed the limit threshold.

    See https://github.com/deepset-ai/haystack/pull/64.
    """
    def __init__(self, limit):
        self.semaphore = Semaphore(limit - 1)

    @contextmanager
    def run(self):
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail="The server is busy processing requests.")
        try:
            yield acquired
        finally:
            self.semaphore.release()
