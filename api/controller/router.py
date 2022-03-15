from fastapi import APIRouter

from api.controller import (
    upload, 
    search,
    feedback, 
    topic
)

router = APIRouter()

router.include_router(search.router, tags=["search"])
router.include_router(topic.router, tags=["topic"])
router.include_router(upload.router, tags=["upload"])
router.include_router(feedback.router, tags=["feedback"])
