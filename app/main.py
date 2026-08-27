import logging

from fastapi import FastAPI

from app.api.routes import router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Dialflo Audio Analysis Service",
    version="1.0.0",
    description="Low-latency audio analysis service",
)


app.include_router(router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "dialflo-audio-service",
    }   