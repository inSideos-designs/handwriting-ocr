from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router, init_service
from backend.core.config import AppConfig
from backend.services.recognition import RecognitionService


def create_app() -> FastAPI:
    app = FastAPI(title="Handwriting OCR API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = AppConfig()
    service = RecognitionService(config)
    init_service(service)

    app.include_router(router)
    return app


app = create_app()
