import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.services.recognition import RecognitionService

router = APIRouter(prefix="/api")
_service: RecognitionService | None = None


def init_service(service: RecognitionService) -> None:
    global _service
    _service = service


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    allowed = ("image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp")
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {file.content_type} not supported")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse image file")

    result = _service.recognize(img)
    return result


@router.post("/recognize-page")
async def recognize_page(file: UploadFile = File(...)):
    allowed = ("image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp")
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {file.content_type} not supported")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse image file")

    result = _service.recognize_page(img)
    return result
