import base64
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from cnnClassifier import logger
from cnnClassifier.components.prediction import PredictionService
from cnnClassifier.entity.prediction import PredictionResponse


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/training/model.h5"))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", "artifacts/training/class_names.yaml"))
IMAGE_SIZE = (224, 224)

app = FastAPI(
    title="Plant Disease Detection API",
    description="FastAPI service for plant disease image classification.",
    version="0.1.0",
)
templates = Jinja2Templates(directory="templates")
prediction_service = PredictionService(
    model_path=MODEL_PATH,
    class_names_path=CLASS_NAMES_PATH,
    image_size=IMAGE_SIZE,
)


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    try:
        decoded_image = prediction_service.decode_image(image.file)
        result = prediction_service.predict(decoded_image)
        logger.info(f"Prediction made: {result['class']} ({result['confidence']})")

        accept_header = request.headers.get("accept", "")
        if "text/html" in accept_header:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "prediction": result,
                    "image_data": image_to_base64(decoded_image),
                },
            )

        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Prediction failed")
        accept_header = request.headers.get("accept", "")
        if "text/html" in accept_header:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": str(exc)},
                status_code=500,
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
