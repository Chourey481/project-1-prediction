from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from typing import Optional

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Load the model
model_path = r"C:\Users\HP\Downloads\potato project\project\potatoes.h5"
MODEL = load_model(model_path)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/", response_class=HTMLResponse)
async def render_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=3000)
