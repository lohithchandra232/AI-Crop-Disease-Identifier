from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download trained model from Hugging Face
model_path = hf_hub_download(
    repo_id="creator-hub1/crop-disease-model",
    filename="crop_disease_model.keras"
)

# Load model
model = load_model(model_path)

# Class names
class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]


@app.get("/")
def home():
    return {
        "message": "AI Crop Disease Identifier API is working!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_data = await file.read()

    # Open image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to array
    image_array = np.array(image) / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)

    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(confidence * 100, 2)
    }