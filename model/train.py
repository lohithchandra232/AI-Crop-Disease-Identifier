from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load trained model
model = load_model("model/crop_disease_model.keras")

# Class names
class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

@app.get("/")
def home():
    return {"message": "AI Crop Disease Identifier API is working!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to array
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(confidence * 100, 2)
    }