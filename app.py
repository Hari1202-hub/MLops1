from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import uvicorn

# -----------------------------
# Define request body structure
# -----------------------------
class InputData(BaseModel):
    data: list[float]  # expecting a list of numeric values

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title="MLOps Demo API", description="FastAPI app serving an ML model", version="1.0")

# -----------------------------
# Load trained model
# -----------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -----------------------------
# Define routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the MLOps Demo API!"}

@app.post("/predict")
def predict(input_data: InputData):
    """
    Send JSON like:
    {
      "data": [0.02, 0.03, 0.05, 0.01, 0.1, 0.05, 0.03, 0.02, 0.05, 0.04]
    }
    """
    try:
        if model is None:
            return {"error": "Model not loaded"}

        # Convert list to numpy array
        data = np.array(input_data.data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(data)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": str(e)}

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
