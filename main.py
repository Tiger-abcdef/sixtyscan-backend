# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference_utils import load_model, predict_from_audio_bytes

# uvicorn looks for THIS name:
app = FastAPI(
    title="SixtyScan Parkinson Backend",
    description="FastAPI backend for voice-based Parkinson prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL, DEVICE = load_model(weights_path="best_model.pth", device="cpu")


@app.get("/")
async def root():
    return {"message": "SixtyScan backend is running."}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a single audio file and return Parkinson prediction.
    Frontend should send as form-data with key 'file'.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.content_type is None or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        pd_prob, label = predict_from_audio_bytes(MODEL, DEVICE, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "probability": pd_prob,
        "percent": round(pd_prob * 100, 2),
        "label": label,
    }
