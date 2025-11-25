# main.py

import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference_utils import load_model, predict_from_audio_bytes


app = FastAPI(
    title="SixtyScan Backend",
    description="FastAPI backend for SixtyScan Parkinson detection",
    version="1.0.0",
)

# -------------------------------------------------
# CORS: allow frontend (Vercel, custom domain, local)
# For simplicity and to avoid random CORS bugs, we allow all origins.
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Load model on startup
# -------------------------------------------------
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

try:
    MODEL, DEVICE = load_model(weights_path="best_model.pth", device=MODEL_DEVICE)
except Exception as e:
    # If this fails, the service will crash and Render logs will show the error.
    print("[FATAL] Failed to load model:", e)
    raise


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "SixtyScan backend is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a single audio file and return Parkinson prediction.

    Frontend should send as form-data with key "file".
    """
    try:
        audio_bytes = await file.read()
    finally:
        await file.close()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        pd_prob, label = predict_from_audio_bytes(MODEL, DEVICE, audio_bytes)
    except Exception as e:
        # This will catch ffmpeg / decoding / inference errors
        print("[ERROR] Inference failed:", repr(e))
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    percent = round(pd_prob * 100)

    return {
        "probability": pd_prob,  # 0.0–1.0
        "percent": percent,      # 0–100
        "label": label,          # "Parkinson" or "Non-Parkinson"
    }


if __name__ == "__main__":
    # For local debugging only; Render uses its own start command.
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        reload=True,
    )
