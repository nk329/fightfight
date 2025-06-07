from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil, os

from predictor import load_model, predict_winner
from face_crop import crop_face

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "model.pt"))

# Static mount
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

@app.post("/api/upload")
async def upload_images(imageA: UploadFile = File(...), imageB: UploadFile = File(...)):
    pathA = os.path.join(UPLOAD_DIR, "playerA.jpg")
    pathB = os.path.join(UPLOAD_DIR, "playerB.jpg")

    with open(pathA, "wb") as f:
        shutil.copyfileobj(imageA.file, f)
    with open(pathB, "wb") as f:
        shutil.copyfileobj(imageB.file, f)

    try:
        crop_face(pathA, os.path.join(STATIC_DIR, "faceA.png"))
        crop_face(pathB, os.path.join(STATIC_DIR, "faceB.png"))
    except Exception as e:
        print("❌ 얼굴 자르기 실패:", str(e))
        raise HTTPException(status_code=400, detail=f"얼굴 자르기 실패: {str(e)}")

    prob = predict_winner(model, height_diff=5.0, weight_diff=0.0, reach_diff=7.0)

    return JSONResponse({
        "faceA": "/static/faceA.png",
        "faceB": "/static/faceB.png",
        "probability": round(prob, 4)
    })
