from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from predictor import load_model, predict_winner
from face_crop import crop_face
from mediapipe_utils import predictor_mediapipe
from mediapipe_utils import GetInfo

try:
    from fastapi import FastAPI
    app = FastAPI()
except Exception as e:
    print("❌ 초기화 중 에러 발생:", e)
    raise

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
WEB_DIR = os.path.join(BASE_DIR, "unity_web")

# Static mount
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/unity", StaticFiles(directory=WEB_DIR, html=True), name="unity")

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

    # data = predictor_mediapipe(pathA, pathB)  #미디어 파이프 연결
    # prob = predict_winner(model, height_diff=data[0], weight_diff=data[1], reach_diff=data[2])  # 예측 모델 연결

    try:
        data = predictor_mediapipe(pathA, pathB)  # 미디어 파이프로 신체 정보 추출
        print("📏 키 차이:", data[0])
        print("⚖️  몸통 면적 차이:", data[1])
        print("🤜 리치 차이:", data[2])
    except Exception as e:
        print("❌ 포즈 분석 실패:", str(e))
        raise HTTPException(status_code=400, detail=f"포즈 분석 실패: {str(e)}")

    prob = predict_winner(model, height_diff=data[0], weight_diff=data[1], reach_diff=data[2])
    print("🧠 예측된 승률 (A 기준):", round(prob, 4))

    return JSONResponse({
        "faceA": "/static/faceA.png",
        "faceB": "/static/faceB.png",
        "probability": round(prob, 4)
    })
