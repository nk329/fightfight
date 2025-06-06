from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException


from predictor import load_model, predict_winner
from face_crop import crop_face

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중엔 전체 허용, 배포 시엔 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 (잘라낸 얼굴 이미지) 제공
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 업로드 디렉토리 준비
os.makedirs("backend/app/uploads", exist_ok=True)
os.makedirs("backend/app/static", exist_ok=True)

# 모델 미리 로드
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "model.pt"))
model = load_model(MODEL_PATH)


@app.post("/api/upload")
async def upload_images(imageA: UploadFile = File(...), imageB: UploadFile = File(...)):
    # 저장 경로
    pathA = "backend/app/uploads/playerA.jpg"
    pathB = "backend/app/uploads/playerB.jpg"

    with open(pathA, "wb") as f:
        shutil.copyfileobj(imageA.file, f)

    with open(pathB, "wb") as f:
        shutil.copyfileobj(imageB.file, f)

    try:
        crop_face(pathA, "backend/app/static/faceA.png")
        crop_face(pathB, "backend/app/static/faceB.png")
    except Exception as e:
        print("❌ 얼굴 자르기 실패:", str(e))
        raise HTTPException(status_code=400, detail=f"얼굴 자르기 실패: {str(e)}")

    # 🧪 예시 키/리치 차이 (MediaPipe 미적용 상태)
    height_diff = 5.0
    weight_diff = 0.0
    reach_diff = 7.0

    prob = predict_winner(model, height_diff, weight_diff, reach_diff)

    return JSONResponse({
        "faceA": "/static/faceA.png",
        "faceB": "/static/faceB.png",
        "probability": round(prob, 4)
    })
