import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker.task")

# MediaPipe 모델 생성 함수
def get_pose_landmarker():
    try:
        bo = python.BaseOptions(model_asset_path=MODEL_PATH)
        opt = vision.PoseLandmarkerOptions(base_options=bo, output_segmentation_masks=True)
        return vision.PoseLandmarker.create_from_options(opt)
    except Exception as e:
        print("MediaPipe 모델 초기화 실패:", e)
        raise

# 두 이미지로부터 포즈 차이 계산
def predictor_mediapipe(fileA, fileB):
    P1 = GetInfo(fileA)
    P2 = GetInfo(fileB)
    diff = [(P2[0] - P1[0]) * 100, (P2[1] - P1[1]) * 100, (P2[2] - P1[2]) * 100]
    return diff

# 거리 계산 함수들
def dist(pos1, pos2):
    return math.sqrt((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2 + (pos2.z - pos1.z)**2)

def vector(a, b):
    return [b.x - a.x, b.y - a.y, b.z - a.z]

def cross_product(v1, v2):
    return [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]

def magnitude(v):
    return math.sqrt(sum(x**2 for x in v))

def triangle_area(p1, p2, p3):
    v1 = vector(p1, p2)
    v2 = vector(p1, p3)
    cross = cross_product(v1, v2)
    return 0.5 * magnitude(cross)

def quad_area(p1, p2, p3, p4):
    area1 = triangle_area(p1, p2, p3)
    area2 = triangle_area(p4, p2, p3)
    return area1 + area2

# 이미지 파일에서 키, 면적(근사 체중), 리치 계산
def GetInfo(file):
    PL = get_pose_landmarker()
    image = mp.Image.create_from_file(file)
    detection_result = PL.detect(image)
    arr = detection_result.pose_world_landmarks[0]

    Height_Weight_Reach = []

    left_height = dist(arr[5], arr[12]) + dist(arr[12], arr[24]) + dist(arr[24], arr[26]) + dist(arr[26], arr[28]) + dist(arr[28], arr[30])
    right_height = dist(arr[2], arr[11]) + dist(arr[11], arr[23]) + dist(arr[23], arr[25]) + dist(arr[25], arr[27]) + dist(arr[27], arr[29])
    Height_Weight_Reach.append(max(left_height, right_height))

    Height_Weight_Reach.append(quad_area(arr[11], arr[12], arr[23], arr[24]))

    left_arm = dist(arr[12], arr[14]) + dist(arr[14], arr[16])
    right_arm = dist(arr[11], arr[13]) + dist(arr[13], arr[15])
    Height_Weight_Reach.append(max(left_arm, right_arm))

    return Height_Weight_Reach

# 테스트용 실행 코드
if __name__ == "__main__":
    print("모델 경로:", MODEL_PATH)
    print("파일 존재 여부:", os.path.exists(MODEL_PATH))

    try:
        P1 = GetInfo("P1.png")
        P2 = GetInfo("P2.png")

        height_diff = (P2[0] - P1[0]) * 100
        weight_diff = (P2[1] - P1[1]) * 100
        reach_diff = (P2[2] - P1[2]) * 100

        print("키 차이:", height_diff)
        print("몸통 면적 차이:", weight_diff)
        print("리치 차이:", reach_diff)
    except Exception as e:
        print("테스트 중 오류:", e)
