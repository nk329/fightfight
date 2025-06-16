import cv2
import os

def crop_face(image_path, output_path):
    # 절대경로로 모델 파일 위치 계산
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(base_dir, "models", "haarcascade_frontalface_default.xml")

    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"얼굴 검출 모델 파일을 찾을 수 없음: {cascade_path}")

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise ValueError("CascadeClassifier가 비어 있음. 경로 오류 또는 파일 손상.")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"❌ 얼굴 감지 실패: {image_path}")  # 로그 추가
        raise ValueError("얼굴을 찾을 수 없습니다.")

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, face_img)
    print(f"얼굴 이미지 저장 완료: {output_path}")
