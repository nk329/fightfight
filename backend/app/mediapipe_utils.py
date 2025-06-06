import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_body_metrics(image_path):
    """
    입력: 이미지 파일 경로 (예: 'playerA.jpg')
    출력: dict → {"height": float, "reach": float}
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            raise ValueError("사람 포즈 인식 실패")

        lm = results.pose_landmarks.landmark

        # 신체 keypoint 인덱스
        nose = lm[mp_pose.PoseLandmark.NOSE]
        left_heel = lm[mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = lm[mp_pose.PoseLandmark.RIGHT_HEEL]

        left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # height = nose.y - 평균 발.y
        foot_y = (left_heel.y + right_heel.y) / 2
        height = abs(nose.y - foot_y)

        # reach = 손목 사이 x 거리
        reach = abs(left_wrist.x - right_wrist.x)

        return {
            "height": height,
            "reach": reach
        }

# 테스트
if __name__ == "__main__":
    metrics = extract_body_metrics("playerA.jpg")
    print(metrics)  # 예: {'height': 0.65, 'reach': 0.44}
