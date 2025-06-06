import os

path = r"C:\Users\전남규\AppData\Roaming\Python\Python311\site-packages\mediapipe\modules\pose_landmark\pose_landmark_cpu.binarypb"
print(" 파일 존재함" if os.path.exists(path) else " 파일 없음")
