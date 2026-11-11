#포커 확률 계산기
from ultralytics import YOLO

# 1. runs 폴더 깊숙한 곳에 있는 최고 성능의 모델을 불러옵니다.
# (폴더 이름이 train, train2, train3 등일 수 있으니 실제 폴더명을 확인하세요)
model = YOLO("runs/detect/train-2/weights/best.pt")

# 2. 테스트용 이미지에 모델을 적용하여 결과를 확인합니다.
# show=True: 인식된 결과 화면을 팝업으로 띄워줍니다.
results = model.predict(source="test_image.jpg", show=False, save=True,conf=0.1)