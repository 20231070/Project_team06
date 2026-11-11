from ultralytics import YOLO

def main():
    # 1. 사전 학습된 가장 가벼운 모델 불러오기
    model = YOLO("yolov8n.pt")

    # 2. 모델 학습 시작
    # data: 앞서 다운로드 받은 데이터셋 폴더 안의 yaml 파일 경로
    results = model.train(
        data="dataset/data.yaml",  # 폴더명이 다르다면 수정해 주세요.
        epochs=25,                 # 전체 데이터셋 반복 학습 횟수
        imgsz=640,                 # 이미지 크기 (기본값 640)
        batch=16,                  # 한 번에 처리할 이미지 장수
        patience=15,               # 15번 동안 성능 개선이 없으면 조기 종료
        device="cpu"               # GPU 사용 시 0, CPU만 있다면 "cpu"로 변경
    )

if __name__ == '__main__':
    main()