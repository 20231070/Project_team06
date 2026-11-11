# download_data.py
from roboflow import Roboflow

# 주의: 깃허브에 올릴 코드이므로 실제 API 키를 코드에 직접 노출하지 않는 것이 좋습니다.
# 팀원 각자 Roboflow에 가입하여 발급받은 본인 API 키를 사용하도록 안내하세요.
api_key = "각자 API 사용 요망"

rf = Roboflow(api_key=api_key)
project = rf.workspace("dankrad-mo3mt").project("card-5qrpg")
version = project.version(1)

# 'dataset'이라는 폴더 안에 YOLOv8 형식으로 다운로드합니다.
dataset = version.download("yolov8", location="dataset")

print(f"데이터셋이 {dataset.location} 경로에 성공적으로 다운로드되었습니다.")