import sys
import subprocess
import importlib.util

# 필요한 패키지
packages = {
    "streamlit": "streamlit",
    "PIL": "pillow"
}

# 패키지 설치 여부 확인
for module, package in packages.items():

    if importlib.util.find_spec(module) is None:
        print(f"[INSTALL] {package} 설치 중...")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package
        ])

    else:
        print(f"[OK] {package} 설치 확인")

print("모든 패키지 확인 완료!")

# 여기부터 웹 화면 작성
import streamlit as st

st.title("Poker Win Calculator")

st.write("텍사스 홀덤 승리 확률 계산기")

st.info("웹 화면 테스트 중입니다.")
