
from pathlib import Path
from collections import deque, Counter
from threading import Lock

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

import poker_equity as poker
from poker_equity_web import (
    estimate_equity,
    evaluate_visible_hand,
    get_hand_features,
    get_recommendation,
)
from ultralytics import YOLO


# ============================================================
# 1. 기본 설정
# ============================================================

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
MODEL_PATH = ROOT / "best.pt"

CONFIDENCE = 0.25
IMAGE_SIZE = 960
IOU = 0.5

# 화면 세로 좌표 기준
# 이 값보다 위쪽에 있는 카드는 커뮤니티 카드,
# 아래쪽에 있는 카드는 내 패로 분류합니다.
DEFAULT_SPLIT_Y = 0.60

DEFAULT_OPPONENTS = 1
DEFAULT_TRIALS = 1400

# 분석 주기: 0.6초
ANALYSIS_INTERVAL = 0.6

# 최근 인식 결과 중 같은 결과가 2번 이상 나오면 안정된 결과로 판단
STABLE_HISTORY_SIZE = 5
STABLE_MIN_COUNT = 2

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9",
         "10", "J", "Q", "K", "A"]
SUITS = ["s", "h", "d", "c"]

SUIT_SYMBOLS = {
    "s": "♠",
    "h": "♥",
    "d": "♦",
    "c": "♣",
}

SUIT_COLORS = {
    "s": "#222222",
    "h": "#e53935",
    "d": "#e53935",
    "c": "#222222",
}

DECK = [rank + suit for rank in RANKS for suit in SUITS]


# ============================================================
# 2. Streamlit 설정
# ============================================================

st.set_page_config(
    page_title="Poker Simulator | 실시간 승률 분석",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

h1, h2, h3 {
    font-weight: 800;
}

.panel {
    background: #171b24;
    border: 1px solid #303744;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}

.panel-title {
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 12px;
}

.card-row {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    align-items: center;
    min-height: 110px;
}

.poker-card {
    width: 67px;
    height: 94px;
    background: #ffffff;
    border-radius: 9px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 7px;
    box-sizing: border-box;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    font-weight: 800;
}

.card-rank {
    font-size: 18px;
    line-height: 20px;
}

.card-suit {
    font-size: 31px;
    text-align: center;
    line-height: 35px;
}

.card-back {
    width: 67px;
    height: 94px;
    border-radius: 9px;
    background: repeating-linear-gradient(
        45deg,
        #193e78,
        #193e78 5px,
        #28599c 5px,
        #28599c 10px
    );
    border: 4px solid white;
    box-sizing: border-box;
}

.empty-card {
    width: 67px;
    height: 94px;
    border: 2px dashed #596273;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #8993a4;
    font-size: 24px;
}

.metric-box {
    background: #202633;
    border: 1px solid #394252;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}

.metric-label {
    color: #aeb8c8;
    font-size: 13px;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
}

.advice-box {
    background: #202633;
    border-left: 5px solid #38bdf8;
    border-radius: 9px;
    padding: 15px;
    margin-top: 10px;
}

.status-text {
    color: #aeb8c8;
    font-size: 13px;
}

div[data-testid="stImage"] img {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. 카드 이미지 출력
# ============================================================

def render_card(card, width=67):
    """카드 이미지 파일이 있으면 Streamlit 이미지로 출력합니다."""

    image_path = ASSET_DIR / f"{card}.png"

    if image_path.exists():
        st.image(str(image_path), width=width)
        return

    rank = card[:-1]
    suit = card[-1]
    symbol = SUIT_SYMBOLS.get(suit, "?")
    color = SUIT_COLORS.get(suit, "#222222")

    st.markdown(
        f"""
        <div class="poker-card" style="width:{width}px;">
            <div class="card-rank" style="color:{color};">{rank}</div>
            <div class="card-suit" style="color:{color};">{symbol}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cards(cards, empty_count=0):
    """카드들을 가로로 배치합니다."""

    items = list(cards) + [None] * empty_count

    if not items:
        return

    columns = st.columns(len(items))

    for column, card in zip(columns, items):
        with column:
            if card is not None:
                render_card(card)
            else:
                st.markdown(
                    '<div class="empty-card">?</div>',
                    unsafe_allow_html=True,
                )


def render_card_back(count=2):
    """카드 뒷면 이미지를 출력합니다."""

    image_path = ASSET_DIR / "back.png"
    columns = st.columns(count)

    for column in columns:
        with column:
            if image_path.exists():
                st.image(str(image_path), width=67)
            else:
                st.markdown(
                    '<div class="card-back"></div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# 4. YOLO 모델 로딩
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO 모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# ============================================================
# 5. 웹캠 영상 처리
# ============================================================

class PokerVideoProcessor(VideoProcessorBase):
    """
    웹캠 영상은 그대로 반환합니다.

    YOLO를 여기서 실행하지 않습니다.
    최신 프레임만 저장하고 오른쪽 분석 영역에서 가져갑니다.
    """

    def __init__(self):
        self.lock = Lock()
        self.latest_frame = None
        self.frame_count = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.latest_frame = image.copy()
            self.frame_count += 1

        # 원본 프레임을 그대로 반환
        return frame

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.frame_count

            return self.latest_frame.copy(), self.frame_count


# ============================================================
# 6. 카드 인식
# ============================================================

def remove_duplicates(detections):
    """
    같은 카드가 여러 번 검출되면 confidence가 높은 것만 남깁니다.

    detections:
        [(card_name, confidence, cx), ...]
    """

    best = {}

    for card, confidence, cx in detections:
        if card not in best or confidence > best[card][0]:
            best[card] = (confidence, cx)

    result = []

    for card, (confidence, cx) in best.items():
        result.append((card, confidence, cx))

    result.sort(key=lambda x: x[2])

    return result


def detect_cards(model, image, split_y, confidence_threshold=CONFIDENCE):
    """
    YOLO로 카드를 검출하고 세로 위치만으로 구분합니다.

    cy < split_y : 커뮤니티 카드
    cy >= split_y: 내 카드
    """

    height, width = image.shape[:2]

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=confidence_threshold,
        iou=IOU,
        verbose=False,
    )

    hand_detections = []
    board_detections = []

    debug_image = image.copy()

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = map(int, xyxy)

            class_name = model.names[class_id]
            card_name = poker.norm(str(class_name))

            if card_name not in DECK:
                continue

            cx = (x1 + x2) / 2
            cy = ((y1 + y2) / 2) / height

            if cy < split_y:
                board_detections.append(
                    (card_name, confidence, cx)
                )
                box_color = (0, 200, 255)
                label = f"BOARD {card_name}"
            else:
                hand_detections.append(
                    (card_name, confidence, cx)
                )
                box_color = (0, 255, 100)
                label = f"HAND {card_name}"

            cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            cv2.putText(
                debug_image,
                f"{label} {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

    # 세로 분할선
    split_pixel = int(height * split_y)

    cv2.line(
        debug_image,
        (0, split_pixel),
        (width, split_pixel),
        (0, 0, 255),
        2,
    )

    cv2.putText(
        debug_image,
        "COMMUNITY CARDS",
        (10, max(25, split_pixel - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )

    cv2.putText(
        debug_image,
        "MY HAND",
        (10, min(height - 10, split_pixel + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 100),
        2,
    )

    hand_detections = remove_duplicates(hand_detections)
    board_detections = remove_duplicates(board_detections)

    hand_cards = [item[0] for item in hand_detections]
    board_cards = [item[0] for item in board_detections]

    return hand_cards, board_cards, debug_image


# ============================================================
# 7. 카드 인식 결과 안정화
# ============================================================

def get_stable_cards(history):
    """
    최근 인식 결과 중 반복해서 나타나는 카드 구성을 반환합니다.

    안정된 결과가 없으면 None을 반환합니다.
    빈 카드 구성도 정상적인 인식 결과로 취급합니다.
    """

    if not history:
        return None

    counts = Counter(history)

    stable_result, count = counts.most_common(1)[0]

    if count < STABLE_MIN_COUNT:
        return None

    return stable_result


# ============================================================
# 8. 카드 유효성 검사
# ============================================================

def validate_cards(hand_cards, board_cards):
    if len(hand_cards) != 2:
        return False, "내 카드는 정확히 2장이 인식되어야 합니다."

    if len(board_cards) not in [0, 3, 4, 5]:
        return False, "커뮤니티 카드는 0장, 3장, 4장, 5장이어야 합니다."

    all_cards = hand_cards + board_cards

    if len(all_cards) != len(set(all_cards)):
        return False, "같은 카드가 중복으로 인식되었습니다."

    return True, "카드 인식 완료"


# ============================================================
# 9. 세션 상태 초기화
# ============================================================

if "hand_cards" not in st.session_state:
    st.session_state.hand_cards = []

if "board_cards" not in st.session_state:
    st.session_state.board_cards = []

if "debug_image" not in st.session_state:
    st.session_state.debug_image = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "card_history" not in st.session_state:
    st.session_state.card_history = deque(
        maxlen=STABLE_HISTORY_SIZE
    )

if "last_frame_count" not in st.session_state:
    st.session_state.last_frame_count = -1

if "last_analysis_key" not in st.session_state:
    st.session_state.last_analysis_key = None

if "last_detect_status" not in st.session_state:
    st.session_state.last_detect_status = "웹캠 연결을 기다리는 중입니다."


# ============================================================
# 10. 화면 제목
# ============================================================

st.title("♠ Texas Hold'em Poker Simulator")
st.caption("실시간 카드 인식 및 승률 분석 시스템")

st.divider()


# ============================================================
# 11. 화면 레이아웃
# ============================================================

left_col, right_col = st.columns(
    [1.15, 1],
    gap="large",
)


# ============================================================
# 12. 왼쪽: 실시간 웹캠
# ============================================================

with left_col:

    st.subheader("📷 실시간 웹캠")

    st.markdown(
<<<<<<< HEAD
        '<div class="section-title">'
        '📷 포커 사진 촬영'
=======
        '<div class="status-text">'
        '웹캠 영상은 YOLO 분석과 분리되어 실시간으로 출력됩니다.'
>>>>>>> upstream/main
        '</div>',
        unsafe_allow_html=True,
    )

<<<<<<< HEAD
    uploaded = st.camera_input(
        "웹캠으로 포커 테이블을 촬영하세요.",
        key="card_camera"
=======
    rtc_configuration = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    })

    ctx = webrtc_streamer(
        key="poker-webcam",
        video_processor_factory=PokerVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30, "max": 30},
            },
            "audio": False,
        },
        async_processing=True,
>>>>>>> upstream/main
    )

    st.caption(
        "웹캠 시작 버튼을 눌러 카메라를 연결하세요. "
        "브라우저에서 카메라 사용 권한을 허용해야 합니다."
    )

    st.markdown("### 카메라 배치 안내")

    st.markdown("""
    - 화면 위쪽: 커뮤니티 카드
    - 화면 아래쪽: 내 카드
    - 카드가 서로 겹치지 않도록 배치
    - 카메라가 테이블을 내려다보도록 고정
    """)

    split_y = st.slider(
        "카드 영역 세로 분할 위치",
        min_value=0.20,
        max_value=0.90,
        value=DEFAULT_SPLIT_Y,
        step=0.01,
        help="이 위치보다 위쪽은 커뮤니티 카드, 아래쪽은 내 카드로 인식합니다.",
    )

    st.markdown("### YOLO 인식 설정")

    confidence = st.slider(
        "인식 신뢰도",
        min_value=0.10,
        max_value=0.90,
        value=CONFIDENCE,
        step=0.05,
    )

    opponents = st.slider(
        "상대 플레이어 수",
        min_value=1,
        max_value=9,
        value=DEFAULT_OPPONENTS,
    )

    trials = st.select_slider(
        "승률 계산 반복 횟수",
        options=[300, 500, 800, 1000, 1400, 2000, 3000],
        value=DEFAULT_TRIALS,
    )


# ============================================================
# 13. 오른쪽: 카드 인식 및 승률 분석
# ============================================================

with right_col:

    st.subheader("🃏 카드 인식 및 승률 분석")

    @st.fragment(run_every=ANALYSIS_INTERVAL)
    def analysis_panel():

        # ----------------------------------------------------
        # A. 웹캠 연결 확인
        # ----------------------------------------------------

        if not ctx.state.playing:
            st.info("웹캠을 시작하면 카드 인식이 자동으로 진행됩니다.")
            return

        processor = ctx.video_processor

        if processor is None:
            st.warning("웹캠 프로세서를 초기화하는 중입니다.")
            return

        frame, frame_count = processor.get_latest_frame()

        if frame is None:
            st.info("웹캠 프레임을 기다리는 중입니다.")
            return

        # 같은 프레임이면 YOLO를 다시 실행하지 않음
        if frame_count == st.session_state.last_frame_count:
            st.caption("새로운 카메라 프레임을 기다리는 중입니다.")
        else:
            st.session_state.last_frame_count = frame_count

            try:
                model = load_model()

                # ------------------------------------------------
                # B. YOLO 카드 인식
                # ------------------------------------------------

                hand, board, debug = detect_cards(
                    model,
                    frame,
                    split_y,
                    confidence_threshold=confidence,
                )       


                st.session_state.debug_image = debug

                st.session_state.card_history.append(
                    (tuple(hand), tuple(board))
                )

                stable_result = get_stable_cards(
                    st.session_state.card_history
                )

                if stable_result is not None:
                    stable_hand, stable_board = stable_result

                st.session_state.hand_cards = list(stable_hand)
                st.session_state.board_cards = list(stable_board)

                st.session_state.last_detect_status = (
                    f"최근 프레임 인식: 내 카드 {len(hand)}장 / "
                    f"커뮤니티 카드 {len(board)}장"
                )

            except Exception as e:
                st.error(f"카드 인식 중 오류가 발생했습니다: {e}")
                return

        # ----------------------------------------------------
        # C. 현재 카드 표시
        # ----------------------------------------------------

        hand_cards = st.session_state.hand_cards
        board_cards = st.session_state.board_cards

        st.markdown("#### 내 카드")

        if hand_cards:
            render_cards(hand_cards, max(0, 2 - len(hand_cards)))
        else:
            render_card_back(2)

        st.markdown("#### 커뮤니티 카드")

        render_cards(
            board_cards,
            max(0, 5 - len(board_cards)),
        )

        st.caption(st.session_state.last_detect_status)

        # ----------------------------------------------------
        # D. YOLO 디버그 화면
        # ----------------------------------------------------

        with st.expander("YOLO 카드 인식 결과 확인", expanded=False):

            if st.session_state.debug_image is not None:
                debug_rgb = cv2.cvtColor(
                    st.session_state.debug_image,
                    cv2.COLOR_BGR2RGB,
                )

                st.image(
                    debug_rgb,
                    caption="카드 검출 박스 및 세로 분할선",
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # E. 카드 유효성 검사
        # ----------------------------------------------------

        valid, message = validate_cards(
            hand_cards,
            board_cards,
        )

        if not valid:
            st.warning(message)
            st.session_state.analysis_result = None
            st.session_state.last_analysis_key = None
            return

        st.success(message)

        # ----------------------------------------------------
        # F. 승률 분석
        # ----------------------------------------------------

        analysis_key = (
            tuple(hand_cards),
            tuple(board_cards),
            opponents,
            trials,
        )

        if analysis_key != st.session_state.last_analysis_key:

            try:
                equity = estimate_equity(
                    hand_cards,
                    board_cards,
                    opponents=opponents,
                    trials=trials,
                )

                current_hand = evaluate_visible_hand(
                    hand_cards,
                    board_cards,
                )

                grade, advice_text = get_recommendation(
                    equity,
                    opponents,
                )

                hand_features = get_hand_features(hand_cards)

                st.session_state.analysis_result = {
                    "equity": equity,
                    "current_hand": current_hand,
                    "grade": grade,
                    "advice_text": advice_text,
                    "hand_features": hand_features,
                }

                st.session_state.last_analysis_key = analysis_key

            except Exception as e:
                st.error(f"승률 계산 중 오류가 발생했습니다: {e}")
                return

        result = st.session_state.analysis_result

        if result is None:
            st.info("카드 인식 후 승률 분석이 진행됩니다.")
            return

        # ----------------------------------------------------
        # G. 승률 결과 출력
        # ----------------------------------------------------

        st.markdown("#### 📊 승률 분석 결과")

        equity = result["equity"]

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">예상 승률</div>
                    <div class="metric-value" style="color:#38bdf8;">
                        {equity:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_col2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">추천 등급</div>
                    <div class="metric-value">
                        {result["grade"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # H. 현재 핸드 및 특징
        # ----------------------------------------------------

        st.markdown("#### 현재 핸드")

        st.write(result["current_hand"])

        st.markdown("#### 내 카드 특징")

        st.write(result["hand_features"])

        # ----------------------------------------------------
        # I. 행동 추천
        # ----------------------------------------------------

        st.markdown("#### 💡 추천 행동")

        st.markdown(
            f"""
            <div class="advice-box">
                {result["advice_text"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"상대 플레이어: {opponents}명 · "
            f"시뮬레이션: {trials:,}회"
        )

        st.caption(
            "승률은 현재 인식된 카드와 설정된 상대 수를 기준으로 "
            "몬테카를로 방식으로 추정한 값입니다."
        )

    analysis_panel()


# ============================================================
# 14. 하단 안내
# ============================================================

st.divider()

st.caption("""
주의: 이 프로그램은 카드 인식 및 확률 추정을 위한 프로젝트입니다.
카드 인식 결과가 실제 테이블 상황과 다를 수 있으므로
승률 분석 전에 YOLO 인식 결과를 확인하세요.
""")
