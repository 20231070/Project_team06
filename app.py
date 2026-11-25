import random
from pathlib import Path

import streamlit as st
from poker_equity_web import estimate_equity, evaluate_visible_hand, get_hand_features, get_recommendation


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"

st.set_page_config(
    page_title="Poker Simulator | 승률 계산기",
    page_icon="♠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- 카드 표시용 상수 --------------------
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["s", "h", "d", "c"]
SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_COLOR = {"s": "#20252d", "c": "#20252d", "h": "#dc3545", "d": "#dc3545"}
DECK = [rank + suit for rank in RANKS for suit in SUITS]

def card_image_name(card):
    return f"{card[:-1]}{card[-1]}.png"

def render_card(card, width=82):
    path = ASSET_DIR / card_image_name(card)
    if path.exists():
        st.image(str(path), width=width)
    else:
        suit = SUIT_SYMBOL[card[-1]]
        color = SUIT_COLOR[card[-1]]
        st.markdown(
            f'<div class="playing-card" style="color:{color}">'
            f'<b>{card[:-1]}</b><span>{suit}</span></div>',
            unsafe_allow_html=True,
        )

def render_back(width=82):
    path = ASSET_DIR / "back.png"
    if path.exists():
        st.image(str(path), width=width)
    else:
        st.markdown('<div class="card-back">♠</div>', unsafe_allow_html=True)

def card_caption(card):
    return f"{card[:-1]}{SUIT_SYMBOL[card[-1]]}"

# -------------------- 스타일 --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
    .block-container { max-width: 1500px; padding-top: 2.0rem; padding-bottom: 2rem; }
    h1 { font-size: 2.05rem !important; font-weight: 800 !important; color: #202938; }
    h2, h3 { color: #202938; font-weight: 750 !important; }
    .subtitle { color: #64748b; font-size: .88rem; margin-top: -.45rem; }
    .section-title { font-size: 1.22rem; font-weight: 750; color: #202938; margin: .2rem 0 .65rem; }
    .hint { background: #eef5ff; border-radius: 9px; padding: 13px 15px; color: #2862b9; font-size: .82rem; line-height: 1.55; }
    .upload-box { border: 1.5px dashed #c8ced8; border-radius: 10px; background: #fafbfc; padding: 10px; }
    .metric-label { color: #64748b; font-size: .82rem; margin-bottom: 2px; }
    .metric-value { color: #182230; font-size: 2rem; font-weight: 750; line-height: 1.25; }
    .advice { background: #effaf5; color: #08784d; border-radius: 9px; padding: 13px 15px; font-weight: 650; font-size: .9rem; }
    .grade { color: #344054; font-size: 1.05rem; font-weight: 700; }
    .small-muted { color: #98a2b3; font-size: .75rem; }
    div[data-testid="stFileUploader"] { border: 1px dashed #c8ced8; border-radius: 10px; padding: 12px; background: #fafbfc; }
    div.stButton > button { border-radius: 8px; min-height: 2.55rem; }
    hr { border-color: #e5e7eb; }
    .playing-card { width: 66px; height: 96px; border: 1px solid #d7dce2; border-radius: 7px; background: white; display:flex; flex-direction:column; justify-content:space-between; padding:7px; font-size:19px; }
    .playing-card span { font-size:30px; align-self:center; }
    .card-back { width:66px;height:96px;border-radius:7px;border:2px solid white;outline:1px solid #ddd;background:#d85a62;color:white;display:flex;align-items:center;justify-content:center;font-size:34px; }
    @media (max-width: 800px) { .block-container { padding: 1rem; } h1 { font-size: 1.5rem !important; } }
</style>
""", unsafe_allow_html=True)

# -------------------- 세션 상태 --------------------
def new_deal():
    cards = random.sample(DECK, 5)
    st.session_state.hole_cards = cards[:2]
    st.session_state.flop_cards = cards[2:5]

if "hole_cards" not in st.session_state:
    new_deal()
if "flop_cards" not in st.session_state:
    new_deal()
if "opponents" not in st.session_state:
    st.session_state.opponents = 1

# -------------------- 헤더 --------------------
st.markdown("# ♠ Poker Simulator (승률 계산기)")
st.markdown('<div class="subtitle">텍사스 홀덤 내 패와 커뮤니티 카드의 승리 확률 및 추천 행동을 보여줍니다.</div>', unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([0.88, 1.12], gap="large")

with left:
    st.markdown('<div class="section-title">📷 카드 사진 (YOLO 인식 영역)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">💡 <b>카드 이미지 업로드</b><br>'
        '카드 사진을 업로드할 수 있습니다. 현재 배포본은 화면 및 승률 시뮬레이션만을 제공하며, '
        'YOLO 모델 가중치(best.pt)가 연결되면 자동 카드 인식 기능을 연동할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
         "카드 사진 업로드",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="card_photo_upload"
    )
    if uploaded is not None:
        st.image(uploaded, caption="업로드한 카드 사진", use_container_width=True)
        st.info("이미지 업로드가 완료되었습니다. YOLO 모델 연결 전에는 카드 자동 판독이 수행되지 않습니다.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎴 시뮬레이션 설정</div>', unsafe_allow_html=True)
    opponents = st.slider("상대 플레이어 수", min_value=1, max_value=9, value=int(st.session_state.opponents))
    st.session_state.opponents = opponents
    st.caption("상대방의 카드는 알 수 없으므로 남은 카드에서 무작위로 생성합니다.")

    if st.button("🎲 새 무작위 카드 뽑기", use_container_width=True, type="secondary"):
        new_deal()
        st.rerun()

with right:
    hole = st.session_state.hole_cards
    flop = st.session_state.flop_cards
    known = hole + flop
    equity = estimate_equity(hole, flop, opponents=opponents, trials=1400)
    current_hand = evaluate_visible_hand(hole, flop)
    grade, advice_text = get_recommendation(equity, opponents)
    hand_features = get_hand_features(hole)

    st.markdown('<div class="section-title">내 카드</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2.2])
    with c1:
        render_card(hole[0], 78)
        st.caption(card_caption(hole[0]))
    with c2:
        render_card(hole[1], 78)
        st.caption(card_caption(hole[1]))
    with c3:
        st.markdown("**카드 평가**")
        st.markdown(f'<div class="grade">{grade}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-muted">특징: {hand_features}</div>', unsafe_allow_html=True)
        st.caption(f"현재 족보: {current_hand}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">커뮤니티 카드</div>', unsafe_allow_html=True)
    board_cols = st.columns(5)
    for i, col in enumerate(board_cols):
        with col:
            if i < 3:
                render_card(flop[i], 70)
                st.caption(card_caption(flop[i]))
            else:
                render_back(70)
                st.caption("비공개")

    st.markdown("---")
    st.markdown('<div class="section-title">승리 확률</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">예상 승률 (무승부는 0.5승 반영)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{equity:.1%}</div>', unsafe_allow_html=True)
    st.progress(float(max(0.0, min(1.0, equity))))
    st.caption(f"{opponents}명 상대 · poker_equity.py 기반 몬테카를로 시뮬레이션 1,400회")
    st.markdown('<div class="section-title">추천 행동</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="advice">👉 {advice_text}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("※ 승률은 현재 내 카드 2장과 공개된 커뮤니티 카드를 기준으로, 남은 커뮤니티 카드 2장 및 상대 카드를 무작위로 생성해 추정한 값입니다.")
