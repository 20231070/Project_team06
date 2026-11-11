import random
import itertools
from collections import Counter

# ---------------- 설정 ----------------
NUM_OPPONENTS = 1     # 상대 인원 (패는 모름 → 랜덤으로 가정)
ITERATIONS = 20000    # 시뮬레이션 횟수

RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = [r + s for r in RANKS for s in SUITS]
VALUE = {r: i + 2 for i, r in enumerate(RANKS)}

HAND_NAMES = [
    "하이카드", "원페어", "투페어", "트리플", "스트레이트",
    "플러시", "풀하우스", "포카드", "스트레이트 플러시", "로열 플러시",
]


def rank5(cards):
    """카드 5장의 족보를 (족보번호, 키커...) 튜플로 반환. 튜플이 클수록 강한 패."""
    vals = sorted((VALUE[c[0]] for c in cards), reverse=True)
    is_flush = len({c[1] for c in cards}) == 1

    # 스트레이트 판정 (A-2-3-4-5 도 인정)
    straight_high = 0
    uniq = sorted(set(vals), reverse=True)
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_high = 5

    # 같은 숫자 묶음: (개수, 숫자) 큰 순서
    groups = sorted(Counter(vals).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    counts = [cnt for _, cnt in groups]
    ordered = [v for v, _ in groups]

    if is_flush and straight_high == 14:
        return (9,)                      # 로열 플러시
    if is_flush and straight_high:
        return (8, straight_high)        # 스트레이트 플러시
    if counts == [4, 1]:
        return (7, *ordered)             # 포카드
    if counts == [3, 2]:
        return (6, *ordered)             # 풀하우스
    if is_flush:
        return (5, *vals)                # 플러시
    if straight_high:
        return (4, straight_high)        # 스트레이트
    if counts == [3, 1, 1]:
        return (3, *ordered)             # 트리플
    if counts == [2, 2, 1]:
        return (2, *ordered)             # 투페어
    if counts == [2, 1, 1, 1]:
        return (1, *ordered)             # 원페어
    return (0, *vals)                    # 하이카드


def best_rank(cards):
    """5장 이상이면 그중 가장 강한 5장 조합의 족보를 반환 (5장이면 그대로)."""
    return max(rank5(combo) for combo in itertools.combinations(cards, 5))


def my_win_rate(my_cards, iterations=ITERATIONS, num_opponents=NUM_OPPONENTS):
    """내 카드 2장만 알고, 나머지 커뮤니티 카드 3장과 상대 패를 랜덤으로 뽑아 승률을 추정."""
    rest = [c for c in DECK if c not in my_cards]
    score = 0.0
    for _ in range(iterations):
        draw = random.sample(rest, 3 + 2 * num_opponents)
        board = draw[:3]
        mine = best_rank(my_cards + board)

        best_opp = max(
            best_rank(draw[3 + 2 * i: 5 + 2 * i] + board) for i in range(num_opponents)
        )
        if mine > best_opp:
            score += 1          # 승리
        elif mine == best_opp:
            score += 0.5        # 무승부는 절반으로 계산
    return score / iterations


def describe_hand(my_cards):
    """내 카드 2장의 특징을 한 줄로 설명."""
    (r1, s1), (r2, s2) = my_cards
    v1, v2 = VALUE[r1], VALUE[r2]
    tags = []
    if v1 == v2:
        tags.append("포켓 페어")
    if s1 == s2:
        tags.append("같은 무늬(플러시 가능성)")
    if v1 != v2 and abs(v1 - v2) <= 2:
        tags.append("숫자가 이어짐(스트레이트 가능성)")
    if max(v1, v2) >= 11:
        tags.append("높은 카드 포함")
    return ", ".join(tags) if tags else "특별한 조합 없음"


def advice(rate, num_opponents=NUM_OPPONENTS):
    """승률을 '공평한 몫(1/전체 인원)'과 비교해서 카드 평가와 행동 추천을 반환."""
    fair = 1 / (num_opponents + 1)
    ratio = rate / fair

    if ratio >= 1.4:
        return "매우 좋은 카드", "적극적으로 베팅/레이즈하세요."
    if ratio >= 1.1:
        return "좋은 카드", "계속 게임하세요 (콜 또는 가벼운 베팅)."
    if ratio >= 0.9:
        return "보통 카드", "신중하게 계속하세요. 큰 베팅이 들어오면 포기도 고려하세요."
    if ratio >= 0.7:
        return "약한 카드", "되도록 포기하세요. 공짜로 볼 수 있을 때만 계속하세요."
    return "매우 약한 카드", "포기(폴드)를 추천합니다."


# ---------------- 실행: 실행할 때마다 내 카드가 새로 뽑힘 ----------------
my_cards = random.sample(DECK, 2)
rate = my_win_rate(my_cards)
grade, action = advice(rate)

print("내 카드:", my_cards)
print("카드 특징:", describe_hand(my_cards))
print(f"내 승률: {rate:.1%}")
print("카드 평가:", grade)
print("추천 행동:", action)