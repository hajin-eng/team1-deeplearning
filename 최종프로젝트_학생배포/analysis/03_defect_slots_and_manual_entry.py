"""03. 원본 불량 기록의 _1/_2 의미 추적 + 수기 입력 가설 점검

[A~E] 참고_원본불량기록_train.csv의 _1/_2 열이 무엇을 뜻하는지 단서를 찾는다.
[F~H] 공장·공기압·냉각수 열이 '사람이 주기적으로 적은 값'처럼 보이는지 점검한다.
[I~L] _1/_2 의미 후보 3가지를 가르는 관찰값을 계산한다 (맨 아래 '가설 검증').
      A 검사 위치(면) 2곳 / B 검사 단계 2회(1차·2차) / C 부품(캐비티) 2개
      어느 가설이 맞는지 '확정'하는 것이 아니라, 가설마다 예상하는 모습과 실제 관찰값을 나란히 놓고
      유리하거나 불리한 단서가 있는지 확인하는 것이다. 원본 불량 기록은 읽는 용도로만 쓴다.

위 세 가지(_1/_2 의미 추적, 수기 입력 점검, 가설 검증) 모두 확정이 아니라 단서 수집이다. 결론은 전처리_결정_정리.md로 확인한다.
읽기만 하고 아무 파일도 만들지 않는다.
실행: python analysis/03_defect_slots_and_manual_entry.py
"""

# 실행 결과 요약
# _1과 _2는 같은 값의 사본이 아님
# _1/_2 열의 의미(검사 위치 / 검사 단계 / 부품(캐비티))는 데이터로 확정 불가

# 코드 메모
# 1. HGB(HistGradientBoostingClassifier)(K 섹션)
# _1열에만 / _2열에만 찍힌 불량을 모델이 똑같이 잡는지 보려면 공정 조건으로 불량을 맞히는 모델이 필요함
# 허용된 라이브러리(scikit-learn)에 들어 있고, 쌍둥이 그룹 분할에서 RF보다 점수가 높았음 (0.28~0.29 vs 0.23)
# 2. Mann-Whitney 검정(J 섹션)
# 두 그룹(_1열에만 vs _2열에만 찍힌 행)의 공정 조건 값이 다른지 비교하는데, 값을 순위로 바꿔 비교하므로 정규분포를 가정하지 않음
# 비스킷 두께처럼 정수라 같은 값이 많거나 설비 정지 같은 극단값이 있어도 영향을 덜 받음
# 두 그룹의 행 수가 달라도(예: 247행 vs 57행) 사용할 수 있음
# 한계: 쌍둥이 행이 서로 독립이 아니라서 p값이 실제보다 작게 나올 수 있고, 시기 차이는 통제하지 않았음
# 3. Bonferroni 보정(J 섹션)
# 검정을 20번 하면 차이가 없어도 우연히 p < 0.05가 한 번 이상 나올 확률이 약 64%(1 - 0.95^20)
# 유의 기준을 0.05 / 20 = 0.0025로 낮춰서 우연히 유의하게 나오는 것을 줄임
# 단점: 기준이 엄격해서 실제 차이도 놓칠 수 있음 (Exfoliation p=0.0042는 0.05 기준은 통과하지만 0.0025는 통과하지 못함)

import common
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier

from preprocess import load_data

common.announce("03 _1/_2 가설 검증, 수기 입력 점검", "20초", "20초")

ID_COLS = ["shot_id", "id", "Shot"]

# 원본 기록의 불량 유형 13종 (열 이름은 '<유형>_1', '<유형>_2')
# 슬롯 = 각 유형의 _1 열과 _2 열 (코드에서는 번호 1, 2로 부름)
DEFECT_TYPES = [
    "Short_Shot",
    "Bubble",
    "Exfoliation",
    "Blow_Hole",
    "Stain",
    "Dent",
    "Deformation",
    "Contamination",
    "Impurity",
    "Crack",
    "Scratch",
    "Buring_Mark",
    "Inclusions",
]

# 7종 라벨로 접을 때의 우선순위 (앞쪽이 우선). 여기 없는 유형은 '기타불량'.
# 이 우선순위는 원본 기록과 정답을 대조해서 5,274샷 전부 일치하는 것으로 확인한 규칙.
PRIORITY = ["Short_Shot", "Blow_Hole", "Exfoliation", "Deformation", "Stain"]

# 한 불량 유형(예: Short_Shot)이 '어느 열에 찍혔는가'를 구분하는 이름. E, J, K 섹션이 같은 이름을 사용.
# 그 불량이 _1 열에만 찍힌 행 (_1 > 0 이고 _2 = 0)
ONLY_1 = "_1열에만"
# 그 불량이 _2 열에만 찍힌 행 (_2 > 0 이고 _1 = 0)
ONLY_2 = "_2열에만"
# 그 불량이 _1 열과 _2 열 모두에 찍힌 행
BOTH = "두 열 모두"
# 그 불량이 두 열 모두 0인 행 (다른 유형의 불량은 있을 수 있음)
NEITHER = "어느 열에도 없음"

X = pd.read_csv("data/주조공정_입력.csv", header=1, encoding="utf-8-sig")
X.columns = X.columns.str.strip()
defects = pd.read_csv("참고_원본불량기록_train.csv", encoding="utf-8-sig")
labels = pd.read_csv("정답_train.csv", encoding="utf-8-sig")

# ===============================================================
# A. 유형별로 _1 / _2 건수와 겹침
#    목적: _1과 _2가 '같은 불량을 두 번 적은 것(재검사)'인지, '서로 다른 것을 센 것'인지 가림.
#         P(_2|_1) = _1 열에 찍힌 행 중 같은 유형이 _2 열에도 찍힌 비율 (P(_1|_2)는 그 반대 방향).
#         이 비율이 높으면 같은 불량을 두 번 적은 것, 낮으면 서로 다른 것.
# ===============================================================
print("=== A. 유형별 _1/_2 건수와 겹침 ===")
for t in DEFECT_TYPES:
    in1 = defects[f"{t}_1"] > 0
    in2 = defects[f"{t}_2"] > 0
    both = (in1 & in2).sum()
    print(
        f"  {t:14s} _1={in1.sum():4d} _2={in2.sum():4d} {BOTH}={both:4d}",
        f" P(_2|_1)={both / max(in1.sum(), 1):.2f}",
        f" P(_1|_2)={both / max(in2.sum(), 1):.2f}",
    )

# ===============================================================
# B. 한 행에 불량이 2개 이상 찍힌 행은 _1/_2를 어떻게 나눠 쓰나
#    목적: _2가 '두 번째 불량을 적는 칸'이라면 서로 다른 유형은 _1과 _2에 나뉘어 찍힐 것.
# ===============================================================
print("\n=== B. 불량이 2개 이상 찍힌 행의 조합 ===")
combo_count = {}
examples = []
for _, row in defects.iterrows():
    # 이 행에서 0보다 큰 칸을 (유형, 슬롯) 목록으로 모음.
    active = []
    for t in DEFECT_TYPES:
        for s in (1, 2):
            if row[f"{t}_{s}"] > 0:
                active.append((t, s))
    if len(active) < 2:
        continue

    # 찍힌 칸들에 나온 불량 유형의 종류와, 쓰인 슬롯(_1/_2)의 종류를 모음.
    types = set()
    slots = set()
    for t, s in active:
        types.add(t)
        slots.add(s)
    if len(types) == 1:
        kind = "같은 유형이 _1·_2 둘 다"
    elif slots == {1}:
        kind = "다른 유형, 모두 _1"
    elif slots == {2}:
        kind = "다른 유형, 모두 _2"
    else:
        kind = "다른 유형, _1과 _2에 나뉨"

    combo_count[kind] = combo_count.get(kind, 0) + 1
    if len(types) > 1 and len(examples) < 5:
        examples.append(active)

print(" ", combo_count)
print("  예:", examples)


# ===============================================================
# C. 정답 라벨은 _1과 _2 중 어디를 보고 만들어졌나
#    목적: 정답 라벨이 _1 열만 보고 만들어졌는지, _2 열만 보고 만들어졌는지,
#         두 열을 모두 보고 만들어졌는지 가림.
#         우선순위 규칙으로 접은 결과가 정답과 다르면 '불일치'로 체크하여 셈.
# ===============================================================
def fold_label(row, slots):
    """주어진 슬롯(_1/_2)의 불량만 보고 7종 라벨로 접는다."""
    found = set()
    for t in DEFECT_TYPES:
        for s in slots:
            if row[f"{t}_{s}"] > 0:
                found.add(t)
    if not found:
        return "정상"
    # 우선순위가 높은 유형부터 확인해서 가장 먼저 걸린 것을 라벨로 함
    for t in PRIORITY:
        if t in found:
            return "불량_" + t
    return "기타불량"  # 우선순위 목록에 없는 유형만 있는 경우


print("\n=== C. 정답 라벨은 _1·_2 중 어디를 보고 만들어졌나 ===")
merged = defects.merge(labels, on="shot_id")
for name, slots in [
    ("_1 열만 사용", (1,)),
    ("_2 열만 사용", (2,)),
    ("두 열 모두 사용", (1, 2)),
]:
    # 행마다 fold_label로 7종 라벨을 만들어 정답과 비교할 목록을 생성.
    folded = []
    for _, merged_row in merged.iterrows():
        folded.append(fold_label(merged_row, slots))
    folded = pd.Series(folded, index=merged.index)
    print(f"  {name}: 정답과 불일치 {int((folded != merged['label']).sum())}건")

# ===============================================================
# D. 행 단위 _1/_2 불량 비율을 제품·구간별로
#    목적: _1과 _2가 제품·시기에 따라 같은 방식으로 변하는지 확인.
#         다르게 변한다면 서로 다른 현상을 가리킨다는 단서가 됨.
# ===============================================================
print("\n=== D. 행 단위 _1/_2 비율 (제품별, 1000샷 구간별) ===")
rate = defects.merge(X[["shot_id", "Product_Type"]], on="shot_id")

# _1 열 이름들과 _2 열 이름들을 따로 모음 (예: Short_Shot_1, Bubble_1, ...).
slot1_names = []
slot2_names = []
for t in DEFECT_TYPES:
    slot1_names.append(f"{t}_1")
    slot2_names.append(f"{t}_2")

# has1 / has2: 그 행에서 _1 열(또는 _2 열) 중 하나라도 불량이 찍혔는지 (True / False)
rate["has1"] = rate[slot1_names].sum(axis=1) > 0
rate["has2"] = rate[slot2_names].sum(axis=1) > 0

print(rate.groupby("Product_Type")[["has1", "has2"]].mean().round(3).to_string())

rate["block"] = (rate["shot_id"] - 1) // 1000  # 1000샷 단위 구간 번호
print(rate.groupby("block")[["has1", "has2"]].mean().round(3).T.to_string())

# ===============================================================
# E. 같은 유형이라도 _1열에만 / _2열에만 / 두 열 모두에 찍혔을 때 공정 조건이 다른가 (중앙값)
#    목적: _1과 _2가 서로 다른 공정 조건에서 생기는 불량이라면 조건 값이 달라야 함.
#         공정 조건 값이 다르면 _1/_2가 서로 다른 현상을 가리킨다는 뜻이고, 같으면 입력(공정 조건)만으로는 구분할 수 없음.
# ===============================================================
print("\n=== E. _1열에만 / _2열에만 / 두 열 모두에 찍힌 행의 공정 조건 (중앙값) ===")
joined = defects.merge(X, on="shot_id")
process_cols = [
    "Casting_Pressure",
    "Cycle_Time",
    "Melting_Furnace_Temp",
    "Biscuit_Thickness",
]

# 제품마다 공정 조건이 크게 달라서(압력 1052 vs 596) 제품별로 따로 확인.
for defect, product in [
    ("Short_Shot", 2),
    ("Short_Shot", 1),
    ("Blow_Hole", 2),
    ("Exfoliation", 1),
]:
    sub = joined[joined["Product_Type"] == product]
    # np.select는 조건을 위에서부터 확인해 처음 맞는 것을 고름.
    # 두 열 모두에 찍힌 행은 첫 조건에서 BOTH로 빠지고, 둘째·셋째 조건에는 한 열에만 찍힌 행만 남게 됨.
    kind = np.select(
        [
            (sub[f"{defect}_1"] > 0) & (sub[f"{defect}_2"] > 0),
            sub[f"{defect}_1"] > 0,
            sub[f"{defect}_2"] > 0,
        ],
        [BOTH, ONLY_1, ONLY_2],
        default=NEITHER,
    )
    table = sub.groupby(kind)[process_cols].median()
    table["n"] = pd.Series(kind).value_counts()
    print(f"  {defect} / Type {product}")
    print(table.round(2).to_string())

# ===============================================================
# 여기부터는 수기 입력 가설. 중복 행은 빼고(한 번 기록한 값으로) 볼 것.
# 목적: 공장 온도·습도 등이 사람이 주기적으로 적은 값이라면 거기에 맞는 흔적이 있어야 함.
#      (확정이 아니라 '모순되지 않는 흔적'을 찾을 것)
# ===============================================================
print("\n############ 수기 입력 가설 점검 ############")
constant_cols = []
for c in X.columns:
    if X[c].nunique(dropna=True) <= 1:
        constant_cols.append(c)

feature_cols = []
for c in X.columns:
    if c not in ID_COLS + constant_cols:
        feature_cols.append(c)
values = X[feature_cols].fillna(-999).values
same_as_prev = np.r_[False, (values[1:] == values[:-1]).all(axis=1)]
# 입력이 같은 행은 같은 값이 복사된 것이라 변화 횟수를 부풀리므로 첫 행만 남김.
unique_rows = X[~same_as_prev].reset_index(drop=True)

# ---------------------------------------------------------------
# F. 소수 첫째 자리 분포
#    목적: 사람이 숫자를 적으면 .0 이나 .5 처럼 떨어지는 값으로 쏠리는 경향이 있음.
#         (소수 첫째 자리가 0~9 각각일 기대 비율은 10%)
# ---------------------------------------------------------------
print("\n=== F. 소수 첫째 자리 분포 ===")
for col in ["Factory_Temp", "Factory_Humidity", "Air_Pressure", "Coolant_Temp"]:
    v = unique_rows[col].dropna()
    first_decimal = (v * 10).round().astype(int).mod(10)
    dist = first_decimal.value_counts(normalize=True).sort_index().round(2).to_dict()
    print(f"  {col:17s}", dist)

# ---------------------------------------------------------------
# G. 열들이 '같은 시점에' 값이 바뀌는가
#    목적: 같은 때 한꺼번에 적은 값이라면 서로의 변화 시점이 겹침.
#          '우연 기대치'는 서로 무관하다고 가정했을 때 겹칠 횟수.
# ---------------------------------------------------------------
print("\n=== G. 값이 바뀌는 시점이 같이 움직이나 ===")


def changed(col):
    """바로 앞 행과 값이 달라진 행 (결측은 제외)."""
    s = unique_rows[col]
    return (s != s.shift()) & s.notna() & s.shift().notna()


sensor_cols = [
    "Factory_Temp",
    "Factory_Humidity",
    "Air_Pressure",
    "Coolant_Temp",
    "Coolant_Pressure",
]
change = {}
for c in sensor_cols:
    change[c] = changed(c)

pairs = [
    ("Factory_Temp", "Factory_Humidity"),
    ("Air_Pressure", "Coolant_Temp"),
    ("Factory_Temp", "Air_Pressure"),
    ("Factory_Temp", "Coolant_Temp"),
    ("Coolant_Temp", "Coolant_Pressure"),
]
for a, b in pairs:
    # 기대치 = 두 열의 변화 횟수를 각각 전체 행 수로 나눈 비율의 곱 x 전체 행 수
    expected = change[a].sum() * change[b].sum() / len(unique_rows)
    print(
        f"  {a} 변화 {change[a].sum()}회, {b} 변화 {change[b].sum()}회,",
        f"동시 {(change[a] & change[b]).sum()}회, 우연 기대치 {expected:.0f}회",
    )

# ---------------------------------------------------------------
# H. 주변 15샷의 중앙값에서 크게 벗어난 값 (입력 실수 후보)
#    목적: 사람이 잘못 적은 값은 주변 값과 동떨어진 값으로 나타나므로 그런 값이 어디에 있는지 찾음.
# ---------------------------------------------------------------
print("\n=== H. 주변 15샷 중앙값에서 크게 벗어난 값 ===")
for col, threshold in [("Factory_Temp", 2.0), ("Factory_Humidity", 6.0)]:
    s = unique_rows[col]
    # 앞뒤 15샷의 중앙값(그 값 자신 포함)을 구하고, 자기 값이 거기서 얼마나 벗어났는지 확인.
    local_median = s.rolling(15, center=True, min_periods=5).median()
    deviation = (s - local_median).abs()
    odd = deviation[deviation > threshold].index
    print(
        f"  {col} (차이 > {threshold}): {len(odd)}개",
        unique_rows.loc[odd, ["shot_id", col]].head(15).values.tolist(),
    )

# ===============================================================
# 가설 검증: _1/_2 의미 후보 3가지
#   A. 검사 위치(면) 2곳
#   B. 검사 단계 2회 (1차 검사 / 2차 검사)
#   C. 부품(캐비티) 2개
#
# 방식: 가설마다 '이런 모습이어야 한다'를 먼저 적어 두고, 세 가설을 구분할 수 있는 숫자를 계산.
#      숫자가 어느 가설과 맞는지는 전처리_결정_정리.md 3번 항목의 표에서 해석.
# ===============================================================
print("\n############ 가설 검증 (_1/_2 의미 후보) ############")

MAIN_TYPES = PRIORITY  # 이름 있는 5종. 기타 유형은 건수가 너무 적어서 제외.
blocks = (
    defects["shot_id"] - 1
) // 1000  # 1000샷 구간 번호 (시기 효과를 통제하려고 사용)

# ---------------------------------------------------------------
# I-1. 두 열의 건수 대칭성
#      부품(캐비티) 2개가 서로 같은 모양이면 두 열의 건수가 비슷해야 함(C의 예상).
#      위치나 단계라면 달라도 무관함 (A, B).
# ---------------------------------------------------------------
n_slot1 = {}
n_slot2 = {}
for t in MAIN_TYPES:
    n_slot1[t] = int((defects[f"{t}_1"] > 0).sum())
    n_slot2[t] = int((defects[f"{t}_2"] > 0).sum())

print("\n=== I-1. 두 열의 건수 ===")
for t in MAIN_TYPES:
    print(f"  {t:12s} _1 {n_slot1[t]:4d} / _2 {n_slot2[t]:4d}")
print(
    f"  5종 합계 _1 {sum(n_slot1.values())} / _2 {sum(n_slot2.values())}",
    f"-> _1이 {sum(n_slot1.values()) / sum(n_slot2.values()):.2f}배",
)

# ---------------------------------------------------------------
# I-2. 같은 유형이 _1과 _2에 함께 찍힌 횟수: 관찰 vs 독립일 때의 기대
#      기대값은 '같은 1000샷 구간 안에서 두 열이 서로 무관하다'고 가정해 계산.
#      (불량이 많은 시기에는 우연히도 두 열에 같이 찍히기 쉬우므로 시기 효과를 빼려는 것)
#      - B(단계): 1차에서 걸린 불량은 빠지므로 관찰이 기대보다 '적을' 것으로 예상
#      - A·C    : 같은 원인이 두 곳에 같이 나타나면 관찰이 기대보다 '많을' 수 있음
# ---------------------------------------------------------------
print("\n=== I-2. 같은 유형이 _1과 _2에 함께 찍힌 횟수: 관찰 vs 독립일 때 기대 ===")
total_observed = 0
total_expected = 0.0
for t in MAIN_TYPES:
    in1 = defects[f"{t}_1"] > 0
    in2 = defects[f"{t}_2"] > 0
    observed = int((in1 & in2).sum())

    expected = 0.0
    for b in sorted(blocks.unique()):
        in_block = blocks == b
        # 구간 안에서 _1 건수 x _2 건수 / 구간 행 수 = 두 열이 무관할 때 겹칠 기대 횟수
        expected += in1[in_block].sum() * in2[in_block].sum() / in_block.sum()

    total_observed += observed
    total_expected += expected
    print(f"  {t:12s} 관찰 {observed:3d} / 기대 {expected:5.1f}")
print(
    f"  5종 합계 관찰 {total_observed} / 기대 {total_expected:.1f}",
    f"-> 관찰이 기대의 {total_observed / total_expected:.1f}배",
)

# ---------------------------------------------------------------
# I-3. 제품이 Type 1 -> 2로 바뀔 때 _1과 _2의 불량 비율 변화
#      두 열이 같은 현상을 서로 다른 위치에서 본 것이라면 제품이 바뀔 때 같이 변하기 쉽고,
#      다른 현상이라면 서로 다르게 변할 수 있음. (D 섹션에서 만든 rate 표, 즉 행 단위 _1/_2 불량 비율을 재사용)
# ---------------------------------------------------------------
print("\n=== I-3. 제품 변화에 대한 반응 ===")
by_product = rate.groupby("Product_Type")[["has1", "has2"]].mean()
print(by_product.round(3).to_string())
print(
    f"  Type 2 / Type 1 비율: _1 {by_product.loc[2, 'has1'] / by_product.loc[1, 'has1']:.2f}배,",
    f"_2 {by_product.loc[2, 'has2'] / by_product.loc[1, 'has2']:.2f}배",
)

# ---------------------------------------------------------------
# J. 같은 불량이 _1에만 / _2에만 찍힌 행의 공정 조건 차이 (Mann-Whitney 검정)
#    목적: _1열에만 찍힌 불량과 _2열에만 찍힌 불량이 서로 다른 공정 조건에서 생기는지 확인.
#          검정을 여러 번 하면 우연히 유의하게 나오는 것이 생기므로 기준을 낮춰서(Bonferroni) 확인.
# ---------------------------------------------------------------
print(f"\n=== J. {ONLY_1} / {ONLY_2} 찍힌 행의 공정 조건 차이 ===")
tests = []
for defect in ["Short_Shot", "Blow_Hole", "Exfoliation", "Deformation"]:
    for product in [1, 2]:
        sub = joined[joined["Product_Type"] == product]
        only1 = sub[(sub[f"{defect}_1"] > 0) & (sub[f"{defect}_2"] == 0)]
        only2 = sub[(sub[f"{defect}_2"] > 0) & (sub[f"{defect}_1"] == 0)]
        if len(only1) < 30 or len(only2) < 30:  # 건수가 너무 적으면 검정이 의미 없다
            continue

        for col in process_cols:
            p_value = mannwhitneyu(only1[col], only2[col]).pvalue
            tests.append(
                (
                    defect,
                    product,
                    col,
                    len(only1),
                    len(only2),
                    only1[col].median(),
                    only2[col].median(),
                    p_value,
                )
            )

bonferroni = 0.05 / len(tests)
print(
    f"  검정 {len(tests)}번 수행 -> 유의 기준을 0.05 대신 {bonferroni:.4f}로 낮춰서 본다"
)
n_significant = 0
for test in tests:
    if test[7] < bonferroni:
        n_significant += 1
print(f"  기준을 통과한 검정: {n_significant}개 / p가 작은 순 6개:")


def p_value_of(test):
    """검정 결과 튜플에서 p값(마지막 칸)을 꺼낸다. 정렬 기준으로 쓴다."""
    return test[7]


for t in sorted(tests, key=p_value_of)[:6]:
    if t[7] < bonferroni:
        mark = "통과"
    else:
        mark = "-"
    print(
        f"    {t[0]:12s} Type {t[1]} {t[2]:20s}",
        f"{ONLY_1}(n={t[3]}) 중앙값 {t[5]:.2f} vs {ONLY_2}(n={t[4]}) {t[6]:.2f} | p={t[7]:.4f} {mark}",
    )

# ---------------------------------------------------------------
# K. 모델이 _1열에만 찍힌 불량과 _2열에만 찍힌 불량을 똑같이 잡는가
#    목적: _2 열의 기록이 잡음이라면 공정 조건으로 설명되지 않아 모델이 잘 못 잡을 것.
#          (원본 불량 기록은 모델 입력이 아니라, 예측이 끝난 뒤에 행을 나눠 보는 용도로만 읽음)
#    HGB(HistGradientBoosting) 모델, 쌍둥이 그룹 분할 out-of-fold 예측 (시드 0)
#    out-of-fold: 모든 행이 '자기를 학습하지 않은 모델'에게 한 번씩 예측받은 결과
# ---------------------------------------------------------------
print(f"\n=== K. 모델의 재현율: 불량이 {ONLY_1} / {ONLY_2} / {BOTH} 찍힌 행 ===")
data = load_data()
X_train = data["X_train"].values
y_train = data["y_train"].values
groups_train = data["groups_train"].values

split = pd.read_csv("분할표.csv", encoding="utf-8-sig")
# preprocess의 train 행 순서(= shot_id 순)와 같은 순서로 shot_id를 맞춤.
train_ids = split.loc[split["split"] == "train", "shot_id"].sort_values().values
slots = defects.set_index("shot_id").loc[train_ids]

oof_pred = np.empty(len(y_train), dtype=object)
for _, train_idx, valid_idx in common.group_folds(y_train, groups_train, seeds=[0]):
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=0,
    )
    model.fit(X_train[train_idx], y_train[train_idx])
    oof_pred[valid_idx] = model.predict(X_train[valid_idx])

print(f"  {'라벨':16s}{ONLY_1:>20s}{ONLY_2:>20s}{BOTH:>16s}")
for t in MAIN_TYPES:
    label = "불량_" + t
    in1 = slots[f"{t}_1"].values > 0
    in2 = slots[f"{t}_2"].values > 0

    cells = []
    for mask in [in1 & ~in2, in2 & ~in1, in1 & in2]:
        # mask(어느 열에 찍혔는지)와 정답 라벨이 label인 조건을 모두 만족하는 행
        selected = mask & (y_train == label)
        n = int(selected.sum())
        hit = int((oof_pred[selected] == label).sum())
        if n:
            cells.append(f"{hit / n:6.1%} ({hit}/{n})")
        else:
            cells.append("-")  # 해당하는 행이 없으면 비율을 구할 수 없음

    # 칸마다 폭 20으로 오른쪽 정렬해서 한 줄로 이어 붙임.
    line = f"  {label:16s}"
    for c in cells:
        line += f"{c:>20s}"
    print(line)

# ---------------------------------------------------------------
# L. 행 층: 쌍둥이 쌍의 불량 기록 일치
#    목적: 같은 Shot이 2행으로 있는 이유(행 층)와 _1/_2 열의 의미(열 층)는 별개의 질문.
#          두 행이 독립된 두 부품이라면 불량 기록이 자주 달라야 함.
#          기록이 거의 같다면 두 행은 같은 샷 단위 기록을 공유한다고 볼 수 있음.
# ---------------------------------------------------------------
print("\n=== L. 행 층: 쌍둥이 쌍(2행)의 불량 기록 ===")
X["group"] = np.cumsum(~same_as_prev) - 1  # 입력이 같은 행들의 묶음 번호
slot_cols = []
for t in DEFECT_TYPES:
    for s in (1, 2):
        slot_cols.append(f"{t}_{s}")
pair_rows = defects.merge(X[["shot_id", "group"]], on="shot_id").merge(
    labels, on="shot_id"
)

n_pairs = 0
n_same_record = 0
n_same_label = 0
for _, pair in pair_rows.groupby("group"):
    if len(pair) != 2:  # 두 행이 모두 train에 있는 쌍만 볼 것
        continue
    n_pairs += 1
    n_same_record += int(
        (pair.iloc[0][slot_cols].values == pair.iloc[1][slot_cols].values).all()
    )
    n_same_label += int(pair.iloc[0]["label"] == pair.iloc[1]["label"])

# 독립일 때 라벨이 같을 확률 = 클래스 비율의 제곱합 (두 행이 서로 무관하게 뽑힌다고 가정)
label_ratio = labels["label"].value_counts(normalize=True)
independent_agreement = float((label_ratio**2).sum())
print(f"  두 행이 모두 있는 쌍 {n_pairs}개")
print(
    f"  불량 기록 26열까지 완전히 같은 쌍: {n_same_record}개 ({n_same_record / n_pairs:.1%})"
)
print(
    f"  라벨이 같은 쌍: {n_same_label}개 ({n_same_label / n_pairs:.1%}) | 독립일 때 기대 {independent_agreement:.1%}"
)

# ---------------------------------------------------------------
# 가설별 예상 (위 I~L 결과의 숫자와 대조해서 읽는다)
# ---------------------------------------------------------------
print("\n=== 가설별 예상 ===")
print(
    "  A 위치(면) : 유형마다 잘 생기는 위치가 달라 한쪽 열에만 있는 유형이 있을 수 있다 (I-1)."
)
print(
    "               같은 원인이 두 곳에 같이 나타나면 겹침이 기대보다 많을 수 있다 (I-2)."
)
print(
    "  B 단계(1·2차): 1차에서 걸린 불량은 빠지므로 겹침이 기대보다 적고 _2가 _1보다 적다 (I-1, I-2)."
)
print(
    "  C 부품(캐비티): 두 부품이 같은 모양이면 건수가 비슷하고(I-1) 같은 원인이면 겹침이 많다 (I-2)."
)
print("  행 층       : 행이 독립된 부품이면 두 행의 기록이 자주 달라야 한다 (L).")
