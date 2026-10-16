"""01. 데이터 기본 점검

확인하는 것
  - 파일 정합성 (분할표 / 정답 / 제출 예시가 서로 맞는가)
  - 상수 열, 결측, 0값, 너무 긴 사이클 시간
  - 쌍둥이(식별자를 뺀 입력값이 바로 앞 행과 같은 행)가 얼마나 있는가
  - 제품(Product_Type)이 어느 shot에서 바뀌는가

01번 점검 결과가 preprocess.py의 처리 규칙이 된다.
  상수 열 -> 제거 / 결측 -> 채우기 / 0값·긴 사이클 -> 표시 열 / 쌍둥이 -> group 번호

읽기만 하고 아무 파일도 만들지 않는다.
실행: python analysis/01_data_verify.py
"""


# 실행 결과 요약
# 파일 정합성: 정답의 shot_id는 분할표의 train과, 제출 예시의 shot_id는 public+private과 일치함
# 상수 열 8개(Air_Pressure·Coolant_Temp·Factory_Temp·Factory_Humidity의 Min/Max)는 값이 하나뿐이라 모델 입력에서 제외함
# 결측은 Factory_Temp·Factory_Humidity와 각각의 Min/Max, 총 6개 열의 같은 90행(shot 6914~7067 범위)에 몰려 있음
# 식별 열 3개(shot_id, id, Shot)와 상수 열 8개를 뺀 특징 열은 21개
# 속도·승압시간이 0인 shot은 3348·3349·3420·3421, 비스킷 두께가 0인 shot은 2498~2501
# Cycle_Time이 60초를 넘는 행은 11개이고, 설비 정지 후 재가동으로 추정됨
# 입력이 바로 앞 행과 같은 행(쌍둥이)이 3,174개(42%)이고, 테스트 2,261행 중 1,305행은 입력이 같은 행이 train에도 있음
# train에서 입력이 같은 그룹 중 라벨이 모두 일치하는 그룹 1,429개, 라벨이 다른 그룹 57개
# 라벨은 정상 4,092개(77.6%)로 불균형이 심하고, Product_Type은 shot 4207까지 1, 4208부터 2로 바뀜

import common
import numpy as np
import pandas as pd

common.announce("01 데이터 기본 점검", "몇 초", "몇 초")

# 식별 열: 번호표일 뿐 공정 정보가 아니라서 특징에서 제외함(PDF 규칙: 사용하면 실격).
ID_COLS = ["shot_id", "id", "Shot"]

# ---------------------------------------------------------------
# 파일 읽기
# ---------------------------------------------------------------
# 목적: 원본 CSV 파일을 수정하지 않고 읽기만 함. 정리는 메모리에서 하고 파일은 건드리지 않음.
split = pd.read_csv("분할표.csv", encoding="utf-8-sig")
labels = pd.read_csv("정답_train.csv", encoding="utf-8-sig")
submit = pd.read_csv("제출_예시.csv", encoding="utf-8-sig")

# 입력 파일은 헤더가 두 줄(1행: 대분류, 2행: 변수명)
# 둘째 줄(header=1)을 열 이름으로 사용하는 것으로 결정함.
X = pd.read_csv("data/주조공정_입력.csv", header=1, encoding="utf-8-sig")
X.columns = X.columns.str.strip()

# ---------------------------------------------------------------
# 1. 파일 정합성
#    목적: 분할표·정답·제출 예시·입력 파일이 같은 shot_id 체계를 쓰는지 확인한다. 어긋나 있으면 이후 모든 분석이 틀어진다.
# ---------------------------------------------------------------
print("=== 1. 파일 정합성 ===")
print("분할표 구성:", split["split"].value_counts().to_dict())
print("입력 크기:", X.shape)

train_ids = set(split.loc[split["split"] == "train", "shot_id"])
test_ids = set(split.loc[split["split"] != "train", "shot_id"])

print("정답 shot_id == 분할표 train:", set(labels["shot_id"]) == train_ids)
print("제출 예시 shot_id == public+private:", set(submit["shot_id"]) == test_ids)

# ---------------------------------------------------------------
# 2. 상수 열 / 결측
#    목적: 정보가 없는 열(상수)과 값이 빠진 곳을 찾아서 처리 대상으로 삼는다.
# ---------------------------------------------------------------
print("\n=== 2. 상수 열 / 결측 ===")
# 고유값이 1개 이하인 열은 모든 행이 같은 값이라 모델이 배울 것이 없음.
constant_cols = []
for c in X.columns:
    if X[c].nunique(dropna=True) <= 1:
        constant_cols.append(c)
print("상수 열:", constant_cols)

na_count = X.isna().sum()
print("결측이 있는 열:", na_count[na_count > 0].to_dict())

# 결측이 어느 샷에 몰려 있는지 보면 원인(센서 끊김 등)을 짐작할 수 있을 것으로 추론.
rows_with_na = X[X.isna().any(axis=1)]
print(
    "결측 행 수:",
    len(rows_with_na),
    "| shot 범위:",
    rows_with_na["shot_id"].min(),
    "~",
    rows_with_na["shot_id"].max(),
)

# 식별 열과 상수 열을 뺀 것이 실제 모델이 쓰는 특징.
feature_cols = []
for c in X.columns:
    if c not in ID_COLS + constant_cols:
        feature_cols.append(c)
print("특징 열 수(식별자·상수 열 제외):", len(feature_cols))

# ---------------------------------------------------------------
# 3. 0값, 너무 긴 사이클
#    목적: 물리적으로 말이 안 되는 값(속도 0 등)이 어느 샷에 있는지 찾는다.
#         설비가 멈췄다 재가동된 샷일 수 있어서 삭제하지 않고 preprocess.py의 표시 열(is_restart, bad_biscuit, long_cycle)로 남긴다.
# ---------------------------------------------------------------
print("\n=== 3. 0값 / 긴 사이클 ===")
for col in [
    "High_Velocity",
    "Rapid_Rise_Time",
    "Pressure_Rise_Time",
    "Biscuit_Thickness",
]:
    print(f"{col} == 0 인 shot:", X.loc[X[col] == 0, "shot_id"].tolist())

# 보통 사이클은 20~36초인데 60초를 넘는 행은 '설비 정지 후 재가동'인 것으로 생각됨.
print("Cycle_Time > 60초 행 수:", int((X["Cycle_Time"] > 60).sum()))

# ---------------------------------------------------------------
# 4. 쌍둥이 (입력이 바로 앞 행과 완전히 같은 행)
#    목적: 같은 입력이 반복되는 정도를 재서, 검증을 무작위로 나눠도 되는지 판단.
#         많이 반복되면 학습과 검증에 같은 입력이 갈라져 점수가 부풀려질 가능성 있음.
# ---------------------------------------------------------------
print("\n=== 4. 쌍둥이 ===")
# NaN == NaN은 False라서 결측끼리도 같은 값으로 인식되지 않으므로, 결측(NaN)을 -999로 바꿔서 비교함.
values = X[feature_cols].fillna(-999).values
# 각 행이 바로 앞 행과 동일한지 확인하기 위해 values[1:]와 values[:-1]을 비교.
same_as_prev = (values[1:] == values[:-1]).all(axis=1)
print("바로 앞 행과 입력이 같은 행:", int(same_as_prev.sum()))

# 같은 입력이 이어지는 행들을 한 그룹으로 묶음(그룹 번호는 0부터).
# 앞 행과 다르면(~same_as_prev) 새 그룹이 시작되므로 누적합(cumsum)이 그룹 번호가 됨.
X["group"] = np.r_[0, np.cumsum(~same_as_prev)]
group_size = X.groupby("group").size()
print("그룹 크기별 그룹 수:", group_size.value_counts().to_dict())

# 테스트 행 중 입력이 같은 행이 train에도 있는 행의 개수: 많을수록 리더보드 점수가 부풀려지기 쉬움.
train_groups = set(X.loc[X["shot_id"].isin(train_ids), "group"])
is_test = X["shot_id"].isin(test_ids)
test_with_twin = int((is_test & X["group"].isin(train_groups)).sum())
print(f"테스트 {len(test_ids)}행 중 같은 입력이 train에 있는 행: {test_with_twin}")

# train 안에서 같은 입력의 행들이 같은 라벨을 갖는지: 다르면 입력만으로는 구분할 수 없는 불량이 있다는 뜻.
merged = X.merge(labels, on="shot_id")
per_group = merged.groupby("group")["label"].agg(n_labels="nunique", size="size")
per_group = per_group[per_group["size"] > 1]  # 행이 2개 이상인 그룹만
print(
    "train 쌍둥이 그룹 라벨 일치:",
    int((per_group["n_labels"] == 1).sum()),
    "/ 불일치:",
    int((per_group["n_labels"] > 1).sum()),
)

# ---------------------------------------------------------------
# 5. 라벨 분포 / 제품 구간
#    목적: 클래스 불균형 정도(정상이 77%)와 제품이 바뀌는 지점을 확인.
# ---------------------------------------------------------------
print("\n=== 5. 라벨 분포 ===")
print(labels["label"].value_counts().to_string())

print("\n=== 6. 제품(Product_Type)별 shot 범위 ===")
print(X.groupby("Product_Type")["shot_id"].agg(["min", "max", "count"]).to_string())
