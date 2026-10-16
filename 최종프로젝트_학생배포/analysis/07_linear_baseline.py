"""07. 선형 모델(로지스틱 회귀) 기준선

PDF의 공개 기준선과 같은 방식(class_weight)으로 로지스틱 회귀를 돌려 본다.
  공개 기준선: 전부 정상 0.1248 / 로지스틱 0.1313 / 로지스틱 class_weight 0.2517

이 실험이 필요한 이유 (보고서 3번: 베이스라인 대조)
  - 우리 파이프라인이 PDF의 기준선과 같은 방식으로 동작하는지 확인한다.
  - 이후 모델(트리, MLP)이 얼마나 나은지 비교할 출발점을 만든다.

조건 비교
  - 특징: 전체 25개 vs 환경 열 6개 제외
  - class_weight: balanced vs 없음
  - 검증: 무작위 / 쌍둥이 그룹 / 시간순

스케일링(StandardScaler)은 전처리 코드에 없어서 모델 안에서 한다.
(파이프라인 안에서 하므로 매번 학습 구간에만 맞춰진다 = 누수 없음)
시간순 분할은 처음 실험과 같은 값을 재현하려고 '모든 클래스 평균' 방식을 쓴다.
실행: python analysis/07_linear_baseline.py    (--quick: 시드 1개)
"""


# 실행 결과 요약
# class_weight=balanced로 쌍둥이 그룹 분할 0.223(랜덤 0.228), 없으면 0.130으로 공개 기준선(0.1313)과 비슷
# 우리 파이프라인이 기준선과 같은 방식으로 동작하는 것을 확인함
# 환경 열 6개를 빼면 balanced 쌍둥이 그룹 점수가 0.223 -> 0.175로 떨어짐 (06번의 RF 결과와 같은 방향)
# 클래스별로는 불량 재현율이 높고(Stain 0.83, Deformation 0.72) 정밀도가 매우 낮아(0.04~0.19) 불량을 많이 잡는 대신 정상을 불량으로 크게 오인함 (정상 재현율 0.229)
# 시간순 80/20은 한 번뿐이고 검증 구간에 없는 클래스를 0점으로 세는 방식이라 값이 흔들려(0.092 / 0.199 / 0.052 / 0.292) 해석하지 않음

# 코드 메모
# load_data(**options)
# 조건마다 넘기는 옵션이 다르고 아예 없을 수도 있음
# {}이면 load_data()와 같고, dict(drop_cols=ENV_COLS)이면 load_data(drop_cols=ENV_COLS)와 같음
# 옵션을 딕셔너리에 담아 두고 **로 풀어서 같은 한 줄로 반복 호출함

import common
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import ENV_COLS, load_data

common.announce("07 로지스틱 기준선", "1분", "20초")


def logistic(class_weight):
    """class_weight를 정해서 모델을 만드는 함수를 돌려준다."""

    def make_model():
        # 로지스틱 회귀는 열의 크기(단위)에 민감해서 먼저 StandardScaler로 맞춤.
        # 이 둘을 파이프라인으로 묶으면 검증 폴드마다 학습 구간에만 맞춰짐.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight=class_weight, max_iter=5000, random_state=0
            ),
        )

    return make_model


# ---------------------------------------------------------------
# 1. 조건별 macro F1
#    목적: 클래스 가중치와 환경 열이 선형 모델 점수에 어떻게 영향을 주는지 봄.
# ---------------------------------------------------------------
print("=== macro F1 (로지스틱 회귀) ===")
CONDITIONS = [
    ("전체 25개 특징", {}),
    ("환경 열 6개 제외(19개)", dict(drop_cols=ENV_COLS)),
]

for label, options in CONDITIONS:
    # **options: 딕셔너리를 풀어서 '이름=값' 인자로 넘김 (위의 코드 메모 참고)
    data = load_data(**options)
    X = data["X_train"].values
    y = data["y_train"].values
    groups = data["groups_train"].values

    for class_weight in ["balanced", None]:
        make_model = logistic(class_weight)

        # 세 가지 분할을 모두 재서, 분할에 따라 점수가 어떻게 달라지는지 봄.
        random_scores = common.random_cv_scores(make_model, X, y)
        group_scores = common.group_cv_scores(make_model, X, y, groups)
        time_score = common.time_split_score(make_model, X, y, present_only=False)

        print(
            f"{label:20s} class_weight={str(class_weight):8s}",
            f"랜덤 {common.fmt(random_scores)} |",
            f"쌍둥이 그룹 {common.fmt(group_scores)} |",
            f"시간순 {time_score:.3f}",
            flush=True,
        )

print(
    "\n공개 기준선(리더보드): 전부 정상 0.1248 / 로지스틱 0.1313 / 로지스틱 class_weight 0.2517"
)

# ---------------------------------------------------------------
# 2. 클래스별 성능 (전체 특징, balanced, 쌍둥이 그룹 분할 out-of-fold)
#    목적: macro F1 한 숫자로는 어떤 불량을 잘 잡고 못 잡는지 알 수 없음.
#          클래스별 정밀도·재현율로 현장 말로 옮길 재료를 만듦.
#    out-of-fold: 모든 행이 '자기를 학습하지 않은 모델'에게 한 번씩 예측받은 결과
# ---------------------------------------------------------------
data = load_data()
X = data["X_train"].values
y = data["y_train"].values
groups = data["groups_train"].values

pred = np.empty(len(y), dtype=object)
for _, train_idx, valid_idx in common.group_folds(y, groups, seeds=[0]):
    model = logistic("balanced")().fit(X[train_idx], y[train_idx])
    pred[valid_idx] = model.predict(X[valid_idx])

print(
    "\n=== 클래스별 (전체 특징, class_weight=balanced, 쌍둥이 그룹 분할 out-of-fold) ==="
)
print(classification_report(y, pred, digits=3, zero_division=0))
