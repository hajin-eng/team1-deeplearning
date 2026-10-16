"""08. 모델 5종 비교: 로지스틱 / RF / ExtraTrees / HGB / MLP(sklearn)

모두 기본 설정이고 따로 조정하지 않았다.

이 비교가 필요한 이유 (보고서 3번: 베이스라인 대조, 6번: 분할 비교)
  - 모델마다 어느 분할에서 점수가 부풀려지는지 확인한다 (트리는 랜덤 분할에서 크게 부풀려짐).
  - 같은 점수여도 현장에서 쓰는 모습(불량 탐지율, 정상 헛경보율)이 다른지 본다.

출력 열
  랜덤(1회)  : 무작위 5-fold (쌍둥이 때문에 점수가 부풀려진다. 비교용)
  그룹 평균  : 쌍둥이 그룹 5-fold 평균 (모델 비교의 기준 점수)
  시간순*    : 앞 80% 학습 -> 뒤 20% 검증, 검증 구간에 있는 클래스만 평균
  불량 탐지율 : 실제 불량 중 '정상이 아니라고' 예측한 비율 (어떤 불량인지는 안 본다)
  헛경보율   : 실제 정상 중 '불량이라고' 예측한 비율
  (탐지율·헛경보율은 쌍둥이 그룹 분할 out-of-fold 예측으로 계산)

MLP는 확률 보정이 실패해 의미 없는 결과가 나온다. PyTorch로 설계할 MLP와는 다르니 무시한다.
실행: python analysis/08_model_comparison.py    (--quick: 시드 1개)
"""


# 실행 결과 요약
# 트리 모델은 랜덤 분할에서 0.53~0.55로 부풀려지고, 쌍둥이 그룹 분할에서는 0.23~0.28이라 모델 비교에 쓸 수 없음
# 그룹 분할 1위는 HGB 0.283이지만 표준편차(0.02~0.03) 때문에 나머지 모델 사이는 구분이 어려움
# 로지스틱은 많이 잡고 많이 틀리고(탐지율 95%, 헛경보율 77%), 트리는 거의 안 틀리지만 거의 못 잡음(탐지율 10~30%, 헛경보율 3~14%)
# MLP(sklearn)는 확률 보정 실패로 무효 (PyTorch로 설계할 MLP와는 별개)

import common
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import load_data

common.announce("08 모델 5종 비교", "2분", "1분")

data = load_data()
X = data["X_train"].values
y = data["y_train"].values
groups = data["groups_train"].values


class PriorFix:
    """class_weight를 못 주는 모델(MLP)용: 예측 확률을 학습 데이터의 클래스 비율로 나눈다.

    비율이 작은 클래스의 확률을 키워서 소수 클래스를 살리려는 시도다.
    (기타불량처럼 비율이 아주 작으면 노이즈가 커져서 오히려 망가질 수 있다.)
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        # 학습 데이터에서 클래스별 비율을 구함 (모델이 쓰는 클래스 순서에 맞춰 정렬).
        ratio = pd.Series(y).value_counts(normalize=True)
        self.ratio = ratio.reindex(self.model.classes_).values
        return self

    def predict(self, X):
        proba = self.model.predict_proba(X)
        # 확률 / 비율: 흔한 클래스(정상)는 낮추고 드문 클래스는 올림.
        return self.model.classes_[np.argmax(proba / self.ratio, axis=1)]


# 모델 만드는 함수 흐름
# 1) make_ 함수 5개: 각각 모델 하나를 새로 만들어 돌려줌 (학습은 하지 않음)
#    폴드마다 새 모델로 학습해야 앞 폴드의 학습이 새어 들어가지 않으므로 모델 자체가 아닌 '만드는 함수'로 둠
# 2) 아래 MODELS 딕셔너리에 이름과 함수를 묶어 둠
# 3) 맨 아래 반복문이 MODELS를 하나씩 꺼내 make_model()로 모델을 만들고 분할별 점수를 계산함
# 클래스 불균형 대응: 로지스틱·RF·ExtraTrees·HGB는 class_weight를 주고, MLP는 그 옵션이 없어 PriorFix로 대신함
def make_logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0),
    )


def make_rf():
    return RandomForestClassifier(
        300, class_weight="balanced_subsample", n_jobs=-1, random_state=0
    )


def make_extra_trees():
    return ExtraTreesClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0)


def make_hgb():
    return HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=150, learning_rate=0.08, random_state=0
    )


def make_mlp():
    return PriorFix(
        make_pipeline(
            StandardScaler(),
            MLPClassifier(
                (64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0
            ),
        )
    )


MODELS = {
    "로지스틱(balanced)": make_logistic,
    "RandomForest": make_rf,
    "ExtraTrees": make_extra_trees,
    "HistGradientBoosting": make_hgb,
    "MLP(sklearn, 확률 보정)": make_mlp,
}

print(
    f"{'모델':24s}{'랜덤(1회)':>9s}{'그룹 평균':>9s}{'그룹 sd':>8s}{'시간순*':>8s}",
    f"| {'불량 탐지율':>9s}{'정상 헛경보율':>12s}",
    flush=True,
)

is_normal = y == "정상"

for name, make_model in MODELS.items():
    # 무작위 분할은 대조용이라 시드 0 한 번만 (점수가 얼마나 부풀려지는지만 보면 됨).
    random_score = common.random_cv_scores(make_model, X, y, seeds=[0]).mean()

    # 그룹 분할: 점수는 모든 시드, 현장 지표용 예측은 시드 0 것만 모음.
    # (시드마다 예측이 다르면 한 행에 예측이 여러 개라서 현장 지표를 계산할 수 없기 때문)
    group_scores = []
    pooled_pred = np.empty(len(y), dtype=object)
    for seed, train_idx, valid_idx in common.group_folds(y, groups):
        pred = make_model().fit(X[train_idx], y[train_idx]).predict(X[valid_idx])
        group_scores.append(common.macro_f1(y[valid_idx], pred))
        if seed == 0:
            pooled_pred[valid_idx] = pred

    time_score = common.time_split_score(make_model, X, y, present_only=True)

    # 현장 지표: macro F1은 평균 점수라서 '불량을 얼마나 잡고 정상을 얼마나 잘못 경보하는지'는 따로 셈.
    detect = ((pooled_pred != "정상") & ~is_normal).sum() / (~is_normal).sum()
    false_alarm = ((pooled_pred != "정상") & is_normal).sum() / is_normal.sum()

    print(
        f"{name:24s}{random_score:9.3f}{np.mean(group_scores):9.3f}{np.std(group_scores):8.3f}",
        f"{time_score:8.3f} | {detect:9.1%}{false_alarm:12.1%}",
        flush=True,
    )

print("* 시간순 = 앞 80% 학습 -> 뒤 20% 검증, 검증 구간에 실제로 있는 클래스만 평균")
