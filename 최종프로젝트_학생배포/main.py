"""
2조 - 다이캐스팅 불량유형 예측
포스코 K-뉴딜아카데미 딥러닝 최종 프로젝트

실행 방법: 학생배포 폴더에서  python main.py
결과 파일: 제출_2조.csv
"""

# ============================================================
# 0. 라이브러리
# ============================================================
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocess import load_data  # 데이터 파트 전처리 (2~3단계)


# ============================================================
# 1. 설정 (현장 수정 요청이 오면 주로 여기를 고친다)
# ============================================================
SEED = 42
TEAM_NAME = "2조"

BASE_DIR = Path(__file__).resolve().parent  # main.py가 있는 폴더
SUBMIT_PATH = BASE_DIR / f"제출_{TEAM_NAME}.csv"

ID_COLS = ["shot_id", "id", "Shot"]  # 피처로 쓰면 실격인 열 (검사용)
DROP_COLS = ()  # 추가로 뺄 열 (현장 수정: "이 열 빼고 돌려 보세요" → 여기에 이름 추가)
TARGET_COL = "label"
LABELS = [
    "정상",
    "불량_Short_Shot",
    "불량_Blow_Hole",
    "불량_Exfoliation",
    "불량_Stain",
    "불량_Deformation",
    "기타불량",
]  # 제출 가능한 라벨 7종

N_TRAIN, N_TEST = 5274, 2261  # 있어야 할 행 수

VALID_FOLDS = 5  # train을 5조각으로 나눠 조각마다 한 번씩 검증용으로 쓴다 (교차검증)

CLASS_WEIGHT_POWER = (
    0.5  # 소수 클래스 가중치 강도 (0 = 끔, 1 = balanced 그대로, 0.5 = 그 중간)
)

MODEL_NAME = "rf"  # 사용할 모델: "logistic" (로지스틱 회귀), "rf" (랜덤포레스트), "hgb" (HistGradientBoosting), "rf_et" (랜덤포레스트 + ExtraTrees 확률 평균), "rf_et_2stage" (rf_et + 두 단계 모델 혼합)
MAX_ITER = 1000  # [로지스틱] 학습 반복 최대 횟수 (기본 100회로는 학습이 덜 끝남)
RF_N_TREES = 300  # [RF] 나무 수 (많을수록 안정적이지만 느림)
RF_MIN_LEAF = 1  # [RF] 잎 하나에 있어야 할 최소 샷 수 (클수록 덜 외움, 1 = 기본값 = E안)
RF_MAX_FEATURES = "sqrt"  # [RF] 질문 하나를 고를 때 보는 열의 수 ("sqrt" = 기본 = E안, 0.5 = 열의 절반)
RFET_N_TREES = 900  # [rf_et] 랜덤포레스트와 ExtraTrees 각각의 나무 수
HGB_MAX_LEAF_NODES = 15  # [HGB] 나무 하나의 잎 최대 개수 (데이터 파트 추천 설정 B)
HGB_LEARNING_RATE = 0.05  # [HGB] 앞 나무의 오답을 한 번에 얼마나 고칠지 (설정 B)
HGB_MAX_ITER = 100  # [HGB] 나무 수 (설정 B)

PUBLIC_BASELINE = 0.2517  # 강사 공개 기준선 (로지스틱 + class_weight, public 점수)

SAMPLE_PATH = BASE_DIR / "제출_예시.csv"  # 형식 검사에 쓰는 강사 제공 예시


# ============================================================
# 2. 단계별 함수
# ============================================================
def set_seed(seed):
    """모든 난수 생성기의 시드를 고정한다. 같은 코드 → 같은 결과."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass  # torch는 MLP 단계부터 사용한다


def check_data(data):
    """[2~3단계] 데이터 파트 load_data()의 결과가 있어야 할 모습인지 확인한다."""
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, groups = data["X_test"], data["groups_train"]

    assert len(X_train) == len(y_train) == len(groups) == N_TRAIN, "train 행 수가 다름"
    assert len(X_test) == len(data["test_ids"]) == N_TEST, "제출 대상 행 수가 다름"
    assert list(X_train.columns) == list(X_test.columns), "train과 test의 열이 다름"
    assert not set(ID_COLS) & set(X_train.columns), "식별 열이 피처에 들어감"
    assert "group" not in X_train.columns, "그룹 번호가 피처에 들어감"
    assert y_train.isin(LABELS).all(), "알 수 없는 라벨이 있음"

    print(
        f"[2~3단계] 데이터 파트 load_data() 완료: train {len(X_train)}행 / "
        f"제출 대상 {len(X_test)}행 / 피처 {X_train.shape[1]}개 / "
        f"남은 결측 {int(X_train.isna().sum().sum() + X_test.isna().sum().sum())}개 / "
        f"쌍둥이 그룹 {groups.nunique()}개"
    )


def split_validation(y, groups):
    """[4단계] train을 5조각으로 나눠, (학습용, 검증용) 5쌍을 만든다.

    각 조각이 한 번씩 돌아가며 검증용이 된다 (교차검증).
    쌍둥이는 같은 조각에 넣고(Group), 클래스 비율은 조각마다 같게 맞춘다(Stratified).
    """
    sgkf = StratifiedGroupKFold(n_splits=VALID_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    counts = {}
    for k, (fit_pos, valid_pos) in enumerate(sgkf.split(y, y, groups), start=1):
        fit_idx, valid_idx = y.index[fit_pos], y.index[valid_pos]

        # 검증
        assert len(fit_idx) + len(valid_idx) == N_TRAIN, (
            "학습용 + 검증용이 train 수와 다름"
        )
        assert not set(groups.loc[fit_idx]) & set(groups.loc[valid_idx]), (
            "쌍둥이가 양쪽에 나뉘어 들어감"
        )

        folds.append((fit_idx, valid_idx))
        counts[f"조각{k}"] = y.loc[valid_idx].value_counts()

    # 검증: 모든 train 샷이 정확히 한 번씩 검증용이 되었는가
    all_valid = np.concatenate([valid_idx for _, valid_idx in folds])
    assert sorted(all_valid) == sorted(y.index), (
        "검증용이 train을 정확히 한 번씩 덮지 않음"
    )

    print(f"[4단계] 교차검증 분할: {VALID_FOLDS}조각. 조각별 검증용 클래스 수:")
    print(pd.DataFrame(counts).reindex(LABELS).to_string())
    return folds


def cross_validate(X, y, folds):
    """[5~6단계] 5쌍을 돌며 학습·예측하고, 검증 예측을 하나로 모은다.

    반환: 모은 예측(train 전체에 하나씩), 조각별 macro F1 목록
    """
    oof_pred = pd.Series(index=y.index, dtype=object)
    fold_scores = []
    for k, (fit_idx, valid_idx) in enumerate(folds, start=1):
        y_fit = y.loc[fit_idx]
        model = build_model(y_fit)
        model.fit(X.loc[fit_idx], y_fit)
        pred = model.predict(X.loc[valid_idx])
        oof_pred.loc[valid_idx] = pred

        score = f1_score(
            y.loc[valid_idx],
            pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        fold_scores.append(score)
        print(
            f"[5단계] 조각{k}: 학습 {len(fit_idx)}행 "
            f"→ 검증 {len(valid_idx)}행 macro F1 = {score:.4f}"
        )

    assert oof_pred.notna().all(), "검증 예측이 비어 있는 샷이 있음"
    print(
        f"[6단계] 조각별 macro F1: 평균 {np.mean(fold_scores):.4f} "
        f"± {np.std(fold_scores):.4f} (최저 {min(fold_scores):.4f} / 최고 {max(fold_scores):.4f})"
    )
    return oof_pred, fold_scores


def make_class_weight(y):
    """[5단계] 학습용 정답으로 클래스별 가중치를 계산한다.

    가중치 = (balanced 가중치) ** 강도. 적은 클래스일수록 가중치가 크다.
    학습용 정답만으로 계산해야 검증용 정보가 새어 들어가지 않는다.
    """
    labels = np.array(LABELS)
    balanced = compute_class_weight("balanced", classes=labels, y=y)
    return {label: w**CLASS_WEIGHT_POWER for label, w in zip(labels, balanced)}


class AverageProbaClassifier(ClassifierMixin, BaseEstimator):
    """여러 분류기를 각각 학습하고, 예측 확률을 평균해 가장 큰 라벨을 고른다.

    글자 라벨 그대로 학습시키려고 sklearn의 VotingClassifier(라벨을 숫자로 바꿔서
    글자 키 가중치 사전과 맞지 않음) 대신 직접 만들었다. Pipeline의 마지막 단계로 쓴다.
    """

    def __init__(self, members):
        self.members = members  # [(이름, 분류기), ...]

    def fit(self, X, y):
        for _, member in self.members:
            member.fit(X, y)
        self.classes_ = self.members[0][1].classes_  # 모든 분류기가 같은 순서(가나다순)
        return self

    def predict_proba(self, X):
        return sum(member.predict_proba(X) for _, member in self.members) / len(self.members)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TwoStageMixClassifier(ClassifierMixin, BaseEstimator):
    """RF+ExtraTrees 다중분류와 '두 단계 모델'의 예측 확률을 절반씩 섞는다 (G안).

    - 다중분류: 7종 라벨을 한 번에 맞히는 RF와 ExtraTrees의 확률 평균 (rf_et와 같음)
    - 두 단계: 1단계 '불량인가'(RF+ET 평균) → 2단계 '불량 6종 중 무엇인가'(불량 샷만으로 학습한 RF+ET 평균)
      확률 = 불량 확률 x 유형 확률, 정상 확률 = 1 - 불량 확률
    - 최종 확률 = (1 - mix) x 다중분류 + mix x 두 단계, 가장 큰 라벨을 고른다.
    글자 라벨 그대로 학습한다. Pipeline의 마지막 단계로 쓴다.
    """

    def __init__(self, n_trees=900, min_leaf=1, max_features="sqrt", class_weight=None, mix=0.5, seed=42):
        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.mix = mix
        self.seed = seed

    def _pair(self, class_weight):
        common = dict(
            n_estimators=self.n_trees,
            min_samples_leaf=self.min_leaf,
            max_features=self.max_features,
            class_weight=class_weight,
            random_state=self.seed,
            n_jobs=-1,
        )
        return [RandomForestClassifier(**common), ExtraTreesClassifier(**common)]

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        self.normal_ = int(np.where(self.classes_ == "정상")[0][0])
        defect = y != "정상"
        self.multi_ = [m.fit(X, y) for m in self._pair(self.class_weight)]
        self.stage1_ = [m.fit(X, defect) for m in self._pair("balanced_subsample")]
        self.stage2_ = [m.fit(X[defect], y[defect]) for m in self._pair("balanced")]
        return self

    def predict_proba(self, X):
        n = len(X)
        multi = sum(m.predict_proba(X) for m in self.multi_) / len(self.multi_)
        p_defect = sum(m.predict_proba(X)[:, 1] for m in self.stage1_) / len(self.stage1_)
        by_type = np.zeros((n, len(self.classes_)))
        for m in self.stage2_:
            proba = m.predict_proba(X)
            for j, label in enumerate(m.classes_):
                by_type[:, int(np.where(self.classes_ == label)[0][0])] += proba[:, j] / len(self.stage2_)
        two_stage = by_type * p_defect[:, None]
        two_stage[:, self.normal_] = 1 - p_defect
        return (1 - self.mix) * multi + self.mix * two_stage

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_model(y):
    """[5단계] 학습할 모델을 만든다. y는 학습용 정답 (클래스 가중치 계산용)

    MODEL_NAME에 따라 로지스틱 또는 랜덤포레스트를 Pipeline으로 묶는다.
    묶어 두면 중앙값·평균 같은 값이 학습용 데이터에서만 계산된다.
    """
    class_weight = make_class_weight(y)

    if MODEL_NAME == "logistic":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight=class_weight, max_iter=MAX_ITER)),
        ]
    elif MODEL_NAME == "rf":
        steps = [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # 나무는 크기 맞추기가 필요 없다
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=RF_N_TREES,
                    min_samples_leaf=RF_MIN_LEAF,
                    max_features=RF_MAX_FEATURES,
                    class_weight=class_weight,
                    random_state=SEED,
                    n_jobs=-1,  # CPU 코어를 모두 써서 빠르게 (결과는 같다)
                ),
            ),
        ]
    elif MODEL_NAME == "rf_et":
        # 두 모델(잎 최소 RF_MIN_LEAF, 열 RF_MAX_FEATURES)의 예측 확률을 평균해 가장 큰 라벨을 고른다.
        # ExtraTrees는 분할 기준을 무작위로 골라 RF보다 덜 공격적으로 불량을 부른다.
        members = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=RFET_N_TREES,
                    min_samples_leaf=RF_MIN_LEAF,
                    max_features=RF_MAX_FEATURES,
                    class_weight=class_weight,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=RFET_N_TREES,
                    min_samples_leaf=RF_MIN_LEAF,
                    max_features=RF_MAX_FEATURES,
                    class_weight=class_weight,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
        steps = [("clf", AverageProbaClassifier(members))]
    elif MODEL_NAME == "rf_et_2stage":
        steps = [
            (
                "clf",
                TwoStageMixClassifier(
                    n_trees=RFET_N_TREES,
                    min_leaf=RF_MIN_LEAF,
                    max_features=RF_MAX_FEATURES,
                    class_weight=class_weight,
                    mix=0.5,
                    seed=SEED,
                ),
            )
        ]
    elif MODEL_NAME == "hgb":
        # HGB는 결측을 스스로 처리하고 크기 맞추기도 필요 없다.
        # 글자 라벨로 만든 가중치 사전을 받지 못해서 "balanced"(강도 1)로 고정한다 (데이터 파트 추천 설정 그대로).
        steps = [
            (
                "clf",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_leaf_nodes=HGB_MAX_LEAF_NODES,
                    learning_rate=HGB_LEARNING_RATE,
                    max_iter=HGB_MAX_ITER,
                    random_state=SEED,
                ),
            ),
        ]
    else:
        raise ValueError(f"알 수 없는 MODEL_NAME: {MODEL_NAME}")

    return Pipeline(steps)


def evaluate(y_true, y_pred):
    """[6단계] 검증 점수를 계산하고 출력한다."""
    score = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    print(
        f"[6단계] 합쳐서 채점한 macro F1 = {score:.4f} (공개 기준선 {PUBLIC_BASELINE})"
    )
    print(
        classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    short = [name.replace("불량_", "") for name in LABELS]
    cm_table = pd.DataFrame(cm, index=short, columns=short)
    print("혼동행렬 (행 = 실제, 열 = 예측)")
    print(cm_table.to_string())
    return score


def make_submission(model, X_test, test_ids):
    """[7단계] 제출용 예측 표를 만든다."""
    pred = model.predict(X_test)
    sub = pd.DataFrame({"shot_id": test_ids.values, TARGET_COL: pred})
    return sub


def check_submission(sub):
    """[7단계] 제출 형식이 규칙에 맞는지 검사한다."""
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")

    assert len(sub) == N_TEST, f"제출 행 수가 {len(sub)}개"
    assert list(sub.columns) == ["shot_id", TARGET_COL], (
        f"열 이름이 {list(sub.columns)}"
    )
    assert sub["shot_id"].tolist() == sample["shot_id"].tolist(), (
        "shot_id가 제출 예시와 다름"
    )
    assert sub[TARGET_COL].isin(LABELS).all(), "허용되지 않은 라벨이 있음"

    print("[7단계] 제출 형식 검사 통과. 예측 분포:")
    print(sub[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).to_string())


# ============================================================
# 3. 전체 실행 순서
# ============================================================
def main():
    start = time.time()
    set_seed(SEED)

    data = load_data(drop_cols=DROP_COLS)  # 2~3단계 (데이터 파트 preprocess.py)
    check_data(data)
    X_train, y_train = data["X_train"], data["y_train"]
    folds = split_validation(y_train, data["groups_train"])  # 4단계
    oof_pred, fold_scores = cross_validate(X_train, y_train, folds)  # 5~6단계
    evaluate(y_train, oof_pred)  # 6단계

    final_model = build_model(y_train)
    final_model.fit(X_train, y_train)
    weight = "balanced" if MODEL_NAME == "hgb" else f"강도 {CLASS_WEIGHT_POWER}"
    print(
        f"[7단계] 최종 학습 완료: train 전체 {len(X_train)}행 "
        f"({MODEL_NAME}, class_weight {weight})"
    )

    sub = make_submission(final_model, data["X_test"], data["test_ids"])
    check_submission(sub)
    sub.to_csv(SUBMIT_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"[7단계] 저장 완료: {SUBMIT_PATH.name} ({len(sub)}행)")

    print(f"실행 시간: {time.time() - start:.1f}초")


if __name__ == "__main__":
    main()
