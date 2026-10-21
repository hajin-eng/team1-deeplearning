"""
동동문하 - 다이캐스팅 불량유형 예측
포스코 K-뉴딜아카데미 딥러닝 최종 프로젝트

실행 방법: 학생배포 폴더에서  python main.py
결과 파일: 제출_동동문하.csv
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ============================================================
# 1. 설정 (현장 수정 요청이 오면 주로 여기를 고친다)
# ============================================================
SEED = 42
TEAM_NAME = "동동문하"

BASE_DIR = Path(__file__).resolve().parent  # main.py가 있는 폴더
INPUT_PATH = BASE_DIR / "data" / "주조공정_입력.csv"
SPLIT_PATH = BASE_DIR / "분할표.csv"
LABEL_PATH = BASE_DIR / "정답_train.csv"
SUBMIT_PATH = BASE_DIR / f"제출_{TEAM_NAME}.csv"

ID_COLS = ["shot_id", "id", "Shot"]  # 피처로 쓰면 실격인 열
CONSTANT_COLS = [
    "Air_Pressure_Min",
    "Air_Pressure_Max",  # 모든 행에서 값이 하나뿐인 열
    "Coolant_Temp_Min",
    "Coolant_Temp_Max",  # (관리 기준 범위로 보임)
    "Factory_Temp_Min",
    "Factory_Temp_Max",
    "Factory_Humidity_Min",
    "Factory_Humidity_Max",
]
EXTRA_DROP_COLS = []  # 추가로 뺄 열 (현장 수정: "이 열 빼고 돌려 보세요" → 여기에 이름 추가)
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

N_TOTAL, N_TRAIN, N_TEST = 7535, 5274, 2261  # 있어야 할 행 수

VALID_FOLDS = 5  # train을 5조각으로 나눠 조각마다 한 번씩 검증용으로 쓴다 (교차검증)

CLASS_WEIGHT_POWER = (
    0.5  # 소수 클래스 가중치 강도 (0 = 끔, 1 = balanced 그대로, 0.5 = 그 중간)
)
MAX_ITER = 1000  # 학습 반복 최대 횟수 (기본 100회로는 학습이 덜 끝남)

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


def load_data():
    """[2단계] 입력·분할표·정답을 읽어 하나의 표로 합친다."""
    # (1) 파일 읽기
    inputs = pd.read_csv(INPUT_PATH, header=1, encoding="utf-8-sig")
    inputs.columns = inputs.columns.str.strip()  # 컬럼명 앞뒤 공백 제거
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    labels = pd.read_csv(LABEL_PATH, encoding="utf-8-sig")

    # (2) shot_id 기준으로 합치기
    df = inputs.merge(split, on="shot_id", how="left", validate="one_to_one")
    df = df.merge(labels, on="shot_id", how="left", validate="one_to_one")
    df = df.sort_values("shot_id").reset_index(drop=True)

    # (3) 검증: 있어야 할 모습과 다르면 즉시 멈춘다
    is_train = df["split"] == "train"
    assert len(df) == N_TOTAL, f"전체 행 수가 {len(df)}개"
    assert df["split"].notna().all(), "분할 정보가 없는 샷이 있음"
    assert is_train.sum() == N_TRAIN, f"train 행 수가 {is_train.sum()}개"
    assert (~is_train).sum() == N_TEST, f"제출 대상 행 수가 {(~is_train).sum()}개"
    assert df.loc[is_train, TARGET_COL].notna().all(), "정답이 없는 train 샷이 있음"
    assert df.loc[~is_train, TARGET_COL].isna().all(), "제출 대상에 정답이 붙어 있음"
    assert df.loc[is_train, TARGET_COL].isin(LABELS).all(), "알 수 없는 라벨이 있음"

    print(
        f"[2단계] 데이터 불러오기 완료: 전체 {len(df)}행 × {df.shape[1]}열 "
        f"(train {is_train.sum()} / 제출 대상 {(~is_train).sum()})"
    )
    return df


def preprocess(df):
    """[3단계] 모델에 넣을 피처 표를 만든다. (데이터 파트 결과로 교체될 부분)

    여기서는 규칙형 처리(열 빼기)만 한다.
    결측 채우기처럼 데이터에서 값을 계산하는 처리는 5단계 Pipeline 안에서 한다.
    """
    non_feature_cols = ID_COLS + ["split", TARGET_COL]
    drop_cols = non_feature_cols + CONSTANT_COLS + EXTRA_DROP_COLS

    # 상수열이 정말 상수인지 확인 (결측은 빼고 센다)
    for col in CONSTANT_COLS:
        assert df[col].nunique() == 1, f"{col}은 상수열이 아님"

    X = df.drop(columns=drop_cols)

    # 검증
    assert len(X) == N_TOTAL, "전처리 중에 행 수가 바뀜"
    assert not set(ID_COLS) & set(X.columns), "식별 열이 피처에 들어감"
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns), (
        "숫자가 아닌 열이 있음"
    )

    n_missing = X.isna().sum()
    print(
        f"[3단계] 전처리 완료: 피처 {X.shape[1]}개 (제외 {len(drop_cols)}개) "
        f"/ 결측이 있는 열: {n_missing[n_missing > 0].to_dict()}"
    )
    return X


def make_twin_groups(df):
    """[4단계] 같은 샷이 두 번 기록된 쌍둥이 행에 같은 그룹 번호를 붙인다.

    쌍둥이 = 바로 앞 행과 입력값(Shot 포함)이 전부 같은 행
    """
    input_cols = [
        c for c in df.columns if c not in ("shot_id", "id", "split", TARGET_COL)
    ]
    values = df[input_cols].fillna(-1)  # 결측끼리도 같다고 비교하기 위해
    same_as_prev = values.eq(values.shift()).all(
        axis=1
    )  # 바로 앞 행과 모든 열이 같은가
    groups = (~same_as_prev).cumsum()  # 앞 행과 다를 때마다 그룹 번호 +1

    print(
        f"[4단계] 쌍둥이 그룹: 바로 앞 행과 같은 행 {same_as_prev.sum()}개 "
        f"→ 그룹 {groups.nunique()}개"
    )
    return groups


def split_validation(df, groups):
    """[4단계] train을 5조각으로 나눠, (학습용, 검증용) 5쌍을 만든다.

    각 조각이 한 번씩 돌아가며 검증용이 된다 (교차검증).
    쌍둥이는 같은 조각에 넣고(Group), 클래스 비율은 조각마다 같게 맞춘다(Stratified).
    """
    train_idx = df.index[df["split"] == "train"]
    y = df.loc[train_idx, TARGET_COL]
    g = groups.loc[train_idx]

    sgkf = StratifiedGroupKFold(n_splits=VALID_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    counts = {}
    for k, (fit_pos, valid_pos) in enumerate(sgkf.split(train_idx, y, g), start=1):
        fit_idx, valid_idx = train_idx[fit_pos], train_idx[valid_pos]

        # 검증
        assert len(fit_idx) + len(valid_idx) == N_TRAIN, (
            "학습용 + 검증용이 train 수와 다름"
        )
        assert not set(groups.loc[fit_idx]) & set(groups.loc[valid_idx]), (
            "쌍둥이가 양쪽에 나뉘어 들어감"
        )

        folds.append((fit_idx, valid_idx))
        counts[f"조각{k}"] = df.loc[valid_idx, TARGET_COL].value_counts()

    # 검증: 모든 train 샷이 정확히 한 번씩 검증용이 되었는가
    all_valid = np.concatenate([valid_idx for _, valid_idx in folds])
    assert sorted(all_valid) == sorted(train_idx), (
        "검증용이 train을 정확히 한 번씩 덮지 않음"
    )

    print(f"[4단계] 교차검증 분할: {VALID_FOLDS}조각. 조각별 검증용 클래스 수:")
    print(pd.DataFrame(counts).reindex(LABELS).to_string())
    return folds


def cross_validate(df, X, folds):
    """[5~6단계] 5쌍을 돌며 학습·예측하고, 검증 예측을 하나로 모은다.

    반환: 모은 예측(train 전체에 하나씩), 조각별 macro F1 목록
    """
    oof_pred = pd.Series(index=df.index[df["split"] == "train"], dtype=object)
    fold_scores = []
    for k, (fit_idx, valid_idx) in enumerate(folds, start=1):
        y_fit = df.loc[fit_idx, TARGET_COL]
        model = build_model(y_fit)
        model.fit(X.loc[fit_idx], y_fit)
        pred = model.predict(X.loc[valid_idx])
        oof_pred.loc[valid_idx] = pred

        score = f1_score(
            df.loc[valid_idx, TARGET_COL],
            pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        fold_scores.append(score)
        n_iter = model.named_steps["clf"].n_iter_[0]
        print(
            f"[5단계] 조각{k}: 학습 {len(fit_idx)}행 (반복 {n_iter}회) "
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


def build_model(y):
    """[5단계] 학습할 모델을 만든다. y는 학습용 정답 (클래스 가중치 계산용)

    결측 채우기 → 크기 맞추기 → 로지스틱 회귀를 하나로 묶는다(Pipeline).
    묶어 두면 중앙값·평균 같은 값이 학습용 데이터에서만 계산된다.
    """
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight=make_class_weight(y), max_iter=MAX_ITER
                ),
            ),
        ]
    )
    return model


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

    # 아래는 단계를 진행하면서 하나씩 채운다
    df = load_data()  # 2단계
    X = preprocess(df)  # 3단계
    groups = make_twin_groups(df)  # 4단계
    folds = split_validation(df, groups)  # 4단계
    oof_pred, fold_scores = cross_validate(df, X, folds)  # 5~6단계
    train_idx = df.index[df["split"] == "train"]
    evaluate(df.loc[train_idx, TARGET_COL], oof_pred.loc[train_idx])  # 6단계

    test_idx = df.index[df["split"] != "train"]
    y_train = df.loc[train_idx, TARGET_COL]
    final_model = build_model(y_train)
    final_model.fit(X.loc[train_idx], y_train)
    print(
        f"[7단계] 최종 학습 완료: train 전체 {len(train_idx)}행 "
        f"(class_weight 강도 {CLASS_WEIGHT_POWER})"
    )

    sub = make_submission(final_model, X.loc[test_idx], df.loc[test_idx, "shot_id"])
    check_submission(sub)
    sub.to_csv(SUBMIT_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"[7단계] 저장 완료: {SUBMIT_PATH.name} ({len(sub)}행)")

    print(f"실행 시간: {time.time() - start:.1f}초")


if __name__ == "__main__":
    main()
