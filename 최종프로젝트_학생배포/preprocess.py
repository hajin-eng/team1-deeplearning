"""전처리. 학생배포/ 폴더에서 실행하고, 모든 경로는 상대경로.

원본 CSV는 수정하지 않고, 읽을 때마다 여기서 정리한다.
사용: from preprocess import load_data

데이터 확인에서 알게 된 것 (코드에 반영하지 않은 것 포함)
- Type 1 안에서도 설정이 바뀐다: shot 3480~3512 사이에 Casting_Pressure 1052 -> 1155,
  Cylinder_Pressure 217 -> 239, Cycle_Time 20.8 -> 22.6. 입력값 수준으로 드러나므로 별도 처리하지 않는다.
- Shot 열은 shot_id 1100, 3414, 4208(제품 전환과 같음), 4808, 6258, 6266에서 리셋된다.
  식별 열이라 특징으로 쓰지 않는다 (구간으로 묶어 쓰는 것도 규칙상 애매하니 강사에게 확인).
- 센서 고정(같은 값이 최장 196~270행): 값이 바뀌는 지점은 Shot 리셋과 무관하고 오류로 볼 근거가 없어 그대로 둔다.
- Spray_Time은 Spray_1 + Spray_2가 아니다 (차이 3.3~8.9, 일정하지 않음). 정의를 모르므로 합계 파생 변수는 만들지 않는다.
- Cylinder_Pressure와 Casting_Pressure는 제품 안에서도 거의 같은 값이다 (상관 Type 1 1.00, Type 2 0.89).
  빼도 점수가 일관되게 달라지지 않아서(RF -0.003, HGB +0.008, 표준편차 안쪽, 10번) 유지한다.
- Clamping_Force, Spray_1/2의 높은 상관은 제품(Type)을 따라 움직이는 것이고, 제품 안에서는 낮다.
- 중복 행: Shot 번호가 같은 연속 행은 항상 2행이고 입력이 같다. 입력이 같은 행은 3,174개(그룹 3,663개).
  가중치(1/그룹 크기)·그룹당 1행 모두 all 대비 차이가 표준편차 안쪽이고 일관되게 낫지도 않아서
  (전체 모드 ±0.008 이내, 빠른 모드 HGB 최대 +0.013, 05번) 행은 그대로 둔다.
  group 번호는 검증 분할 전용이다 (특징에 넣지 않는다).
- 환경 열(공장 온도·습도, 공기압, 냉각수)은 그룹 분할에서만 점수를 올리고 시간순에서는 효과가 없다.
  RF·ExtraTrees·HGB 3종 모두 같은 방향이라 (06·09번) 시기를 알려 주는 표지(시계) 역할로 추정한다.
  유지하되, 모델 비교 때 drop_cols=ENV_COLS 버전도 같이 잰다.
- 결측 채우기는 팀이 정했다: 결측 덩어리 앞뒤 실측값 30개씩의 중앙값 (fill="neighbor", window=30).
  방식·window에 따른 점수 차이는 없었다 (RF 그룹 분할 F1 0.231~0.235, 04번).
- 스케일링은 전처리에서 하지 않는다. 필요한 모델(로지스틱·MLP)이 파이프라인 안에서 StandardScaler로 한다 (07·08번).
- Product_Type은 shot 4207까지 1, 4208부터 2다. 열은 그대로 쓰고 제품별로 나누지는 않았다.
- 참고_원본불량기록은 읽지 않는다 (PDF 규칙: 특징·라벨 재생성 금지).
"""

import numpy as np
import pandas as pd

# 파일 경로. 모두 상대경로라서 학생배포/ 폴더에서 실행해야 한다 (절대경로를 쓰면 실격).
DATA_PATH = "data/주조공정_입력.csv"
SPLIT_PATH = "분할표.csv"
LABEL_PATH = "정답_train.csv"

# 설명서 규칙: 식별 열은 특징으로 쓰지 않는다 (shot_id는 정렬·키 용도로만)
ID_COLS = ["shot_id", "id", "Shot"]
# 표시 열: 이상한 값이나 결측을 지우지 않고 1/0으로 표시해 두는 열 (add_flags에서 만든다)
FLAG_COLS = ["is_restart", "bad_biscuit", "long_cycle", "factory_missing"]
# 삭제 대상 후보로 보는 표시 열 (factory_missing은 센서 결측이라 제외)
OUTLIER_FLAGS = ["is_restart", "bad_biscuit", "long_cycle"]
# 환경 열(시기를 알려 주는 표지 역할로 추정). 빼 보고 점수를 비교할 때: load_data(drop_cols=ENV_COLS)
ENV_COLS = (
    "Factory_Temp",
    "Factory_Humidity",
    "factory_missing",
    "Air_Pressure",
    "Coolant_Temp",
    "Coolant_Pressure",
)


def load_raw():
    """헤더가 두 줄(1행 = 대분류)이라 header=1, 열 이름에 섞인 공백 제거."""
    df = pd.read_csv(DATA_PATH, header=1, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    # shot_id 순 = 생산 순서. 시간순 검증이 이 순서에 기대므로 항상 정렬해서 돌려줌.
    return df.sort_values("shot_id").reset_index(drop=True)


def feature_cols(df):
    """식별 열과 상수 열(고유값 1개, 8개)을 뺀 특징 목록."""
    # 상수 열(Min/Max 8개)은 모든 행이 같은 값이라 모델이 배울 것이 없어서 제외.
    # 공장 온도·습도의 Min/Max는 값도 실측과 맞지 않음.
    const = []
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 1:
            const.append(c)

    # 식별 열은 설명서 규칙상 특징으로 쓸 수 없음.
    feat = []
    for c in df.columns:
        if c not in ID_COLS + const:
            feat.append(c)

    return feat, const


def make_groups(df, feat):
    """바로 앞 샷과 입력값이 완전히 같으면 같은 그룹 번호 (쌍둥이 샷).

    삭제하지 않고 검증 분할(GroupKFold)에서만 쓴다. 특징에 넣으면 안 된다.
    결측은 -999로 바꿔 비교해야 NaN끼리도 같다고 본다.
    """
    # 식별 열(id 등)은 행마다 달라서 포함하면 같은 입력을 못 잡으므로 feat만 비교.
    v = df[feat].fillna(-999).values
    # v[1:]와 v[:-1]을 비교하면 '각 행이 바로 앞 행과 같은지'를 한 번에 알 수 있음.
    same = (v[1:] == v[:-1]).all(axis=1)
    # 앞 행과 달라지는 곳마다 새 그룹이 시작되므로 누적합이 그룹 번호가 됨 (첫 행은 0).
    return np.r_[0, np.cumsum(~same)]


def add_flags(df):
    """이상치는 지우지 않고 표시 열만 추가 (삭제 vs 유지는 점수로 비교해서 결정).

    is_restart: 속도·승압시간이 0 (설비 정지·재가동 추정, 4샷. Cylinder_Pressure도 107~108로 낮음)
    bad_biscuit: 비스킷 두께 0 (4샷)
    long_cycle: 사이클 시간 60초 초과 (11샷)
    factory_missing: 공장 온도·습도 결측 (센서 끊김 추정, 90샷)
    """
    out = df.copy()
    # 속도가 0이면 사출이 제대로 안 된 샷 (설비 정지 후 재가동으로 추정).
    out["is_restart"] = (out["High_Velocity"] == 0).astype(int)
    # 비스킷 두께 0은 물리적으로 있을 수 없는 값.
    out["bad_biscuit"] = (out["Biscuit_Thickness"] == 0).astype(int)
    # 보통 사이클은 20~36초인데 60초를 넘으면 설비 정지로 봄.
    out["long_cycle"] = (out["Cycle_Time"] > 60).astype(int)
    # 결측을 채우고 나면 어디가 비었었는지 알 수 없으므로 채우기 전에 표시.
    out["factory_missing"] = out["Factory_Temp"].isna().astype(int)
    return out


def fit_fill_values(train_df, cols):
    """결측을 채울 중앙값. 누수를 막으려고 train에서만 계산한다."""
    return train_df[cols].median()


def apply_fill(df, fill_values):
    """fit_fill_values로 구한 값으로 결측을 채운다 (train과 test에 같은 값을 쓴다)."""
    out = df.copy()
    out[fill_values.index] = out[fill_values.index].fillna(fill_values)
    return out


def interpolate_fill(df, cols):
    """결측을 shot 순서상 앞뒤 샷의 입력값으로 채운다 (선형 보간, 가장자리는 가까운 값).

    공장 온도·습도는 시기에 따라 달라서(습도 49 -> 69) 가까운 샷 값이 더 비슷할 수 있다.
    다만 결측 구간의 진짜 값을 모르므로 어느 쪽이 나은지는 점수로 비교해서 정한다.
    입력값만 쓰고 정답은 쓰지 않는다. df는 shot_id 순이어야 한다.
    """
    out = df.copy()
    out[cols] = out[cols].interpolate(limit_direction="both")
    return out


def neighbor_median_fill(df, cols, window=30):
    """결측 덩어리마다 앞뒤 실측값 window개씩(최대 2*window개)의 중앙값으로 채운다.

    결측 구간 주변 값이 전체 중앙값보다 가깝다 (공장 습도: 전체 중앙값 63.0, 주변 약 65~71).
    구간 끝에 튀는 값 2개(습도 54.0, 온도 27.4)가 있어도 중앙값이라 덜 흔들린다.
    입력값만 쓰고 정답은 쓰지 않는다. df는 shot_id 순이고 인덱스가 0부터 연속이어야 한다.
    """
    out = df.copy()
    for c in cols:
        s = df[
            c
        ]  # 채우기 전 원본 값만 참조 (앞 덩어리에서 채운 값을 뒤 덩어리가 쓰지 않게)
        miss = s.isna()
        # 연속해서 비어 있는 행들을 한 덩어리로 묶음 (값이 비었는지가 바뀔 때마다 새 번호).
        run = (miss != miss.shift()).cumsum()
        for idx in s[miss].groupby(run[miss]).groups.values():
            # 덩어리 바로 앞과 뒤에서 값이 있는 행을 window개씩 가져와 중앙값을 계산.
            before = s.loc[: idx[0] - 1].dropna().tail(window)
            after = s.loc[idx[-1] + 1 :].dropna().head(window)
            out.loc[idx, c] = pd.concat([before, after]).median()
    return out


def load_data(fill="neighbor", drop_cols=(), drop_flagged_train=False, window=30):
    """train/test 입력, 정답, 쌍둥이 그룹을 한 번에 돌려준다.

    반환: dict(X_train, y_train, groups_train, X_test, test_ids, feat)
    - 행 순서는 shot_id 순(= 생산 순서). 시간순 분할에 그대로 쓸 수 있다.

    옵션 (기본값은 처음 정한 방식 그대로. 바꾼 뒤에는 점수를 비교해서 근거를 남긴다)
    - fill: "neighbor"(결측 덩어리 앞뒤 실측값 window개씩의 중앙값, 팀이 정한 기본값),
      "median"(train 중앙값), "interp"(앞뒤 샷 선형 보간)
    - window: fill="neighbor"에서 앞뒤로 몇 개의 실측값을 볼지 (기본 30)
    - drop_cols: 특징에서 뺄 열 (표시 열 포함). 예: ("Cylinder_Pressure",) 또는 ENV_COLS
    - drop_flagged_train: True면 train에서 is_restart·bad_biscuit·long_cycle 행을 뺀다.
      test는 2,261행 전부 예측해야 해서 건드리지 않는다.
    """
    if fill not in ("neighbor", "median", "interp"):
        raise ValueError("fill은 'neighbor', 'median', 'interp' 중 하나")

    # 1) 원본을 읽고 쓸 특징을 정함 (식별 열·상수 열 제외).
    raw = load_raw()
    feat, _ = feature_cols(raw)
    # drop_cols에 적은 열 이름이 실제로 있는지 확인 (오타를 일찍 알려 주려고).
    unknown = []
    for c in drop_cols:
        if c not in feat + FLAG_COLS:
            unknown.append(c)
    if unknown:
        raise ValueError(f"특징에 없는 열: {unknown}")

    # 2) 쌍둥이 그룹 번호. 입력이 같은 행을 찾는 것이라서, 표시 열이나 채운 값이 섞이기 전의
    #    원본 값으로 계산. 행은 지우지 않고 번호만 붙여 두었다가 검증 분할에서 사용.
    groups = make_groups(raw, feat)

    # 3) 이상한 값은 지우지 않고 표시 열만 붙임 (삭제 여부는 점수로 비교해서 정한다).
    df = add_flags(raw)
    df["group"] = groups
    # 모델이 쓰는 열 = 특징 + 표시 열. group은 특징에 넣지 않음.
    cols = []
    for c in feat + FLAG_COLS:
        if c not in drop_cols:
            cols.append(c)

    # 4) 분할표로 train / public / private 구분. 정답은 train에만 존재.
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    labels = pd.read_csv(LABEL_PATH, encoding="utf-8-sig")
    df = df.merge(split, on="shot_id", how="left")

    # 5) 결측 채우기. 기본(neighbor)은 앞뒤 실측값의 중앙값이고,
    #    median은 train 통계만 써서 test 정보가 새어 들어가지 않게 함.
    is_train = df["split"] == "train"
    # 결측이 하나라도 있는 열만 채우기 대상.
    na_cols = []
    for c in feat:
        if df[c].isna().any():
            na_cols.append(c)
    if fill == "neighbor":
        df = neighbor_median_fill(df, na_cols, window)
    elif fill == "median":
        df = apply_fill(df, fit_fill_values(df[is_train], na_cols))
    else:
        df = interpolate_fill(df, na_cols)

    # 6) train에만 정답을 붙이고, train과 test(public + private)로 나눔.
    train = df[is_train].merge(labels, on="shot_id", how="left")
    test = df[~is_train]
    if drop_flagged_train:
        # 이상한 값을 가진 행은 train에서만 제외. test는 2,261행을 모두 예측해야 해서 그대로 둘 것.
        train = train[train[OUTLIER_FLAGS].sum(axis=1) == 0]
    # reset_index는 행 번호를 0부터 다시 매길 뿐이고, 행 순서(shot_id 순)는 그대로.
    return {
        "X_train": train[cols].reset_index(drop=True),
        "y_train": train["label"].reset_index(drop=True),
        "groups_train": train["group"].reset_index(drop=True),
        "X_test": test[cols].reset_index(drop=True),
        "test_ids": test["shot_id"].reset_index(drop=True),
        "feat": cols,
    }


# 직접 실행(python preprocess.py)했을 때만 돈다. 정리된 데이터는 저장하지 않고 요약만 출력한다.
# 다른 파일에서 `from preprocess import load_data`로 가져다 쓸 때는 실행되지 않는다.
if __name__ == "__main__":
    d = load_data()
    print("train", d["X_train"].shape, "/ test", d["X_test"].shape)
    print(
        "특징 수",
        len(d["feat"]),
        "/ 남은 결측",
        int(d["X_train"].isna().sum().sum()),
        int(d["X_test"].isna().sum().sum()),
    )
    print("쌍둥이 그룹 수(train)", d["groups_train"].nunique())
    print(d["y_train"].value_counts().to_string())
