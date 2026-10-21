"""
모델 비교 실험
같은 데이터(데이터 파트 load_data)와 같은 교차검증 분할에서 모델별 점수를 비교한다.
분할 운을 줄이려고 시드 여러 개로 반복한다.

실행 방법: 학생배포 폴더에서  python 실험_모델비교.py
제출 파일은 만들지 않는다 (비교만 한다)
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import main  # main.py를 불러온다 (main()은 실행되지 않는다)

MODELS = ["logistic", "rf", "hgb"]  # 비교할 모델 (설정값은 main.py 그대로)
SEEDS = [42, 0, 1, 2]  # 분할·모델 시드

start = time.time()

# 데이터 준비는 모델·시드와 상관없으니 한 번만 한다
data = main.load_data(drop_cols=main.DROP_COLS)  # 데이터 파트 preprocess.py
main.check_data(data)
X = data["X_train"]
y_true = data["y_train"]

rows = []
for seed in SEEDS:
    main.SEED = seed  # 분할(split_validation)과 모델(random_state)이 이 값을 쓴다
    main.set_seed(seed)
    folds = main.split_validation(y_true, data["groups_train"])  # 시드마다 분할이 달라진다
    for model_name in MODELS:
        main.MODEL_NAME = model_name
        print(f"\n===== 시드 {seed}, 모델 {model_name} =====")
        oof_pred, fold_scores = main.cross_validate(X, y_true, folds)

        is_normal = y_true == "정상"
        rows.append(
            {
                "시드": seed,
                "모델": model_name,
                "합쳐서 macro F1": f1_score(
                    y_true, oof_pred, labels=main.LABELS, average="macro", zero_division=0
                ),
                "정상 recall": (oof_pred[is_normal] == "정상").mean(),
            }
        )

result = pd.DataFrame(rows)
print("\n===== 합쳐서 macro F1 (행 = 모델, 열 = 시드) =====")
table = result.pivot(index="모델", columns="시드", values="합쳐서 macro F1")[SEEDS]
table["평균"] = table[SEEDS].mean(axis=1)
table["표준편차"] = table[SEEDS].std(axis=1, ddof=0)
print(table.reindex(MODELS).round(4).to_string())

print("\n===== 정상 recall 평균 =====")
print(result.groupby("모델")["정상 recall"].mean().reindex(MODELS).round(3).to_string())
print(f"\n(설정: RF 잎 최소 {main.RF_MIN_LEAF}, 강도 {main.CLASS_WEIGHT_POWER} / HGB 설정 B, balanced)")
print(f"실행 시간: {time.time() - start:.1f}초")
