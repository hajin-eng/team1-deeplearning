"""
랜덤포레스트 실험
main.py의 RF_MIN_LEAF(잎 최소 샷 수)와 CLASS_WEIGHT_POWER(가중치 강도)를
여러 값으로 바꿔 끼우며 교차검증 점수를 비교한다.

실행 방법: 학생배포 폴더에서  python 실험_RF.py
제출 파일은 만들지 않는다 (비교만 한다)
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import main  # main.py를 불러온다 (main()은 실행되지 않는다)

MIN_LEAVES = [1, 5, 10, 20, 40]  # 비교할 잎 최소 샷 수
POWERS = [0.25, 0.5, 0.75, 1]  # 비교할 강도

start = time.time()
main.set_seed(main.SEED)
main.MODEL_NAME = "rf"  # 이 실험은 RF만 본다

# 데이터 준비는 설정과 상관없으니 한 번만 한다
df = main.load_data()
X = main.preprocess(df)
groups = main.make_twin_groups(df)
folds = main.split_validation(df, groups)
train_idx = df.index[df["split"] == "train"]
y_true = df.loc[train_idx, main.TARGET_COL]

rows = []
for min_leaf in MIN_LEAVES:
    for power in POWERS:
        main.RF_MIN_LEAF = min_leaf  # 다이얼 1
        main.CLASS_WEIGHT_POWER = power  # 다이얼 2
        print(f"\n===== 잎 최소 {min_leaf}, 강도 {power} =====")
        oof_pred, fold_scores = main.cross_validate(df, X, folds)
        pred = oof_pred.loc[train_idx]

        is_normal = y_true == "정상"
        rows.append(
            {
                "잎 최소": min_leaf,
                "강도": power,
                "합쳐서 macro F1": f1_score(
                    y_true, pred, labels=main.LABELS, average="macro", zero_division=0
                ),
                "조각 평균": np.mean(fold_scores),
                "조각 표준편차": np.std(fold_scores),
                "정상 recall": (pred[is_normal] == "정상").mean(),
            }
        )

result = pd.DataFrame(rows)
print("\n===== 설정별 결과 =====")
print(result.to_string(index=False, float_format="%.4f"))

print("\n===== 합쳐서 macro F1 한눈에 보기 (행 = 잎 최소, 열 = 강도) =====")
print(result.pivot(index="잎 최소", columns="강도", values="합쳐서 macro F1").round(4))
print(f"실행 시간: {time.time() - start:.1f}초")
