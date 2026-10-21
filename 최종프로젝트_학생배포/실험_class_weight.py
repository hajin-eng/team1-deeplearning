"""
class_weight 강도 실험
main.py의 CLASS_WEIGHT_POWER를 여러 값으로 바꿔 끼우며 교차검증 점수를 비교한다.

실행 방법: 학생배포 폴더에서  python 실험_class_weight.py
제출 파일은 만들지 않는다 (비교만 한다)
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import main  # main.py를 불러온다 (main()은 실행되지 않는다)

POWERS = [0, 0.25, 0.5, 0.6, 0.75, 1]  # 비교할 강도 목록

start = time.time()
main.set_seed(main.SEED)

# 데이터 준비는 강도와 상관없으니 한 번만 한다
data = main.load_data(drop_cols=main.DROP_COLS)  # 데이터 파트 preprocess.py
main.check_data(data)
X = data["X_train"]
y_true = data["y_train"]
folds = main.split_validation(y_true, data["groups_train"])

rows = []
for power in POWERS:
    main.CLASS_WEIGHT_POWER = power  # 다이얼 돌리기
    print(f"\n===== 강도 {power} =====")
    oof_pred, fold_scores = main.cross_validate(X, y_true, folds)
    pred = oof_pred

    is_normal = y_true == "정상"
    rows.append(
        {
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
print("\n===== 강도별 결과 =====")
print(result.to_string(index=False, float_format="%.4f"))
print(f"실행 시간: {time.time() - start:.1f}초")
