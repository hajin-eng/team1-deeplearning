"""
중간 제출 후보 CSV 만들기
A·B·C·D 네 가지 RF 설정으로 각각 train 전체를 학습해, 제출 형식 검사를 통과한 CSV를 제출후보/ 폴더에 저장한다.
후보끼리 예측이 얼마나 같은지도 출력한다.

실행 방법: 학생배포 폴더에서  python 실험_제출후보.py   (약 10초)
저장 위치: 제출후보/제출_{후보 이름}.csv  (git에 올리지 않는다, .gitignore에 등록됨)
최종 제출 파일은 이 스크립트가 아니라 python main.py가 만든다 (main.py 기본값 = C안).
"""

from pathlib import Path

import main  # main.py를 불러온다 (main()은 실행되지 않는다)

OUT_DIR = Path(__file__).resolve().parent / "제출후보"

# 후보 이름 → (잎 최소, 열 비율, 가중치 강도). 나무 수는 main.py 값(300) 그대로
CANDIDATES = {
    "A_잎5_열sqrt_강도0.5": (5, "sqrt", 0.5),
    "B_잎5_열0.5_강도0.5": (5, 0.5, 0.5),
    "C_잎4_열0.5_강도0.5": (4, 0.5, 0.5),
    "D_잎5_열sqrt_강도0.75": (5, "sqrt", 0.75),
    "E_잎1_열sqrt_강도0.5": (1, "sqrt", 0.5),  # 쌍둥이를 가장 많이 외우는 표준 RF 설정 (그룹 CV 낮음)
}

main.MODEL_NAME = "rf"
data = main.load_data(drop_cols=main.DROP_COLS)
main.check_data(data)
X, y = data["X_train"], data["y_train"]

OUT_DIR.mkdir(exist_ok=True)
preds = {}
for name, (leaf, max_features, power) in CANDIDATES.items():
    main.RF_MIN_LEAF = leaf
    main.RF_MAX_FEATURES = max_features
    main.CLASS_WEIGHT_POWER = power
    main.set_seed(main.SEED)  # 후보마다 같은 시드에서 시작

    model = main.build_model(y)
    model.fit(X, y)
    sub = main.make_submission(model, data["X_test"], data["test_ids"])
    main.check_submission(sub)  # 행 수, 열 이름, shot_id 순서, 라벨 7종
    sub.to_csv(
        OUT_DIR / f"제출_{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    preds[name] = sub["label"].values
    print(f"저장: 제출후보/제출_{name}.csv")

base = preds["C_잎4_열0.5_강도0.5"]
print("\n=== C안과 같은 예측 비율 / 정상 예측 수 ===")
for name, p in preds.items():
    print(f"{name}: C와 일치 {(p == base).mean():.3f} | 정상 예측 {(p == '정상').sum()}개 ({(p == '정상').mean():.1%})")
