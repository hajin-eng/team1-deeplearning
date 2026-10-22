"""
중간 제출 후보 CSV 만들기
후보별 설정으로 각각 train 전체를 학습해, 제출 형식 검사를 통과한 CSV를 제출후보/ 폴더에 저장한다.
후보끼리 예측이 얼마나 같은지도 출력한다.

실행 방법: 학생배포 폴더에서  python 실험_제출후보.py   (약 3분)
저장 위치: 제출후보/제출_{후보 이름}.csv  (git에 올리지 않는다, .gitignore에 등록됨)
최종 제출 파일은 이 스크립트가 아니라 python main.py가 만든다 (main.py 기본값 = 제출하기로 확정한 모델).
"""

from pathlib import Path

import main  # main.py를 불러온다 (main()은 실행되지 않는다)

OUT_DIR = Path(__file__).resolve().parent / "제출후보"

# 후보 이름 → main.py 설정값. 쓰지 않는 설정은 main.py 기본값 그대로
CANDIDATES = {
    # E: RF 한 개 (중간 제출 2회차, public 0.5853)
    "E_RF_잎1_열sqrt_강도0.5": dict(MODEL_NAME="rf", RF_MIN_LEAF=1, RF_MAX_FEATURES="sqrt", CLASS_WEIGHT_POWER=0.5),
    # F: RF와 ExtraTrees(각 900그루)의 예측 확률 평균
    "F_RF+ET_잎1_열sqrt_강도0.5": dict(MODEL_NAME="rf_et", RF_MIN_LEAF=1, RF_MAX_FEATURES="sqrt", CLASS_WEIGHT_POWER=0.5),
    # G: F와 '두 단계 모델'(불량인가 → 불량 6종 중 무엇인가)의 예측 확률을 절반씩 섞음
    "G_F+두단계_잎1_열sqrt_강도0.5": dict(MODEL_NAME="rf_et_2stage", RF_MIN_LEAF=1, RF_MAX_FEATURES="sqrt", CLASS_WEIGHT_POWER=0.5),
}

data = main.load_data(drop_cols=main.DROP_COLS)
main.check_data(data)
X, y = data["X_train"], data["y_train"]

OUT_DIR.mkdir(exist_ok=True)
preds = {}
for name, cfg in CANDIDATES.items():
    for key, value in cfg.items():
        setattr(main, key, value)
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

base = preds["E_RF_잎1_열sqrt_강도0.5"]
print("\n=== E안과 같은 예측 비율 / 정상 예측 수 ===")
for name, p in preds.items():
    print(f"{name}: E와 일치 {(p == base).mean():.3f} | 정상 예측 {(p == '정상').sum()}개 ({(p == '정상').mean():.1%})")
