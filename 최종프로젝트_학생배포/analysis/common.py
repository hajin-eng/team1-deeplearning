"""분석 스크립트 공통 도구.

모든 분석 스크립트는 프로젝트 폴더(학생배포/)에서 실행한다고 가정한다.
    python analysis/04_fill_window.py
    python analysis/08_model_comparison.py --quick    (시드 1개로 빠르게 확인)

각 스크립트의 맨 위에서 이 파일을 가장 먼저 import 해야 한다.
(경로, 한글 출력, 경고 설정이 import 하는 순간 적용되기 때문)

이 파일이 하는 일
  1) 환경 설정: 한글 출력, import 경로, 경고 숨김
  2) 점수 계산: macro F1 (두 가지 방식)
  3) 검증 분할: 쌍둥이 그룹 / 무작위 / 시간순
  4) 위를 묶은 한 줄짜리 평가 함수

왜 따로 뺐나: 실험마다 같은 분할·같은 점수 계산을 써야 숫자를 서로 비교할 수 있다.
복사해서 쓰다 보면 스크립트마다 조금씩 달라지므로 한 곳에서 관리한다.
"""


# 코드 메모
# 설정: QUICK(--quick이면 True), SEEDS(빠른 모드 [0], 전체 [0, 1, 2]), N_FOLDS(5)
# announce: 스크립트 시작 때 예상 소요 시간을 한 줄로 출력
# macro_f1: 클래스별 F1 평균 / macro_f1_present: 검증 구간에 있는 클래스만 평균
# fmt: 점수 배열을 '평균 (±표준편차)' 문자열로 변환
# group_folds: 쌍둥이 그룹 5-fold 분할 / random_folds: 무작위 5-fold 분할 / time_split: 앞 80% 학습, 뒤 20% 검증 분할
# group_cv_scores, random_cv_scores, time_split_score: 위 분할로 모델을 학습하고 macro F1을 돌려주는 평가 함수

import sys
import warnings

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# [환경 설정] 윈도우 터미널(cp949)에서 한글이 깨지지 않게 출력 인코딩을 UTF-8로 바꿈.
sys.stdout.reconfigure(encoding="utf-8")

# [환경 설정] 실행 위치(= 프로젝트 폴더)에 있는 preprocess.py를 import 할 수 있게 함.
# analysis/ 안의 스크립트를 실행하면 기본 경로는 analysis/ 이라 그대로는 preprocess를 못 찾음.
sys.path.insert(0, ".")

# 수렴 경고처럼 결과와 상관없는 경고는 숨겨서 출력을 읽기 쉽게 함.
warnings.filterwarnings("ignore")

# [빠른 모드] --quick을 붙이면 시드를 1개만 써서 몇 배 빠르게 돌림.
# 코드가 오류 없이 도는지 확인할 때 쓰고, 숫자는 덜 안정적임.
QUICK = "--quick" in sys.argv
if QUICK:
    SEEDS = [0]
else:
    SEEDS = [0, 1, 2]

N_FOLDS = 5


def announce(title, full_time, quick_time):
    """스크립트가 시작할 때 무엇을 하는지, 얼마나 걸리는지 한 줄로 알려 준다.

    오래 걸리는 스크립트는 한참 조용해서 멈춘 것처럼 보이므로, 먼저 예상 시간을 보여 준다.
    시간은 이 컴퓨터에서 잰 대략값이다.
    """

    def about(t):
        # '몇 초'처럼 이미 어림하는 말에는 '약'을 또 붙이지 않음.
        if t.startswith(("몇", "약")):
            return t
        return f"약 {t}"

    if full_time == quick_time:
        # --quick을 붙여도 빨라지지 않는 스크립트(시드를 쓰지 않는 점검)
        print(f"[{title}] {about(full_time)}", flush=True)
    elif QUICK:
        print(f"[{title}] 빠른 모드(시드 1개), {about(quick_time)}", flush=True)
    else:
        print(
            f"[{title}] 전체 모드, {about(full_time)} (빨리 보려면 --quick, {about(quick_time)})",
            flush=True,
        )


# ---------------------------------------------------------------
# 점수
# ---------------------------------------------------------------
def macro_f1(y_true, y_pred):
    """macro F1: 클래스별 F1의 단순 평균.

    정답이나 예측에 한 번이라도 나온 클래스는 모두 평균에 들어간다.
    """
    return f1_score(y_true, y_pred, average="macro")


def macro_f1_present(y_true, y_pred):
    """검증 구간의 정답에 실제로 있는 클래스만 평균한 macro F1.

    시간순 분할은 검증 구간에 없는 불량 유형이 생긴다.
    모델이 그 유형을 예측하면 F1=0이 평균에 들어가 점수가 불공정하게 깎이는데, 이를 막는다.
    """
    # labels=np.unique(y_true): 평균에 넣을 클래스를 '정답에 있는 것'으로 제한함.
    return f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))


def fmt(scores):
    """점수 배열을 '평균 (±표준편차)' 문자열로."""
    # 표준편차를 같이 보여 주는 이유: 점수 차이가 표준편차보다 작으면 노이즈로 볼 수 있음.
    return f"{np.mean(scores):.3f} (±{np.std(scores):.3f})"


# ---------------------------------------------------------------
# 검증 분할
# ---------------------------------------------------------------
def group_folds(y, groups, seeds=None):
    """쌍둥이 그룹 단위 5-fold. 같은 그룹은 학습과 검증에 갈라지지 않는다.

    시드마다 5번씩 (seed, 학습 인덱스, 검증 인덱스)를 돌려준다.
    """
    if seeds is None:
        seeds = SEEDS
    for seed in seeds:
        # 목적: 입력이 같은 행(쌍둥이)이 학습과 검증에 나뉘면 모델이 답을 외워서 맞히게 됨.
        #       그룹 단위로 나누면 이런 누수 없이 '처음 보는 입력'에 대한 성능을 잴 수 있음.
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        # 분할은 y와 groups만으로 정해지므로 X 자리에는 길이만 맞는 빈 배열을 넣음.
        for train_idx, valid_idx in cv.split(np.zeros(len(y)), y, groups):
            yield seed, train_idx, valid_idx


def random_folds(y, seeds=None):
    """무작위 5-fold. 쌍둥이가 학습과 검증에 갈라질 수 있어 점수가 부풀려진다."""
    if seeds is None:
        seeds = SEEDS
    for seed in seeds:
        # 목적: 그룹 분할과 비교해서 '쌍둥이 때문에 점수가 얼마나 부풀려지는지' 보여 주기 위한 대조군.
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, valid_idx in cv.split(np.zeros(len(y)), y):
            yield seed, train_idx, valid_idx


def time_split(n, ratio=0.8):
    """shot 순서(= 행 순서)의 앞 80%로 학습, 뒤 20%로 검증."""
    # 목적: 실제 현장처럼 '과거로 학습해 미래를 맞히는' 상황을 흉내 냄.
    # 전처리에서 행을 shot_id 순으로 정렬해 두었으므로 앞에서부터 자르면 시간순이 됨.
    cut = int(n * ratio)
    return np.arange(cut), np.arange(cut, n)


# ---------------------------------------------------------------
# 한 번에 점수 내기
# ---------------------------------------------------------------
def group_cv_scores(make_model, X, y, groups, seeds=None):
    """쌍둥이 그룹 분할로 시드 x 5-fold macro F1 배열을 돌려준다."""
    # make_model은 '모델을 새로 만들어 주는 함수'임.
    # 폴드마다 새 모델로 학습해야 앞 폴드의 학습이 다음 폴드에 새어 들어가지 않음.
    scores = []
    for _, train_idx, valid_idx in group_folds(y, groups, seeds):
        model = make_model().fit(X[train_idx], y[train_idx])
        scores.append(macro_f1(y[valid_idx], model.predict(X[valid_idx])))
    return np.array(scores)


def random_cv_scores(make_model, X, y, seeds=None):
    """무작위 분할로 시드 x 5-fold macro F1 배열을 돌려준다."""
    scores = []
    for _, train_idx, valid_idx in random_folds(y, seeds):
        model = make_model().fit(X[train_idx], y[train_idx])
        scores.append(macro_f1(y[valid_idx], model.predict(X[valid_idx])))
    return np.array(scores)


def time_split_score(make_model, X, y, present_only=True):
    """시간순 분할(앞 80% 학습 / 뒤 20% 검증) macro F1 한 번.

    present_only=True : 검증 구간에 있는 클래스만 평균 (권장)
    present_only=False: 처음 실험과 같은 값을 재현할 때 쓰는 방식
    """
    train_idx, valid_idx = time_split(len(y))
    model = make_model().fit(X[train_idx], y[train_idx])
    pred = model.predict(X[valid_idx])
    if present_only:
        score_fn = macro_f1_present
    else:
        score_fn = macro_f1
    return score_fn(y[valid_idx], pred)
