"""보고서 그림 4·5를 만든다 (그림 1~3은 보고서_그림_만들기.py).

그림 4: 새 입력 행의 혼동행렬 + 위험 순위 검사 곡선 (최종 H안, 그룹 교차검증)
그림 5: 불량 종류별 구간 불량률 (원인이 아니라 상관임을 그림 제목에 적는다)

실행: python 보고서_그림2_만들기.py        (처음엔 H안 그룹 교차검증을 돌려 보고서_그림/H_그룹CV확률.npy 에 저장, 약 2분)
      python 보고서_그림2_만들기.py --force (확률을 다시 계산)
"""
import copy
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.model_selection import StratifiedGroupKFold

import main

warnings.filterwarnings("ignore")
rcParams["font.family"] = "Malgun Gothic"  # 맥이면 'AppleGothic'
rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "보고서_그림"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "H_그룹CV확률.npy"
BLUE, RED, GRAY = "#2b6cb0", "#d9480f", "#868e96"
main.RFET_N_TREES = 300
main.RF_N_TREES = 300
LABELS = main.LABELS
NORMAL = "정상"


def load():
    base = main.load_data(drop_cols=())
    nb = main.add_neighbor_features(copy.deepcopy(base))
    return base["X_train"].reset_index(drop=True), nb["X_train"], base["y_train"].reset_index(drop=True), base["groups_train"].reset_index(drop=True).values


def oof_proba(X65, y, g):
    """처음 보는 입력(그룹 교차검증)에서 샷마다 7개 라벨 확률을 얻는다"""
    if CACHE.exists() and "--force" not in sys.argv:
        return np.load(CACHE)
    main.MODEL_NAME = "rf_et_2stage"
    P = np.zeros((len(y), len(LABELS)))
    for k, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=main.SEED).split(X65, y, g)):
        model = main.build_model(y.iloc[tr]); model.fit(X65.iloc[tr], y.iloc[tr])
        cols = [LABELS.index(c) for c in model.classes_]
        P[np.ix_(va, cols)] = model.predict_proba(X65.iloc[va])
        print(f"조각 {k + 1}/5 완료", flush=True)
    np.save(CACHE, P)
    return P


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_fig4(P, y):
    yt = y.values
    pred = np.array(LABELS)[P.argmax(1)]
    cm = pd.crosstab(pd.Categorical(yt, LABELS), pd.Categorical(pred, LABELS), dropna=False).values
    short = [l.replace("불량_", "") for l in LABELS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
    a.set_xticks(range(7)); a.set_xticklabels(short, rotation=40, ha="right"); a.set_yticks(range(7)); a.set_yticklabels(short)
    a.set_xlabel("모델이 말한 라벨"); a.set_ylabel("실제 라벨")
    bad = yt != NORMAL
    caught = (pred[bad] != NORMAL).sum(); fa = ((yt == NORMAL) & (pred != NORMAL)).sum()
    a.set_title(f"처음 보는 입력에서 불량 {bad.sum()}건 중 {caught}건만 잡았다 ({caught / bad.sum():.0%})\n정상 {(yt == NORMAL).sum()}건 중 {fa}건을 불량이라 했다", fontsize=11, loc="left")
    # 위험 순위 곡선
    risk = 1 - P[:, LABELS.index(NORMAL)]
    o = np.argsort(-risk); cum = np.cumsum(bad[o]) / bad.sum(); x = np.arange(1, len(o) + 1) / len(o)
    b.plot(x * 100, cum * 100, color=BLUE, lw=2.2, label="위험 높은 순으로 검사")
    b.plot([0, 100], [0, 100], color=GRAY, ls="--", label="무작위로 검사")
    for q in (.1, .2, .3):
        v = cum[int(q * len(o)) - 1] * 100
        b.scatter([q * 100], [v], color=RED, zorder=3); b.annotate(f"{q:.0%} 검사 → 불량 {v:.0f}% 발견", (q * 100, v), textcoords="offset points", xytext=(14, -22), fontsize=9)
    b.set_xlabel("검사한 샷 비율 (%)"); b.set_ylabel("찾아낸 불량 비율 (%)"); b.legend(frameon=False, loc="lower right"); style(b)
    b.set_title("모델은 불량을 단정하지 못하지만\n위험 순위로는 무작위보다 2배 이상 잘 찾는다", fontsize=11, loc="left")
    fig.text(0.01, 0.01, "학습 데이터 그룹 교차검증(같은 입력이 학습에 없는 샷), 1안(최종 제출안). 리더보드 점수는 사용하지 않음.", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "그림4_혼동행렬_위험순위.png", dpi=160); plt.close(fig)


def plot_fig5(Xraw, y):
    t2 = (Xraw["Product_Type"] == 2).values
    specs = [("불량_Short_Shot", "High_Velocity", t2, "유형 2", [2.469, 2.515, 2.524, 2.537, 2.744]),
             ("불량_Blow_Hole", "High_Velocity", t2, "유형 2", [2.469, 2.511, 2.521, 2.528, 2.543, 2.744]),
             ("불량_Exfoliation", "Spray_Time", ~t2, "유형 1", [6.99, 7.9, 8.0, 8.1, 8.4, 10.8])]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (t, col, m, tn, edges) in zip(axs, specs):
        d = pd.DataFrame({"v": Xraw.loc[m, col].values, "y": (y[m] == t).astype(int).values})
        d["b"] = pd.cut(d.v, edges, include_lowest=True)
        r = d.groupby("b", observed=True).y.agg(["mean", "size"]); base = d.y.mean()
        lab = [(f"{iv.left:.3f}~{iv.right:.3f}" if col == "High_Velocity" else f"{iv.left:.1f}~{iv.right:.1f}") + f"\n(n={n})" for iv, n in zip(r.index, r["size"])]
        ax.bar(range(len(r)), r["mean"] * 100, color=[RED if v < base * 0.4 or v > base * 1.5 else BLUE for v in r["mean"]])
        ax.axhline(base * 100, color=GRAY, ls="--", lw=1)
        for i, v in enumerate(r["mean"]): ax.text(i, v * 100 + .2, f"{v:.1%}", ha="center", fontsize=9)
        ax.set_xticks(range(len(r))); ax.set_xticklabels(lab, fontsize=8); ax.set_xlabel(col); ax.set_ylabel(f"{t.replace('불량_', '')} 비율 (%)"); style(ax)
        ax.set_title(f"{tn}: {col} 구간별 {t.replace('불량_', '')}\n(점선 = {tn} 전체 평균 {base:.1%})", fontsize=10, loc="left")
    fig.suptitle("조건 구간에 따라 불량률이 크게 갈린다 — 단, 같은 데이터에서 고른 구간의 상관이며 원인은 시험 생산으로 확인해야 한다", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "그림5_구간별_불량률.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    Xraw, X65, y, g = load()
    P = oof_proba(X65, y, g)
    plot_fig4(P, y); plot_fig5(Xraw, y)
    print("저장:", OUT)
