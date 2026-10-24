"""
1조 - 보고서 그림 만들기 (보고서 5·6번: 학습곡선 2종, 분할 비교)

실행: 학생배포 폴더에서  python 보고서_그림_만들기.py
      (계산은 처음 한 번만, 숫자는 보고서_그림/결과수치.json에 저장. 그림만 다시 그리려면 --plot)
결과: 보고서_그림/ 아래 PNG 3장 + 결과수치.json

- 그림 1  epoch 곡선        MLP의 train / validation 손실 (어디서부터 외우기 시작하나)
- 그림 2  데이터양 곡선     최종 모델(H안)을 학습 데이터 20·40·60·80·100%로 학습했을 때 검증 점수
- 그림 3  분할 비교         무작위 80/20 · 쌍둥이 묶음 80/20 · 시간순(앞 80% 학습 → 뒤 20% 검증)

속도를 위해 트리 수는 300그루로 줄여서 잰다 (최종 제출은 900그루, 점수 차이는 0.001 이내).
"""

import copy
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

import main

warnings.filterwarnings("ignore")
rcParams["font.family"] = "Malgun Gothic"  # 맥이면 'AppleGothic'
rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "보고서_그림"
OUT.mkdir(exist_ok=True)
JSON_PATH = OUT / "결과수치.json"
BLUE, RED, GRAY = "#2b6cb0", "#d9480f", "#868e96"  # 색은 차분한 파랑 + 강조 주황 + 회색

main.RFET_N_TREES = 300
main.RF_N_TREES = 300
LABELS = main.LABELS
NORMAL = "정상"


# ------------------------------------------------------------
# 데이터와 점수
# ------------------------------------------------------------
def load():
    base = main.load_data(drop_cols=())
    with_neighbors = main.add_neighbor_features(copy.deepcopy(base))
    y = base["y_train"].reset_index(drop=True)
    g = base["groups_train"].reset_index(drop=True).values
    return base["X_train"].reset_index(drop=True), with_neighbors["X_train"], y, g


def scores(y_true, y_pred, weight=None):
    """macro F1, 불량 탐지율(불량 중 '정상이 아니라고' 한 비율), 정확도"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    w = np.ones(len(y_true)) if weight is None else np.asarray(weight)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0, sample_weight=w)
    bad = y_true != NORMAL
    det = (w[bad] * (y_pred[bad] != NORMAL)).sum() / w[bad].sum()
    acc = (w * (y_true == y_pred)).sum() / w.sum()
    return {"macro_f1": float(f1), "탐지율": float(det), "정확도": float(acc)}


def fit_predict(model_name, X_fit, y_fit, X_val):
    main.MODEL_NAME = model_name
    model = main.build_model(y_fit)
    model.fit(X_fit, y_fit)
    return model.predict(X_val)


# ------------------------------------------------------------
# MLP (설계 카드: 입력→128→64→7, ReLU, 드롭아웃 0.2, 클래스 가중 손실, Adam 1e-3, weight decay 1e-4)
# ------------------------------------------------------------
def make_mlp_data(X_fit, y_fit, X_val, classes):
    lo, hi = np.quantile(X_fit, [0.005, 0.995], axis=0)
    clipped = np.clip(X_fit, lo, hi)
    mu, sd = clipped.mean(0), clipped.std(0) + 1e-8
    prep = lambda A: (np.clip(A, lo, hi) - mu) / sd
    return prep(X_fit), np.searchsorted(classes, y_fit), prep(X_val)


def build_net(n_in, n_out):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, n_out),
    )


def mlp_epoch_curve(X, y, g, seed, epochs=60):
    """train 손실과 validation 손실을 epoch마다 잰다 (validation = 쌍둥이를 묶은 20%)"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = np.unique(y)
    yv = np.asarray(y)
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, yv, g))
    Xa, ya, Xv = make_mlp_data(X.values[tr], yv[tr], X.values[va], classes)
    yva = np.searchsorted(classes, yv[va])
    Xa, Xv = torch.tensor(Xa, dtype=torch.float32), torch.tensor(Xv, dtype=torch.float32)
    ya_t, yva_t = torch.tensor(ya), torch.tensor(yva)
    w = compute_class_weight("balanced", classes=np.arange(len(classes)), y=ya) ** 0.5
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    net = build_net(Xa.shape[1], len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loss, val_loss = [], []
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 128):
            b = perm[i : i + 128]
            opt.zero_grad()
            lossf(net(Xa[b]), ya_t[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            train_loss.append(lossf(net(Xa), ya_t).item())
            val_loss.append(lossf(net(Xv), yva_t).item())
    return train_loss, val_loss


def mlp_predict(X_fit, y_fit, X_val, seed):
    """조기 중단(안쪽 10% 검증 손실이 10 epoch 안 좋아지면 멈춤)을 쓰는 MLP 예측"""
    import copy as cp

    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = np.unique(y_fit)
    yf = np.asarray(y_fit)
    Xs, ys, Xv = make_mlp_data(X_fit, yf, X_val, classes)
    inner = StratifiedKFold(10, shuffle=True, random_state=seed)
    itr, iva = next(inner.split(Xs, ys))
    w = compute_class_weight("balanced", classes=np.arange(len(classes)), y=ys) ** 0.5
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    Xa, ya = torch.tensor(Xs[itr], dtype=torch.float32), torch.tensor(ys[itr])
    Xi, yi = torch.tensor(Xs[iva], dtype=torch.float32), torch.tensor(ys[iva])
    net = build_net(Xs.shape[1], len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = 1e9, None, 0
    for _ in range(80):
        net.train()
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 128):
            b = perm[i : i + 128]
            opt.zero_grad()
            lossf(net(Xa[b]), ya[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xi), yi).item()
        if vl < best - 1e-4:
            best, state, bad = vl, cp.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xv, dtype=torch.float32)).argmax(1).numpy()
    return classes[pred]


# ------------------------------------------------------------
# 계산 1: epoch 곡선
# ------------------------------------------------------------
def compute_epoch(X25, y, g):
    runs = [mlp_epoch_curve(X25, y, g, seed) for seed in (42, 0, 1)]
    return {"train_loss": np.mean([r[0] for r in runs], 0).tolist(), "val_loss": np.mean([r[1] for r in runs], 0).tolist(),
            "val_loss_seeds": [r[1] for r in runs]}


# ------------------------------------------------------------
# 계산 2: 데이터양 곡선 (H안)
# ------------------------------------------------------------
def compute_volume(X65, y, g):
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    main.NEIGHBOR_FEATURES = True
    out = {str(f): {"무작위 CV": [], "쌍둥이 묶음 CV": [], "리더보드풍": []} for f in fractions}
    yv = np.asarray(y)
    for seed in (42, 0):
        rng = np.random.default_rng(seed)
        for mode in ("무작위 CV", "쌍둥이 묶음 CV"):
            spl = (StratifiedKFold(5, shuffle=True, random_state=seed).split(X65, yv) if mode == "무작위 CV"
                   else StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X65, yv, g))
            folds = list(spl)
            for f in fractions:
                pred = np.empty(len(yv), dtype=object)
                twin = np.zeros(len(yv), bool)
                for fit, val in folds:
                    if f < 1.0:
                        if mode == "무작위 CV":
                            fit = train_test_split(fit, train_size=f, stratify=yv[fit], random_state=int(rng.integers(1e6)))[0]
                        else:  # 쌍둥이 묶음은 쪼개지 않고 묶음째로 줄인다
                            ug = np.unique(g[fit])
                            keep = rng.choice(ug, size=int(len(ug) * f), replace=False)
                            fit = fit[np.isin(g[fit], keep)]
                    twin[val] = np.isin(g[val], g[fit])
                    pred[val] = fit_predict("rf_et_2stage", X65.iloc[fit], y.iloc[fit], X65.iloc[val])
                out[str(f)][mode].append(scores(yv, pred)["macro_f1"])
                if mode == "무작위 CV":  # 쌍둥이가 학습에 있는 행 : 없는 행 = 58.4 : 41.6 으로 맞춰 다시 채점
                    wgt = np.where(twin, 0.584 / twin.sum(), 0.416 / (~twin).sum()) * len(yv)
                    out[str(f)]["리더보드풍"].append(scores(yv, pred, wgt)["macro_f1"])
        print("  데이터양 곡선 시드", seed, "완료", flush=True)
    return {f: {k: float(np.mean(v)) for k, v in d.items()} for f, d in out.items()}


# ------------------------------------------------------------
# 계산 3: 분할 비교
# ------------------------------------------------------------
def compute_split(X25, X65, y, g):
    yv = np.asarray(y)
    n = len(yv)
    cut = int(n * 0.8)  # train은 shot_id 순서로 들어 있으므로 앞 80% = 이른 시기
    specs = {"로지스틱": ("logistic", X25, False), "RF(E안)": ("rf", X25, False), "H안(최종)": ("rf_et_2stage", X65, True), "MLP": ("mlp", X25, False)}
    res = {name: {} for name in specs}
    info = {"시간순 검증 20% 불량률": float((yv[cut:] != NORMAL).mean()), "시간순 학습 80% 불량률": float((yv[:cut] != NORMAL).mean()),
            "전체 불량률": float((yv != NORMAL).mean())}
    for name, (model_name, X, nb) in specs.items():
        main.NEIGHBOR_FEATURES = nb
        rnd, grp = [], []
        for seed in (42, 0, 1, 2, 3):
            tr, va = train_test_split(np.arange(n), test_size=0.2, stratify=yv, random_state=seed)
            sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            gtr, gva = next(sgkf.split(X, yv, g))
            for store, (a, b) in ((rnd, (tr, va)), (grp, (gtr, gva))):
                p = (mlp_predict(X.values[a], yv[a], X.values[b], seed) if model_name == "mlp"
                     else fit_predict(model_name, X.iloc[a], y.iloc[a], X.iloc[b]))
                store.append(scores(yv[b], p))
        a, b = np.arange(cut), np.arange(cut, n)
        p = (mlp_predict(X.values[a], yv[a], X.values[b], 42) if model_name == "mlp"
             else fit_predict(model_name, X.iloc[a], y.iloc[a], X.iloc[b]))
        tm = scores(yv[b], p)
        agg = lambda lst, k: [float(np.mean([r[k] for r in lst])), float(np.std([r[k] for r in lst]))]
        res[name] = {"무작위 80/20": {k: agg(rnd, k) for k in rnd[0]}, "쌍둥이 묶음 80/20": {k: agg(grp, k) for k in grp[0]},
                     "시간순 80/20": {k: [v, 0.0] for k, v in tm.items()}}
        print("  분할 비교", name, "완료", flush=True)
    return {"결과": res, "정보": info}


# ------------------------------------------------------------
# 그림
# ------------------------------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_epoch(d):
    tl, vl = np.array(d["train_loss"]), np.array(d["val_loss"])
    best = int(np.argmin(vl)) + 1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, len(tl) + 1), tl, color=BLUE, lw=2, label="학습 데이터 손실")
    ax.plot(range(1, len(vl) + 1), vl, color=RED, lw=2, label="검증 데이터 손실 (처음 보는 입력)")
    ax.axvline(best, color=GRAY, ls="--")
    seeds_best = ", ".join(str(int(np.argmin(s)) + 1) for s in d["val_loss_seeds"])
    ax.annotate(f"평균 곡선은 epoch {best}에서 가장 낮음\n(시드 3개의 최저점: epoch {seeds_best})", (best, vl[best - 1]), (best + 8, 1.25), arrowprops=dict(arrowstyle="->", color=GRAY), fontsize=10)
    ax.set_xlabel("epoch (전체 학습 데이터를 한 바퀴 본 횟수)")
    ax.set_ylabel("손실 (낮을수록 잘 맞춤)")
    ax.set_title(f"epoch {best}부터 검증 손실이 올라갑니다 = 외우기 시작한 지점", fontsize=12, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "그림1_epoch곡선.png", dpi=170)
    plt.close(fig)
    return best


def plot_volume(d):
    fr = sorted(d, key=float)
    xs = [int(float(f) * 100) for f in fr]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [("무작위 CV", "무작위로 나눈 검증 (같은 입력이 섞여 점수가 부풀음)", GRAY, ":"), ("리더보드풍", "리더보드와 비슷한 구조로 가중", BLUE, "-"), ("쌍둥이 묶음 CV", "처음 보는 입력만 검증 (가장 정직)", RED, "-")]
    for key, label, color, ls in series:
        ys = [d[f][key] for f in fr]
        ax.plot(xs, ys, color=color, ls=ls, lw=2, marker="o", label=label)
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), (6, 0), textcoords="offset points", fontsize=10, color=color, va="center")
    ax.set_xticks(xs)
    ax.set_xlabel("학습에 쓴 데이터의 비율 (%)")
    ax.set_ylabel("검증 macro F1")
    g_gain = d[fr[-1]]["쌍둥이 묶음 CV"] - d[fr[0]]["쌍둥이 묶음 CV"]
    l_gain = d[fr[-1]]["리더보드풍"] - d[fr[0]]["리더보드풍"]
    ax.set_title(f"데이터를 20%→100%로 늘리면: 처음 보는 입력 {g_gain:+.2f}, 리더보드풍 {l_gain:+.2f}", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_xlim(15, 108)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "그림2_데이터양곡선.png", dpi=170)
    plt.close(fig)
    return g_gain, l_gain


def plot_split(d):
    res = d["결과"]
    models = list(res)
    modes = ["무작위 80/20", "쌍둥이 묶음 80/20", "시간순 80/20"]
    colors = [GRAY, BLUE, RED]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    w = 0.26
    for j, (mode, c) in enumerate(zip(modes, colors)):
        vals = [res[m][mode]["macro_f1"][0] for m in models]
        bars = ax.bar(np.arange(len(models)) + (j - 1) * w, vals, w, color=c, label=mode)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("macro F1")
    h = res["H안(최종)"]
    ax.set_title(f"H안: 무작위 {h[modes[0]]['macro_f1'][0]:.2f} → 시간순 {h[modes[2]]['macro_f1'][0]:.2f}. 현장 배포 성능은 시간순 쪽에 가깝습니다", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(0.75, ax.get_ylim()[1]))
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "그림3_분할비교.png", dpi=170)
    plt.close(fig)


def plot_all(R):
    out = {}
    out["epoch"] = plot_epoch(R["epoch"])
    out["데이터양"] = plot_volume(R["volume"])
    plot_split(R["split"])
    return out


def main_run():
    R = json.loads(JSON_PATH.read_text(encoding="utf-8")) if JSON_PATH.exists() else {}
    if "--plot" not in sys.argv:
        X25, X65, y, g = load()
        t0 = time.time()
        for key, fn in (("epoch", lambda: compute_epoch(X25, y, g)), ("volume", lambda: compute_volume(X65, y, g)), ("split", lambda: compute_split(X25, X65, y, g))):
            if key in R and "--force" not in sys.argv:
                continue
            print(f"[{key}] 계산 시작", flush=True)
            R[key] = fn()
            JSON_PATH.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{key}] 완료 ({time.time() - t0:.0f}초)", flush=True)
    print("그림 저장:", plot_all(R))


if __name__ == "__main__":
    main_run()
