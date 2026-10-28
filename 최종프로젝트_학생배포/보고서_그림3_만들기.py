"""그림 4: 새 입력(쌍둥이 묶음 5분할) 기준 정상/불량 2x2 표, 1안(H안)과 2안(D안) 비교. 입력: 시드별 검증 확률 캐시(blockcv2.py)"""
import pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams
rcParams["font.family"] = "Malgun Gothic"; rcParams["axes.unicode_minus"] = False
R, y = pickle.load(open(r"C:/Users/swrkd/AppData/Local/Temp/claude/c--Users-swrkd-Desktop-team1-deeplearning-main/7a761312-f706-4c12-ac6e-669e680f4db3/scratchpad/cache_block_p.pkl", "rb"))
bad = y != "정상"
GOOD, MISS, FALSE, OKN = "#2f9e6e", "#d9480f", "#e8a33d", "#dde5ee"   # 잡음 / 놓침 / 헛경보 / 정상 통과
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, (k, name) in zip(axes, [("H", "1안 (최종 제출안)"), ("D", "2안 (현장성을 높인 또 다른 안)")]):
    c = np.zeros((2, 2))
    for seed in (7, 123, 2026):
        P, cl = R[(k, "그룹", seed)]; alarm = np.asarray(cl)[P.argmax(1)] != "정상"
        c += [[(bad & alarm).sum(), (bad & ~alarm).sum()], [(~bad & alarm).sum(), (~bad & ~alarm).sum()]]
    c = np.round(c / 3).astype(int); rate = c / c.sum(1, keepdims=True)
    cells = [[("불량을 잡음", GOOD, "white"), ("불량을 놓침", MISS, "white")], [("헛경보", FALSE, "black"), ("정상 통과", OKN, "#333")]]
    for i in range(2):
        for j in range(2):
            lab, col, tc = cells[i][j]
            ax.add_patch(FancyBboxPatch((j + 0.03, 1 - i + 0.03), 0.94, 0.94, boxstyle="round,pad=0,rounding_size=0.06", fc=col, ec="none"))
            ax.text(j + .5, 1 - i + .76, lab, ha="center", va="center", fontsize=12, color=tc, fontweight="bold")
            ax.text(j + .5, 1 - i + .46, f"{c[i,j]:,}건", ha="center", va="center", fontsize=22, color=tc, fontweight="bold")
            ax.text(j + .5, 1 - i + .2, f"{'불량' if i == 0 else '정상'}의 {rate[i,j]:.0%}", ha="center", va="center", fontsize=11, color=tc)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal"); ax.axis("off")
    ax.text(.5, 2.06, "모델: 경보", ha="center", fontsize=10, color="#555"); ax.text(1.5, 2.06, "모델: 정상", ha="center", fontsize=10, color="#555")
    ax.text(-.04, 1.5, f"실제 불량\n{c[0].sum():,}건", ha="right", va="center", fontsize=11)
    ax.text(-.04, .5, f"실제 정상\n{c[1].sum():,}건", ha="right", va="center", fontsize=11)
    ax.set_title(name, fontsize=14, fontweight="bold", pad=34)
fig.tight_layout(); fig.savefig("보고서_그림/그림6_2x2.png", dpi=170); print("ok")
