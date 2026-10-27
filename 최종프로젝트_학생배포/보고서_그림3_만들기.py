"""그림 6: 새 입력(쌍둥이 묶음 5분할) 기준 정상/불량 2x2 표, H·D·A 비교. 입력: 시드별 검증 확률 캐시(blockcv2.py)"""
import pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "Malgun Gothic"; rcParams["axes.unicode_minus"] = False
R, y = pickle.load(open(r"C:/Users/swrkd/AppData/Local/Temp/claude/c--Users-swrkd-Desktop-team1-deeplearning-main/7a761312-f706-4c12-ac6e-669e680f4db3/scratchpad/cache_block_p.pkl", "rb"))
bad = y != "정상"
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
for a, (k, name) in zip(ax, [("H", "H안 (제출)"), ("D", "D안 (균형형)"), ("A", "A안 (기본형)")]):
    c = np.zeros((2, 2))
    for seed in (7, 123, 2026):
        P, cl = R[(k, "그룹", seed)]; alarm = np.asarray(cl)[P.argmax(1)] != "정상"
        c += [[(bad & alarm).sum(), (bad & ~alarm).sum()], [(~bad & alarm).sum(), (~bad & ~alarm).sum()]]
    c = np.round(c / 3).astype(int); rate = c / c.sum(1, keepdims=True)
    a.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{c[i,j]:,}건\n({rate[i,j]:.0%})", ha="center", va="center", fontsize=10, color="white" if rate[i, j] > .5 else "black")
    a.set_xticks([0, 1]); a.set_xticklabels(["경보(불량이라 함)", "정상이라 함"], fontsize=9)
    a.set_yticks([0, 1]); a.set_yticklabels([f"실제 불량\n{c[0].sum():,}건", f"실제 정상\n{c[1].sum():,}건"], fontsize=9)
    a.set_title(f"{name}\n잡음 {c[0,0]} · 놓침 {c[0,1]:,} · 헛경보 {c[1,0]}", fontsize=10)
fig.tight_layout(); fig.savefig("보고서_그림/그림6_2x2.png", dpi=170); print("ok")
