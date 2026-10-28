"""보고서/보고서_1조.html(원본)을 발표 자료·웹 보고서와 같은 분위기로 꾸며 보고서/보고서_1조.pdf로 만든다.

사용: python 보고서_PDF_만들기.py
- 원본 HTML의 <style>을 아래 CSS로 바꿔 저장하고(내용은 그대로),
  인쇄용 사본에서만 문장 나눔·요약 카드·막대 그래프·비용 곡선을 넣어 Edge로 PDF 변환한다.
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "보고서"
SRC = DIR / "보고서_1조.html"
PDF = DIR / "보고서_1조.pdf"
DATA = json.loads((ROOT / "보고서_그림" / "비용표_데이터.json").read_text(encoding="utf-8"))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

NAVY, BLUE, ORANGE, GRAY, GREEN, RED = "#0b5a8c", "#1a7fc0", "#e8a33a", "#9aa8b6", "#2e9d6b", "#e0480f"

CSS = """
@page{size:A4;margin:13mm 14mm 15mm;@bottom-right{content:counter(page);font:8pt "Noto Sans KR","Malgun Gothic",sans-serif;color:#7a8da0}}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact;box-sizing:border-box}
body{font-family:"Noto Sans KR","Pretendard","Malgun Gothic","맑은 고딕",sans-serif;font-size:10pt;line-height:1.75;color:#1b2b3a;margin:0;word-break:keep-all}
.hd{background:#0b3556;color:#fff;padding:24px 28px 20px;border-radius:12px;margin-bottom:12px;border-bottom:7px solid #8fd0f0}
.hd h1{font-size:21pt;margin:0 0 8px;letter-spacing:-.5px;color:#fff;line-height:1.3}
.hd .sub{color:#cfe6f6;font-size:9.2pt;margin:0;line-height:1.7}
h2{font-size:14pt;margin:26px 0 12px;padding:9px 18px;background:#0b5a8c;color:#fff;border-radius:9px;border-bottom:5px solid #8fd0f0;break-after:avoid;letter-spacing:-.3px}
h2.pb{break-before:page;margin-top:0}
p{margin:0 0 9px}.sn{display:block;margin-bottom:5px}
ul{margin:4px 0 8px;padding:0;list-style:none}
li{margin:0 0 6px;padding-left:16px;position:relative}
li::before{content:"";position:absolute;left:2px;top:.72em;width:6px;height:6px;border-radius:50%;background:#1a7fc0}
ul.summary li{background:#eaf4fb;border:1px solid #cfe3f0;border-radius:9px;padding:9px 14px 9px 28px;margin:0 0 8px;break-inside:avoid}
ul.summary li::before{left:13px;top:1.05em}
b{color:#0a3c63}code{background:#e6eef6;padding:0 4px;border-radius:3px;font-size:8.8pt;color:#0b5a8c}
u{text-decoration-color:#e0a030;text-underline-offset:2px}
table{border-collapse:separate;border-spacing:0;width:100%;margin:8px 0 14px;font-size:9pt;line-height:1.55;break-inside:avoid;border:1px solid #c9dcec;border-radius:9px;overflow:hidden}
th,td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #dbe8f2}
th{background:#0b5a8c;color:#fff;font-weight:700}
td:first-child{font-weight:600;color:#0a3c63}
tr:nth-child(even) td{background:#f3f8fc}
tr.hl td{background:#d6e9f7!important;color:#0a3c63;font-weight:700}
tr:last-child td{border-bottom:0}
img{width:100%;display:block;margin:10px 0 3px;border:1px solid #cfe0ee;border-radius:10px;background:#fff;padding:5px;break-inside:avoid}
.cap{font-size:8.6pt;color:#5a6b7c;margin:0 0 12px;padding-left:3px;line-height:1.6;break-before:avoid}
.warn{background:#fdeee4;border:1px solid #f3c5a8;border-left:5px solid #e0480f;border-radius:9px;padding:10px 14px;margin:10px 0;break-inside:avoid}
.warn ul{margin-bottom:0}.warn li::before{background:#e0480f}
.cards{display:flex;gap:10px;margin:10px 0 14px}
.card{flex:1;background:#fff;border:2px solid #cfe3f0;border-radius:12px;padding:12px 10px;text-align:center}
.card .n{font-size:19pt;font-weight:800;color:#0b5a8c;line-height:1.25}.card .l{font-size:8.2pt;color:#5a6b7c;line-height:1.5;margin-top:3px}
.take{display:flex;gap:10px;margin:0 0 12px}
.take>div{flex:1;background:#fff;border:2px solid #cfe3f0;border-radius:12px;padding:10px 12px;break-inside:avoid}
.take .n{display:block;font-size:17pt;font-weight:800;color:#0b5a8c;line-height:1.25}
.take .t{display:block;margin-top:2px;font-size:8.4pt;color:#5a6b7c;line-height:1.5}
.take .good .n{color:#2e9d6b}.take .bad .n{color:#e0480f}.take .mid .n{color:#c98410}
.verdict{background:#0b5a8c;color:#fff;border-radius:10px;padding:11px 16px;margin:0 0 12px;font-size:10pt;line-height:1.65;break-inside:avoid}
.verdict b{color:#ffe08a}
.bchart{border:2px solid #cfe3f0;border-radius:12px;background:#fff;padding:12px 16px 6px;margin:0 0 12px;break-inside:avoid}
.bchart h4{margin:0 0 2px;font-size:10.5pt;color:#0a3c63}
.bchart .sub{color:#5a6b7c;font-size:8.3pt;line-height:1.55;margin-bottom:6px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:8.3pt;color:#5a6b7c;margin:0 0 4px}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.brow{display:grid;grid-template-columns:150px 1fr;gap:2px 14px;align-items:center;padding:6px 0;border-top:1px solid #e1ebf3}
.brow .bl{font-size:9pt;color:#0a3c63;font-weight:650;line-height:1.4}.brow .bl small{display:block;font-weight:400;color:#6b7c8d;font-size:7.4pt}
.bar2{display:flex;align-items:center;gap:7px;margin:2px 0}.bar2 i{display:block;height:12px;border-radius:3px;min-width:3px}
.bar2 span{font-size:8.4pt;color:#33475b;white-space:nowrap}
.div{position:relative;height:20px;margin:2px 0;background:#eef3f8;border-radius:5px}
.div .band{position:absolute;top:0;bottom:0;background:rgba(150,160,180,.25);border-radius:3px}
.div .zero{position:absolute;top:-2px;bottom:-2px;left:50%;width:2px;background:#7a8da0}
.div i{position:absolute;top:4px;height:12px;border-radius:3px}
.div span{position:absolute;top:1px;font-size:8pt;color:#0a3c63;font-weight:700}
.chartbox{border:2px solid #cfe3f0;border-radius:12px;background:#fff;padding:10px 14px 4px;margin:0 0 12px;break-inside:avoid}
.chartbox h4{margin:0 0 2px;font-size:10.5pt;color:#0a3c63}.chartbox .sub{color:#5a6b7c;font-size:8.3pt;margin-bottom:4px}
.chartbox svg{display:block;width:100%;height:auto}
.chartbox text{font-family:"Noto Sans KR","Malgun Gothic",sans-serif;fill:#5a6b7c;font-size:11px}.chartbox .val{fill:#0a3c63;font-weight:700}
"""

# ---------- 시각 요소 (웹 보고서와 같은 구성) ----------


def take(items):
    return '<div class="take">' + "".join(
        f'<div class="{c}"><span class="n">{n}</span><span class="t">{t}</span></div>' for c, n, t in items) + "</div>"


def verdict(t):
    return f'<div class="verdict">{t}</div>'


def bars(title, sub, series, rows, mx, fmt):
    leg = "".join(f'<span><i class="dot" style="background:{c}"></i>{n}</span>' for n, c in series)
    out = f'<div class="bchart"><h4>{title}</h4><div class="sub">{sub}</div><div class="legend">{leg}</div>'
    for lab, small, vals in rows:
        bs = "".join(
            f'<div class="bar2"><i style="background:{series[i][1]};width:{max(0.5, v / mx * 100):.1f}%"></i><span>{fmt(v)}</span></div>'
            for i, v in enumerate(vals))
        out += f'<div class="brow"><div class="bl">{lab}<small>{small}</small></div><div>{bs}</div></div>'
    return out + "</div>"


def diverge(title, sub, rows, lim, band):
    out = f'<div class="bchart"><h4>{title}</h4><div class="sub">{sub}</div>'
    bl, bw = 50 - band / lim * 50, band / lim * 100
    for lab, small, v in rows:
        w = abs(v) / lim * 50
        left = 50 if v >= 0 else 50 - w
        col = GREEN if v >= 0 else RED
        lx = (50 + w + 1) if v >= 0 else (50 - w - 13)
        bar = (f'<div class="div"><div class="band" style="left:{bl:.1f}%;width:{bw:.1f}%"></div><div class="zero"></div>'
               f'<i style="background:{col};left:{left:.1f}%;width:{max(w, .4):.1f}%"></i>'
               f'<span style="left:{lx:.1f}%">{v:+.4f}</span></div>')
        out += f'<div class="brow"><div class="bl">{lab}<small>{small}</small></div><div>{bar}</div></div>'
    return out + "</div>"


def cost_chart(ratio=10, split="새 입력", per_day=3800):
    """경보 기준별 하루 총비용 곡선(1안·2안·전수 검사). 웹 계산기와 같은 계산."""
    names = list(DATA["모델"].keys())
    cols = [BLUE, ORANGE]
    k = per_day / 1000
    sc = [o["scale"] for o in DATA["모델"][names[0]][split]]
    ser = [[(ratio * o["missed"] + o["fa"]) * k for o in DATA["모델"][m][split]] for m in names]
    full = DATA["정상"] * k
    W, H, L, R, T, B = 640, 270, 56, 24, 26, 50
    ymax = max(full, *[v for s in ser for v in s]) * 1.1
    X = lambda i: L + i * (W - L - R) / (len(sc) - 1)
    Y = lambda v: T + (1 - v / ymax) * (H - T - B)
    g = ""
    for j in range(5):
        v = ymax * j / 4
        g += f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#dbe6ef"/><text x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v:,.0f}</text>'
    for i, s in enumerate(sc):
        g += f'<text x="{X(i):.1f}" y="{H-B+18}" text-anchor="middle">×{s}</text>'
    g += f'<text x="{(L+W-R)/2}" y="{H-8}" text-anchor="middle">정상 확률 배율 (오른쪽일수록 불량으로 더 자주 부름 → 검사 대상 증가)</text>'
    g += (f'<line x1="{L}" x2="{W-R}" y1="{Y(full):.1f}" y2="{Y(full):.1f}" stroke="#7a8da0" stroke-width="2" stroke-dasharray="6 5"/>'
          f'<text x="{L+6}" y="{Y(full)+16:.1f}" class="val">전수 검사 {full:,.0f}</text>')
    for mi, arr in enumerate(ser):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(arr))
        g += f'<polyline fill="none" stroke="{cols[mi]}" stroke-width="3" points="{pts}"/>'
        for i, v in enumerate(arr):
            g += f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{cols[mi]}"/>'
        bi = arr.index(min(arr))
        end = bi >= len(sc) - 1
        g += (f'<circle cx="{X(bi):.1f}" cy="{Y(arr[bi]):.1f}" r="8" fill="none" stroke="{cols[mi]}" stroke-width="2.5"/>'
              f'<text x="{X(bi)-(10 if end else 0):.1f}" y="{Y(arr[bi])+(24 if mi else -14):.1f}" text-anchor="{"end" if end else "middle"}" class="val">최저 {arr[bi]:,.0f}</text>')
    lg = "".join(f'<span><i class="dot" style="background:{cols[i]}"></i>{n}</span>' for i, n in enumerate(names))
    lg += '<span><i class="dot" style="background:#7a8da0"></i>전수 검사</span>'
    return (f'<div class="chartbox"><h4>경보 기준별 하루 총비용 ({ratio}:1, {split}, 하루 {per_day:,}샷)</h4>'
            f'<div class="sub">낮을수록 좋다. 두 모델 곡선이 같은 수준으로 모이므로 비용은 모델이 아니라 경보 기준이 정한다.</div>'
            f'<div class="legend">{lg}</div><svg viewBox="0 0 {W} {H}">{g}</svg></div>')


f3 = lambda v: f"{v:.3f}"
pct = lambda v: f"{v:g}%"
S1, S2 = ("1안 (최종 제출안)", BLUE), ("2안 (현장성을 높인 또 다른 안)", ORANGE)

VIS = {
    4: take([("good", "+0.0045", "앞뒤 사건 입력 추가: 유일하게 일관된 이득"),
             ("mid", "±0.005", "시드를 바꿔도 점수가 이만큼 흔들림"),
             ("good", "20여 가지", "그 밖의 시도도 모두 ±0.005 이내")])
    + diverge("한 번에 하나만 바꾼 결과 (1안 대비 점수 차이)",
              "회색 띠 = 시드 흔들림(±0.005). 막대가 띠 안에 있으면 의미 있는 차이로 보기 어렵다.",
              [("앞뒤 사건 입력 40개 추가", "없음 → 40개", 0.0045), ("앞 사건만", "", -0.0002), ("뒤 사건만", "", -0.0015),
               ("클래스 가중치 0.3", "기본 0.5", -0.0002), ("클래스 가중치 0.75", "", 0.0006), ("클래스 가중치 1.0", "", -0.0007)],
              0.006, 0.005),
    6: take([("good", "0.56", "무작위 80/20: 같은 입력이 학습에 있어 외우면 높게 나옴"),
             ("mid", "0.26", "쌍둥이를 묶으면(새 입력) 절반 아래로"),
             ("bad", "0.125", "시간순이면 \"전부 정상\" 수준")])
    + verdict("<b>현장 배포 성능에 가까운 쪽은 시간순이다.</b> 리더보드는 시험 샷이 생산 순서에 섞여 있어 무작위 쪽에 가깝다.")
    + bars("검증 방식별 macro F1 (1안 vs 2안)", "시드 5개 평균. 시간순은 shot 순 앞 80%로 학습해 뒤 20%를 검증.",
           [S1, S2], [("무작위 80/20", "리더보드와 가장 비슷", [0.560, 0.504]), ("쌍둥이 묶음 80/20", "새 입력", [0.256, 0.338]),
                      ("시간순 80/20", "현장 배포에 가까움", [0.125, 0.111])], 0.7, f3),
    7: take([("good", "5 : 1", "×0.1이 전수 검사보다 24% 저렴 → 모델이 분명히 도움"),
             ("mid", "10 : 1", "이점 9~14%, 샷의 70~80%를 검사해야 함"),
             ("bad", "20 : 1", "전수 검사와 구별 안 됨 → 모델 이점 없음")])
    + verdict("<b>비용을 정하는 것은 모델이 아니라 경보 기준(비용 비율)이다.</b>")
    + cost_chart(),
    8: take([("good", "15.8→5.3%", "유형 2, High_Velocity &gt; 2.537 → Short_Shot"),
             ("good", "12.3→0.7%", "유형 2, High_Velocity &gt; 2.543 → Blow_Hole"),
             ("good", "9.8→1.8%", "유형 1, Spray_Time &gt; 8.1 → Exfoliation")]),
}
VIS_AFTER_FIG = {  # 해당 절의 그림 뒤에 넣을 요소
    4: bars("1안 vs 2안: 평가 조건에 따라 우열이 뒤집힌다", "학습 데이터 검증(시드 7·123·2026 평균). 탐지율 = 실제 불량 중 잡아낸 비율.",
            [S1, S2], [("public 점수", "실제 제출 결과", [0.5987, 0.5710]), ("새 입력 F1", "같은 입력이 없는 샷", [0.258, 0.323]),
                       ("새 생산 구간 F1", "연속 100샷 통째로", [0.136, 0.235])], 0.7, f3)
    + bars("불량 탐지율", "1안은 새 생산 구간에서 불량을 거의 못 잡는다(1.2%).", [S1, S2],
           [("새 입력", "", [10.3, 48.0]), ("새 생산 구간", "", [1.2, 31.0])], 60, pct),
    8: bars("조건 구간 안과 밖의 불량률", "막대가 짧을수록 불량이 적다. 괄호는 Wilson 95% 신뢰구간(안 / 밖). 구간 경계를 같은 데이터에서 골랐으므로 상관일 뿐 원인이 아니다.",
            [("조건 구간 안", NAVY), ("조건 구간 밖", GRAY)],
            [("Short_Shot", "유형 2 · High_Velocity &gt; 2.537 (3.7~7.4 / 14.2~17.6)", [5.3, 15.8]),
             ("Blow_Hole", "유형 2 · High_Velocity &gt; 2.543 (0.2~1.9 / 10.9~13.9)", [0.7, 12.3]),
             ("Blow_Hole", "유형 2 · Coolant_Temp ≤ 26.4 (20.0~27.3 / 5.2~7.4)", [23.4, 6.2]),
             ("Exfoliation", "유형 1 · Spray_Time &gt; 8.1 (1.2~2.7 / 8.5~11.2)", [1.8, 9.8]),
             ("Short_Shot", "유형 1 · Melting_Furnace_Temp &gt; 711 (1.7~4.0 / 5.5~7.5)", [2.6, 6.4])], 25, pct),
}
PAGE_BREAK = set()

SENT_END = re.compile(r"(?<=[다요음임함됨])\.(</[biu]>)?(\s+)(?=[<\"“(가-힣A-Z0-9①②③\[])")


def split_sentences(html):
    def one(m):
        tag, inner = m.group(1), m.group(2)
        if len(re.sub(r"<.*?>", "", inner)) < 110:
            return m.group(0)
        pieces, last = [], 0
        for s in SENT_END.finditer(inner):
            head = inner[: s.end()]
            if any(head.count(f"<{t}>") != head.count(f"</{t}>") for t in "biu"):
                continue
            pieces.append(inner[last : s.start() + 1 + len(s.group(1) or "")])
            last = s.end()
        pieces.append(inner[last:])
        return f"<{tag}>" + "".join(f'<span class="sn">{p}</span>' for p in pieces if p.strip()) + f"</{tag}>"

    return re.sub(r"<(p|li)>(.*?)</\1>", one, html, flags=re.S)


def decorate(html):
    html = split_sentences(html)
    html = html.replace("<ul>", '<ul class="summary">', 1)
    html = html.replace("<tr><td><b>정상 ×0.1</b>", '<tr class="hl"><td><b>정상 ×0.1</b>', 1)
    # 6절: 그림 3(막대 그래프 이미지)을 웹과 같은 막대 그래프로 대체
    html = re.sub(r'<img src="[^"]*그림3_분할비교[^"]*"><div class="cap">.*?</div>\n', "", html, count=1, flags=re.S)

    def head(m):
        n = int(m.group(2))
        cls = ' class="pb"' if n in PAGE_BREAK else ""
        return f"<h2{cls}>{m.group(1)}</h2>\n{VIS.get(n, '')}"

    html = re.sub(r"<h2>((\d+)\.[^<]*)</h2>", head, html)
    # 4절: 두 번째 표(1안 vs 2안) 앞에 막대 그래프, 8절: 그림 6 앞에 안·밖 막대 그래프
    html = html.replace('<table><tr><th>모델</th><th>public 점수', VIS_AFTER_FIG[4] + '<table><tr><th>모델</th><th>public 점수', 1)
    html = html.replace('<img src="../보고서_그림/그림5_구간별_불량률.png">', VIS_AFTER_FIG[8] + '<img src="../보고서_그림/그림5_구간별_불량률.png">', 1)
    return html


if __name__ == "__main__":
    src = SRC.read_text(encoding="utf-8")
    src = re.sub(r"<style>.*?</style>", lambda m: "<style>" + CSS + "</style>", src, count=1, flags=re.S)
    SRC.write_text(src, encoding="utf-8", newline="\n")

    tmp = DIR / "_인쇄용.html"
    tmp.write_text(decorate(src), encoding="utf-8")
    out = Path(tempfile.mkdtemp()) / "out.pdf"
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Start-Process -Wait -FilePath '{EDGE}' -ArgumentList '--headless','--disable-gpu','--no-pdf-header-footer','--print-to-pdf={out}','{tmp.as_uri()}'"],
        check=True,
    )
    tmp.unlink()
    try:
        shutil.copyfile(out, PDF)
    except PermissionError:  # PDF 뷰어가 파일을 열어 둔 경우
        PDF = PDF.with_name("보고서_1조_new.pdf")
        shutil.copyfile(out, PDF)
        print("기존 PDF가 열려 있어 새 이름으로 저장했습니다. 뷰어를 닫고 이름을 바꾸세요.")
    print("저장:", PDF, f"({PDF.stat().st_size:,} bytes)")
