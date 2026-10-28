"""보고서/보고서_1조.html(PDF 원본)을 읽어 화면용 웹 보고서 보고서/보고서_웹.html 을 만든다.

내용은 그대로 가져오고 디자인만 바꾼다(어두운 포트폴리오 스타일, 상단 메뉴, 요약 카드).
보고서_1조.html 을 고친 뒤 `python 보고서_웹_만들기.py` 를 다시 실행하면 웹 버전도 따라서 바뀐다.
"""
import re
from pathlib import Path

DIR = Path(__file__).resolve().parent / "보고서"
src = (DIR / "보고서_1조.html").read_text(encoding="utf-8")

title = re.search(r"<h1>(.*?)</h1>", src, re.S).group(1)
sub = re.search(r'<div class="sub">(.*?)</div>', src, re.S).group(1)
metrics = re.findall(r'<div class="card"><div class="n">(.*?)</div><div class="l">(.*?)</div></div>', src, re.S)

body = src.split('</div></div>\n', 1)[1] if False else src[src.index("<h2>"):src.rindex("</body>")]
cards_end = src.index("</div>\n<ul>") if "</div>\n<ul>" in src else -1
parts = re.split(r"<h2>(.*?)</h2>", body)  # ['', 제목0, 내용0, 제목1, 내용1, ...]
sections = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]


WIDGET_CSS = """
.calc{border:1px solid var(--accent);background:var(--hover);border-radius:16px;padding:24px 28px;margin:0 0 30px}
.calc h3{margin:0 0 4px;font-size:1.15rem}.calc .hint{color:var(--muted);font-size:.85rem;margin-bottom:16px}
.calc .row{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;margin:12px 0}
.calc .lbl{color:var(--muted);font-size:.82rem;min-width:84px}
.tabs{display:inline-flex;gap:6px;flex-wrap:wrap}.tab{padding:7px 16px;border:1px solid var(--border);border-radius:8px;background:var(--surface);color:var(--muted);font:inherit;font-size:.88rem;cursor:pointer}
.tab.on{background:var(--accent);color:var(--on-accent);border-color:var(--accent);font-weight:700}
.calc input[type=range]{width:260px;accent-color:var(--accent)}
.calc input[type=number]{width:96px;padding:6px 8px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text);font:inherit}
.big{color:var(--accent);font-weight:800;font-size:1.2rem}
.calc table{min-width:620px}.calc tr.best td{background:rgba(121,226,193,.12)}.calc tr.best td:first-child{border-left:3px solid var(--mint)}
.bar{display:inline-block;height:9px;border-radius:5px;background:var(--accent);vertical-align:middle;margin-right:8px;min-width:2px}
.calc .msg{margin-top:14px;padding:12px 16px;border-radius:10px;background:var(--surface-2);font-size:.93rem;color:var(--msg-t)}
"""
WIDGET_HTML = """
<div class="calc" id="calc"><h3>직접 바꿔 보기: 비용 비율을 넣으면 운영 기준이 바뀝니다</h3>
<div class="hint">현장이 "놓침 1건 = 헛경보 몇 건"인지 알려 주면 여기에 넣어 바로 최저 비용 기준을 확인할 수 있습니다. 아래 정적 표(시드 4개 평균)와 달리 이 계산기는 시드 7·123·2026 평균이라 몇 건 차이가 납니다.</div>
<div class="row"><span class="lbl">모델</span><span class="tabs" id="cm"></span></div>
<div class="row"><span class="lbl">검증 조건</span><span class="tabs" id="cs"></span></div>
<div class="row"><span class="lbl">놓침 비용 비율</span><input type="range" id="cr" min="1" max="50" value="10"><span class="big" id="crv"></span><span class="lbl">(헛경보 1건 = 1)</span>
<span class="tabs" id="cq"></span></div>
<div class="row"><span class="lbl">하루 샷 수</span><input type="number" id="cn" value="3800" min="100" step="100"><span class="lbl">사이클 22.6초·24시간이면 약 3,800</span></div>
<div class="chartbox" id="cch"></div>
<div class="table-wrap"><table><thead><tr><th>결정 기준(정상 확률 배율)</th><th>하루 검사 대상</th><th>놓침</th><th>헛경보</th><th>총비용</th></tr></thead><tbody id="cb"></tbody></table></div>
<div class="msg" id="cmsg"></div></div>
<script>
const CALC=__DATA__;
let cm=Object.keys(CALC["모델"])[0],cs="새 입력";
const $=id=>document.getElementById(id), fmt=x=>Math.round(x).toLocaleString();
function tabs(el,items,get,set){el.innerHTML="";items.forEach(v=>{const b=document.createElement("button");b.className="tab"+(get()===v?" on":"");b.textContent=v;b.onclick=()=>{set(v);draw()};el.appendChild(b)})}
function draw(){
 const r=+$("cr").value, N=+$("cn").value||3800, k=N/1000; $("crv").textContent=r+" : 1";
 tabs($("cm"),Object.keys(CALC["모델"]),()=>cm,v=>cm=v);
 tabs($("cs"),["새 입력","새 생산 구간"],()=>cs,v=>cs=v);
 tabs($("cq"),[5,10,20],()=>+$("cr").value,v=>$("cr").value=v);
 const rows=CALC["모델"][cm][cs].map(o=>({name:"정상 ×"+o.scale,ins:(CALC["불량"]-o.missed+o.fa)*k,m:o.missed*k,f:o.fa*k}));
 rows.unshift({name:"검사 안 함",ins:0,m:CALC["불량"]*k,f:0});
 rows.push({name:"전수 검사",ins:1000*k,m:0,f:CALC["정상"]*k});
 rows.forEach(o=>o.c=r*o.m+o.f);
 const full=rows[rows.length-1].c, cand=rows.filter(o=>o.name.startsWith("정상")), best=cand.reduce((a,b)=>a.c<b.c?a:b), max=Math.max(...rows.map(o=>o.c));
 $("cb").innerHTML=rows.map(o=>`<tr class="${o===best?"best":""}"><td>${o.name}${o===best?" ◀ 최저":""}</td><td>${fmt(o.ins)}</td><td>${fmt(o.m)}</td><td>${fmt(o.f)}</td><td><span class="bar" style="width:${Math.max(2,o.c/max*140)}px"></span>${fmt(o.c)}</td></tr>`).join("");
 drawChart(r,k);
 const sv=(1-best.c/full)*100;
 $("cmsg").innerHTML=`<b>${cm}, ${cs}, ${r}:1, 하루 ${fmt(N)}샷</b>: 최저 비용은 <b>${best.name}</b>(검사 ${fmt(best.ins)}샷, 놓침 ${fmt(best.m)}건, 헛경보 ${fmt(best.f)}건, 비용 ${fmt(best.c)}). 전수 검사(${fmt(full)})와 비교해 <b>${sv>=0?sv.toFixed(0)+"% 절감":(-sv).toFixed(0)+"% 더 비쌈"}</b>${sv<5?" → 이 비율에서는 모델 이점이 거의 없습니다.":"."}`;
}
function drawChart(r,k){
 const names=Object.keys(CALC["모델"]),cols=["var(--accent)","var(--yellow)"],W=640,H=280,L=56,R=24,T=30,B=52;
 const sc=CALC["모델"][names[0]][cs].map(o=>o.scale),n=sc.length;
 const ser=names.map(m=>CALC["모델"][m][cs].map(o=>(r*o.missed+o.fa)*k)),full=CALC["정상"]*k;
 const ymax=Math.max(full,...ser.flat())*1.1,X=i=>L+i*(W-L-R)/(n-1),Y=v=>T+(1-v/ymax)*(H-T-B);
 let g="";
 for(let j=0;j<=4;j++){const v=ymax*j/4,y=Y(v);g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" style="stroke:var(--border)"/><text x="${L-8}" y="${y+4}" text-anchor="end">${fmt(v)}</text>`}
 sc.forEach((s,i)=>g+=`<text x="${X(i)}" y="${H-B+18}" text-anchor="middle">×${s}</text>`);
 g+=`<text x="${(L+W-R)/2}" y="${H-8}" text-anchor="middle">정상 확률 배율 (오른쪽일수록 불량으로 더 자주 부름 → 검사 대상 증가)</text>`;
 g+=`<line x1="${L}" x2="${W-R}" y1="${Y(full)}" y2="${Y(full)}" style="stroke:var(--muted);stroke-width:2;stroke-dasharray:6 5"/><text x="${L+6}" y="${Y(full)+16}" text-anchor="start" class="val">전수 검사 ${fmt(full)}</text>`;
 ser.forEach((arr,mi)=>{
  g+=`<polyline fill="none" style="stroke:${cols[mi]};stroke-width:3" points="${arr.map((v,i)=>X(i)+","+Y(v)).join(" ")}"/>`;
  arr.forEach((v,i)=>g+=`<circle cx="${X(i)}" cy="${Y(v)}" r="3.5" style="fill:${cols[mi]}"/>`);
  const bi=arr.indexOf(Math.min(...arr));
  g+=`<circle cx="${X(bi)}" cy="${Y(arr[bi])}" r="8" style="fill:none;stroke:${cols[mi]};stroke-width:2.5"/><text x="${X(bi)-(bi>=n-1?10:0)}" y="${Y(arr[bi])+(mi?24:-14)}" text-anchor="${bi>=n-1?"end":"middle"}" class="val">최저 ${fmt(arr[bi])}</text>`;
 });
 const lg=names.map((m,i)=>`<span><i class="dot" style="background:${cols[i]}"></i>${m}</span>`).join("")+`<span><i class="dot" style="background:var(--muted)"></i>전수 검사</span>`;
 $("cch").innerHTML=`<div class="legend">${lg}</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="경보 기준별 총비용 곡선">${g}</svg>`;
}
$("cr").oninput=draw;$("cn").oninput=draw;draw();
</script>
"""


VIS_CSS = """
.take{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 24px;max-width:920px}
.take>div{flex:1;min-width:190px;padding:16px 18px;border:1px solid var(--border);border-radius:14px;background:var(--surface)}
.take .n{display:block;font-size:1.7rem;font-weight:800;color:var(--accent);line-height:1.25}
.take .t{display:block;margin-top:4px;font-size:.84rem;color:var(--muted);line-height:1.6}
.take .good .n{color:var(--mint)}.take .bad .n{color:#e5684b}.take .mid .n{color:var(--yellow)}
.verdict{max-width:920px;margin:0 0 24px;padding:16px 20px;border-left:4px solid var(--accent);background:var(--hover);border-radius:0 12px 12px 0;font-size:1.02rem;color:var(--strong);line-height:1.75}
.bchart{max-width:920px;border:1px solid var(--border);border-radius:16px;background:var(--surface);padding:22px 26px 14px;margin:0 0 26px}
.bchart h4{margin:0 0 4px;font-size:1.05rem;letter-spacing:-.03em}
.bchart .sub{color:var(--muted);font-size:.84rem;line-height:1.65;margin-bottom:12px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:.82rem;color:var(--muted);margin:0 0 6px}
.dot{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.brow{display:grid;grid-template-columns:170px 1fr;gap:4px 18px;align-items:center;padding:10px 0;border-top:1px solid var(--border)}
.brow .bl{font-size:.9rem;color:var(--strong);font-weight:650;line-height:1.5}.brow .bl small{display:block;font-weight:400;color:var(--muted);font-size:.74rem}
.bar2{display:flex;align-items:center;gap:9px;margin:3px 0}.bar2 i{display:block;height:15px;border-radius:4px;min-width:3px}
.bar2 span{font-size:.82rem;font-variant-numeric:tabular-nums;color:var(--prose);white-space:nowrap}
.c1{background:var(--accent)}.c2{background:var(--yellow)}.c3{background:#8793a6}.c4{background:var(--mint)}.c5{background:#e5684b}
.div{position:relative;height:22px;margin:3px 0;background:var(--surface-2);border-radius:6px}
.div .band{position:absolute;top:0;bottom:0;background:rgba(150,160,180,.20);border-radius:4px}
.div .zero{position:absolute;top:-2px;bottom:-2px;left:50%;width:2px;background:var(--muted)}
.div i{position:absolute;top:4px;height:14px;border-radius:3px}
.div span{position:absolute;top:2px;font-size:.78rem;font-variant-numeric:tabular-nums;color:var(--strong);font-weight:650}
.fold{max-width:920px;margin:6px 0 20px;border:1px solid var(--border);border-radius:14px;background:var(--surface)}
.fold>summary{cursor:pointer;padding:14px 20px;font-weight:650;color:var(--strong);list-style:none;display:flex;justify-content:space-between;gap:10px}
.fold>summary::after{content:"\\25BC  \\D3BC\\CE58\\AE30";color:var(--accent);font-size:.82rem;font-weight:500;white-space:nowrap}
.fold[open]>summary::after{content:"\\25B2  \\C811\\AE30"}
.fold>summary::-webkit-details-marker{display:none}
.fold>.inner{padding:2px 20px 14px}.fold>.inner .prose{max-width:none}
.chartbox{margin:6px 0 18px}.chartbox svg{display:block;width:100%;height:auto}
.chartbox text{font-family:inherit;fill:var(--muted);font-size:11px}.chartbox .val{fill:var(--strong);font-weight:700;font-size:11.5px}
@media(max-width:700px){.brow{grid-template-columns:1fr}}
"""


def take(items):
    return '<div class="take">' + "".join(
        f'<div class="{c}"><span class="n">{n}</span><span class="t">{t}</span></div>' for c, n, t in items) + "</div>"


def verdict(t):
    return f'<div class="verdict">{t}</div>'


def bars(title, sub, series, rows, mx, fmt):
    """series: [(이름, css클래스)], rows: [(라벨, 보조, [값...])]. 막대 길이는 mx 대비 비율."""
    leg = "".join(f'<span><i class="dot {c}"></i>{n}</span>' for n, c in series)
    out = f'<div class="bchart"><h4>{title}</h4><div class="sub">{sub}</div><div class="legend">{leg}</div>'
    for lab, small, vals in rows:
        bs = "".join(
            f'<div class="bar2"><i class="{series[i][1]}" style="width:{max(0.5, v / mx * 100):.1f}%"></i><span>{fmt(v)}</span></div>'
            for i, v in enumerate(vals))
        out += f'<div class="brow"><div class="bl">{lab}<small>{small}</small></div><div>{bs}</div></div>'
    return out + "</div>"


def diverge(title, sub, rows, lim, band):
    out = f'<div class="bchart"><h4>{title}</h4><div class="sub">{sub}</div>'
    bl = 50 - band / lim * 50
    bw = band / lim * 100
    for lab, small, v in rows:
        w = abs(v) / lim * 50
        left = 50 if v >= 0 else 50 - w
        cls = "c4" if v >= 0 else "c5"
        lx = (50 + w + 1) if v >= 0 else (50 - w - 12)
        bar = (f'<div class="div"><div class="band" style="left:{bl:.1f}%;width:{bw:.1f}%"></div><div class="zero"></div>'
               f'<i class="{cls}" style="left:{left:.1f}%;width:{max(w, .4):.1f}%"></i>'
               f'<span style="left:{lx:.1f}%">{v:+.4f}</span></div>')
        out += f'<div class="brow"><div class="bl">{lab}<small>{small}</small></div><div>{bar}</div></div>'
    return out + "</div>"


def fold(html, summary, opened=False):
    return f'<details class="fold"{" open" if opened else ""}><summary>{summary}</summary><div class="inner">{html}</div></details>'


f3 = lambda v: f"{v:.3f}"
pct = lambda v: f"{v:g}%"

INTRO = {
    4: take([("good", "+0.0045", "앞뒤 사건 입력값 추가: 유일하게 일관된 이득"),
             ("mid", "±0.005", "시드를 바꿔도 점수가 이만큼 흔들림(이 안은 사실상 같음)"),
             ("good", "20여 가지", "그 밖의 시도도 모두 ±0.005 이내 → 1안이 평평한 정점")])
    + diverge("한 번에 하나만 바꾼 결과 (1안 대비 점수 차이)",
              "회색 띠 = 시드 흔들림(±0.005). 막대가 띠 안에 있으면 의미 있는 차이라고 보기 어렵다.",
              [("앞뒤 사건 입력 40개 추가", "없음 → 40개", 0.0045), ("앞 사건만", "", -0.0002), ("뒤 사건만", "", -0.0015),
               ("클래스 가중치 0.3", "기본 0.5", -0.0002), ("클래스 가중치 0.75", "", 0.0006), ("클래스 가중치 1.0", "", -0.0007)],
              0.006, 0.005)
    + verdict("<b>리더보드에서는 1안(0.5987 대 0.5710), 처음 보는 입력과 새 생산 구간에서는 2안이 불량을 더 많이 잡는다.</b> 선택은 public 점수로 했고, 검증 수치는 그 한계를 보는 데 썼다.")
    + bars("1안 vs 2안: 평가 조건에 따라 우열이 뒤집힌다", "같은 학습 데이터 검증(시드 7·123·2026 평균). 탐지율 = 실제 불량 중 잡아낸 비율.",
           [("1안 (최종 제출안)", "c1"), ("2안 (현장성을 높인 또 다른 안)", "c2")],
           [("public 점수", "실제 제출 결과", [0.5987, 0.5710]), ("새 입력 F1", "같은 입력이 없는 샷", [0.258, 0.323]),
            ("새 생산 구간 F1", "연속 100샷 통째로", [0.136, 0.235])], 0.7, f3)
    + bars("불량 탐지율", "1안은 새 생산 구간에서 불량을 거의 못 잡는다(1.2%).",
           [("1안 (최종 제출안)", "c1"), ("2안 (현장성을 높인 또 다른 안)", "c2")],
           [("새 입력", "", [10.3, 48.0]), ("새 생산 구간", "", [1.2, 31.0])], 60, pct),
    6: take([("good", "0.56", "무작위 80/20: 같은 입력이 학습에 있어 외우면 높게 나옴"),
             ("mid", "0.26", "쌍둥이를 묶으면(새 입력) 절반 아래로"),
             ("bad", "0.125", "시간순(shot 6009~7535)이면 \"전부 정상\" 수준")])
    + verdict("<b>현장 배포 성능에 가까운 쪽은 시간순이다.</b> 리더보드는 시험 샷이 생산 순서에 섞여 있어 무작위 쪽에 가깝다. 점수 하나만 보고 판단하면 안 되는 이유다.")
    + bars("검증 방식별 macro F1 (1안 vs 2안)", "시드 5개 평균. 시간순은 shot 순 앞 80%로 학습해 뒤 20%를 검증.",
           [("1안 (최종 제출안)", "c1"), ("2안 (현장성을 높인 또 다른 안)", "c2")],
           [("무작위 80/20", "리더보드와 가장 비슷", [0.560, 0.504]), ("쌍둥이 묶음 80/20", "새 입력", [0.256, 0.338]),
            ("시간순 80/20", "현장 배포에 가까움", [0.125, 0.111])], 0.7, f3),
    7: take([("good", "5 : 1", "×0.1이 전수 검사보다 24% 저렴 → 모델이 분명히 도움"),
             ("mid", "10 : 1", "이점 9~14%, 샷의 70~80%를 검사해야 함"),
             ("bad", "20 : 1", "전수 검사와 구별 안 됨 → 모델 이점 없음")])
    + verdict("<b>비용을 정하는 것은 모델이 아니라 경보 기준(비용 비율)이다.</b> 아래 계산기에서 비율을 바꾸면 두 모델의 비용 곡선이 함께 움직인다.")
    ,
    8: take([("good", "15.8 → 5.3%", "유형 2, High_Velocity &gt; 2.537 → Short_Shot"),
             ("good", "12.3 → 0.7%", "유형 2, High_Velocity &gt; 2.543 → Blow_Hole"),
             ("good", "9.8 → 1.8%", "유형 1, Spray_Time &gt; 8.1 → Exfoliation")])
    + bars("조건 구간 안과 밖의 불량률", "막대가 짧을수록 불량이 적다. 괄호 속 구간은 Wilson 95% 신뢰구간. 구간 경계를 같은 데이터에서 골랐으므로 상관일 뿐 원인이 아니다.",
           [("조건 구간 안", "c1"), ("조건 구간 밖", "c3")],
           [("Short_Shot", "유형 2 · High_Velocity &gt; 2.537 (3.7~7.4 / 14.2~17.6)", [5.3, 15.8]),
            ("Blow_Hole", "유형 2 · High_Velocity &gt; 2.543 (0.2~1.9 / 10.9~13.9)", [0.7, 12.3]),
            ("Blow_Hole", "유형 2 · Coolant_Temp ≤ 26.4 (20.0~27.3 / 5.2~7.4)", [23.4, 6.2]),
            ("Exfoliation", "유형 1 · Spray_Time &gt; 8.1 (1.2~2.7 / 8.5~11.2)", [1.8, 9.8]),
            ("Short_Shot", "유형 1 · Melting_Furnace_Temp &gt; 711 (1.7~4.0 / 5.5~7.5)", [2.6, 6.4])], 25, pct),
}
FOLD = {4: "자세한 설명과 표 전체 보기", 7: "가정과 출처, 정적 비용표, 모델 비교 표 전체 보기"}

NAV = ["요약", "데이터", "설계", "베이스라인", "비교 실험", "학습곡선", "분할 비교", "운영 기준", "원인 분석", "현장 언어", "AI 사용"]


SENT_END = re.compile(r"(?<=[다요음임함됨])\.(</[biu]>)?(\s+)(?=[<\"“(가-힣A-Z0-9①②③\[])")


def split_sentences(html):
    """긴 문단을 문장 단위로 나눠 한 문장씩 줄바꿈한다(굵은 글씨 같은 태그가 열린 채로는 나누지 않는다)."""

    def one(m):
        tag, inner = m.group(1), m.group(2)
        if len(re.sub(r"<.*?>", "", inner)) < 120:
            return m.group(0)
        pieces, last = [], 0
        for s in SENT_END.finditer(inner):
            head = inner[: s.end()]
            if any(head.count(f"<{t}>") != head.count(f"</{t}>") for t in "biu"):
                continue
            pieces.append(inner[last : s.start() + 1 + len(s.group(1) or "")])
            last = s.end()
        pieces.append(inner[last:])
        body = "".join(f'<span class="sn">{p}</span>' for p in pieces if p.strip())
        return f"<{tag}>{body}</{tag}>"

    return re.sub(r"<(p|li)>(.*?)</\1>", one, html, flags=re.S)


def convert(html, n):
    html = re.sub(r'<div class="cards">.*?</div>\n(?=<ul>)', "", html, flags=re.S) if n == 0 else html
    if n == 0:
        html = html.replace("<ul>", '<ul class="summary">', 1)
    html = split_sentences(html)
    html = re.sub(r"(<table.*?</table>)", r'<div class="table-wrap">\1</div>', html, flags=re.S)
    html = re.sub(r'(<img [^>]*>)<div class="cap">(.*?)</div>', r"<figure>\1<figcaption>\2</figcaption></figure>", html, flags=re.S)
    html = html.replace('<div class="warn">', '<div class="card warn">')
    return html.strip()


import json
DATA = json.dumps(json.loads((DIR.parent / "보고서_그림" / "비용표_데이터.json").read_text(encoding="utf-8")), ensure_ascii=False)
metric_html = "".join(
    f'<div class="metric"><span class="metric-label">{l}</span><span class="metric-value">{v}</span></div>' for v, l in metrics
)
HERO = f"""<div class="hero"><span class="eyebrow">POSCO K-뉴딜아카데미 · Deep Learning · 1조</span>
<h1>주조 불량 유형 예측<br><span class="hl">1조 최종 보고서</span></h1><p class="hero-description">{sub}</p>
<div class="metrics">{metric_html}</div></div>"""

def body(n, content):
    html = convert(content, n)
    if n == 6:  # 새 막대 그래프가 그림 3과 같은 내용이라 웹에서는 그림 3 제거
        html = re.sub(r"<figure>(?:(?!</figure>).)*?그림3_분할비교(?:(?!</figure>).)*?</figure>", "", html, count=1, flags=re.S)
    if n == 8:
        html = re.sub(r'(<div class="table-wrap">.*?</div>)', lambda m: fold(m.group(1), "구간별 불량률 표 보기"), html, count=1, flags=re.S)
        return f'<div class="prose">{html}</div>'
    if n in FOLD:
        return fold(f'<div class="prose">{html}</div>', FOLD[n])
    return f'<div class="prose">{html}</div>'


pages, nav = [], []
for n, (h, content) in enumerate(sections):
    name = re.sub(r"^\d+\.\s*", "", h)
    label = NAV[n] if n < len(NAV) else name
    nav.append(f'<a href="#s{n}" data-n="{n}"><span class="num">{n:02d}</span>{label}</a>')
    head = HERO if n == 0 else ""
    pages.append(
        f'<section id="s{n}" class="page">{head}<div class="pagehead"><span class="eyebrow">{n:02d} / {label}</span>'
        f'<h2 class="section-title">{name}</h2></div>'
        f'{INTRO.get(n, "")}{WIDGET_HTML.replace("__DATA__", DATA) if n == 7 else ""}{body(n, content)}</section>'
    )

CSS = """
:root{--bg:#090f1c;--surface:#111b2c;--surface-2:#17243a;--border:#2b3c54;--text:#edf3fb;--muted:#a9bad0;--accent:#72b8ff;--mint:#79e2c1;--yellow:#f2ca82;--side:272px;--side-bg:#0c1424;--prose:#d7e1ee;--strong:#fff;--num:#5f7391;--on-accent:#08101d;--metric-bg:rgba(17,27,44,.85);--glow:rgba(43,113,186,.22);--hl:rgba(114,184,255,.22);--warn-t:#f6e3bd;--msg-t:#e7eefb;--hover:rgba(114,184,255,.07)}
:root[data-theme=light]{--bg:#f6f8fb;--surface:#fff;--surface-2:#edf2f8;--border:#d5dee9;--text:#172033;--muted:#566680;--accent:#1d6fd6;--mint:#0c8a68;--yellow:#b7791f;--side-bg:#fff;--prose:#2a364b;--strong:#0b1424;--num:#8a99ae;--on-accent:#fff;--metric-bg:#fff;--glow:rgba(29,111,214,.10);--hl:rgba(29,111,214,.16);--warn-t:#6b4a0c;--msg-t:#1d2a40;--hover:rgba(29,111,214,.05)}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font-family:"Pretendard","Noto Sans KR","Malgun Gothic",-apple-system,"Segoe UI",sans-serif;line-height:1.9;word-break:keep-all}
a{color:inherit;text-decoration:none}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:var(--side);background:var(--side-bg);border-right:1px solid var(--border);padding:26px 16px 22px;overflow-y:auto;display:flex;flex-direction:column;z-index:50}
.brand{font-weight:800;letter-spacing:-.04em;font-size:1.02rem;padding:0 10px}.brand span{color:var(--accent)}.brand small{display:block;margin-top:4px;color:var(--muted);font-weight:500;font-size:.78rem;letter-spacing:0}
.side-label{margin:26px 10px 8px;color:var(--muted);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase}
.sidebar nav a{display:flex;align-items:center;gap:12px;padding:9px 12px;border-radius:9px;color:var(--muted);font-size:.92rem;line-height:1.4;transition:.15s}
.sidebar nav a .num{color:var(--num);font-size:.76rem;font-weight:700;min-width:20px}
.sidebar nav a:hover{background:var(--surface);color:var(--text)}
.sidebar nav a.on{background:rgba(114,184,255,.14);color:var(--strong);font-weight:700}.sidebar nav a.on .num{color:var(--accent)}
.side-links{margin-top:auto;padding:20px 10px 0;display:flex;flex-direction:column;gap:8px;font-size:.84rem;color:var(--muted)}.side-links a:hover{color:var(--accent)}
main{margin-left:var(--side);padding:56px clamp(24px,5vw,72px) 90px;max-width:calc(var(--side) + 1100px)}
.page{display:none}.page.on{display:block}
.hero{padding:6px 0 44px;margin-bottom:38px;border-bottom:1px solid var(--border);background:radial-gradient(ellipse at 80% 0%,var(--glow),transparent 55%)}
.eyebrow{display:inline-block;color:var(--mint);font-size:.8rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
h1,h2,h3{line-height:1.35;letter-spacing:-.045em}h1{margin:18px 0;font-size:clamp(2.1rem,4.6vw,3.6rem);font-weight:850}h1 .hl{color:var(--accent)}
.hero-description{max-width:760px;color:var(--muted);font-size:1rem;margin:0}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:34px}
.metric{padding:20px;border:1px solid var(--border);border-radius:14px;background:var(--metric-bg)}
.metric-label{display:block;color:var(--muted);font-size:.8rem;line-height:1.5}.metric-value{display:block;margin-top:8px;color:var(--accent);font-size:clamp(1.4rem,2.6vw,2rem);font-weight:800;line-height:1.3}
.pagehead{margin-bottom:26px}.section-title{margin:8px 0 0;font-size:clamp(1.7rem,3.4vw,2.4rem)}
.hero,.pagehead,.calc,.prose,.pager{max-width:920px}
td:first-child{font-weight:600;color:var(--strong)}tbody tr:hover td{background:var(--hover)}
figure{cursor:zoom-in}
.progress{position:fixed;left:0;top:0;height:3px;width:0;background:var(--accent);z-index:99;transition:width .1s}
.lightbox{position:fixed;inset:0;background:rgba(5,9,16,.86);display:none;align-items:center;justify-content:center;z-index:100;cursor:zoom-out;padding:24px}
.lightbox.on{display:flex}.lightbox img{max-width:96vw;max-height:94vh;background:#fff;border-radius:10px;padding:10px}
.themebtn{margin:22px 10px 0;align-self:flex-start;padding:8px 12px;border:1px solid var(--border);border-radius:9px;background:var(--surface);color:var(--muted);font:inherit;font-size:.82rem;cursor:pointer;text-align:left}
.themebtn:hover{color:var(--accent);border-color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media print{.sidebar,.pager,.progress,.themebtn{display:none}.page{display:block!important;page-break-before:always}main{margin:0}}
.prose p,.prose li{color:var(--prose);font-size:1.03rem}
.prose p{margin:0 0 22px}.sn{display:block;margin-bottom:.6em}.prose li .sn:last-child,.prose p .sn:last-child{margin-bottom:0}
.prose ul{padding-left:0;list-style:none;margin:0 0 20px}.prose li{position:relative;padding-left:22px;margin-bottom:16px}
.prose li::before{content:"";position:absolute;left:3px;top:.78em;width:7px;height:7px;border-radius:50%;background:var(--mint)}
.prose ul.summary li{padding:16px 20px 16px 38px;background:var(--surface);border:1px solid var(--border);border-radius:14px;margin-bottom:12px}
.prose ul.summary li::before{left:18px;top:1.3em}
b{color:var(--strong);font-weight:700}.prose b{background:linear-gradient(transparent 62%,var(--hl) 62%)}
code{background:var(--surface-2);color:var(--mint);padding:1px 7px;border-radius:5px;font-size:.88em}u{text-decoration-color:var(--yellow);text-underline-offset:3px}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:14px;margin:10px 0 28px}
table{width:100%;border-collapse:collapse;background:var(--surface);min-width:560px}
th,td{padding:14px 16px;text-align:left;border-bottom:1px solid var(--border);font-size:.9rem;line-height:1.65;color:var(--prose);font-variant-numeric:tabular-nums}
th{color:var(--muted);background:var(--surface-2);font-weight:650}tr:last-child td{border-bottom:0}
figure{margin:14px 0 30px;max-width:900px}figure img{display:block;width:100%;background:#fff;border-radius:12px;padding:10px;border:1px solid var(--border)}
figcaption{margin-top:12px;color:var(--muted);font-size:.86rem;line-height:1.7}
.card{padding:24px 28px;background:var(--surface);border:1px solid var(--border);border-radius:16px;margin:16px 0}
.card.warn{border-color:rgba(242,202,130,.55);background:rgba(242,202,130,.06)}.card.warn li,.card.warn b{color:var(--warn-t)}
.pager{display:flex;justify-content:space-between;gap:14px;margin-top:56px;padding-top:26px;border-top:1px solid var(--border)}
.pager button{flex:1;max-width:340px;padding:14px 18px;border:1px solid var(--border);border-radius:12px;background:var(--surface);color:var(--text);font:inherit;font-size:.92rem;cursor:pointer;text-align:left;transition:.15s}
.pager button:hover{border-color:var(--accent);transform:translateY(-2px)}.pager .next{text-align:right;margin-left:auto}.pager small{display:block;color:var(--muted);font-size:.74rem}
@media(max-width:900px){.sidebar{position:static;width:auto;border-right:0;border-bottom:1px solid var(--border);padding:16px}.sidebar nav{display:grid;grid-template-columns:repeat(2,1fr);gap:2px}.side-links{flex-direction:row;padding-top:14px}main{margin-left:0;padding:34px 20px 70px}.metrics{grid-template-columns:repeat(2,1fr)}}
@media(max-width:520px){.metrics{grid-template-columns:1fr}}
"""

JS = """
const pages=[...document.querySelectorAll('.page')],links=[...document.querySelectorAll('.sidebar nav a')],names=%NAMES%;
function show(n,push){n=Math.max(0,Math.min(pages.length-1,n));pages.forEach((p,i)=>p.classList.toggle('on',i===n));links.forEach((a,i)=>a.classList.toggle('on',i===n));
 if(push)history.replaceState(null,'','#s'+n);window.scrollTo(0,0);document.title=names[n]+' | 주조 불량 유형 예측 1조 보고서'}
links.forEach((a,i)=>a.addEventListener('click',e=>{e.preventDefault();show(i,true)}));
document.querySelectorAll('.pager button').forEach(b=>b.addEventListener('click',()=>show(+b.dataset.to,true)));
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const c=pages.findIndex(p=>p.classList.contains('on'));if(e.key==='ArrowRight')show(c+1,true);if(e.key==='ArrowLeft')show(c-1,true)});
const root=document.documentElement,tb=document.getElementById('theme');
function setTheme(t){root.dataset.theme=t;tb.textContent=t==='light'?'🌙 어두운 화면으로':'☀ 밝은 화면으로';try{localStorage.setItem('theme',t)}catch(e){}}
let t0=null;try{t0=localStorage.getItem('theme')}catch(e){}
setTheme(t0||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'));
tb.onclick=()=>setTheme(root.dataset.theme==='light'?'dark':'light');
const bar=document.getElementById('progress');
addEventListener('scroll',()=>{const h=document.documentElement;bar.style.width=(h.scrollTop/Math.max(1,h.scrollHeight-h.clientHeight)*100)+'%'});
const lb=document.getElementById('lb'),lbi=lb.querySelector('img');
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{lbi.src=i.src;lb.classList.add('on')});
lb.onclick=()=>lb.classList.remove('on');
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('on')});
show(parseInt((location.hash.match(/s(\\d+)/)||[])[1]||0),false);
"""

# 페이지 아래 이전/다음 버튼
for i in range(len(pages)):
    prev = f'<button data-to="{i-1}"><small>← 이전</small>{NAV[i-1]}</button>' if i > 0 else "<span></span>"
    nxt = f'<button class="next" data-to="{i+1}"><small>다음 →</small>{NAV[i+1]}</button>' if i < len(pages) - 1 else ""
    pages[i] = pages[i].replace("</section>", f'<div class="pager">{prev}{nxt}</div></section>')

page = (
    '<!DOCTYPE html>\n<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    "<title>주조 불량 유형 예측 | 1조 보고서</title>\n<style>" + CSS + WIDGET_CSS + VIS_CSS + "</style></head><body>\n"
    '<aside class="sidebar"><div class="brand">DIE-CASTING <span>DEFECT LAB</span><small>1조 최종 보고서 · 포스코 K-뉴딜아카데미</small></div>'
    '<div class="side-label">목차</div><nav>' + "".join(nav) + "</nav>"
    '<button class="themebtn" id="theme" type="button"></button><div class="side-links"><a href="보고서_1조.pdf">PDF 보고서</a><a href="https://github.com/hajin-eng/team1-deeplearning">GitHub 코드</a>'
    '<span style="font-size:.72rem">← → 키로도 넘길 수 있습니다</span></div></aside>\n<div class="progress" id="progress"></div><div class="lightbox" id="lb"><img alt=""></div>\n<main>'
    + "".join(pages)
    + "</main>\n<script>" + JS.replace("%NAMES%", json.dumps(NAV, ensure_ascii=False)) + "</script></body></html>"
)
(DIR / "보고서_웹.html").write_text(page, encoding="utf-8")
print("저장:", DIR / "보고서_웹.html", f"({len(page):,}자)")
