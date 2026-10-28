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
.calc{border:1px solid var(--accent);background:rgba(114,184,255,.05);border-radius:16px;padding:24px 28px;margin:0 0 30px}
.calc h3{margin:0 0 4px;font-size:1.15rem}.calc .hint{color:var(--muted);font-size:.85rem;margin-bottom:16px}
.calc .row{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;margin:12px 0}
.calc .lbl{color:var(--muted);font-size:.82rem;min-width:84px}
.tabs{display:inline-flex;gap:6px;flex-wrap:wrap}.tab{padding:7px 16px;border:1px solid var(--border);border-radius:8px;background:var(--surface);color:var(--muted);font:inherit;font-size:.88rem;cursor:pointer}
.tab.on{background:var(--accent);color:#08101d;border-color:var(--accent);font-weight:700}
.calc input[type=range]{width:260px;accent-color:var(--accent)}
.calc input[type=number]{width:96px;padding:6px 8px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text);font:inherit}
.big{color:var(--accent);font-weight:800;font-size:1.2rem}
.calc table{min-width:620px}.calc tr.best td{background:rgba(121,226,193,.12)}.calc tr.best td:first-child{border-left:3px solid var(--mint)}
.bar{display:inline-block;height:9px;border-radius:5px;background:var(--accent);vertical-align:middle;margin-right:8px;min-width:2px}
.calc .msg{margin-top:14px;padding:12px 16px;border-radius:10px;background:var(--surface-2);font-size:.93rem;color:#e7eefb}
"""
WIDGET_HTML = """
<div class="calc" id="calc"><h3>직접 바꿔 보기: 비용 비율을 넣으면 운영 기준이 바뀝니다</h3>
<div class="hint">현장이 "놓침 1건 = 헛경보 몇 건"인지 알려 주면 여기에 넣어 바로 최저 비용 기준을 확인할 수 있습니다. 아래 정적 표(시드 4개 평균)와 달리 이 계산기는 시드 7·123·2026 평균이라 몇 건 차이가 납니다.</div>
<div class="row"><span class="lbl">모델</span><span class="tabs" id="cm"></span></div>
<div class="row"><span class="lbl">검증 조건</span><span class="tabs" id="cs"></span></div>
<div class="row"><span class="lbl">놓침 비용 비율</span><input type="range" id="cr" min="1" max="50" value="10"><span class="big" id="crv"></span><span class="lbl">(헛경보 1건 = 1)</span>
<span class="tabs" id="cq"></span></div>
<div class="row"><span class="lbl">하루 샷 수</span><input type="number" id="cn" value="3800" min="100" step="100"><span class="lbl">사이클 22.6초·24시간이면 약 3,800</span></div>
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
 const sv=(1-best.c/full)*100;
 $("cmsg").innerHTML=`<b>${cm}, ${cs}, ${r}:1, 하루 ${fmt(N)}샷</b>: 최저 비용은 <b>${best.name}</b>(검사 ${fmt(best.ins)}샷, 놓침 ${fmt(best.m)}건, 헛경보 ${fmt(best.f)}건, 비용 ${fmt(best.c)}). 전수 검사(${fmt(full)})와 비교해 <b>${sv>=0?sv.toFixed(0)+"% 절감":(-sv).toFixed(0)+"% 더 비쌈"}</b>${sv<5?" → 이 비율에서는 모델 이점이 거의 없습니다.":"."}`;
}
$("cr").oninput=draw;$("cn").oninput=draw;draw();
</script>
"""

NAV = ["요약", "데이터", "설계", "베이스라인", "비교 실험", "학습곡선", "분할 비교", "운영 기준", "원인 분석", "현장 언어", "AI 사용"]


def convert(html, n):
    html = re.sub(r'<div class="cards">.*?</div>\n(?=<ul>)', "", html, flags=re.S) if n == 0 else html
    html = re.sub(r"(<table.*?</table>)", r'<div class="table-wrap">\1</div>', html, flags=re.S)
    html = re.sub(r'(<img [^>]*>)<div class="cap">(.*?)</div>', r"<figure>\1<figcaption>\2</figcaption></figure>", html, flags=re.S)
    html = html.replace('<div class="warn">', '<div class="card warn">')
    return html.strip()


import json
DATA = json.dumps(json.loads((DIR.parent / "보고서_그림" / "비용표_데이터.json").read_text(encoding="utf-8")), ensure_ascii=False)
out = []
for n, (h, content) in enumerate(sections):
    name = re.sub(r"^\d+\.\s*", "", h)
    out.append(
        f'<section id="s{n}" class="content-section"><div class="container">'
        f'<span class="eyebrow">{n:02d} / {name.split("(")[0].strip()}</span>'
        f'<h2 class="section-title">{name}</h2>{WIDGET_HTML.replace("__DATA__", DATA) if n == 7 else ""}{convert(content, n)}</div></section>'
    )

metric_html = "".join(
    f'<div class="metric"><span class="metric-label">{l}</span><span class="metric-value">{v}</span></div>' for v, l in metrics
)
nav_html = "".join(f'<a href="#s{n}">{NAV[n] if n < len(NAV) else n}</a>' for n in range(len(sections)))

page = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>주조 불량 유형 예측 | 1조 보고서</title>
<style>
:root{{--bg:#090f1c;--surface:#111b2c;--surface-2:#17243a;--border:#2b3c54;--text:#edf3fb;--muted:#a9bad0;--accent:#72b8ff;--mint:#79e2c1;--yellow:#f2ca82;--max:1040px}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;scroll-padding-top:84px}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:"Pretendard","Noto Sans KR","Malgun Gothic",-apple-system,"Segoe UI",sans-serif;line-height:1.8;word-break:keep-all}}
a{{color:inherit;text-decoration:none}}.container{{width:min(calc(100% - 40px),var(--max));margin:0 auto}}
.header{{position:sticky;top:0;z-index:100;background:rgba(9,15,28,.94);border-bottom:1px solid var(--border);backdrop-filter:blur(12px)}}
.nav{{min-height:64px;display:flex;justify-content:space-between;align-items:center;gap:20px}}
.brand{{font-weight:800;letter-spacing:-.04em;white-space:nowrap}}.brand span{{color:var(--accent)}}
.nav-links{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.8rem;color:var(--muted);justify-content:flex-end}}.nav-links a:hover{{color:var(--accent)}}
.hero{{padding:90px 0 70px;background:radial-gradient(ellipse at 75% 20%,rgba(43,113,186,.24),transparent 48%),radial-gradient(ellipse at 15% 85%,rgba(38,133,120,.12),transparent 40%)}}
.eyebrow{{display:inline-block;color:var(--mint);font-size:.8rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}}
h1,h2,h3{{line-height:1.35;letter-spacing:-.045em}}h1{{margin:20px 0;font-size:clamp(2.2rem,5.5vw,4.2rem);font-weight:850}}h1 .hl{{color:var(--accent)}}
.hero-description{{max-width:780px;color:var(--muted);font-size:1.02rem}}
.buttons{{display:flex;flex-wrap:wrap;gap:12px;margin-top:30px}}
.button{{display:inline-flex;align-items:center;min-height:44px;padding:8px 20px;border:1px solid var(--border);border-radius:9px;background:var(--surface);font-weight:650;transition:.2s}}
.button:hover{{transform:translateY(-2px);border-color:var(--accent)}}.button.primary{{color:#08101d;background:var(--accent);border-color:var(--accent)}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:56px}}
.metric{{padding:22px;border:1px solid var(--border);border-radius:14px;background:rgba(17,27,44,.8)}}
.metric-label{{display:block;color:var(--muted);font-size:.8rem;line-height:1.5}}.metric-value{{display:block;margin-top:10px;color:var(--accent);font-size:clamp(1.5rem,3vw,2.1rem);font-weight:800;line-height:1.3}}
section.content-section{{padding:70px 0;border-top:1px solid rgba(43,60,84,.65)}}
.section-title{{margin:10px 0 24px;font-size:clamp(1.6rem,3.6vw,2.3rem)}}
p,li{{color:#d3deec;font-size:.97rem}}p{{margin:0 0 16px}}ul{{padding-left:1.2em}}li{{margin-bottom:10px}}li::marker{{color:var(--mint)}}
b{{color:#fff}}code{{background:var(--surface-2);color:var(--mint);padding:1px 7px;border-radius:5px;font-size:.88em}}u{{text-decoration-color:var(--yellow);text-underline-offset:3px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:14px;margin:8px 0 22px}}
table{{width:100%;border-collapse:collapse;background:var(--surface);min-width:560px}}
th,td{{padding:13px 16px;text-align:left;border-bottom:1px solid var(--border);font-size:.88rem;line-height:1.6;color:#d3deec}}
th{{color:var(--muted);background:var(--surface-2);font-weight:650}}tr:last-child td{{border-bottom:0}}
figure{{margin:10px 0 24px}}figure img{{display:block;width:100%;max-width:880px;background:#fff;border-radius:12px;padding:10px;border:1px solid var(--border)}}
figcaption{{margin-top:10px;color:var(--muted);font-size:.84rem;line-height:1.6}}
.card{{padding:24px 28px;background:var(--surface);border:1px solid var(--border);border-radius:16px;margin:14px 0}}
.card.warn{{border-color:rgba(242,202,130,.55);background:rgba(242,202,130,.06)}}.card.warn li,.card.warn b{{color:#f6e3bd}}
.footer{{padding:40px 0;border-top:1px solid var(--border);color:var(--muted);font-size:.85rem}}
@media(max-width:850px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.nav{{flex-direction:column;align-items:flex-start;padding:10px 0}}.nav-links{{justify-content:flex-start}}}}
@media(max-width:520px){{.metrics{{grid-template-columns:1fr}}.hero{{padding:60px 0 50px}}}}
{WIDGET_CSS}</style></head><body>
<header class="header"><nav class="nav container"><a href="#home" class="brand">DIE-CASTING <span>DEFECT LAB</span></a><div class="nav-links">{nav_html}</div></nav></header>
<main>
<section id="home" class="hero"><div class="container">
<span class="eyebrow">POSCO K-뉴딜아카데미 · Deep Learning · Final Project · 1조</span>
<h1>주조 불량 유형 예측<br><span class="hl">1조 최종 보고서</span></h1>
<p class="hero-description">{sub}</p>
<div class="buttons"><a class="button primary" href="#s4">비교 실험 보기 →</a><a class="button" href="보고서_1조.pdf">PDF 보고서</a><a class="button" href="https://github.com/hajin-eng/team1-deeplearning">GitHub 코드</a></div>
<div class="metrics">{metric_html}</div>
</div></section>
{''.join(out)}
</main>
<footer class="footer"><div class="container">1조 · 포스코 K-뉴딜아카데미 딥러닝 최종 프로젝트 · 이 페이지는 <code>보고서_1조.html</code>(PDF 원본)에서 <code>보고서_웹_만들기.py</code>로 자동 생성됩니다.</div></footer>
</body></html>"""
(DIR / "보고서_웹.html").write_text(page, encoding="utf-8")
print("저장:", DIR / "보고서_웹.html", f"({len(page):,}자)")
