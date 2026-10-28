"""보고서/보고서_1조.html(원본)을 발표 자료와 같은 분위기로 꾸며 보고서/보고서_1조.pdf로 만든다.

사용: python 보고서_PDF_만들기.py
- 원본 HTML의 <style>을 아래 CSS로 바꿔 저장하고(내용은 그대로), 긴 문단은 문장 단위로 나눈 인쇄용 사본을 Edge로 PDF 변환한다.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DIR = Path(__file__).resolve().parent / "보고서"
SRC = DIR / "보고서_1조.html"
PDF = DIR / "보고서_1조.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page{size:A4;margin:13mm 14mm 15mm;@bottom-right{content:counter(page);font:8pt "Noto Sans KR","Malgun Gothic",sans-serif;color:#7a8da0}}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact;box-sizing:border-box}
body{font-family:"Noto Sans KR","Pretendard","Malgun Gothic","맑은 고딕",sans-serif;font-size:10pt;line-height:1.75;color:#1b2b3a;margin:0;word-break:keep-all}
.hd{background:#0b3556;color:#fff;padding:24px 28px 20px;border-radius:12px;margin-bottom:12px;border-bottom:7px solid #8fd0f0}
.hd h1{font-size:21pt;margin:0 0 8px;letter-spacing:-.5px;color:#fff;line-height:1.3}
.hd .sub{color:#cfe6f6;font-size:9.2pt;margin:0;line-height:1.7}
h2{font-size:14pt;margin:26px 0 12px;padding:9px 18px;background:#0b5a8c;color:#fff;border-radius:9px;border-bottom:5px solid #8fd0f0;break-after:avoid;letter-spacing:-.3px}
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
tr:last-child td{border-bottom:0}
img{width:100%;display:block;margin:10px 0 3px;border:1px solid #cfe0ee;border-radius:10px;background:#fff;padding:5px;break-inside:avoid}
.cap{font-size:8.6pt;color:#5a6b7c;margin:0 0 12px;padding-left:3px;line-height:1.6;break-before:avoid}
.warn{background:#fdeee4;border:1px solid #f3c5a8;border-left:5px solid #e0480f;border-radius:9px;padding:10px 14px;margin:10px 0;break-inside:avoid}
.warn ul{margin-bottom:0}.warn li::before{background:#e0480f}
.cards{display:flex;gap:10px;margin:10px 0 14px}
.card{flex:1;background:#fff;border:2px solid #cfe3f0;border-radius:12px;padding:12px 10px;text-align:center}
.card .n{font-size:19pt;font-weight:800;color:#0b5a8c;line-height:1.25}.card .l{font-size:8.2pt;color:#5a6b7c;line-height:1.5;margin-top:3px}
"""

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


src = SRC.read_text(encoding="utf-8")
src = re.sub(r"<style>.*?</style>", lambda m: "<style>" + CSS + "</style>", src, count=1, flags=re.S)
SRC.write_text(src, encoding="utf-8", newline="\n")

body = split_sentences(src)
body = body.replace("<ul>", '<ul class="summary">', 1)
tmp = DIR / "_인쇄용.html"
tmp.write_text(body, encoding="utf-8")
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
