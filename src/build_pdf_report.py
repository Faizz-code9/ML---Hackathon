"""
PathSnap — Convert report/report.md to Publication-Grade PDF
src/build_pdf_report.py

Uses Python `markdown` + clean academic CSS + headless Chrome/Edge
to render report/PathSnap_Technical_Report.pdf with all embedded images.
"""

import re
import subprocess
import markdown
from pathlib import Path

REPORT_DIR = Path(__file__).parent.parent / "report"
MD_PATH = REPORT_DIR / "report.md"
HTML_PATH = REPORT_DIR / "PathSnap_Technical_Report.html"
PDF_PATH = REPORT_DIR / "PathSnap_Technical_Report.pdf"

md_text = MD_PATH.read_text(encoding="utf-8")

# Replace display/inline LaTeX with clean offline-safe HTML math so PDF renders cleanly without CDN
HMM_EQ_HTML = """
<div class="math-block">
  <b>P(s<sub>t</sub> = j | s<sub>t&minus;1</sub> = i) =</b>
  <table class="cases-table">
    <tr><td><b>0.85</b></td><td>if j = i &nbsp;(staying on the same arc at 1 Hz)</td></tr>
    <tr><td><b>0.90 / out_deg(v)</b></td><td>if to_node(i) = from_node(j) &nbsp;(direct 1-hop transition)</td></tr>
    <tr><td><b>0.25 / out_deg(v)</b></td><td>if reachable via 1 intermediate node &nbsp;(short arc skipped)</td></tr>
    <tr><td><b>0.10 / out_deg(v)</b></td><td>if reachable via 2 intermediate nodes</td></tr>
    <tr><td><b>10<sup>&minus;6</sup></b></td><td>otherwise &nbsp;(disconnected segment penalty)</td></tr>
  </table>
</div>
"""

md_text = re.sub(
    r"\$\$P\(s_t = j.*?\$\$" ,
    HMM_EQ_HTML.strip(),
    md_text,
    flags=re.DOTALL,
)

# Clean inline LaTeX expressions into crisp HTML
inline_replacements = {
    r"$t$": "<i>t</i>",
    r"$P(o_t \mid s_t = a) = \hat{p}_a^{(t)} = \text{model.predict\_proba}(x_{t,a})[:, 1]$":
        "<i>P</i>(<i>o<sub>t</sub></i> | <i>s<sub>t</sub></i> = <i>a</i>) = <i>p&#770;<sub>a</sub></i><sup>(<i>t</i>)</sup> = <code>model.predict_proba(x<sub>t,a</sub>)[:, 1]</code>",
    r"$10^{-12}$": "10<sup>&minus;12</sup>",
    r"$O(\text{out-degree})$": "<i>O</i>(out-degree)",
    r"$t_i - t_{i-1} > 5\text{ s}$": "<i>t<sub>i</sub></i> &minus; <i>t</i><sub><i>i</i>&minus;1</sub> &gt; 5 s",
    r"$\arg\max_a V_{t-1}(a)$": "arg max<sub><i>a</i></sub> <i>V</i><sub><i>t</i>&minus;1</sub>(<i>a</i>)",
    r"$\log 10^{-6} \approx -13.82$": "log(10<sup>&minus;6</sup>) &approx; &minus;13.82",
}
for k, v in inline_replacements.items():
    md_text = md_text.replace(k, v)

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc", "nl2br"],
)

CSS = """
@page {
    size: A4;
    margin: 14mm 14mm 15mm 14mm;
}
body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    font-size: 11.2pt;
    line-height: 1.55;
    color: #000000;
    max-width: 100%;
    margin: 0 auto;
}
h1 {
    font-size: 22pt;
    color: #000000;
    border-bottom: 2.5px solid #000000;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 6px;
}
h2 {
    font-size: 15.5pt;
    color: #000000;
    border-bottom: 1.4px solid #000000;
    padding-bottom: 3px;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
}
h3 {
    font-size: 13pt;
    color: #000000;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}
h4 {
    font-size: 11.8pt;
    color: #000000;
    margin-top: 11px;
    margin-bottom: 5px;
    page-break-after: avoid;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px 0;
    font-size: 9.4pt;
    color: #000000;
    page-break-inside: avoid;
}
th {
    background-color: #e5e7eb;
    color: #000000;
    font-weight: 700;
    text-align: left;
    padding: 6px 7px;
    border: 1px solid #000000;
}
td {
    padding: 5.5px 7px;
    border: 1px solid #333333;
    color: #000000;
    vertical-align: top;
}
tr:nth-child(even) td {
    background-color: #f9fafb;
}
img {
    display: block;
    max-width: 98%;
    max-height: 380px;
    width: auto;
    height: auto;
    margin: 10px auto 5px auto;
    border: 1px solid #333333;
    border-radius: 3px;
    page-break-inside: avoid;
}
p em {
    color: #000000;
    font-size: 9.8pt;
}
code {
    font-family: Consolas, 'Courier New', monospace;
    background: #f3f4f6;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9.6pt;
    color: #000000;
}
pre {
    background: #f3f4f6;
    color: #000000;
    border: 1px solid #333333;
    padding: 10px 12px;
    border-radius: 4px;
    font-size: 9.2pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code {
    background: transparent;
    color: #000000;
    padding: 0;
}
blockquote {
    border-left: 3.5px solid #000000;
    background: #f3f4f6;
    margin: 10px 0;
    padding: 7px 12px;
    font-size: 10.2pt;
    color: #000000;
}
.math-block {
    background: #f9fafb;
    border: 1px solid #000000;
    border-radius: 4px;
    padding: 9px 14px;
    margin: 10px 0;
    font-size: 10.2pt;
    color: #000000;
    page-break-inside: avoid;
}
.cases-table {
    width: auto;
    margin: 5px 0 0 14px;
    border: none;
    color: #000000;
}
.cases-table td {
    border: none !important;
    background: transparent !important;
    padding: 2px 10px 2px 0;
    color: #000000;
}
ul, ol {
    margin-top: 5px;
    margin-bottom: 9px;
    padding-left: 22px;
}
li {
    margin-bottom: 4px;
    color: #000000;
}
hr {
    border: none;
    border-top: 1px solid #000000;
    margin: 14px 0;
}
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>PathSnap — Technical Report</title>
  <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

HTML_PATH.write_text(full_html, encoding="utf-8")
print(f"✓ Wrote styled HTML: {HTML_PATH}")

# Render PDF using Chrome or Edge headless
browsers = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
browser = next((b for b in browsers if Path(b).exists()), None)
if not browser:
    raise RuntimeError("Neither Chrome nor Edge found for headless PDF export.")

cmd = [
    browser,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH.resolve().as_uri(),
]
res = subprocess.run(cmd, capture_output=True, text=True)
if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10_000:
    print(f"✅ PDF compiled successfully: {PDF_PATH} ({PDF_PATH.stat().st_size / 1e6:.2f} MB)")
else:
    print("STDERR:", res.stderr)
    raise RuntimeError("PDF generation failed.")
