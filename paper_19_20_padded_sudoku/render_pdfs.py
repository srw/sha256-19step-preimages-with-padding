#!/usr/bin/env python3
"""
Converts PAPER_19_20_PADDED_SHA256.md and COMPANION_SUDOKU_ELI5_ELIU.md
into publication-typeset HTML + KaTeX and renders PDFs via Headless Chrome.
"""

import html
import os
import re
import subprocess
import sys

def clean_tex(tex):
    tex = tex.replace(r"\;\middle|\;", r"\;\vert\;")
    tex = tex.replace(r"\middle|", r"\vert ")
    tex = tex.replace(r"\rotateleft", r"\mathbin{\lll}")
    tex = re.sub(r"`(\[E\d+\])`", r"\\text{\1}", tex)
    return tex

def md_to_html_body(md_text):
    placeholders = []
    def save_placeholder(content):
        idx = len(placeholders)
        placeholders.append(content)
        return f"@@PLACEHOLDER_{idx}@@"

    # 1. Code blocks ```lang ... ```
    def repl_codeblock(m):
        lang = m.group(1).strip()
        code = html.escape(m.group(2))
        return save_placeholder(f'<pre class="code-block"><code class="lang-{lang}">{code}</code></pre>')

    md_text = re.sub(r"```([^\n]*)\n(.*?)```", repl_codeblock, md_text, flags=re.DOTALL)

    # 2. Display math $$...$$ (Process BEFORE inline code so backticks inside math don't break)
    def repl_display_math(m):
        tex = clean_tex(m.group(1).strip())
        return save_placeholder(f'<div class="math-display">\\[{html.escape(tex)}\\]</div>')

    md_text = re.sub(r"\$\$(.*?)\$\$", repl_display_math, md_text, flags=re.DOTALL)

    # 3. Inline code `...`
    def repl_inline_code(m):
        code = html.escape(m.group(1))
        return save_placeholder(f'<code>{code}</code>')

    md_text = re.sub(r"`([^`\n]+)`", repl_inline_code, md_text)

    # 4. Inline math $...$
    def repl_inline_math(m):
        tex = clean_tex(m.group(1))
        return save_placeholder(f'<span class="math-inline">\\({html.escape(tex)}\\)</span>')

    md_text = re.sub(r"(?<!\\)\$([^\$\n]+?)\$", repl_inline_math, md_text)

    def format_inline(text):
        def repl_link(m):
            label = m.group(1)
            url = m.group(2)
            if url.startswith("file://"):
                return f'<span class="file-ref">{label}</span>'
            return f'<a href="{html.escape(url)}">{label}</a>'
        text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", repl_link, text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
        return text

    lines = md_text.splitlines()
    out = []
    i = 0
    in_ul = False
    in_ol = False
    in_blockquote = False

    def close_lists():
        nonlocal in_ul, in_ol, in_blockquote
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False
        if in_blockquote:
            out.append("</blockquote>")
            in_blockquote = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_lists()
            i += 1
            continue

        if re.match(r"^---+$", stripped):
            close_lists()
            out.append("<hr/>")
            i += 1
            continue

        m_head = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m_head:
            close_lists()
            level = len(m_head.group(1))
            title = format_inline(m_head.group(2))
            out.append(f"<h{level}>{title}</h{level}>")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-\|]+\|\s*$", lines[i + 1]):
            close_lists()
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            align_row = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
            aligns = []
            for a in align_row:
                if a.startswith(":") and a.endswith(":"):
                    aligns.append("center")
                elif a.endswith(":"):
                    aligns.append("right")
                else:
                    aligns.append("left")
            out.append('<div class="table-wrap"><table><thead><tr>')
            for idx_c, h in enumerate(headers):
                al = aligns[idx_c] if idx_c < len(aligns) else "left"
                out.append(f'<th style="text-align:{al}">{format_inline(h)}</th>')
            out.append("</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>")
                for idx_c, cell in enumerate(row_cells):
                    al = aligns[idx_c] if idx_c < len(aligns) else "left"
                    out.append(f'<td style="text-align:{al}">{format_inline(cell)}</td>')
                out.append("</tr>")
                i += 1
            out.append("</tbody></table></div>")
            continue

        if stripped.startswith(">"):
            if in_ul or in_ol:
                close_lists()
            if not in_blockquote:
                out.append("<blockquote>")
                in_blockquote = True
            bq_text = re.sub(r"^>\s?", "", stripped)
            out.append(f"<p>{format_inline(bq_text)}</p>")
            i += 1
            continue

        m_ul = re.match(r"^(\s*)[-\*]\s+(.*)$", line)
        if m_ul:
            if in_ol or in_blockquote:
                close_lists()
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            item_lines = [m_ul.group(2).strip()]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and not re.match(r"^\s*[-\*\d]+[\.\)]?\s+", lines[i]):
                if lines[i].strip():
                    item_lines.append(lines[i].strip())
                i += 1
            out.append(f"<li>{format_inline(' '.join(item_lines))}</li>")
            continue

        m_ol = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m_ol:
            num = int(m_ol.group(2))
            if in_ul or in_blockquote:
                close_lists()
            if not in_ol:
                out.append(f'<ol start="{num}">')
                in_ol = True
            item_lines = [m_ol.group(3).strip()]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and not re.match(r"^\s*[-\*\d]+[\.\)]?\s+", lines[i]):
                if lines[i].strip():
                    item_lines.append(lines[i].strip())
                i += 1
            out.append(f"<li>{format_inline(' '.join(item_lines))}</li>")
            continue

        if re.match(r"^@@PLACEHOLDER_\d+@@$", stripped):
            close_lists()
            out.append(stripped)
            i += 1
            continue

        close_lists()
        para_lines = [stripped]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].strip().startswith("#")
            and not lines[i].strip().startswith("|")
            and not lines[i].strip().startswith(">")
            and not re.match(r"^---+$", lines[i].strip())
            and not re.match(r"^\s*[-\*]\s+", lines[i])
            and not re.match(r"^\s*\d+\.\s+", lines[i])
            and not re.match(r"^@@PLACEHOLDER_\d+@@$", lines[i].strip())
        ):
            para_lines.append(lines[i].strip())
            i += 1
        para_html = format_inline(" ".join(para_lines))
        out.append(f"<p>{para_html}</p>")

    close_lists()
    body = "\n".join(out)

    for _ in range(3):
        def restore(m):
            idx = int(m.group(1))
            return placeholders[idx]
        body = re.sub(r"@@PLACEHOLDER_(\d+)@@", restore, body)

    return body


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{
    delimiters: [
      {{left: '\\\\[', right: '\\\\]', display: true}},
      {{left: '\\\\(', right: '\\\\)', display: false}}
    ],
    throwOnError: false
  }});"></script>
<style>
  @page {{
    size: letter;
    margin: 18mm 18mm 20mm 18mm;
  }}
  body {{
    font-family: "Times New Roman", "CMU Serif", Georgia, Cambria, serif;
    font-size: 10.5pt;
    line-height: 1.46;
    color: #111;
    max-width: 100%;
    margin: 0;
    padding: 0;
    text-align: justify;
    hyphens: auto;
  }}
  h1 {{
    font-size: 16.5pt;
    line-height: 1.25;
    text-align: center;
    margin-top: 0;
    margin-bottom: 10pt;
    font-weight: 700;
    color: #000;
  }}
  h2 {{
    font-size: 12.8pt;
    line-height: 1.3;
    margin-top: 15pt;
    margin-bottom: 6pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 2pt;
    color: #000;
    page-break-after: avoid;
    text-align: left;
  }}
  h3 {{
    font-size: 11pt;
    line-height: 1.3;
    margin-top: 11pt;
    margin-bottom: 4pt;
    color: #111;
    page-break-after: avoid;
    text-align: left;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 7pt;
  }}
  ul, ol {{
    margin-top: 2pt;
    margin-bottom: 8pt;
    padding-left: 18pt;
  }}
  li {{
    margin-bottom: 3.5pt;
  }}
  hr {{
    border: none;
    border-top: 1px solid #bbb;
    margin: 12pt 0;
  }}
  code {{
    font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 8.8pt;
    background-color: #f4f5f7;
    padding: 0.8pt 2.5pt;
    border-radius: 2pt;
  }}
  pre.code-block {{
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 3pt;
    padding: 7pt 9pt;
    overflow-x: auto;
    font-size: 8.2pt;
    line-height: 1.35;
    margin: 8pt 0;
    page-break-inside: avoid;
    text-align: left;
  }}
  pre.code-block code {{
    background: none;
    padding: 0;
    font-size: 8.2pt;
  }}
  .math-display {{
    margin: 8pt 0;
    overflow-x: visible;
    page-break-inside: avoid;
    font-size: 9.8pt;
  }}
  .table-wrap {{
    margin: 9pt 0;
    page-break-inside: avoid;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.6pt;
    line-height: 1.3;
  }}
  th, td {{
    border: 1px solid #999;
    padding: 4pt 5pt;
    vertical-align: top;
  }}
  th {{
    background-color: #efefef;
    font-weight: 700;
  }}
  blockquote {{
    margin: 8pt 0;
    padding: 6pt 10pt;
    border-left: 3pt solid #444;
    background-color: #f9f9fb;
  }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

def count_pdf_pages(pdf_path):
    with open(pdf_path, "rb") as f:
        data = f.read()
    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches)

def build_pdf(md_path, html_path, pdf_path, title):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    body = md_to_html_body(md_text)
    full_html = HTML_TEMPLATE.format(title=title, body=body)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        "--virtual-time-budget=5000",
        f"file://{os.path.abspath(html_path)}",
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    pages = count_pdf_pages(pdf_path)
    size_kb = os.path.getsize(pdf_path) / 1024.0
    print(f"Rendered {os.path.basename(pdf_path)}: {pages} pages ({size_kb:.1f} KB)")
    return pages

if __name__ == "__main__":
    base = "/Users/srw/Code/sha-check-eprint"
    p1 = build_pdf(
        os.path.join(base, "PAPER_19_20_PADDED_SHA256.md"),
        os.path.join(base, "PAPER_19_20_PADDED_SHA256.html"),
        os.path.join(base, "PAPER_19_20_PADDED_SHA256.pdf"),
        "Practical Preimages for 19- and 20-Step Reduced SHA-256: Exact Algebraic Padding Inversion, Context Shaping, and the Carry/Slot-Sudoku Framework",
    )
    p2 = build_pdf(
        os.path.join(base, "COMPANION_SUDOKU_ELI5_ELIU.md"),
        os.path.join(base, "COMPANION_SUDOKU_ELI5_ELIU.html"),
        os.path.join(base, "COMPANION_SUDOKU_ELI5_ELIU.pdf"),
        "Carry-Sudoku & Slot-Sudoku in Reduced SHA-256: A Self-Contained ELI5 and ELI-Undergraduate Companion Guide",
    )
    print(f"TOTAL PAGES: Paper = {p1} pages, Companion = {p2} pages, Combined = {p1 + p2} pages")
