"""
Convert Jupyter notebooks to TENSO-styled HTML pages (same style as getting-started.html).
Run:  python convert_notebooks.py
"""
import json, re, html as _h
from pathlib import Path

DOCS   = Path(__file__).parent.parent   # docs/
NB_DIR = Path(__file__).parent          # docs/notebooks/

NOTEBOOKS = [
    {
        "ipynb": "tree_vs_train_annotated.ipynb",
        "out":   "tree_vs_train_annotated.html",
        "title": "Spin-Boson: Train vs Tree — TENSO",
        "h1":    "Spin-Boson Model: Train vs Tree Tensor Network",
        "crumb": "Spin-Boson Example",
    },
    {
        "ipynb": "FMO_tutorial.ipynb",
        "out":   "FMO_tutorial.html",
        "title": "FMO Energy Transfer — TENSO",
        "h1":    "FMO Energy Transfer",
        "crumb": "FMO Example",
    },
    {
        "ipynb": "time_dependent_no_quantity.ipynb",
        "out":   "time_dependent_no_quantity.html",
        "title": "Laser-Driven Dynamics — TENSO",
        "h1":    "Laser-Driven Dynamics",
        "crumb": "Laser-Driven Example",
    },
]

NAV = """\
  <nav class="nav" role="navigation" aria-label="Main navigation">
    <a href="../index.html" class="nav__logo">TENSO</a>
    <ul class="nav__links" role="list">
      <li><a href="../index.html"           class="nav__link">Home</a></li>
      <li><a href="../installation.html"    class="nav__link">Install</a></li>
      <li><a href="../getting-started.html" class="nav__link">Getting Started</a></li>
      <li><a href="../background.html"      class="nav__link">Background</a></li>
      <li><a href="../structure.html"       class="nav__link">Structure</a></li>
      <li><a href="../examples.html"        class="nav__link">Examples</a></li>
      <li><a href="../api.html"             class="nav__link">API</a></li>
      <li><a href="../input-generator.html" class="nav__link">Input Generator</a></li>
      <li><a href="../cite.html"            class="nav__link">Cite</a></li>
      <li><a href="../contact.html"         class="nav__link">Contact</a></li>
    </ul>
    <div class="nav__actions">
      <button class="btn-icon" data-theme-toggle aria-label="Toggle theme">
        <span class="theme-toggle-icon">☀️</span>
      </button>
      <a href="https://github.com/ifgroup/pytenso" class="btn-icon" aria-label="GitHub"
         target="_blank" rel="noopener"><i class="fab fa-github"></i></a>
    </div>
    <button class="hamburger" aria-label="Toggle menu" aria-expanded="false">
      <span></span><span></span><span></span>
    </button>
  </nav>"""

FOOTER = """\
  <footer class="footer">
    <div class="footer__grid">
      <div>
        <div class="footer__brand-name">TENSO</div>
        <p class="footer__tagline">Exact open quantum dynamics via Tree Tensor Network HEOM</p>
        <p style="font-size:0.8rem; color: var(--text-muted);">Franco Group · University of Rochester</p>
      </div>
      <div>
        <div class="footer__col-title">Docs</div>
        <ul class="footer__links">
          <li><a href="../installation.html"    class="footer__link">Installation</a></li>
          <li><a href="../getting-started.html" class="footer__link">Getting Started</a></li>
          <li><a href="../background.html"      class="footer__link">Background</a></li>
          <li><a href="../structure.html"       class="footer__link">Structure</a></li>
        </ul>
      </div>
      <div>
        <div class="footer__col-title">More</div>
        <ul class="footer__links">
          <li><a href="../examples.html" class="footer__link">Examples</a></li>
          <li><a href="../api.html"      class="footer__link">API Reference</a></li>
          <li><a href="../cite.html"     class="footer__link">Cite</a></li>
          <li><a href="../contact.html"  class="footer__link">Contact</a></li>
        </ul>
      </div>
    </div>
    <div class="footer__bottom">
      <span class="footer__copyright">© 2025 Franco Group, University of Rochester. MIT License.</span>
    </div>
  </footer>"""

# ─────────────────────────────────────────────────────────────────────────────
# Markdown → HTML
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text):
    text = re.sub(r'\$.*?\$', '', text)           # strip inline math
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s_]+', '-', text).strip('-') or 'section'

def protect_math(text):
    """Replace math with placeholders so inline processing can't damage it."""
    placeholders = {}
    counter = [0]

    def store(m):
        key = f"\x00MATH{counter[0]}\x00"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    text = re.sub(r'\$\$[\s\S]+?\$\$', store, text)   # display math first
    text = re.sub(r'\$[^$\n]+?\$',     store, text)   # then inline math
    return text, placeholders

def restore_math(text, placeholders):
    for k, v in placeholders.items():
        text = text.replace(k, v)
    return text

def inline_fmt(text, ph):
    """Apply bold, italic, inline-code, links — math already protected."""
    text = re.sub(r'`([^`]+)`',                    r'<code>\1</code>',        text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*',            r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*',                r'<strong>\1</strong>',    text)
    text = re.sub(r'\*(.+?)\*',                    r'<em>\1</em>',            text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',       r'<a href="\2">\1</a>',    text)
    return restore_math(text, ph)

def parse_table(lines):
    """Convert GFM table lines to HTML."""
    rows = []
    for line in lines:
        if re.match(r'^\s*\|[-:| ]+\|\s*$', line):
            continue  # separator row
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rows.append(cells)
    if not rows:
        return ''
    head, body = rows[0], rows[1:]
    th = ''.join(f'<th>{c}</th>' for c in head)
    tbody_rows = ''.join(
        '<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
        for r in body
    )
    return (
        '<div class="table-wrapper">\n'
        f'<table><thead><tr>{th}</tr></thead>'
        f'<tbody>{tbody_rows}</tbody></table>\n'
        '</div>'
    )

def md_cell_to_html(md_text, headings):
    """Convert a markdown cell string to HTML, collecting h2 headings."""
    lines  = md_text.split('\n')
    output = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # ── Fenced display math ──────────────────────────────────────
        if line.strip().startswith('$$'):
            block = [line]
            i += 1
            while i < len(lines) and '$$' not in lines[i]:
                block.append(lines[i]); i += 1
            if i < len(lines):
                block.append(lines[i]); i += 1
            output.append('\n'.join(block))
            continue

        # ── ATX headings ─────────────────────────────────────────────
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            raw   = m.group(2).strip()
            text, ph = protect_math(raw)
            text = inline_fmt(text, ph)
            slug  = slugify(raw)
            if level == 2:
                headings.append((slug, re.sub(r'<[^>]+>', '', text)))
            output.append(f'<h{level} id="{slug}">{text}</h{level}>')
            i += 1
            continue

        # ── Blockquote → callout ─────────────────────────────────────
        if line.startswith('>'):
            bq_lines = []
            while i < len(lines) and lines[i].startswith('>'):
                bq_lines.append(lines[i].lstrip('> ').rstrip())
                i += 1
            inner = ' '.join(bq_lines)
            text, ph = protect_math(inner)
            text = inline_fmt(text, ph)
            output.append(
                '<div class="callout callout--tip">'
                '<span class="callout__icon"><i class="fa fa-lightbulb"></i></span>'
                f'<div>{text}</div></div>'
            )
            continue

        # ── GFM table ────────────────────────────────────────────────
        if '|' in line and re.match(r'^\s*\|', line):
            tbl_lines = []
            while i < len(lines) and '|' in lines[i]:
                tbl_lines.append(lines[i]); i += 1
            output.append(parse_table(tbl_lines))
            continue

        # ── Unordered list ───────────────────────────────────────────
        if re.match(r'^[-*+]\s', line):
            items = []
            while i < len(lines) and re.match(r'^[-*+]\s', lines[i]):
                item_text = lines[i][2:].strip()
                text, ph  = protect_math(item_text)
                items.append(f'<li>{inline_fmt(text, ph)}</li>')
                i += 1
            output.append('<ul>' + ''.join(items) + '</ul>')
            continue

        # ── Ordered list ─────────────────────────────────────────────
        if re.match(r'^\d+\.\s', line):
            items = []
            while i < len(lines) and re.match(r'^\d+\.\s', lines[i]):
                item_text = re.sub(r'^\d+\.\s', '', lines[i]).strip()
                text, ph  = protect_math(item_text)
                items.append(f'<li>{inline_fmt(text, ph)}</li>')
                i += 1
            output.append('<ol>' + ''.join(items) + '</ol>')
            continue

        # ── Horizontal rule ──────────────────────────────────────────
        if re.match(r'^-{3,}$', line.strip()):
            output.append('<hr>')
            i += 1
            continue

        # ── Blank line ───────────────────────────────────────────────
        if not line.strip():
            i += 1
            continue

        # ── Paragraph (gather until blank or structural line) ────────
        para_lines = []
        while i < len(lines):
            l = lines[i]
            if (not l.strip()
                    or re.match(r'^#{1,4}\s', l)
                    or l.startswith('>')
                    or re.match(r'^[-*+]\s', l)
                    or re.match(r'^\d+\.\s', l)
                    or ('|' in l and re.match(r'^\s*\|', l))
                    or l.strip().startswith('$$')):
                break
            para_lines.append(l)
            i += 1
        raw_para = ' '.join(para_lines).strip()
        if raw_para:
            text, ph = protect_math(raw_para)
            text = inline_fmt(text, ph)
            output.append(f'<p>{text}</p>')

    return '\n'.join(output)

# ─────────────────────────────────────────────────────────────────────────────
# Code cell rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_code_cell(src_lines, outputs):
    src = ''.join(src_lines).rstrip()
    escaped = _h.escape(src)
    html = (
        '<div class="code-block reveal">\n'
        f'<pre><code class="language-python">{escaped}</code></pre>\n'
        '</div>'
    )

    for out in outputs:
        otype = out.get('output_type', '')

        # Image output (matplotlib plots)
        if otype in ('display_data', 'execute_result'):
            data = out.get('data', {})
            if 'image/png' in data:
                b64 = data['image/png']
                if isinstance(b64, list):
                    b64 = ''.join(b64)
                html += (
                    '\n<figure class="nb-figure">'
                    f'<img src="data:image/png;base64,{b64.strip()}" '
                    'alt="Simulation output plot" '
                    'style="max-width:100%; border-radius:8px; '
                    'background:#fff; padding:6px; '
                    'box-shadow:0 4px 20px rgba(0,0,0,0.4);">'
                    '</figure>'
                )

        # Short plain-text output (≤ 4 lines, skip long verbose sim logs)
        elif otype in ('stream', 'execute_result'):
            text_data = out.get('text', out.get('data', {}).get('text/plain', []))
            if isinstance(text_data, list):
                text_data = ''.join(text_data)
            lines = text_data.strip().splitlines()
            if 1 <= len(lines) <= 4:
                escaped_out = _h.escape(text_data.strip())
                html += (
                    '\n<div class="nb-output">'
                    f'<pre>{escaped_out}</pre>'
                    '</div>'
                )

    return html

# ─────────────────────────────────────────────────────────────────────────────
# Page assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_toc(headings):
    if not headings:
        return ''
    items = ''.join(
        f'<li class="toc__item">'
        f'<a href="#{slug}" class="toc__link">{text}</a></li>\n'
        for slug, text in headings
    )
    return (
        '<aside class="sidebar">\n'
        '  <nav class="toc" aria-label="Table of contents">\n'
        '    <div class="toc__title">On this page</div>\n'
        f'    <ul class="toc__list">\n{items}    </ul>\n'
        '  </nav>\n'
        '</aside>'
    )

EXTRA_CSS = """\
  <style>
    /* ── Notebook output blocks ─────────────────────────── */
    .nb-output {
      background: rgba(0,10,35,0.55);
      border-left: 3px solid rgba(255,216,43,0.3);
      border-radius: 0 6px 6px 0;
      margin: 0.25rem 0 0.75rem;
      padding: 0.6rem 1rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      color: var(--text-secondary);
    }
    .nb-output pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
    .nb-figure { margin: 1.5rem 0; text-align: center; }
    .nb-figure img { max-width: min(100%, 680px); }
  </style>"""

def build_page(info, nb_dir):
    ipynb_path = nb_dir / info["ipynb"]
    nb = json.loads(ipynb_path.read_text(encoding='utf-8'))
    cells = nb.get('cells', [])

    headings = []   # collected (slug, display_text) tuples
    body_parts = []
    first_h1_skipped = False  # the notebook's own H1 is replaced by our <h1>

    for cell in cells:
        ctype = cell.get('cell_type', '')
        src   = cell.get('source', [])
        if isinstance(src, list):
            src_str = ''.join(src)
        else:
            src_str = src

        if ctype == 'markdown':
            # Skip the notebook's own top-level H1 (we render it separately)
            if not first_h1_skipped and src_str.startswith('# '):
                first_h1_skipped = True
                continue
            body_parts.append(md_cell_to_html(src_str, headings))

        elif ctype == 'code':
            src_lines = cell.get('source', [])
            if isinstance(src_lines, str):
                src_lines = [src_lines]
            outputs = cell.get('outputs', [])
            if ''.join(src_lines).strip():   # skip empty cells
                body_parts.append(render_code_cell(src_lines, outputs))

    toc_html  = build_toc(headings)
    body_html = '\n'.join(body_parts)

    page = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_h.escape(info['title'])}</title>
  <link rel="stylesheet" href="../assets/css/main.css">
  <link rel="stylesheet" href="../assets/css/prism-tenso.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
  <script>MathJax = {{ tex: {{ inlineMath: [['$','$']] }}, svg: {{ fontCache: 'global' }} }};</script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" defer></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" defer></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js" defer></script>
{EXTRA_CSS}
</head>
<body>
  <div class="progress-bar"></div>
{NAV}

  <div class="page-layout">
{toc_html}

    <main class="article" id="main-content">
      <nav class="breadcrumb" aria-label="Breadcrumb">
        <a href="../index.html">Home</a>
        <span class="breadcrumb__sep">/</span>
        <a href="../examples.html">Examples</a>
        <span class="breadcrumb__sep">/</span>
        <span class="breadcrumb__current">{_h.escape(info['crumb'])}</span>
      </nav>

      <h1>{_h.escape(info['h1'])}</h1>

{body_html}

      <div style="margin-top:3rem;">
        <a href="../examples.html" class="btn btn--secondary">
          <i class="fa fa-arrow-left"></i> Back to Examples
        </a>
      </div>
    </main>
  </div>

{FOOTER}

  <script src="../assets/js/search-index.js"></script>
  <script src="../assets/js/main.js"></script>
</body>
</html>"""

    out_path = nb_dir / info["out"]
    out_path.write_text(page, encoding='utf-8')
    print(f"  Written: {info['out']}  ({len(page)//1024} KB)")


if __name__ == '__main__':
    print("Converting notebooks to TENSO-styled HTML...")
    for info in NOTEBOOKS:
        build_page(info, NB_DIR)
    print("Done.")
