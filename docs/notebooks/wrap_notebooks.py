"""
Convert notebooks to static HTML and wrap them in the TENSO site shell.
Run:  python wrap_notebooks.py
This always re-runs nbconvert first so the HTML is a clean slate.
"""
import re
import subprocess
import sys
from pathlib import Path

JUPYTER = r"C:\Users\night\anaconda3\Scripts\jupyter.exe"

NOTEBOOKS = [
    {
        "ipynb": "tree_vs_train_annotated.ipynb",
        "src":   "tree_vs_train_annotated.html",
        "title": "Spin-Boson: Train vs Tree — TENSO",
        "heading": "Spin-Boson Model: Train vs Tree Tensor Network Comparison",
    },
    {
        "ipynb": "FMO_tutorial.ipynb",
        "src":   "FMO_tutorial.html",
        "title": "FMO Complex — TENSO",
        "heading": "The Fenna–Matthews–Olson (FMO) Complex",
    },
    {
        "ipynb": "time_dependent_no_quantity.ipynb",
        "src":   "time_dependent_no_quantity.html",
        "title": "Laser-Driven Dynamics — TENSO",
        "heading": "Laser-Driven Dynamics",
    },
]

NAV_HTML = """\
  <nav class="nav" role="navigation" aria-label="Main navigation">
    <a href="../index.html" class="nav__logo">TENSO</a>
    <ul class="nav__links" role="list">
      <li><a href="../index.html"          class="nav__link">Home</a></li>
      <li><a href="../installation.html"   class="nav__link">Install</a></li>
      <li><a href="../getting-started.html" class="nav__link">Getting Started</a></li>
      <li><a href="../background.html"     class="nav__link">Background</a></li>
      <li><a href="../structure.html"      class="nav__link">Structure</a></li>
      <li><a href="../examples.html"       class="nav__link">Examples</a></li>
      <li><a href="../api.html"            class="nav__link">API</a></li>
      <li><a href="../input-generator.html" class="nav__link">Input Generator</a></li>
      <li><a href="../cite.html"           class="nav__link">Cite</a></li>
      <li><a href="../contact.html"        class="nav__link">Contact</a></li>
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

# CSS injected AFTER the lab template CSS so our overrides win
DARK_OVERRIDE_CSS = """\
<style id="tenso-nb-overrides">
/* ── TENSO dark theme overrides for rendered notebooks ──────────────── */
html, body {
  background: #000b22 !important;
  color: #f0f4ff !important;
  font-family: 'Inter', 'Space Grotesk', sans-serif;
}

/* Remove lab template's own background */
#main-container, .jp-Notebook,
.jp-NotebookPanel, .jp-NotebookPanel-notebook {
  background: transparent !important;
}

/* ── Notebook back-bar ────────────────────────────────────────────── */
.nb-back-bar {
  max-width: 1000px;
  margin: 0 auto;
  padding: 1.25rem 1.5rem 0;
}
.nb-back-bar a {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.875rem;
  color: #FFD82B;
  text-decoration: none;
  border: 1px solid rgba(255,216,43,0.35);
  border-radius: 8px;
  padding: 0.4rem 0.9rem;
  transition: background 0.2s, border-color 0.2s;
}
.nb-back-bar a:hover {
  background: rgba(255,216,43,0.1);
  border-color: rgba(255,216,43,0.7);
}

/* ── Notebook wrapper ─────────────────────────────────────────────── */
.nb-content-wrap {
  max-width: 1000px;
  margin: 0 auto;
  padding: 1.5rem 1.5rem 4rem;
}

/* ── Cells ────────────────────────────────────────────────────────── */
.jp-Cell {
  background: rgba(0,30,95,0.22) !important;
  border: 1px solid rgba(255,216,43,0.08) !important;
  border-radius: 10px !important;
  padding: 0.5rem !important;
  margin: 0.5rem 0 !important;
}
.jp-Cell:hover {
  border-color: rgba(255,216,43,0.18) !important;
}
.jp-MarkdownCell .jp-Cell-inputWrapper,
.jp-MarkdownCell .jp-InputPrompt {
  background: transparent !important;
}

/* ── Markdown text ───────────────────────────────────────────────── */
.jp-RenderedHTMLCommon,
.jp-RenderedMarkdown,
.jp-RenderedHTMLCommon p,
.jp-RenderedHTMLCommon li {
  color: #f0f4ff !important;
}
.jp-RenderedHTMLCommon h1 { color: #f0f4ff !important; font-size: 1.9rem; margin: 1.5rem 0 0.75rem; }
.jp-RenderedHTMLCommon h2 { color: #f0f4ff !important; font-size: 1.4rem; margin: 1.25rem 0 0.6rem; border-bottom: 1px solid rgba(255,216,43,0.2); padding-bottom: 0.3rem; }
.jp-RenderedHTMLCommon h3 { color: #B7D3FF !important; font-size: 1.15rem; margin: 1rem 0 0.5rem; }
.jp-RenderedHTMLCommon a  { color: #FFD82B !important; text-decoration: underline; }
.jp-RenderedHTMLCommon strong { color: #FFE95F !important; }

/* ── Tables — plain markdown style ───────────────────────────────── */
.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  color: #f0f4ff !important;
  margin: 1rem 0;
  width: 100%;
  background: transparent !important;
}
.jp-RenderedHTMLCommon th {
  background: transparent !important;
  color: #f0f4ff !important;
  border: none !important;
  border-bottom: 2px solid rgba(255,255,255,0.3) !important;
  padding: 0.45rem 0.8rem;
  text-align: left;
  font-weight: 600;
}
.jp-RenderedHTMLCommon td {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid rgba(255,255,255,0.1) !important;
  padding: 0.4rem 0.8rem;
  color: #f0f4ff !important;
}

/* ── Code / editor area ─────────────────────────────────────────── */
.jp-InputArea-editor,
.CodeMirror, .cm-editor {
  background: #000d2e !important;
  border-radius: 6px !important;
}
.CodeMirror, .cm-content, .cm-line {
  color: #f0f4ff !important;
}
.jp-InputPrompt, .jp-OutputPrompt {
  color: #FFD82B !important;
  font-size: 0.78rem;
  opacity: 0.7;
}

/* ── Outputs ──────────────────────────────────────────────────────── */
.jp-OutputArea-output {
  background: rgba(0,10,35,0.55) !important;
  border-radius: 6px;
  color: #f0f4ff !important;
}

/* Cap verbose text output (stdout/stderr) and make it scrollable */
.jp-RenderedText {
  max-height: 200px;
  overflow-y: auto;
  border-radius: 4px;
}
.jp-RenderedText pre {
  color: #B7D3FF !important;
  font-size: 0.82rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}

/* Hide the floating Copy buttons on output areas — they're noise here */
.jp-OutputArea .jp-CopyButton,
.jp-OutputArea .jp-Button[data-commandlinker-command="copyToClipboard"],
.jp-OutputArea button[title="Copy to Clipboard"],
.jp-RenderedText .jp-CopyButton { display: none !important; }

/* ── Math (MathJax) ───────────────────────────────────────────────── */
.MathJax { color: #f0f4ff !important; }

/* ── Images from outputs ─────────────────────────────────────────── */
.jp-RenderedImage img,
.jp-OutputArea-output img {
  max-width: 100%;
  border-radius: 8px;
  background: #fff;          /* keep plot backgrounds white */
  padding: 4px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}

/* ── Scrollbars ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #000b22; }
::-webkit-scrollbar-thumb { background: rgba(255,216,43,0.3); border-radius: 3px; }
</style>"""

HEAD_EXTRAS = """\
  <link rel="stylesheet" href="../assets/css/main.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">"""


def nbconvert(info: dict, nb_dir: Path):
    """Run nbconvert to produce a fresh, unwrapped HTML from the .ipynb."""
    ipynb = nb_dir / info["ipynb"]
    print(f"  Converting {info['ipynb']} ...")
    result = subprocess.run(
        [JUPYTER, "nbconvert", "--to", "html", "--template", "lab",
         "--embed-images", "--no-prompt", "--output-dir", str(nb_dir), str(ipynb)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def process(info: dict, nb_dir: Path):
    src_path = nb_dir / info["src"]
    html = src_path.read_text(encoding="utf-8")

    # ── 1. Swap <title> ─────────────────────────────────────────────
    html = re.sub(r'<title>.*?</title>', f'<title>{info["title"]}</title>', html, flags=re.DOTALL)

    # ── 2. Inject HEAD extras (TENSO CSS + FontAwesome) ─────────────
    html = html.replace("</head>", f"{HEAD_EXTRAS}\n{DARK_OVERRIDE_CSS}\n</head>")

    # ── 3. Add data-theme="dark" to <html> ──────────────────────────
    html = re.sub(r'<html([^>]*)>', r'<html\1 data-theme="dark">', html, count=1)

    # ── 4. Inject nav + back bar after <body …> ──────────────────────
    # Find opening <body> tag
    body_match = re.search(r'<body[^>]*>', html)
    if body_match:
        insert_pos = body_match.end()
        injection = (
            f'\n  <div class="progress-bar"></div>\n'
            f'{NAV_HTML}\n'
            f'  <div class="nb-back-bar">'
            f'<a href="../examples.html">&#8592; Back to Examples</a></div>\n'
            f'  <div class="nb-content-wrap">\n'
        )
        html = html[:insert_pos] + injection + html[insert_pos:]

    # ── 5. Close the nb-content-wrap before </body> ──────────────────
    html = html.replace("</body>", '  </div><!-- /.nb-content-wrap -->\n  <script src="../assets/js/main.js"></script>\n</body>')

    # ── 6. Write output (overwrite the nbconvert file in-place) ──────
    src_path.write_text(html, encoding="utf-8")
    print(f"  Wrapped: {info['src']}")


if __name__ == "__main__":
    nb_dir = Path(__file__).parent
    print("Step 1/2 — nbconvert (fresh HTML from .ipynb)...")
    for info in NOTEBOOKS:
        nbconvert(info, nb_dir)
    print("Step 2/2 — Wrapping in TENSO shell...")
    for info in NOTEBOOKS:
        process(info, nb_dir)
    print("Done.")
