"""
app.py — Local Web UI for Supreme Court RAG Pipeline
Two features:
  Tab 1 — Legal Q&A search over 919,518 indexed chunks
  Tab 2 — Upload any judgment PDF and get a structured summary

Run:
    pip install gradio pymupdf
    python app.py

Then open: http://localhost:7860
"""

import re
import sys
import time
import logging
import requests
from pathlib import Path

# ── Suppress noisy logs ───────────────────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# ── Ollama config (reuse from retrieve.py) ────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434"
OLLAMA_MODEL        = "saullm-legal"
LLM_TEMPERATURE     = 0.1

# ── Lazy-load the retrieval pipeline ─────────────────────────────────────────
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from retrieve import LegalRetriever
        _pipeline = LegalRetriever(llm_backend="ollama", ollama_model=OLLAMA_MODEL)
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str, max_chars: int = 12000) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Truncates to max_chars to stay within LLM context window.

    PyMuPDF is the most reliable extractor for court judgment PDFs —
    handles scanned+OCR, multi-column layouts, and footnotes better
    than pdfplumber or pypdf.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Run:\n  pip install pymupdf"
        )

    doc  = fitz.open(pdf_path)
    text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        if len(text) >= max_chars:
            break

    doc.close()

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated for context window ...]"

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Summarization via Ollama
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY_SYSTEM_PROMPT = (
    "You are an expert legal analyst specializing in Supreme Court of India "
    "judgments. When given a judgment text, produce a structured legal summary "
    "with these exact sections:\n\n"
    "**CASE NAME & CITATION:** Full case name and citation if visible.\n\n"
    "**BENCH:** Names of judges on the bench.\n\n"
    "**FACTS:** Key facts of the case in 3-5 sentences.\n\n"
    "**ISSUES:** Legal issues/questions framed by the court.\n\n"
    "**HELD:** What the court decided — the ratio decidendi.\n\n"
    "**LEGAL PRINCIPLES:** Key legal principles laid down or applied.\n\n"
    "**ACTS & PROVISIONS:** Specific Acts, Sections, and Articles cited.\n\n"
    "**SIGNIFICANCE:** Why this judgment matters — its precedential value.\n\n"
    "Be precise, cite specific provisions, and use legal terminology correctly."
)


def summarize_via_ollama(judgment_text: str) -> str:
    """
    Send extracted judgment text to saullm-legal via Ollama
    and return a structured summary.
    """
    prompt = (
        f"<s>[INST] <<SYS>>\n"
        f"{SUMMARY_SYSTEM_PROMPT}\n"
        f"<</SYS>>\n\n"
        f"Please summarize the following Supreme Court judgment:\n\n"
        f"{judgment_text}\n\n"
        f"[/INST]\n\n"
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    "temperature" : LLM_TEMPERATURE,
                    "num_predict" : 800,      # summary can be longer
                    "stop"        : ["</s>", "[INST]"],
                },
            },
            timeout=180,     # summarization takes longer than Q&A
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Start it with:\n  ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama timed out during summarization. "
            "The PDF may be too long. Try a shorter document."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio handler functions
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "What is the test for granting anticipatory bail under Section 438?",
    "Does the right to life under Article 21 include the right to livelihood?",
    "What are the conditions for granting interim injunction?",
    "What writs can the Supreme Court issue under Article 32?",
    "What did the court hold in Maneka Gandhi vs Union of India?",
    "What is the doctrine of promissory estoppel in Indian law?",
    "What is the scope of judicial review of administrative action?",
    "Can a High Court interfere with CBI investigation under Article 226?",
]


def run_query(query: str, num_sources: int):
    """Tab 1 — Legal Q&A over indexed corpus."""
    if not query or not query.strip():
        return "Please enter a query.", "", ""

    query = query.strip()
    t0    = time.time()

    try:
        pipeline = _get_pipeline()
        result   = pipeline.retrieve(query)
    except Exception as e:
        return f"❌ Error: {e}", "", ""

    elapsed = time.time() - t0

    # Format answer
    answer = result.answer or "No answer generated."

    # Format sources
    sources_md = ""
    for i, src in enumerate(result.sources[:int(num_sources)], 1):
        preview = src.get("text_preview", "")[:300].replace("\n", " ").strip()
        sources_md += (
            f"**[{i}] {src['case_title']}** ({src['year']})\n"
            f"📅 {src['judgment_date']} &nbsp;|&nbsp; "
            f"🎯 Rerank: `{src['rerank_score']:.4f}` &nbsp;|&nbsp; "
            f"📂 `{src['source']}`\n\n"
            f"> {preview}…\n\n---\n\n"
        )

    # Format latency
    lms = result.latency_ms
    latency_md = (
        f"⏱ **Total: {elapsed:.1f}s** &nbsp;|&nbsp; "
        f"Encode: {lms.get('encode_ms',0)}ms &nbsp;|&nbsp; "
        f"Dense: {lms.get('dense_ms',0)}ms &nbsp;|&nbsp; "
        f"Sparse: {lms.get('sparse_ms',0)}ms &nbsp;|&nbsp; "
        f"Rerank: {lms.get('rerank_ms',0)}ms &nbsp;|&nbsp; "
        f"LLM: {lms.get('llm_ms',0)}ms"
    )

    return answer, sources_md, latency_md


def run_pdf_summary(pdf_file, summary_mode: str):
    """
    Tab 2 — Upload a judgment PDF and generate a structured summary.

    summary_mode options:
      Full Summary     — all 8 sections
      Quick Brief      — just Facts + Held + Principles (shorter)
      Key Provisions   — focus on Acts, Sections, Articles cited
    """
    if pdf_file is None:
        return "Please upload a PDF file.", "", ""

    t0 = time.time()

    try:
        # Extract text
        pdf_path    = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
        pdf_name    = Path(pdf_path).name

        # Adjust max_chars based on mode
        max_chars   = 12000 if summary_mode == "Full Summary" else 7000
        text        = extract_text_from_pdf(pdf_path, max_chars=max_chars)

        char_count  = len(text)
        word_count  = len(text.split())

    except ImportError as e:
        return str(e), "", ""
    except Exception as e:
        return f"❌ PDF extraction failed: {e}", "", ""

    if not text.strip():
        return "❌ Could not extract text from this PDF. It may be a scanned image without OCR.", "", ""

    # Adjust system prompt based on mode
    global SUMMARY_SYSTEM_PROMPT
    if summary_mode == "Quick Brief":
        prompt_override = (
            "You are an expert legal analyst. Provide a BRIEF summary of this "
            "Supreme Court judgment covering only:\n\n"
            "**CASE NAME:** Full case name.\n\n"
            "**FACTS:** Key facts in 2-3 sentences.\n\n"
            "**HELD:** What the court decided.\n\n"
            "**KEY PRINCIPLE:** The single most important legal principle.\n\n"
            "Be concise — maximum 200 words total."
        )
    elif summary_mode == "Key Provisions":
        prompt_override = (
            "You are an expert legal analyst. From this Supreme Court judgment, "
            "extract ONLY the legal provisions:\n\n"
            "**CASE NAME:** Full case name.\n\n"
            "**CONSTITUTION:** Articles of the Constitution cited.\n\n"
            "**STATUTES:** Acts and their specific Sections cited.\n\n"
            "**CASE LAW:** Key precedents cited (case name + year).\n\n"
            "**RATIO:** One-sentence ratio decidendi.\n\n"
            "List every provision mentioned. Be exhaustive and precise."
        )
    else:
        prompt_override = SUMMARY_SYSTEM_PROMPT

    # Build prompt
    prompt = (
        f"<s>[INST] <<SYS>>\n"
        f"{prompt_override}\n"
        f"<</SYS>>\n\n"
        f"Summarize the following Supreme Court judgment:\n\n"
        f"{text}\n\n"
        f"[/INST]\n\n"
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    "temperature" : LLM_TEMPERATURE,
                    "num_predict" : 800,
                    "stop"        : ["</s>", "[INST]"],
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        summary = response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Start it with: `ollama serve`", "", ""
    except requests.exceptions.Timeout:
        return "❌ Ollama timed out. Try 'Quick Brief' mode for long PDFs.", "", ""
    except Exception as e:
        return f"❌ LLM error: {e}", "", ""

    elapsed = time.time() - t0

    # Document info
    doc_info = (
        f"📄 **{pdf_name}** &nbsp;|&nbsp; "
        f"{char_count:,} characters &nbsp;|&nbsp; "
        f"~{word_count:,} words extracted &nbsp;|&nbsp; "
        f"⏱ {elapsed:.1f}s"
    )

    return summary, doc_info, ""


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Libre+Baskerville:wght@400;700&display=swap');

:root {
    --parchment:  #f5f0e8;
    --ink:        #1a1208;
    --ink-light:  #3d2e0f;
    --gold:       #b8860b;
    --gold-light: #d4a017;
    --red:        #8b1a1a;
    --border:     #c8b89a;
    --shadow:     rgba(26, 18, 8, 0.15);
}

body, .gradio-container {
    background: var(--parchment) !important;
    font-family: 'EB Garamond', Georgia, serif !important;
    color: var(--ink) !important;
}

.header-block {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 2px solid var(--gold);
    margin-bottom: 2rem;
}

.header-block h1 {
    font-family: 'Libre Baskerville', Georgia, serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    letter-spacing: 0.03em;
    margin: 0 0 0.3rem !important;
}

.header-block .subtitle {
    font-family: 'EB Garamond', serif;
    font-style: italic;
    font-size: 1.1rem;
    color: var(--ink-light);
    margin: 0;
}

.header-block .badge {
    display: inline-block;
    margin-top: 0.8rem;
    padding: 0.2rem 0.9rem;
    border: 1px solid var(--gold);
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--gold);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

textarea, input[type="text"] {
    font-family: 'EB Garamond', serif !important;
    font-size: 1.05rem !important;
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--ink) !important;
    padding: 0.8rem !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(184, 134, 11, 0.15) !important;
    outline: none !important;
}

button.primary, .gr-button-primary {
    background: var(--ink) !important;
    color: var(--parchment) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Libre Baskerville', serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 1.8rem !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}

button.primary:hover { background: var(--gold) !important; }

button.secondary, .gr-button-secondary {
    background: transparent !important;
    color: var(--ink-light) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'EB Garamond', serif !important;
}

.answer-box {
    background: #fffdf7 !important;
    border-left: 4px solid var(--gold) !important;
    border-radius: 0 3px 3px 0 !important;
    padding: 1.2rem 1.5rem !important;
    font-size: 1.1rem !important;
    line-height: 1.7 !important;
    min-height: 100px;
}

.sources-box {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

.latency-box {
    background: transparent !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--ink-light) !important;
    padding: 0.3rem 0 !important;
}

label span {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: var(--ink-light) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

.examples-header {
    font-family: 'Libre Baskerville', serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--ink-light);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

table.examples td {
    font-family: 'EB Garamond', serif !important;
    font-size: 0.98rem !important;
    color: var(--red) !important;
    cursor: pointer !important;
    padding: 0.35rem 0.8rem !important;
    border-bottom: 1px solid #ece6d8 !important;
    transition: background 0.15s;
}

table.examples td:hover { background: #f0ead8 !important; }

input[type="range"] { accent-color: var(--gold) !important; }

.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* PDF upload area */
.upload-area {
    border: 2px dashed var(--border) !important;
    border-radius: 4px !important;
    background: #fffdf7 !important;
    padding: 1.5rem !important;
    text-align: center;
}

.pdf-info-box {
    background: transparent !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--ink-light) !important;
}

/* Tab styling */
.tab-nav button {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
}
"""

HEADER_HTML = """
<div class="header-block">
    <h1>⚖ Supreme Court of India</h1>
    <p class="subtitle">Judgment Retrieval &amp; Analysis System</p>
    <span class="badge">919,518 chunks &nbsp;·&nbsp; 1976–2024 &nbsp;·&nbsp; QLoRA Fine-tuned SaulLM-7B</span>
</div>
"""

PDF_INSTRUCTIONS = """
### How to use
1. Upload any Supreme Court judgment PDF
2. Choose your summary type
3. Click **Summarise Judgment**

**Works best with:** text-based PDFs from [Indian Kanoon](https://indiankanoon.org), Supreme Court website, or downloaded judgment PDFs.

**Note:** Scanned PDFs without embedded text may not extract well. Max ~12,000 characters are sent to the LLM context window.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    css=CSS,
    title="Supreme Court RAG",
    theme=gr.themes.Base(
        font=gr.themes.GoogleFont("EB Garamond"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs(elem_classes=["tab-nav"]):

        # ── Tab 1: Legal Q&A ──────────────────────────────────────────────────
        with gr.Tab("🔍  Search Judgments"):

            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Legal Query",
                        placeholder="Enter your legal question, case citation, or legal principle…",
                        lines=3,
                        max_lines=6,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Search & Analyse", variant="primary", scale=3)
                        clear_btn  = gr.Button("Clear", variant="secondary", scale=1)

                    num_sources = gr.Slider(
                        label="Number of source chunks",
                        minimum=3, maximum=10, value=5, step=1,
                    )

                with gr.Column(scale=1):
                    gr.HTML('<div class="examples-header">Example Queries</div>')
                    gr.Examples(
                        examples=[[ex] for ex in EXAMPLES],
                        inputs=query_input,
                        label="",
                    )

            gr.HTML('<hr class="section-divider">')

            answer_output = gr.Markdown(
                label="Answer",
                elem_classes=["answer-box"],
                value="_Your answer will appear here._",
            )

            gr.HTML('<hr class="section-divider">')

            sources_output = gr.Markdown(
                label="Source Judgments",
                elem_classes=["sources-box"],
                value="",
            )

            latency_output = gr.Markdown(
                label="", elem_classes=["latency-box"], value="",
            )

            submit_btn.click(
                fn=run_query,
                inputs=[query_input, num_sources],
                outputs=[answer_output, sources_output, latency_output],
            )
            query_input.submit(
                fn=run_query,
                inputs=[query_input, num_sources],
                outputs=[answer_output, sources_output, latency_output],
            )
            clear_btn.click(
                fn=lambda: ("_Your answer will appear here._", "", ""),
                outputs=[answer_output, sources_output, latency_output],
            )

        # ── Tab 2: PDF Summarization ──────────────────────────────────────────
        with gr.Tab("📄  Summarise a Judgment PDF"):

            with gr.Row():
                with gr.Column(scale=2):
                    pdf_upload = gr.File(
                        label="Upload Judgment PDF",
                        file_types=[".pdf"],
                        elem_classes=["upload-area"],
                    )

                    summary_mode = gr.Radio(
                        label="Summary Type",
                        choices=["Full Summary", "Quick Brief", "Key Provisions"],
                        value="Full Summary",
                        info=(
                            "Full Summary — all 8 legal sections  |  "
                            "Quick Brief — 200-word overview  |  "
                            "Key Provisions — Acts, Sections, Articles only"
                        ),
                    )

                    with gr.Row():
                        summarise_btn = gr.Button("Summarise Judgment", variant="primary", scale=3)
                        clear_pdf_btn = gr.Button("Clear", variant="secondary", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown(PDF_INSTRUCTIONS)

            gr.HTML('<hr class="section-divider">')

            pdf_info_output = gr.Markdown(
                label="", elem_classes=["pdf-info-box"], value="",
            )

            summary_output = gr.Markdown(
                label="Summary",
                elem_classes=["answer-box"],
                value="_Upload a PDF and click Summarise Judgment._",
            )

            summarise_btn.click(
                fn=run_pdf_summary,
                inputs=[pdf_upload, summary_mode],
                outputs=[summary_output, pdf_info_output, gr.Textbox(visible=False)],
            )
            clear_pdf_btn.click(
                fn=lambda: ("_Upload a PDF and click Summarise Judgment._", ""),
                outputs=[summary_output, pdf_info_output],
            )


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Supreme Court RAG — Local Web UI")
    print("  Tab 1: Search over 919,518 indexed chunks")
    print("  Tab 2: Upload any judgment PDF for summarization")
    print("  Open:  http://localhost:7860")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )