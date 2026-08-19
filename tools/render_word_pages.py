"""Rasterize the final Word PDF export for DOCX visual QA."""

from pathlib import Path

import pypdfium2 as pdfium


OUT = Path(r"D:\rs-project\fgvc\runs\_analysis\meeting_doc_render_word")
PDF = OUT / "meeting_brief_final.pdf"


if not PDF.exists():
    raise FileNotFoundError(PDF)

pdf = pdfium.PdfDocument(str(PDF))
n_pages = len(pdf)
for i in range(n_pages):
    image = pdf[i].render(scale=150 / 72).to_pil()
    image.save(OUT / f"page-{i + 1:02d}.png")
pdf.close()

print(f"rendered {n_pages} pages -> {OUT}")
