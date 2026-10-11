"""Usage:  python -m src.ingest.run_ingest

Put reports (.pdf or .docx) in data/reports/ named  <Company>_<Year>.<ext>
(e.g. Infosys_FY26.pdf, Microsoft_FY25.docx).
Output: data/processed/chunks.jsonl  (one chunk per line).
"""
import json
import logging
import sys
import os
from pathlib import Path
import fitz
import pytesseract
from PIL import Image

from .section_tagger import tag_page
from .chunker import build_chunks

REPORTS = Path("data/reports")
OUT = Path("data/processed/chunks.jsonl")


def parse_name(path: Path):
    stem = path.stem
    if "_" in stem:
        company, year = stem.rsplit("_", 1)
    else:
        company, year = stem, "unknown"
    return company, year

def ocr_pdf_page(page):
    """Use Tesseract OCR to extract text from a scanned PDF page."""
    import fitz
    import pytesseract
    from PIL import Image

    tesseract_path = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if tesseract_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

    scale = 200 / 72
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )

    return pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 3",
        timeout=45,
    ).strip()

def read_report(path: Path, company: str, year: str):
    """Read a report, OCR scanned PDF pages, and build unique chunks."""

    is_pdf = path.suffix.lower() == ".pdf"

    if is_pdf:
        from .pdf_loader import iter_pages

        # First, use the existing PDF loader.
        try:
            pages = list(iter_pages(path))
        except Exception as exc:
            logging.warning(
                "PDF loader failed for %s: %s. Trying OCR fallback.",
                path.name,
                exc,
            )
            pages = None

        # If the PDF loader fails, read the PDF directly.
        if pages is None:
            import fitz

            pages = []
            with fitz.open(str(path)) as pdf:
                for page_no, page in enumerate(pdf, start=1):
                    try:
                        text = page.get_text("text", sort=True) or ""
                    except Exception:
                        text = ""

                    pages.append((page_no, text, []))

        # OCR pages with little or no extractable text.
        import fitz

        with fitz.open(str(path)) as pdf:
            processed_pages = []

            for page_no, text, tables in pages:
                text = text or ""
                tables = tables or []

                if len(text.strip()) < 30:
                    try:
                        if 1 <= page_no <= len(pdf):
                            ocr_text = ocr_pdf_page(pdf[page_no - 1])

                            if len(ocr_text) > len(text.strip()):
                                text = ocr_text

                            logging.info(
                                "OCR checked %s page %s: %d characters",
                                path.name,
                                page_no,
                                len(ocr_text),
                            )
                    except Exception as exc:
                        logging.warning(
                            "OCR failed for %s page %s: %s",
                            path.name,
                            page_no,
                            exc,
                        )

                processed_pages.append((page_no, text.strip(), tables))

        pages = processed_pages

    else:
        from .docx_loader import iter_pages
        pages = list(iter_pages(path))

    # Preserve the existing section tagging and chunking pipeline.
    chunks, counts = [], {}

    for page_no, text, tables in pages:
        text = text or ""
        tables = tables or []

        if not text.strip() and not tables:
            continue

        section = tag_page(text)
        counts[section] = counts.get(section, 0) + 1

        page_chunks = build_chunks(
            company, year, page_no, section, text, tables
        )

        for idx, ch in enumerate(page_chunks):
            ch_dict = ch.to_dict() if hasattr(ch, "to_dict") else dict(ch)
            ch_dict["chunk_id"] = (
                f"{company}_{year}_p{page_no}_c{idx + 1}"
            )
            chunks.append(ch_dict)

    return chunks, counts



def main():
    files = sorted(
        p for p in REPORTS.iterdir()
        if p.suffix.lower() in (".pdf", ".docx") and not p.name.startswith("~$")
    ) if REPORTS.exists() else []
    if not files:
        sys.exit(f"No .pdf or .docx files found in {REPORTS}/")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=OUT.parent / "ingest.log", level=logging.INFO, filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    total, failed = 0, []
    with OUT.open("w", encoding="utf-8") as f:
        for path in files:
            company, year = parse_name(path)
            try:
                got = read_report(path, company, year)   # whole file read first: no half-written reports
            except Exception as e:
                logging.exception("could not read %s", path.name)
                print(f"[ERROR] {path.name}: {type(e).__name__}: {e} (details in {OUT.parent / 'ingest.log'})")
                failed.append(path.name)
                continue
            if got is None:
                print(f"[SKIP] {path.name}: looks scanned or has no text layer (needs OCR) - "
                      f"check it with: python -m src.tools pdfs")
                logging.warning("skipped %s: no text layer", path.name)
                continue
            chunks, counts = got
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total += len(chunks)
            note = " (docx: page numbers are approximate)" if path.suffix.lower() == ".docx" else ""
            logging.info("read %s: %d chunks %s", path.name, len(chunks), counts)
            print(f"[OK] {path.name}: {len(chunks)} chunks, pages by section: {counts}{note}")
    print(f"Done. {total} chunks -> {OUT}" + (f"  ({len(failed)} file(s) failed: {', '.join(failed)})" if failed else ""))


if __name__ == "__main__":
    main()
