import re
from pathlib import Path

from pypdf import PdfReader

project_root = Path(__file__).resolve().parent
pdf_dir = project_root / "pdfs"
out_file = project_root / "all_papers.txt"

def remove_references_section(text):
    """Remove the References section and everything after it."""
    # Common patterns for references sections
    patterns = [
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCE\s*\n',
        r'\n\s*Reference\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    return text

def main():
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Number of PDFs accessed: {len(pdf_files)}")

    all_text = []
    failed_count = 0

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            text_chunks = []

            for page_num, page in enumerate(reader.pages):
                try:
                    text_chunks.append(page.extract_text() or "")
                except Exception as page_error:
                    print(f"Warning: {pdf_file.name} - Page {page_num + 1} failed: {str(page_error)}")
                    continue

            if text_chunks:
                text = "\n".join(text_chunks)

                try:
                    text = remove_references_section(text)
                except Exception:
                    print(f"Warning: {pdf_file.name} - Reference removal failed, continuing...")

                all_text.append(f"--- {pdf_file.name} ---\n{text}\n")
                print(f"✓ Processed: {pdf_file.name}")
            else:
                print(f"✗ Failed: {pdf_file.name} - No text extracted")
                failed_count += 1

        except Exception as e:
            print(f"✗ Failed: {pdf_file.name} - {str(e)}")
            failed_count += 1

    if not all_text:
        print("\nWarning: No PDFs were successfully extracted!")
        print("Creating empty output file...")

    try:
        out_file.write_text("\n".join(all_text), encoding="utf-8")
    except UnicodeEncodeError as e:
        print(f"\n⚠️  Unicode encoding error: {str(e)}")
        print("  → Retrying with character replacement...")
        with open(out_file, "w", encoding="utf-8", errors="replace") as f:
            f.write("\n".join(all_text))

    print(f"\nSuccessfully processed: {len(all_text)} PDFs")
    if failed_count > 0:
        print(f"Failed: {failed_count} PDFs")
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
