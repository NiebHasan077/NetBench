#!/usr/bin/env python3
"""
Paper2Corpus pipeline.
Extracts, cleans, and converts PDF research papers to JSON format for LLM training.

Usage:
    python3 pipeline.py
    
Or with custom paths:
    python3 pipeline.py --pdf-dir ./pdfs --output research_corpus.json
"""

import re
import json
import sys
import argparse
from pathlib import Path
from pypdf import PdfReader


def remove_references_section(text):
    """Remove the References section and everything after it."""
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


def extract_pdfs_to_text(pdf_dir, output_file):
    """Step 1: Extract text from all PDFs in directory."""
    pdf_dir = Path(pdf_dir)
    output_file = Path(output_file)
    
    if not pdf_dir.exists():
        print(f"Error: PDF directory '{pdf_dir}' does not exist.")
        sys.exit(1)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{pdf_dir}'")
        sys.exit(1)
    
    print(f"[Step 1/3] Extracting text from {len(pdf_files)} PDFs...")
    
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
                    print(f"  ⚠️  {pdf_file.name} - Page {page_num + 1}: {str(page_error)}")
                    continue
            
            if text_chunks:  # Only process if we got some text
                text = "\n".join(text_chunks)
                
                # Remove references section
                try:
                    text = remove_references_section(text)
                except Exception as ref_error:
                    print(f"  ⚠️  {pdf_file.name}: Reference removal failed, continuing...")
                
                all_text.append(f"--- {pdf_file.name} ---\n{text}\n")
                print(f"  ✓ {pdf_file.name}")
            else:
                print(f"  ✗ {pdf_file.name}: No text extracted")
                failed_count += 1
                
        except Exception as e:
            print(f"  ✗ {pdf_file.name}: {str(e)}")
            failed_count += 1
    
    if not all_text:
        print(f"\n⚠️  Warning: No PDFs were successfully extracted!")
        print(f"  Failed: {failed_count}/{len(pdf_files)} PDFs")
        print(f"  Creating empty output file...")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_file.write_text("\n".join(all_text), encoding="utf-8")
    except UnicodeEncodeError as e:
        print(f"  ⚠️  Unicode encoding error: {str(e)}")
        print(f"  → Retrying with character replacement...")
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            f.write("\n".join(all_text))
    
    print(f"  ✓ Successfully extracted: {len(all_text)}/{len(pdf_files)} PDFs")
    if failed_count > 0:
        print(f"  ✗ Failed: {failed_count} PDFs")
    
    print(f"  → Saved to {output_file}")
    return output_file


def remove_boilerplate_sections(text):
    """Remove common academic paper boilerplate sections."""
    sections_to_remove = [
        r'\n\s*ACKNOWLEDGMENTS?\s*\n.*?(?=\n\s*[A-Z]{2,}|\Z)',
        r'\n\s*Keywords?:.*?\n',
        r'\n\s*CCS CONCEPTS\s*\n.*?(?=\n\s*KEYWORDS|\Z)',
        r'\n\s*ACM Reference Format:.*?(?=\n\s*[A-Z]{2,}|\Z)',
        r'Permission to make digital.*?(?=\n\s*[A-Z]{2,})',
        r'Copyright.*?ACM\.',
        r'ISBN.*?\n',
        r'DOI:.*?\n',
        r'https?://doi\.org/.*?\n',
    ]
    
    for pattern in sections_to_remove:
        text = re.sub(pattern, '\n', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text


def remove_author_info(text):
    """Remove author names and affiliations from paper headers."""
    text = re.sub(r'\S+@\S+\.(com|edu|org|net|io|ai)', '', text)
    text = re.sub(r'(University|Institute|Laboratory|Inc\.|Corp\.|Ltd\.) of \w+', '', text)
    text = re.sub(r'\w+ (University|Institute|College|Laboratory)', '', text)
    return text


def fix_hyphenated_words(text):
    """Rejoin words split across lines with hyphens."""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text


def clean_math_notation(text):
    """Clean mathematical symbols and formulas."""
    text = re.sub(r'\b[αβγδεζηθικλμνξοπρστυφχψω]\b', '', text)
    text = re.sub(r'\s+[∈∉⊂⊆∪∩∑∏∫≈≠≤≥]+\s+', ' ', text)
    text = re.sub(r'Equation\s*\d+', '', text)
    text = re.sub(r'Formula\s*\d+', '', text)
    return text


def remove_table_figure_content(text):
    """Remove table contents."""
    text = re.sub(r'(\|.*?\|.*?\n){2,}', '\n', text)
    return text


def normalize_sentences(text):
    """Ensure proper sentence boundaries."""
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    text = re.sub(r'[.!?]{2,}', '.', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def clean_text(text):
    """Clean and normalize text for language model training."""
    
    # Apply all cleaning steps in order
    text = remove_boilerplate_sections(text)
    text = remove_author_info(text)
    text = fix_hyphenated_words(text)
    
    # Remove special formatting characters and artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[{}()\[\]|@,;]', ' ', text)  # Remove special chars
    text = re.sub(r'[.]{2,}', '.', text)  # Replace multiple dots
    text = re.sub(r'[-]{2,}', '-', text)  # Replace multiple dashes
    text = re.sub(r'[_]{2,}', '_', text)  # Replace multiple underscores
    
    # Clean mathematical notation
    text = clean_math_notation(text)
    
    # Remove tables
    text = remove_table_figure_content(text)
    
    # Remove citation patterns
    text = re.sub(r'\[[0-9,\-\s]+\]', '', text)  # Remove [1], [1-5], etc.
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize sentence boundaries
    text = normalize_sentences(text)
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean paragraph breaks
    
    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 20 and not line.isupper():  # Skip short lines and all-caps headers
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def split_by_pdf(text):
    """Split text by PDF file markers to keep each PDF separate."""
    pdf_pattern = r'---\s+(.+?\.pdf)\s+---'
    parts = re.split(pdf_pattern, text)
    
    pdfs = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            pdf_name = parts[i].strip()
            pdf_content = parts[i + 1].strip()
            
            try:
                # Clean the content
                cleaned_content = clean_text(pdf_content)
                
                # Only include if substantial content remains
                if len(cleaned_content) > 100:
                    pdfs.append((pdf_name, cleaned_content))
                else:
                    print(f"  ⚠️  {pdf_name}: Too short after cleaning ({len(cleaned_content)} chars), skipping")
            except Exception as e:
                print(f"  ✗ {pdf_name}: Cleaning failed - {str(e)}")
                continue
    
    return pdfs


def clean_extracted_text(input_file, output_file):
    """Step 2: Clean the extracted text."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    print(f"[Step 2/3] Cleaning extracted text...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pdfs = split_by_pdf(content)
    
    if not pdfs:
        print(f"\n⚠️  Warning: No PDFs remained after cleaning!")
        print(f"  This might be due to:")
        print(f"    - All PDFs being too short after cleaning")
        print(f"    - Errors during the cleaning process")
        print(f"  Creating empty output file...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for pdf_name, cleaned_content in pdfs:
                try:
                    f.write(f'PDF_START:{pdf_name}\n')
                    f.write(cleaned_content)
                    f.write(f'\nPDF_END\n\n')
                except Exception as e:
                    print(f"  ✗ {pdf_name}: Failed to write - {str(e)}")
                    continue
    except UnicodeEncodeError as e:
        print(f"  ⚠️  Unicode encoding error: {str(e)}")
        print(f"  → Retrying with character replacement...")
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            for pdf_name, cleaned_content in pdfs:
                try:
                    f.write(f'PDF_START:{pdf_name}\n')
                    f.write(cleaned_content)
                    f.write(f'\nPDF_END\n\n')
                except Exception as e:
                    print(f"  ✗ {pdf_name}: Failed to write - {str(e)}")
                    continue
    
    print(f"  ✓ Cleaned {len(pdfs)} PDFs")
    print(f"  → Saved to {output_file}")
    return output_file


def convert_to_json(input_file, output_file):
    """Step 3: Convert cleaned text to JSON format."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    print(f"[Step 3/3] Converting to JSON format...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by PDF markers
    pdf_blocks = content.split('PDF_START:')
    
    # Create JSON structure
    corpus = []
    for block in pdf_blocks[1:]:  # Skip the first empty split
        if 'PDF_END' in block:
            try:
                # Extract PDF name and content
                parts = block.split('\n', 1)
                pdf_name = parts[0].strip()
                
                # Get content between PDF_START and PDF_END
                content_parts = parts[1].split('PDF_END')
                pdf_content = content_parts[0].strip()
                
                if pdf_content:
                    corpus.append({"text": pdf_content})
                    print(f"  ✓ {pdf_name}")
                else:
                    print(f"  ⚠️  {pdf_name}: Empty content, skipping")
            except Exception as e:
                print(f"  ✗ Error processing block: {str(e)}")
                continue
    
    if not corpus:
        print(f"\n⚠️  Warning: No documents to convert to JSON!")
        print(f"  Creating empty JSON array...")
    
    # Write to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Converted {len(corpus)} PDFs to JSON")
        print(f"  → Saved to {output_file}")
    except UnicodeEncodeError as e:
        print(f"  ⚠️  Unicode encoding error: {str(e)}")
        print(f"  → Retrying with character replacement...")
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Converted {len(corpus)} PDFs to JSON (with character replacements)")
        print(f"  → Saved to {output_file}")
    except Exception as e:
        print(f"\n❌ Error writing JSON file: {str(e)}")
        raise
    
    # Print statistics
    if corpus:
        lengths = [len(doc['text']) for doc in corpus]
        print(f"\n📊 Statistics:")
        print(f"  Total documents: {len(corpus)}")
        print(f"  Average length: {sum(lengths) // len(lengths):,} characters")
        print(f"  Total size: {sum(lengths):,} characters")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Process research paper PDFs into JSON format for LLM training.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python3 pipeline.py
  
  # Specify custom paths
  python3 pipeline.py --pdf-dir ./papers --output my_corpus.json
  
  # Keep intermediate files
  python3 pipeline.py --keep-intermediate
        """
    )
    
    parser.add_argument(
        '--pdf-dir',
        default='pdfs',
        help='Directory containing PDF files (default: pdfs)'
    )
    
    parser.add_argument(
        '--output',
        default='research_corpus.json',
        help='Output JSON file (default: research_corpus.json)'
    )
    
    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep intermediate files (all_papers.txt, all_papers_cleaned.txt)'
    )
    
    args = parser.parse_args()
    
    # Intermediate file names
    raw_text_file = Path('all_papers.txt')
    cleaned_text_file = Path('all_papers_cleaned.txt')
    
    print("=" * 60)
    print("Paper2Corpus Pipeline")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Extract PDFs
        extract_pdfs_to_text(args.pdf_dir, raw_text_file)
        print()
        
        # Step 2: Clean text
        clean_extracted_text(raw_text_file, cleaned_text_file)
        print()
        
        # Step 3: Convert to JSON
        convert_to_json(cleaned_text_file, args.output)
        print()
        
        # Cleanup intermediate files unless requested to keep
        if not args.keep_intermediate:
            if raw_text_file.exists():
                raw_text_file.unlink()
            if cleaned_text_file.exists():
                cleaned_text_file.unlink()
            print("🧹 Cleaned up intermediate files")
            print()
        
        print("=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        print(f"Output: {args.output}")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
