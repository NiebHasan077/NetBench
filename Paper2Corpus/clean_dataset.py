#!/usr/bin/env python3

import re
import sys

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
    # Remove email patterns more thoroughly
    text = re.sub(r'\S+@\S+\.(com|edu|org|net|io|ai)', '', text)
    # Remove affiliation-like patterns (University, Inc., Corp., etc.)
    text = re.sub(r'(University|Institute|Laboratory|Inc\.|Corp\.|Ltd\.) of \w+', '', text)
    text = re.sub(r'\w+ (University|Institute|College|Laboratory)', '', text)
    return text

def fix_hyphenated_words(text):
    """Rejoin words split across lines with hyphens."""
    # Match word-\n and recombine
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text

def clean_math_notation(text):
    """Clean mathematical symbols and formulas."""
    # Remove standalone Greek letters and math symbols that lost context
    text = re.sub(r'\b[αβγδεζηθικλμνξοπρστυφχψω]\b', '', text)
    # Remove isolated math operators
    text = re.sub(r'\s+[∈∉⊂⊆∪∩∑∏∫≈≠≤≥]+\s+', ' ', text)
    # Remove formula markers
    text = re.sub(r'Equation\s*\d+', '', text)
    text = re.sub(r'Formula\s*\d+', '', text)
    return text

def remove_table_figure_content(text):
    """Remove table contents."""
    # Remove table contents (rows of numbers/data)
    text = re.sub(r'(\|.*?\|.*?\n){2,}', '\n', text)
    return text

def normalize_sentences(text):
    """Ensure proper sentence boundaries."""
    # Fix missing spaces after periods
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # Remove multiple punctuation
    text = re.sub(r'[.!?]{2,}', '.', text)
    
    # Ensure paragraphs have proper spacing
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
    # Split by the PDF file markers
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
                    print(f"  Warning: {pdf_name} - Too short after cleaning, skipping")
            except Exception as e:
                print(f"  Error cleaning {pdf_name}: {str(e)}")
                continue
    
    return pdfs

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 clean_dataset.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process each PDF separately
        pdfs = split_by_pdf(content)
        
        if not pdfs:
            print("Warning: No PDFs remained after cleaning!")
            print("Creating empty output file...")
        
        # Write cleaned output with PDF markers
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for pdf_name, cleaned_content in pdfs:
                    try:
                        f.write(f'PDF_START:{pdf_name}\n')
                        f.write(cleaned_content)
                        f.write(f'\nPDF_END\n\n')
                    except Exception as e:
                        print(f"  Error writing {pdf_name}: {str(e)}")
                        continue
        except UnicodeEncodeError as e:
            print(f"\n⚠️  Unicode encoding error: {str(e)}")
            print(f"  → Retrying with character replacement...")
            with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                for pdf_name, cleaned_content in pdfs:
                    try:
                        f.write(f'PDF_START:{pdf_name}\n')
                        f.write(cleaned_content)
                        f.write(f'\nPDF_END\n\n')
                    except Exception as e:
                        print(f"  Error writing {pdf_name}: {str(e)}")
                        continue
        
        print(f"Successfully cleaned {len(pdfs)} PDFs")
        print(f"Output written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()