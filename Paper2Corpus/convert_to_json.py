#!/usr/bin/env python3

import json
import sys

def convert_to_json(input_file, output_file):
    """Convert text chunks (one PDF per entry) to JSON format for LLM training."""
    
    try:
        # Read all content from input file
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
                    else:
                        print(f"Warning: {pdf_name} - Empty content, skipping")
                except Exception as e:
                    print(f"Error processing block: {str(e)}")
                    continue
        
        if not corpus:
            print("\nWarning: No documents to convert to JSON!")
            print("Creating empty JSON array...")
        
        # Write to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
        except UnicodeEncodeError as e:
            print(f"\n⚠️  Unicode encoding error: {str(e)}")
            print(f"  → Retrying with character replacement...")
            with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
            print(f"Note: Some characters were replaced due to encoding issues")
        
        print(f"Successfully converted {len(corpus)} PDFs to JSON format")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error converting to JSON: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert_to_json.py input_file output_file.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_to_json(input_file, output_file)

if __name__ == "__main__":
    main()
