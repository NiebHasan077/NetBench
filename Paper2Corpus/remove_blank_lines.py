#!/usr/bin/env python3

import sys
import argparse

def remove_blank_lines(input_file, output_file=None):
    """
    Remove blank lines from a text file and save to output file.
    If no output file is specified, it will overwrite the input file.
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove blank lines (including those with only whitespace)
        non_blank_lines = [line for line in lines if line.strip()]
        
        # Determine output file
        out_file = output_file if output_file else input_file
        
        # Write the filtered content
        with open(out_file, 'w', encoding='utf-8') as f:
            f.writelines(non_blank_lines)
            
        print(f"Successfully processed file. Removed {len(lines) - len(non_blank_lines)} blank lines.")
        print(f"Output written to: {out_file}")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Remove blank lines from a text file.')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('-o', '--output', help='Path to the output file (optional)', default=None)
    
    args = parser.parse_args()
    remove_blank_lines(args.input_file, args.output)

if __name__ == "__main__":
    main()