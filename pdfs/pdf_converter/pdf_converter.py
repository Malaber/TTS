import argparse
import sys
from pathlib import Path
from docling.converter import DocumentConverter


def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown for TTS input using Docling."
    )
    parser.add_argument(
        "input_file",
        help="Path to the PDF file you want to convert."
    )

    args = parser.parse_args()
    input_path = Path(args.input_file)

    # 2. Validation Checks
    if not input_path.exists():
        print(f"Error: The file '{input_path}' does not exist.")
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print(f"Error: '{input_path}' is not a PDF file.")
        sys.exit(1)

    # 3. Derive Output Filename (e.g., document.pdf -> document.md)
    output_path = input_path.with_suffix(".md")

    # 4. Conversion Process
    print(f"Converting {input_path.name}...")

    try:
        converter = DocumentConverter()
        result = converter.convert(str(input_path))
        markdown_output = result.document.export_to_markdown()

        # 5. Save to File
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)

        print(f"Success! Saved text to: {output_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
