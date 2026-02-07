import argparse
import sys
import io  # Added for binary stream handling
from pathlib import Path
import requests
import re
import numpy as np  # Added for concatenation
import soundfile as sf  # Added for audio decoding
from tqdm import tqdm

# Logic imports
from docling.document_converter import DocumentConverter


def clean_markdown_for_tts(text):
    """Strips Markdown and HTML comments for a smoother listening experience."""
    # Remove HTML comments (like )
    text = re.sub(r'', '', text)
    # Remove Bold/Italic
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Remove Header hashes
    text = re.sub(r'#+\s', '', text)
    # Remove links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'[-*_]{3,}', '', text)
    # Remove HTML image comments
    text = text.replace("<!-- image -->", "")
    # Remove unknown chars
    text = text.replace("/uniF6B7", "")
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="PDF to Speech Pipeline (Buffered Chunks)")
    parser.add_argument("input_file", help="Path to the PDF file")
    parser.add_argument("--url", default="http://localhost:5002/api/tts", help="Coqui TTS API URL")
    parser.add_argument("--language", default="de", help="Language code (de, en, etc.)")
    parser.add_argument("--max_chars", type=int, default=500, help="Wait for this many chars before chunking")
    parser.add_argument("--snippet", type=int, nargs='?', const=3, help="Only process the first N chunks (default: 3)")

    args = parser.parse_args()
    input_path = Path(args.input_file)
    md_path = input_path.with_suffix(".md")

    suffix = "-snippet.wav" if args.snippet is not None else ".wav"
    audio_output = input_path.parent / (input_path.stem + suffix)

    # --- Step 1: Text Extraction ---
    if md_path.exists():
        print(f"♻️  Found cached Markdown: {md_path.name}")
        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        print(f"🔍 Extracting PDF text...")
        doc_converter = DocumentConverter()
        doc_result = doc_converter.convert(str(input_path))
        raw_text = doc_result.document.export_to_markdown()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

    text_to_read = clean_markdown_for_tts(raw_text)

    # --- Step 2: Buffered Chunking Logic ---
    paragraphs = [p.strip() for p in text_to_read.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) < args.max_chars:
            current_chunk += p + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    if args.snippet is not None:
        chunks = chunks[:args.snippet]

    print(f"🎙️  Sending {len(chunks)} large chunks to TTS server (Max {args.max_chars} chars each)...")

    all_audio_segments = []
    final_sr = None

    try:
        for chunk_text in tqdm(chunks, desc="Synthesizing"):
            payload = {
                'text': chunk_text,
                'language_id': args.language
            }

            # Use a longer timeout because 10k chars takes time to process on CPU
            response = requests.get(args.url, params=payload, timeout=300)

            if response.status_code == 200:
                # Decode the binary WAV response into a NumPy array
                data, sr = sf.read(io.BytesIO(response.content))
                all_audio_segments.append(data)
                final_sr = sr
            else:
                print(f"\n⚠️ Error {response.status_code} on chunk starting with: {chunk_text[:50]}...")

        if all_audio_segments:
            # Stitch all segments into one array (this works now because they are NumPy arrays, not bytes)
            combined_audio = np.concatenate(all_audio_segments)
            sf.write(audio_output, combined_audio, final_sr)
            print(f"✨ Success! Audio saved to: {audio_output}")
        else:
            print("❌ No audio generated.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
