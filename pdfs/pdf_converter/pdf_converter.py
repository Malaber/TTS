import argparse
import sys
from pathlib import Path
import torch
import soundfile as sf
from tqdm import tqdm
import re
import numpy as np  # Added for audio concatenation

# Logic imports
from docling.document_converter import DocumentConverter
from qwen_tts import Qwen3TTSModel


def clean_markdown_for_tts(text):
    """Strips Markdown structural characters for a smoother listening experience."""
    # Remove Bold/Italic stars
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Remove Header hashes
    text = re.sub(r'#+\s', '', text)
    # Remove links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'[-*_]{3,}', '', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="PDF to Speech Pipeline with Caching")
    parser.add_argument("input_file", help="Path to the PDF file")
    parser.add_argument("--speaker", default="Lenn", help="Speaker: Lenn (DE), Serena (EN), Ryan (EN), etc.")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", help="Qwen3 model path")
    # Added snippet argument
    parser.add_argument("--snippet", action="store_true", help="Only process the first 5 snippets for testing")

    args = parser.parse_args()
    input_path = Path(args.input_file)
    md_path = input_path.with_suffix(".md")

    # Adjust output name if in snippet mode
    suffix = "-snippet.wav" if args.snippet else ".wav"
    audio_output = input_path.parent / (input_path.stem + suffix)

    # --- Step 1: Text Extraction (with Cache Check) ---
    if md_path.exists():
        print(f"♻️  Found existing Markdown: {md_path.name}. Skipping PDF extraction.")
        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        if not input_path.exists() or input_path.suffix.lower() != ".pdf":
            print(f"Error: {input_path} is not a valid PDF file.")
            sys.exit(1)

        print(f"🔍 Extracting text from {input_path.name}...")
        try:
            doc_converter = DocumentConverter()
            doc_result = doc_converter.convert(str(input_path))
            raw_text = doc_result.document.export_to_markdown()

            # Save the cache so we can skip this next time
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"💾 Cached extracted text to {md_path.name}")
        except Exception as e:
            print(f"Text extraction failed: {e}")
            sys.exit(1)

    # Prepare text for TTS
    text_to_read = clean_markdown_for_tts(raw_text)

    # --- Step 2: Initialize TTS ---
    print(f"🚀 Loading TTS Model...")
    # Optimized for Mac: float16 is faster on Metal (MPS) than bfloat16
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.bfloat16
    )

    # --- Step 3: Generate Audio (with Paragraph Chunking) ---
    print(f"🎙️  Generating speech for {audio_output.name}...")

    # Split text into paragraphs based on double newlines
    paragraphs = [p.strip() for p in text_to_read.split('\n\n') if p.strip()]

    all_audio_segments = []
    final_sr = None
    processed_count = 0

    try:
        # Loop through paragraphs with a progress bar
        for i, para in enumerate(tqdm(paragraphs, desc="Synthesizing")):

            # Check if we should stop early in snippet mode
            if args.snippet and processed_count >= 5:
                break

            # --- SUB-CHUNKING LOGIC ---
            # If a paragraph is very long, the GPU slows down exponentially.
            # We split long paragraphs into sub-chunks (approx 450 chars) to maintain speed.
            sub_chunks = []
            if len(para) > 500:
                # Split by sentence-ending punctuation to keep it natural
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for s in sentences:
                    if len(current_chunk) + len(s) < 450:
                        current_chunk += " " + s
                    else:
                        sub_chunks.append(current_chunk.strip())
                        current_chunk = s
                sub_chunks.append(current_chunk.strip())
            else:
                sub_chunks = [para]

            for chunk in sub_chunks:
                if not chunk: continue
                if args.snippet and processed_count >= 5:
                    break

                # Generate audio for the sub-chunk
                wavs, sr = model.generate_custom_voice(
                    text=chunk,
                    language="auto",
                    speaker=args.speaker
                )
                all_audio_segments.append(wavs[0])
                final_sr = sr
                processed_count += 1

        if all_audio_segments:
            # Stitch all segments into one array
            combined_audio = np.concatenate(all_audio_segments)
            sf.write(audio_output, combined_audio, final_sr)
            print(f"✨ Success! Audio saved to: {audio_output}")
        else:
            print("⚠️ No text chunks were found to process.")

    except Exception as e:
        print(f"TTS generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
