import argparse
import sys
import io
import re
from pathlib import Path

import torch
import requests
import numpy as np
import soundfile as sf
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
    parser = argparse.ArgumentParser(description="Adaptive PDF to Speech Pipeline")
    parser.add_argument("input_file", help="Path to the PDF file")

    # Mode Toggle
    parser.add_argument("--mode", choices=["api", "local"], default="local", help="Synthesize via API or locally")

    # API Settings
    parser.add_argument("--url", default="http://localhost:5002/api/tts", help="Coqui TTS API URL")
    parser.add_argument("--language", default="German", help="Language for local (German) or API (de)")

    # Local Settings
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Local Qwen3 model path")
    parser.add_argument("--ref_audio", default="german_narrator.wav", help="Fixed reference voice for local mode")
    parser.add_argument("--ref_text", default="Dies ist die feste Stimme für meine Ernährungsbildungs-Präsentation. Ich erläutere ihnen hiermit die komplizierten Fakten des Lebens ganz simpel und mit etwas Witz.",
                        help="Transcription of ref_audio")

    # Processing Settings
    parser.add_argument("--max_chars", type=int, default=500, help="Wait for this many chars before chunking")
    parser.add_argument("--snippet", type=int, nargs='?', const=3, help="Only process the first N chunks")

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

    # --- Step 2: Buffered Chunking ---
    paragraphs = [p.strip() for p in text_to_read.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < args.max_chars:
            current_chunk += p + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk: chunks.append(current_chunk.strip())
    if args.snippet is not None: chunks = chunks[:args.snippet]

    # --- Step 3: Initialization ---
    model = None
    if args.mode == "local":
        from qwen_tts import Qwen3TTSModel
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🚀 Loading Local Model on {device}...")
        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map={"": device},
            torch_dtype=torch.bfloat16
        )
    else:
        print(f"🌐 Using API at {args.url}...")

    # --- Step 4: Synthesis Loop ---
    all_audio_segments = []
    final_sr = None

    try:
        for chunk_text in tqdm(chunks, desc=f"Synthesizing ({args.mode})"):
            if args.mode == "api":
                # --- API MODE ---
                payload = {'text': chunk_text, 'language_id': 'de'}
                response = requests.get(args.url, params=payload, timeout=300)
                if response.status_code == 200:
                    data, sr = sf.read(io.BytesIO(response.content))
                    all_audio_segments.append(data)
                    final_sr = sr
                else:
                    print(f"\n⚠️ API Error {response.status_code}")

            else:
                # --- LOCAL MODE ---
                # Using the Base model cloning for absolute consistency
                wavs, sr = model.generate_voice_clone(
                    text=chunk_text,
                    language=args.language,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text
                )
                all_audio_segments.append(wavs[0])
                final_sr = sr

        if all_audio_segments:
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
