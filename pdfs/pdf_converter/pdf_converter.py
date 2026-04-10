import argparse
import sys
import io
import re
import gc
import os
from pathlib import Path

import torch
import requests
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Optional: for memory monitoring
try:
    import psutil
except ImportError:
    psutil = None

# Logic imports
from docling.document_converter import DocumentConverter

# Suppress "Setting `pad_token_id` to `eos_token_id`" warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import transformers
transformers.logging.set_verbosity_error()


def clean_markdown_for_tts(text):
    """Strips Markdown, HTML comments, and PDF artifacts for a smoother listening experience."""
    # 1. Remove HTML comments
    text = re.sub(r'', '', text, flags=re.DOTALL)

    # 2. Markdown formatting removal
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

    # --- PDF Artifact Cleanup ---

    text = text.replace("­ ", "")
    text = text.replace("  ", " ")

    # 3. Remove invisible soft hyphens (\xad)
    text = text.replace('\xad', '')

    # 4. Fix line-break hyphenation ("Lebensmittelzu- bereitung" -> "Lebensmittelzubereitung")
    # Matches a word char, a hyphen, 1+ whitespace chars (including newlines), and a word char
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # 5. Fix multiple horizontal spaces (justified text artifacts)
    # Using [ \t]+ instead of \s+ so we don't destroy \n\n paragraph breaks!
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Fix hard-wrapped lines inside paragraphs
    # Replaces single newlines with a space, but leaves double newlines (\n\n) alone
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

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
        text_to_read = clean_markdown_for_tts(raw_text)
    else:
        print(f"🔍 Extracting PDF text...")
        doc_converter = DocumentConverter()
        doc_result = doc_converter.convert(str(input_path))
        raw_text = doc_result.document.export_to_markdown()
        text_to_read = clean_markdown_for_tts(raw_text)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text_to_read)

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

        # Determine the best precision for the hardware
        dtype = torch.float16 if device == "mps" else torch.bfloat16

        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map={"": device},
            torch_dtype=dtype,  # Optimized for Apple Silicon
            attn_implementation="sdpa"  # Enforce PyTorch native attention!
        )
        # Suppress "Setting `pad_token_id` to `eos_token_id`" warning
        if hasattr(model.model, "config"):
            model.model.config.pad_token_id = model.model.config.eos_token_id
    else:
        print(f"🌐 Using API at {args.url}...")

    # --- Step 4: Synthesis Loop (Memory & Disk Optimized) ---
    print(f"🎙️ Starting synthesis. Mode: {args.mode}")

    first_chunk = True
    output_file = None

    try:
        pbar = tqdm(chunks, desc=f"Synthesizing ({args.mode})")
        for chunk_text in pbar:
            # --- 0. Memory Monitoring (Optional) ---
            if psutil:
                ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
                stats = {"RAM": f"{ram_gb:.1f}GB"}
                if torch.backends.mps.is_available():
                    mps_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
                    stats["MPS"] = f"{mps_gb:.1f}GB"
                pbar.set_postfix(stats)

            # --- 1. Generation Logic ---
            if args.mode == "api":
                payload = {'text': chunk_text, 'language_id': 'de'}
                response = requests.get(args.url, params=payload, timeout=300)
                if response.status_code == 200:
                    data, sr = sf.read(io.BytesIO(response.content))
                else:
                    continue
            else:
                # 🛑 CRITICAL FIX: Tell PyTorch NOT to track gradients/memory history!
                with torch.inference_mode():
                    wavs, sr = model.generate_voice_clone(
                        text=chunk_text,
                        language=args.language,
                        ref_audio=args.ref_audio,
                        ref_text=args.ref_text
                    )
                
                # Detach from any residual graphs and convert to CPU/Numpy immediately
                if isinstance(wavs[0], torch.Tensor):
                    data = wavs[0].detach().cpu().numpy()
                else:
                    data = wavs[0]

            # --- 2. Disk Logic ---
            if first_chunk:
                output_file = sf.SoundFile(audio_output, mode='w', samplerate=sr,
                                           channels=1, subtype='PCM_16')
                output_file.write(data)
                first_chunk = False
            else:
                output_file.write(data)

            # --- 3. AGGRESSIVE RAM Cleanup ---
            # Delete EVERYTHING heavy from this specific loop iteration
            if args.mode == "local":
                del wavs
            del data
            
            # Force Python to destroy unreferenced objects NOW
            gc.collect()
            
            # NOW tell Apple Silicon to release that destroyed memory back to the OS
            if args.mode == "local" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        if output_file:
            output_file.close()
            print(f"✨ Success! Audio saved to: {audio_output}")

    except Exception as e:
        if output_file: output_file.close()
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
