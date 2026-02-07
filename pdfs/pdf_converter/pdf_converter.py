import argparse
import sys
from pathlib import Path
import torch
import soundfile as sf
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description="PDF to Speech Pipeline")
    parser.add_argument("input_file", help="Path to the PDF file")
    parser.add_argument("--speaker", default="Serena", help="Speaker: Vivian, Serena, Ryan, etc.")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", help="Qwen3 model path")

    args = parser.parse_args()
    input_path = Path(args.input_file)

    if not input_path.exists() or input_path.suffix.lower() != ".pdf":
        print(f"Error: {input_path} is not a valid PDF file.")
        sys.exit(1)

    audio_output = input_path.with_suffix(".wav")

    # --- Step 1: Extract Text ---
    print(f"🔍 Extracting text from {input_path.name}...")
    try:
        doc_converter = DocumentConverter()
        doc_result = doc_converter.convert(str(input_path))
        # Export to markdown but strip some structural noise for smoother reading
        text_to_read = doc_result.document.export_to_markdown()
    except Exception as e:
        print(f"Text extraction failed: {e}")
        sys.exit(1)

    # --- Step 2: Initialize TTS ---
    print(f"🚀 Loading TTS Model ({args.model})...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="auto",  # This will pick 'mps' automatically on Mac
        torch_dtype=torch.float16  # Mac MPS prefers float16 over bfloat16 usually
    )

    # --- Step 3: Generate Audio ---
    print(f"🎙️ Generating speech for {audio_output.name}...")
    try:
        # Qwen3-TTS handles long text well, but for very long PDFs,
        # consider splitting by paragraph to avoid memory issues.
        wavs, sr = model.generate_custom_voice(
            text=text_to_read,
            language="auto",
            speaker=args.speaker
        )

        # Save the first result in the batch
        sf.write(audio_output, wavs[0], sr)
        print(f"✨ Success! Audio saved to: {audio_output}")

    except Exception as e:
        print(f"TTS generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
