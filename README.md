# Insanely Fast Whisper  - Transcription and Conversion Tool

This Python project provides a command-line interface (CLI) for transcribing audio files using the OpenAI Whisper ASR model and converting the transcription output to various formats such as SRT, VTT, and TXT.

## Features

- Transcribe audio files (MP3, WAV, M4A) using the OpenAI Whisper ASR model
- Convert transcription output to SRT, VTT, and TXT formats
- Support for batch processing of multiple audio files in a directory
- Configurable output directory for generated files

## Prerequisites

- Python 3.6 or higher
- PyTorch (with CUDA support if available)
- Transformers library
- Flash Attention (if available)

## Installation

Install insanely-fast-whisper with pipx (pip install pipx or brew install pipx):
```
pipx install insanely-fast-whisper
```
 Clone the repository:

```bash
git clone git@github.com:glows/Transcription_Convert.git
cd Transcription_Convert
```

## License
This project is licensed under the [MIT License](https://github.com/ochen1/insanely-fast-whisper-cli/blob/main/LICENSE).

## Acknowledgments
This tool is powered by Hugging Face's ASR models, primarily Whisper by OpenAI.  
Optimizations are developed by Vaibhavs10/insanely-fast-whisper.  
Developed by @ochen1.