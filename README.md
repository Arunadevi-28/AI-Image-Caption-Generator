# AI Image Caption Generator

An AI-powered image captioning application that generates natural-language descriptions for uploaded images using **BLIP (Bootstrapping Language-Image Pre-training)**. Built with **Python, PyTorch, Hugging Face Transformers, and Gradio**, it supports multiple caption styles, multilingual translation, downloadable caption files, and text-to-speech audio generation.

## Overview

Describing images manually can be time-consuming. This project uses a pretrained vision-language model to analyze uploaded images and generate meaningful text descriptions through an interactive Gradio interface.

The application extends basic image captioning with customizable caption styles, adjustable generation parameters, caption diversity estimation, multilingual translation, multiple export formats, and speech generation.

## Features

- **AI Image Captioning:** Generate image descriptions using the pretrained BLIP model.
- **Multiple Caption Styles:** Choose Short, Creative, Detailed, Social, or Poetic styles.
- **Configurable Generation:** Adjust caption length, creativity, and number of captions.
- **Caption Diversity Score:** Estimate textual differences between generated captions.
- **Multilingual Translation:** Translate captions using supported MarianMT models.
- **Multiple Export Formats:** Download captions in TXT, JSON, or SRT format.
- **Text-to-Speech:** Generate speech audio using gTTS.
- **Interactive Interface:** Upload images and configure settings through Gradio.
- **Hardware Detection:** Use CUDA when available, otherwise fall back to CPU.

## Technologies Used

- **Python** – Main programming language
- **PyTorch** – Deep learning framework
- **Hugging Face Transformers** – Pretrained model integration
- **BLIP** – Image caption generation
- **MarianMT** – Machine translation
- **Gradio** – Interactive web interface
- **Pillow** – Image processing
- **gTTS** – Text-to-speech generation
- **difflib** – Text-based caption diversity estimation

## System Architecture

```text
             Image Upload
                  |
                  v
          Gradio Web Interface
                  |
                  v
          Image Preprocessing
                  |
                  v
          BLIP Captioning Model
                  |
                  v
          Generated Captions
                  |
                  v
          Caption Style Formatting
                  |
          +-------+--------+
          |       |        |
          v       v        v
      Diversity Translation Export
        Score     MarianMT  TXT/JSON/SRT
                           |
                           v
                     gTTS Audio
                           |
                           v
                    Downloadable MP3
```

## How It Works

### 1. Image Upload

The user uploads an image through the Gradio interface.

### 2. Image Preprocessing

The BLIP processor converts the image into the input format required by the pretrained model.

### 3. Caption Generation

The BLIP model generates natural-language descriptions based on the uploaded image and selected generation settings.

### 4. Caption Style Formatting

The application provides five caption styles:

- **Short:** Concise image descriptions.
- **Creative:** Expressive descriptions.
- **Detailed:** More descriptive captions.
- **Social:** Captions suitable for social media.
- **Poetic:** Stylized, expressive captions.

These styles are application-level formatting options rather than separate trained models.

### 5. Caption Diversity Score

The application compares captions using Python's `difflib.SequenceMatcher` and estimates average textual dissimilarity as a percentage.

This score measures differences in text, not semantic diversity or caption accuracy.

### 6. Multilingual Translation

The application uses supported MarianMT models to translate captions into the selected target language.

### 7. Caption Export

Generated captions can be exported as:

- **TXT** – Plain-text captions.
- **JSON** – Structured caption data.
- **SRT** – Subtitle-style text output.

### 8. Text-to-Speech

The gTTS library converts caption text into speech and generates an MP3 audio file.

## Model and Algorithms

### BLIP Image Captioning

**Model:** `Salesforce/blip-image-captioning-base`

The application uses `BlipProcessor` to prepare image inputs and `BlipForConditionalGeneration` to generate captions.

Generation parameters can include temperature, top-k, top-p, and beam-search settings. Actual results depend on the image and selected configuration.

### MarianMT Translation

The application uses language-specific MarianMT models from the Helsinki-NLP model family.

**Model naming pattern:**

`Helsinki-NLP/opus-mt-en-{target_language}`

The selected translation model must be available for the target language.

### Caption Diversity Estimation

The application uses pairwise string comparisons to calculate average textual dissimilarity between generated captions.

This is a simple text-based estimate and does not evaluate factual correctness.

### Text-to-Speech

The gTTS library generates speech audio from caption text. Internet access is required for the online speech service.

## Project Structure

```text
AI-Image-Caption-Generator/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

- **`app.py`** – Main application, including image captioning, caption formatting, translation, diversity estimation, file exports, text-to-speech, and Gradio interface.
- **`requirements.txt`** – Lists the Python packages required by the application.
- **`README.md`** – Project documentation and usage instructions.
- **`.gitignore`** – Excludes virtual environments, Python caches, and other files that should not be committed.

## Requirements

### Software

- Python 3.10 or 3.11 recommended
- pip package manager
- Internet connection for downloading pretrained models and using online services

### Hardware

- A computer capable of running Python and PyTorch
- Sufficient RAM for model loading and inference
- CUDA-compatible GPU optional

## Installation

### 1. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

**Command Prompt:**

```cmd
.venv\Scripts\activate
```

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `requirements.txt` contains all packages required by `app.py`.

## Run the Application

From the project root directory, execute:

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal.

## How to Use

1. Launch the application.
2. Upload an image.
3. Select a caption style.
4. Adjust the available generation settings.
5. Click **Generate Captions**.
6. Review the generated descriptions and diversity score.
7. Select a supported target language if translation is required.
8. Choose TXT, JSON, or SRT for export.
9. Download the generated caption file.
10. Generate speech audio if needed.

The available controls depend on the implementation in `app.py`.

## Example Input and Output

**Example input:** An image of a dog sitting on green grass.

**Illustrative captions:**

- A dog sitting on the grass.
- A brown dog resting outdoors on green grass.
- A dog sitting in a grassy field.

These are illustrative examples, not measured outputs from a recorded execution.

## Applications

- Automatic image description
- Accessibility support for image content
- Social media caption generation
- Image metadata generation
- Multilingual image descriptions
- Educational demonstrations of vision-language models
- AI-powered content creation

## Future Improvements

- Add batch image processing.
- Introduce image history and saved caption collections.
- Improve multilingual error handling.
- Add caption-quality evaluation.
- Support additional pretrained captioning models.
- Improve accessibility and interface design.
- Add automated tests for captioning, translation, and exports.
- Improve error handling and input validation.
- Optimize model loading and inference.
- Deploy the application as a hosted web service.

## Project Summary

**AI Image Caption Generator** combines computer vision, natural-language generation, machine translation, and speech synthesis in an interactive Python application.

```text
Image Upload
     |
     v
BLIP Caption Generation
     |
     v
Caption Formatting
     |
     +----> Diversity Estimation
     |
     +----> MarianMT Translation
     |
     +----> TXT / JSON / SRT Export
     |
     +----> gTTS Speech Generation
```

This project demonstrates the integration of pretrained vision-language models and NLP tools to generate, translate, export, and listen to image descriptions.
