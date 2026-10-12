AI Image Caption Generator

An AI-powered image captioning application that generates multiple descriptions for an uploaded image. Users can customize caption styles, adjust generation settings, translate captions into supported languages, export caption text, and generate speech audio.

Overview

Describing images manually can be time-consuming. This project uses a pretrained vision-language model to generate natural-language captions from images through an interactive Gradio interface.

The application also includes translation, caption formatting, a text-based diversity score, downloadable output files, and text-to-speech functionality.

Features

AI Image Captioning: Generate image descriptions using the pretrained BLIP model.
Multiple Caption Styles: Choose Short, Creative, Detailed, Social, or Poetic formatting.
Configurable Generation: Adjust caption length, creativity, and number of captions.
Caption Diversity Score: Estimate textual differences among generated captions.
Multilingual Translation: Translate captions using supported MarianMT translation models.
Multiple Export Formats: Download captions in TXT, JSON, or SRT format.
Text-to-Speech: Generate audio from caption text using gTTS.
Interactive Interface: Upload images and control the application through Gradio.
Hardware Detection: Use CUDA when available, otherwise run on CPU.

Technologies Used

Python
PyTorch
Hugging Face Transformers
BLIP (Salesforce/blip-image-captioning-base)
MarianMT (Helsinki-NLP/opus-mt-en-{target_language})
Gradio
Pillow
Google Text-to-Speech (gTTS)
Python standard libraries: os, json, uuid, datetime, and difflib

How It Works

The user uploads an image through the Gradio interface.
The BLIP processor converts the image into model inputs.
The pretrained BLIP model generates one or more captions.
The application formats captions according to the selected style.
A pairwise text-similarity calculation estimates caption diversity.
Users can translate captions, export them, or generate speech audio.

Installation

Prerequisites

Python 3.10 or 3.11 recommended

pip

Internet access to download pretrained models and use online text-to-speech
Sufficient RAM for model inference; a compatible CUDA GPU is optional
Start the application.
Upload an image.
Choose a caption style.
Adjust the creativity, caption length, and number of captions.
Click Generate Captions.
Review the generated captions and diversity score.
Select a supported output language.
Choose TXT, JSON, or SRT and click Download File.
Click Generate Audio to create speech from the generated caption text.

Example Input and Output

Example input: An image containing a dog sitting on grass.

Illustrative caption output:

1. A dog sitting on the grass.
2. A brown dog resting outdoors on green grass.
3. A dog sitting in a grassy field.

These captions are illustrative examples, not measured outputs from a recorded test.

Example JSON export:

{
  "generated_time": "2026-10-12 10:30:00",
  "captions": [
    "1. A dog sitting on the grass.",
    "2. A brown dog resting outdoors on green grass.",
    "3. A dog sitting in a grassy field."
  ]
}

The timestamp and captions above are examples. Actual values depend on the run.

Model and Algorithm

BLIP Image Captioning

The project uses Salesforce/blip-image-captioning-base, a pretrained vision-language model designed for image caption generation.

The BlipProcessor prepares image inputs, while BlipForConditionalGeneration generates text conditioned on the image.

Caption generation uses sampling settings, including temperature, top-k, and top-p, alongside beam-search parameters. Results can vary with the input image and generation settings.

MarianMT Translation

The application loads a language-specific MarianMT model when translation is requested. Loaded tokenizer-model pairs are cached during the running session to reduce repeated loading.

Translation is applied to caption text when the selected language is not English.

Caption Diversity Score

The application compares generated captions in pairs using Python's difflib.SequenceMatcher.

The displayed score represents the average textual dissimilarity between pairs, expressed as a percentage. It is a simple string-based measure, not a semantic diversity metric or an evaluation of caption correctness.

Text-to-Speech

The gTTS library converts text into speech audio and saves the result as an MP3 file.

Future Improvements

Add automated caption-quality evaluation using suitable metrics and human review.
Improve multilingual error handling and validate model availability for every language option.
Add batch image processing and image-history support.
Improve accessibility and interface design.
Add tests for caption generation, translation, exports, and error handling.
Add configuration options for model selection and deployment.
Add safer output rendering and more robust exception handling.

Project Structure

Image/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore