import os
import json
import uuid
from PIL import Image
import gradio as gr
import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer
)

from gtts import gTTS
from difflib import SequenceMatcher
from datetime import datetime

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL
print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("Model Loaded!")

# TRANSLATION CACHE
translation_cache = {}

# TRANSLATE FUNCTION
def translate_text(text, tgt_lang):

    if tgt_lang == "en" or not text:
        return text

    model_name = f"Helsinki-NLP/opus-mt-en-{tgt_lang}"

    if model_name not in translation_cache:

        tokenizer = MarianTokenizer.from_pretrained(model_name)

        trans_model = MarianMTModel.from_pretrained(
            model_name
        ).to(device)

        translation_cache[model_name] = (
            tokenizer,
            trans_model
        )

    tokenizer, trans_model = translation_cache[model_name]

    batch = tokenizer(
        [text],
        return_tensors="pt",
        padding=True
    ).to(device)

    generated = trans_model.generate(
        **batch,
        max_length=80
    )

    output = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

    return output

# DIVERSITY SCORE
def diversity_score(captions):

    n = len(captions)

    if n < 2:
        return 0.0

    total = 0
    pairs = 0

    for i in range(n):
        for j in range(i + 1, n):

            sim = SequenceMatcher(
                None,
                captions[i],
                captions[j]
            ).ratio()

            total += (1 - sim)
            pairs += 1

    return round((total / pairs) * 100, 1)

# GENERATE CAPTIONS
def generate_captions(
    image,
    style,
    creativity,
    max_len,
    num_captions
):

    if image is None:
        return "Please upload image", "", "0%"

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=int(max_len),
        num_return_sequences=int(num_captions),
        do_sample=True,
        temperature=float(creativity),
        top_k=50,
        top_p=0.95,
        num_beams=5,
        early_stopping=True
    )

    captions = [
        processor.decode(
            out,
            skip_special_tokens=True
        ).strip()
        for out in outputs
    ]

    styled_captions = []

    for cap in captions:

        if style == "Short":
            text = cap.split(",")[0]

        elif style == "Creative":
            text = f"✨ {cap.capitalize()} ✨"

        elif style == "Detailed":
            text = f"This image shows {cap}."

        elif style == "Social":
            words = [
                w.lower()
                for w in cap.split()
                if len(w) > 3
            ]

            hashtags = " ".join(
                [f"#{w}" for w in words[:3]]
            )

            text = f"{cap} {hashtags}"

        elif style == "Poetic":
            text = cap + "..."

        else:
            text = cap

        styled_captions.append(text)

    html = ""

    for i, cap in enumerate(styled_captions):

        html += f"""
        <div style="
            padding:15px;
            margin:10px;
            border-radius:10px;
            background:#f0f0f0;
            font-size:18px;
        ">
        <b>Caption {i+1}</b><br>
        {cap}
        </div>
        """

    plain_text = "\n".join(
        [
            f"{i+1}. {c}"
            for i, c in enumerate(styled_captions)
        ]
    )

    score = diversity_score(captions)

    return html, plain_text, f"{score}%"

# SAVE FILE
def save_file(plain_text, lang, fmt):

    if not plain_text:
        return None

    translated_text = plain_text

    if lang != "en":

        lines = plain_text.splitlines()

        translated = []

        for line in lines:

            try:
                translated.append(
                    translate_text(line, lang)
                )

            except:
                translated.append(line)

        translated_text = "\n".join(translated)

    uid = str(uuid.uuid4())[:8]

    if fmt == "txt":

        file_path = f"captions_{uid}.txt"

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(translated_text)

        return file_path

    elif fmt == "json":

        file_path = f"captions_{uid}.json"

        data = {
            "generated_time": str(datetime.now()),
            "captions": translated_text.splitlines()
        }

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

        return file_path

    else:

        file_path = f"captions_{uid}.srt"

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as f:

            for i, line in enumerate(
                translated_text.splitlines()
            ):

                start = i * 3
                end = start + 3

                f.write(f"{i+1}\n")
                f.write(
                    f"00:00:{start:02d},000 --> "
                    f"00:00:{end:02d},000\n"
                )
                f.write(f"{line}\n\n")

        return file_path

# TEXT TO SPEECH
def generate_tts(text, lang):

    if not text:
        return None

    uid = str(uuid.uuid4())[:8]

    mp3_path = f"audio_{uid}.mp3"

    tts = gTTS(text=text, lang=lang)

    tts.save(mp3_path)

    return mp3_path

# UI
with gr.Blocks() as demo:

    gr.Markdown(
        "# AI Image Caption Generator"
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Image"
            )

            style_input = gr.Radio(
                [
                    "Short",
                    "Creative",
                    "Detailed",
                    "Social",
                    "Poetic"
                ],
                value="Detailed",
                label="Caption Style"
            )

            creativity_input = gr.Slider(
                0.1,
                1.5,
                value=0.8,
                step=0.1,
                label="Creativity"
            )

            max_len_input = gr.Slider(
                10,
                60,
                value=30,
                step=5,
                label="Caption Length"
            )

            num_input = gr.Slider(
                1,
                5,
                value=3,
                step=1,
                label="Number of Captions"
            )

            generate_btn = gr.Button(
                "Generate Captions"
            )

        with gr.Column():

            output_html = gr.HTML()

            diversity_output = gr.Textbox(
                label="Diversity Score"
            )

            hidden_text = gr.Textbox(
                visible=False
            )

            lang_input = gr.Dropdown(
                [
                    "en",
                    "hi",
                    "fr",
                    "es",
                    "de",
                    "zh"
                ],
                value="en",
                label="Language"
            )

            format_input = gr.Dropdown(
                [
                    "txt",
                    "json",
                    "srt"
                ],
                value="txt",
                label="File Format"
            )

            download_btn = gr.Button(
                "Download File"
            )

            file_output = gr.File()

            tts_btn = gr.Button(
                "Generate Audio"
            )

            audio_output = gr.Audio()

    generate_btn.click(
        fn=generate_captions,
        inputs=[
            image_input,
            style_input,
            creativity_input,
            max_len_input,
            num_input
        ],
        outputs=[
            output_html,
            hidden_text,
            diversity_output
        ]
    )

    download_btn.click(
        fn=save_file,
        inputs=[
            hidden_text,
            lang_input,
            format_input
        ],
        outputs=file_output
    )

    tts_btn.click(
        fn=generate_tts,
        inputs=[
            hidden_text,
            lang_input
        ],
        outputs=audio_output
    )

# RUN APP
demo.launch(
    share=True,
    debug=True
)