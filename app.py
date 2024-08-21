from flask import Flask, request, jsonify, send_file
import os
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

app = Flask(__name__)

# Initialize the model and tokenizer
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model

en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)
ip = IndicProcessor(inference=True)

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        del inputs
        torch.cuda.empty_cache()
    return translations

@app.route('/process', methods=['POST'])
def process_text():
    input_file = request.files['file']
    input_path = "input_eng.txt"
    output_path_tel = "input_tel.txt"
    output_path_eng_audio = "/Users/prasanth/PycharmProjects/APIindicTTS/output/outputen.wav"
    output_path_tel_audio = "/Users/prasanth/PycharmProjects/APIindicTTS/output/outputte.wav"

    # Save the input file
    input_file.save(input_path)

    # Read input sentences
    with open(input_path, 'r', encoding='utf-8') as file:
        en_sents = [line.strip() for line in file.readlines()]

    # Translate to Telugu
    src_lang, tgt_lang = "eng_Latn", "tel_Telu"
    tel_translations = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

    # Write translations to output file
    with open(output_path_tel, 'w', encoding='utf-8') as file:
        for translation in tel_translations:
            file.write(f"{translation}\n")

    # Generate Telugu audio
    os.system(f"python3 -m TTS.bin.synthesize --text \"$(cat {output_path_tel})\" "
              f"--model_path /Users/prasanth/PycharmProjects/APIindicTTS/te/fastpitch/best_model.pth "
              f"--config_path /Users/prasanth/PycharmProjects/APIindicTTS/te/fastpitch/config.json "
              f"--vocoder_path /Users/prasanth/PycharmProjects/APIindicTTS/te/hifigan/best_model.pth "
              f"--vocoder_config_path /Users/prasanth/PycharmProjects/APIindicTTS/te/hifigan/config.json "
              f"--out_path {output_path_tel_audio} "
              f"--speaker_idx 'female'")

    # Generate English audio
    os.system(f"python3 -m TTS.bin.synthesize --text \"$(cat {input_path})\" "
              f"--model_path /Users/prasanth/PycharmProjects/APIindicTTS/en/fastpitch/best_model.pth "
              f"--config_path /Users/prasanth/PycharmProjects/APIindicTTS/en/fastpitch/config.json "
              f"--vocoder_path /Users/prasanth/PycharmProjects/APIindicTTS/en/hifigan/best_model.pth "
              f"--vocoder_config_path /Users/prasanth/PycharmProjects/APIindicTTS/en/hifigan/config.json "
              f"--out_path {output_path_eng_audio} "
              f"--speaker_idx 'male'")

    return jsonify({
        "status": "success",
        "message": "Processing completed",
        "english_audio": output_path_eng_audio,
        "telugu_audio": output_path_tel_audio
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

