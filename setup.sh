#!/bin/bash

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y libsndfile1-dev ffmpeg enchant libenchant1c2a libenchant-dev

# Install PyTorch
pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Clone necessary repositories
git clone https://github.com/AI4Bharat/IndicTrans2.git
git clone https://github.com/VarunGumma/IndicTransTokenizer.git
git clone https://github.com/gokulkarthik/Trainer.git
git clone https://github.com/gokulkarthik/TTS.git
git clone https://github.com/AI4Bharat/Indic-TTS.git

# Install Python dependencies
cd IndicTrans2/huggingface_interface

pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece
cd /Users/prasanth/PycharmProjects/APIindicTTS/IndicTransTokenizer

pip install --editable ./
cd /Users/prasanth/PycharmProjects/APIindicTTS

# Setup Trainer and TTS
cd /Users/prasanth/PycharmProjects/APIindicTTS/Trainer

pip install -e .
cd /Users/prasanth/PycharmProjects/APIindicTTS/TTS

pip install -e .
cp TTS/bin/synthesize.py /content/TTS/TTS/bin


# Download TTS checkpoints
cd /Users/prasanth/PycharmProjects/APIindicTTS
wget https://github.com/AI4Bharat/Indic-TTS/archive/refs/tags/v1-checkpoints-release.zip
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/te.zip
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/en.zip
unzip v1-checkpoints-release.zip
unzip te.zip
unzip en.zip

## Install additional Python packages
#pip install pyenchant
#pip install -r TTS/requirements.txt
