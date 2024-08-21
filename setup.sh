#!/bin/bash



# Update and install system dependencies
echo "Updating package list..."
sudo apt-get update

echo "Installing system dependencies..."
sudo apt-get install -y libsndfile1-dev ffmpeg enchant libenchant1c2a libenchant-dev

# Install PyTorch
echo "Installing PyTorch..."
pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Clone necessary repositories
echo "Cloning repositories..."
git clone https://github.com/AI4Bharat/IndicTrans2.git
git clone https://github.com/VarunGumma/IndicTransTokenizer.git
git clone https://github.com/gokulkarthik/Trainer.git
git clone https://github.com/gokulkarthik/TTS.git
git clone https://github.com/AI4Bharat/Indic-TTS.git

# Install Python dependencies
echo "Installing Python dependencies for IndicTrans2..."
cd IndicTrans2/huggingface_interface || exit
pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece
cd - || exit

echo "Installing Python dependencies for IndicTransTokenizer..."
cd IndicTransTokenizer || exit
pip install --editable ./
cd - || exit

# Setup Trainer and TTS
echo "Installing Python dependencies for Trainer..."
cd Trainer || exit
pip install -e .
cd - || exit

echo "Installing Python dependencies for TTS..."
cd TTS || exit
pip install -e .
cp TTS/bin/synthesize.py /content/TTS/TTS/bin || { echo "Failed to copy synthesize.py"; exit 1; }
cd - || exit

# Download TTS checkpoints
echo "Downloading TTS checkpoints..."
cd .. || exit
wget https://github.com/AI4Bharat/Indic-TTS/archive/refs/tags/v1-checkpoints-release.zip
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/te.zip
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/en.zip
unzip v1-checkpoints-release.zip
unzip te.zip
unzip en.zip

echo "Setup complete!"
