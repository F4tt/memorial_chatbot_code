#!/bin/bash
set -e  # stop if any command fails

echo "Starting training and evaluation pipeline..."

# 1. Pull the latest code
if [ ! -d "memorial_chatbot_code/.git" ]; then
    echo "Cloning repository..."
    git clone https://github.com/F4tt/memorial_chatbot_code.git memorial_chatbot_code
else
    echo "Repository exists. Pulling latest changes..."
    cd memorial_chatbot_code || exit 1
    git pull origin main
    cd ..
fi


cd memorial_chatbot_code

# 2. Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | xargs)
else
    echo "Missing .env file. Exiting."
    exit 1
fi

# 3. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Hugging Face login (via token from environment)
echo "Logging into Hugging Face..."
huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN

# 5. Weights & Biases login (via API key from environment)
echo "Logging into Weights & Biases..."
wandb login $WANDB_API_KEY

# 6. Fine-tuning
echo "Running fine-tuning..."
python scripts/fine_tune.py --config configs/training_config.yaml

# 7. Evaluation
echo "Running evaluation..."
python scripts/evaluation.py \
    --model_path ./models/finetuned \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --val_file ./data/processed/val.jsonl \
    --output evaluation_report.json

echo "Pipeline completed successfully."
