"""
Main script for fine-tuning an LLM with LoRA
"""
import os
import yaml
import torch
import wandb
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse


class LLMFineTuner:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        model_name = self.config['model']['name']

        # Detect if GPU is available
        use_cuda = torch.cuda.is_available()
        bnb_config = None

        # Enable quantization only if GPU is available and load_in_4bit is True
        if use_cuda and self.config['model']['load_in_4bit']:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                print("bitsandbytes not found, running without quantization.")
                bnb_config = None

        # Choose appropriate dtype
        dtype = torch.float16 if use_cuda else torch.float32

        # Load model
        print(f"Loading model {model_name}...")
        if use_cuda and bnb_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=self.config['model']['device_map'],
                torch_dtype=dtype,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if use_cuda else None,
                torch_dtype=dtype,
                trust_remote_code=True
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare model for training if quantization is enabled
        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        print("Model and tokenizer loaded successfully!")

    def setup_lora(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("LoRA configuration set up successfully!")

    def apply_chat_template(self, messages):
        """
        Apply chat template for different models.
        By default: OpenAI-style <|im_start|> and <|im_end|>.
        """
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text.strip()

    def load_datasets(self):
        """Load and preprocess datasets from HuggingFace Hub"""

        def preprocess_function(examples):
            texts = []
            for messages in examples['messages']:
                if len(messages) >= 2:
                    text = self.apply_chat_template(messages)
                    texts.append(text)
                else:
                    texts.append("")

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config['data']['max_length'],
                return_tensors=None
            )

            tokenized['labels'] = tokenized['input_ids'].copy()

            return tokenized

        print("📥 Loading datasets from HuggingFace Hub...")

        dataset_name = self.config['data']['hf_dataset']  # ví dụ: "F4tt/memorial_chatbot"
        raw_datasets = load_dataset(dataset_name)

        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )

        val_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names
        )

        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Validation dataset: {len(val_dataset)} samples")

        return train_dataset, val_dataset

    def setup_trainer(self, train_dataset, val_dataset):
        """Setup HuggingFace Trainer"""
        use_cuda = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            learning_rate=self.config['training']['learning_rate'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            save_strategy=self.config['training']['save_strategy'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            remove_unused_columns=self.config['training']['remove_unused_columns'],
            report_to=self.config['training']['report_to'],
            run_name=f"personal-chatbot-{wandb.util.generate_id()}",
            bf16=use_cuda  # enable bf16 only if GPU is available
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        print("Trainer set up successfully!")

    def train(self):
        """Start training process"""
        print("Starting training...")
        self.trainer.train()

        print("Saving LoRA adapter...")
        self.model.save_pretrained(self.config['training']['output_dir'])
        self.tokenizer.save_pretrained(self.config['training']['output_dir'])

        artifact = wandb.Artifact("chatbot-checkpoints", type="model")
        artifact.add_dir(self.config['training']['output_dir'])
        wandb.log_artifact(artifact)

        print("Training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Config file path")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    if not args.no_wandb:
        wandb.init(project="personal-chatbot-finetune")

    fine_tuner = LLMFineTuner(args.config)

    try:
        fine_tuner.setup_model_and_tokenizer()
        fine_tuner.setup_lora()

        train_dataset, val_dataset = fine_tuner.load_datasets()

        fine_tuner.setup_trainer(train_dataset, val_dataset)
        fine_tuner.train()

    except Exception as e:
        print(f"Error during training: {e}")
        raise e

    finally:
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
