import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        encodings['labels'] = encodings['input_ids'].clone()
        return encodings


class FineTuner:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Loading model: {model_name} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto'
        )

    def apply_lora(self, r: int = 8, lora_alpha: int = 16, target_modules: list = None):
        if target_modules is None:
            target_modules = ['q_proj', 'v_proj']

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )

        self.model = get_peft_model(self.base_model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")

        return self.model

    def train(self, dataset_path: str, epochs: int = 3, batch_size: int = 8):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='logs',
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
        )

        data_collator = DataCollator(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_test_split['train'],
            eval_dataset=train_test_split['test'],
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()

        self.model.save_pretrained(f"{self.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final")

        metrics = trainer.evaluate()
        logger.info(f"Final metrics: {metrics}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLM')
    parser.add_argument('--model', required=True, help='Base model name')
    parser.add_argument('--data', required=True, help='Dataset path (JSONL)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')

    args = parser.parse_args()

    tuner = FineTuner(args.model, args.output)
    tuner.apply_lora(r=args.lora_r)
    metrics = tuner.train(args.data, epochs=args.epochs, batch_size=args.batch_size)

    with open(f"{args.output}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
