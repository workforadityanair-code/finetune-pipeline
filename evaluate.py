import argparse
import json
import math
from typing import Dict, List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


class Evaluator:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto'
        )
        self.model.eval()

    def compute_perplexity(self, texts: List[str]) -> float:
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                encodings = self.tokenizer(text, return_tensors='pt').to(self.device)
                outputs = self.model(**encodings, labels=encodings['input_ids'])
                loss = outputs.loss

                total_loss += loss.item() * encodings['input_ids'].shape[1]
                total_tokens += encodings['input_ids'].shape[1]

        perplexity = math.exp(total_loss / total_tokens)
        return perplexity

    def compute_accuracy(self, prompts: List[str], expected: List[str]) -> float:
        correct = 0

        with torch.no_grad():
            for prompt, expected_text in zip(prompts, expected):
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.95
                )
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if expected_text.lower() in generated.lower():
                    correct += 1

        return correct / len(prompts) if prompts else 0.0

    def evaluate(self, dataset_path: str, metrics: List[str]) -> Dict[str, float]:
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        texts = dataset['text'][:100]

        results = {}

        if 'perplexity' in metrics:
            ppl = self.compute_perplexity(texts)
            results['perplexity'] = ppl

        if 'accuracy' in metrics and 'prompt' in dataset.column_names:
            prompts = dataset['prompt'][:20]
            expected = dataset.get('expected', [''] * len(prompts))
            acc = self.compute_accuracy(prompts, expected)
            results['accuracy'] = acc

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model')
    parser.add_argument('--model', required=True, help='Model path')
    parser.add_argument('--test', required=True, help='Test dataset path')
    parser.add_argument('--metrics', nargs='+', default=['perplexity'], help='Metrics to compute')
    parser.add_argument('--output', default='eval_results.json', help='Output file')

    args = parser.parse_args()

    evaluator = Evaluator(args.model)
    results = evaluator.evaluate(args.test, args.metrics)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to {args.output}")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()
