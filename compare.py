import argparse
import json
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelComparator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto'
        )
        return tokenizer, model

    def generate_samples(self, model, tokenizer, prompts: List[str], max_tokens: int = 50) -> List[str]:
        outputs = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=1
                )
                text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                outputs.append(text)

        return outputs

    def compare(self, baseline_path: str, finetuned_path: str, prompts: List[str]) -> Dict:
        print(f"Loading baseline model: {baseline_path}")
        baseline_tokenizer, baseline_model = self.load_model(baseline_path)
        baseline_model.eval()

        print(f"Loading fine-tuned model: {finetuned_path}")
        ft_tokenizer, ft_model = self.load_model(finetuned_path)
        ft_model.eval()

        print("Generating samples from baseline...")
        baseline_outputs = self.generate_samples(baseline_model, baseline_tokenizer, prompts)

        print("Generating samples from fine-tuned model...")
        ft_outputs = self.generate_samples(ft_model, ft_tokenizer, prompts)

        comparison = {
            'prompts': prompts,
            'baseline_outputs': baseline_outputs,
            'finetuned_outputs': ft_outputs,
            'comparison_samples': []
        }

        for i, (prompt, baseline, finetuned) in enumerate(zip(prompts, baseline_outputs, ft_outputs)):
            comparison['comparison_samples'].append({
                'prompt': prompt,
                'baseline': baseline,
                'finetuned': finetuned,
                'baseline_length': len(baseline.split()),
                'finetuned_length': len(finetuned.split())
            })

        return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and fine-tuned models')
    parser.add_argument('--baseline', required=True, help='Baseline model path')
    parser.add_argument('--finetuned', required=True, help='Fine-tuned model path')
    parser.add_argument('--prompts', nargs='+', default=['Hello,', 'The legal document states'], help='Test prompts')
    parser.add_argument('--output', default='comparison.json', help='Output file')

    args = parser.parse_args()

    comparator = ModelComparator()
    results = comparator.compare(args.baseline, args.finetuned, args.prompts)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison saved to {args.output}")
    print(f"Compared {len(args.prompts)} prompts")


if __name__ == '__main__':
    main()
