import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    def __init__(self, domain: str = 'general'):
        self.domain = domain

    def load_texts(self, input_dir: str) -> List[str]:
        texts = []
        input_path = Path(input_dir)

        for file_path in input_path.glob('*.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
                texts.append(content)

        logger.info(f"Loaded {len(texts)} text files from {input_dir}")
        return texts

    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = ' '.join(text.split())
        return text

    def chunk_texts(self, texts: List[str], chunk_size: int = 512) -> List[str]:
        chunks = []

        for text in texts:
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks of size ~{chunk_size} words")
        return chunks

    def prepare(self, input_dir: str, output_file: str, chunk_size: int = 512):
        texts = self.load_texts(input_dir)
        cleaned_texts = [self.clean_text(text) for text in texts]
        chunks = self.chunk_texts(cleaned_texts, chunk_size)

        dataset = []
        for i, chunk in enumerate(chunks):
            dataset.append({
                'text': chunk,
                'domain': self.domain,
                'chunk_id': i
            })

        with open(output_file, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

        logger.info(f"Saved {len(dataset)} training samples to {output_file}")
        return len(dataset)


def main():
    parser = argparse.ArgumentParser(description='Prepare data for fine-tuning')
    parser.add_argument('--input', required=True, help='Input directory with text files')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--domain', default='general', help='Data domain')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size in words')

    args = parser.parse_args()

    preparator = DataPreparator(domain=args.domain)
    count = preparator.prepare(args.input, args.output, args.chunk_size)

    print(f"Data preparation complete: {count} samples")


if __name__ == '__main__':
    main()
