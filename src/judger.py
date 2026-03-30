"""
judger.py — Dual-model cross-validation using Llama and Qwen.

This module runs two language models (a Llama-based model and a Qwen-based model)
independently on each test example and uses majority voting / consistency scoring
to produce a final, verified prediction for ArgMining 2026.

Usage:
    python judger.py --llama_model <llama_path_or_id> \
                     --qwen_model  <qwen_path_or_id>  \
                     --test_file   data/test.jsonl    \
                     --output_file data/predictions.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-model judger (Llama + Qwen) for ArgMining 2026"
    )
    parser.add_argument(
        "--llama_model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Llama model identifier or local path.",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen2-7B-Instruct",
        help="Qwen model identifier or local path.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test.jsonl",
        help="JSONL file with test examples (each line must have a 'text' field).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/predictions.jsonl",
        help="Path to write final predictions.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def load_pipeline(model_name: str, max_new_tokens: int):
    """Load a text-generation pipeline for the given model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device_map="auto",
    )


def build_prompt(text: str) -> str:
    """Construct a zero-shot argument-mining prompt."""
    return (
        "You are an expert in argument mining. Classify the following text and "
        "briefly explain whether it contains a pro-argument, con-argument, or is "
        "not an argument.\n\n"
        f"Text: {text}\n\n"
        "Classification:"
    )


def extract_label(generated_text: str, prompt: str) -> str:
    """Extract the label from the generated continuation."""
    continuation = generated_text[len(prompt):].strip().lower()
    for label in ("pro", "con", "non-argument", "not an argument"):
        if label in continuation:
            return label
    return continuation.split("\n")[0].strip()


def adjudicate(label_a: str, label_b: str) -> Dict[str, Any]:
    """
    Compare predictions from two models.

    Returns a dict with:
      - 'final_label': agreed label, or 'conflict' when they disagree.
      - 'agreement': True if both models agree, False otherwise.
      - 'llama_label': raw Llama prediction.
      - 'qwen_label': raw Qwen prediction.
    """
    agree = label_a == label_b
    return {
        "llama_label": label_a,
        "qwen_label": label_b,
        "final_label": label_a if agree else "conflict",
        "agreement": agree,
    }


def main() -> None:
    args = parse_args()

    print("Loading Llama pipeline …")
    llama_pipe = load_pipeline(args.llama_model, args.max_new_tokens)
    print("Loading Qwen pipeline …")
    qwen_pipe = load_pipeline(args.qwen_model, args.max_new_tokens)

    # Load test examples
    examples: List[Dict[str, Any]] = []
    with open(args.test_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {lineno} of {args.test_file!r}: {exc}"
                    ) from exc
                if "text" not in obj:
                    raise KeyError(
                        f"Missing 'text' field on line {lineno} of {args.test_file!r}."
                    )
                examples.append(obj)
    print(f"Loaded {len(examples)} test examples from {args.test_file}")

    # Run dual-model inference
    results = []
    for i, example in enumerate(examples):
        text = example["text"]
        prompt = build_prompt(text)

        llama_out = llama_pipe(prompt)[0]["generated_text"]
        qwen_out = qwen_pipe(prompt)[0]["generated_text"]

        llama_label = extract_label(llama_out, prompt)
        qwen_label = extract_label(qwen_out, prompt)

        verdict = adjudicate(llama_label, qwen_label)
        result = {**example, **verdict}
        results.append(result)

        status = "✓ agree" if verdict["agreement"] else "✗ conflict"
        print(
            f"[{i+1}/{len(examples)}] {status} | "
            f"Llama: {llama_label!r} | Qwen: {qwen_label!r} | "
            f"Final: {verdict['final_label']!r}"
        )

    # Write predictions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    agreed = sum(1 for r in results if r["agreement"])
    print(
        f"\nDone. Agreement rate: {agreed}/{len(results)} "
        f"({100 * agreed / len(results):.1f}%)"
    )
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
