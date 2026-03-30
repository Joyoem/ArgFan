"""
train_lora.py — Post-training with LoRA using Hugging Face PEFT.

This module fine-tunes a base language model on argument-mining data
(ArgMining 2026 / UZH Shared Task) using Low-Rank Adaptation (LoRA)
via the PEFT library and the TRL SFTTrainer.

Usage:
    python train_lora.py --model_name_or_path <base_model> \
                         --train_file data/train.jsonl \
                         --output_dir models/lora_adapter
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for ArgMining 2026")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="HuggingFace model identifier or local path.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train.jsonl",
        help="Path to the JSONL training file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/lora_adapter",
        help="Directory to save the LoRA adapter.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Return a LoraConfig targeting the attention projection layers."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )


def main() -> None:
    args = parse_args()

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
    )

    # Wrap model with LoRA adapter
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset from JSONL
    if not Path(args.train_file).exists():
        raise FileNotFoundError(
            f"Training file not found: {args.train_file!r}. "
            "Expected a JSONL file where each line is a JSON object with a 'text' field."
        )
    try:
        dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load training data from {args.train_file!r}. "
            "Ensure the file is valid JSONL with a 'text' field on each line."
        ) from exc

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
