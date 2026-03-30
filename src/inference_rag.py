"""
inference_rag.py — Inference-time Retrieval-Augmented Generation (RAG) using FAISS.

This module builds a FAISS vector index from a knowledge corpus (JSON/JSONL),
retrieves relevant passages for a given query, and feeds them as context to
a language model to generate an argument-mining prediction.

Usage:
    python inference_rag.py --model_name_or_path <model_or_adapter> \
                            --corpus_file data/corpus.jsonl \
                            --query "Is renewable energy beneficial?"
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG inference for ArgMining 2026")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/lora_adapter",
        help="Path to the fine-tuned model or LoRA adapter.",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model for encoding passages.",
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        default="data/corpus.jsonl",
        help="JSONL file where each line has a 'text' field to index.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The argument-mining query to answer.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of passages to retrieve from the FAISS index.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate.",
    )
    return parser.parse_args()


def load_corpus(corpus_file: str) -> List[str]:
    """Load passages from a JSONL file (each line must have a 'text' key)."""
    passages = []
    with open(corpus_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {lineno} of {corpus_file!r}: {exc}"
                    ) from exc
                if "text" not in obj:
                    raise KeyError(
                        f"Missing 'text' field on line {lineno} of {corpus_file!r}."
                    )
                passages.append(obj["text"])
    return passages


def build_faiss_index(embedder: SentenceTransformer, passages: List[str]) -> faiss.IndexFlatL2:
    """Encode passages and build an L2 FAISS index."""
    embeddings = embedder.encode(passages, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.IndexFlatL2,
    passages: List[str],
    top_k: int,
) -> List[str]:
    """Return the top-k passages most similar to the query."""
    query_vec = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(query_vec, top_k)
    return [passages[i] for i in indices[0]]


def build_prompt(query: str, retrieved: List[str]) -> str:
    """Combine retrieved context and query into a prompt."""
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(retrieved))
    return (
        "You are an argument-mining expert. Use the following retrieved passages to "
        "answer the query.\n\n"
        f"### Retrieved Passages:\n{context}\n\n"
        f"### Query:\n{query}\n\n"
        "### Answer:\n"
    )


def main() -> None:
    args = parse_args()

    # Load embedding model and corpus
    embedder = SentenceTransformer(args.embedder)
    passages = load_corpus(args.corpus_file)
    print(f"Loaded {len(passages)} passages from {args.corpus_file}")

    # Build FAISS index
    index = build_faiss_index(embedder, passages)

    # Retrieve relevant passages
    retrieved = retrieve(args.query, embedder, index, passages, args.top_k)
    print(f"\nTop-{args.top_k} retrieved passages:")
    for i, p in enumerate(retrieved, 1):
        print(f"  [{i}] {p[:120]}...")

    # Build prompt and generate answer
    prompt = build_prompt(args.query, retrieved)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    generator = pipeline(
        "text-generation",
        model=args.model_name_or_path,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        device_map="auto",
    )

    output = generator(prompt)[0]["generated_text"]
    answer = output[len(prompt):]
    print(f"\n### Generated Answer:\n{answer}")


if __name__ == "__main__":
    main()
