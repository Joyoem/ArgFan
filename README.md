# ArgFan

**Team ArgFan** — UZH Shared Task at The 13th Workshop on Argument Mining and Reasoning (ArgMining 2026)

---

## Methodology

ArgFan's pipeline combines three complementary techniques to achieve robust argument detection and classification:

### 1. Post-training via LoRA (Low-Rank Adaptation)
We fine-tune a pre-trained causal language model on argument-mining data using **LoRA** (Hu et al., 2021) via the Hugging Face [PEFT](https://github.com/huggingface/peft) library. LoRA injects trainable low-rank matrices into the attention layers while keeping the base model frozen, making the fine-tuning both parameter-efficient and memory-friendly.

- Script: [`src/train_lora.py`](src/train_lora.py)
- Key libraries: `transformers`, `peft`, `trl`

### 2. Inference-time Retrieval-Augmented Generation (RAG)
At inference time we augment the model's context with the most relevant passages retrieved from a pre-built knowledge corpus using a **FAISS** flat vector index. Passages are encoded with a lightweight Sentence-Transformer and the top-*k* nearest neighbours are prepended to the prompt before generation.

- Script: [`src/inference_rag.py`](src/inference_rag.py)
- Key libraries: `sentence-transformers`, `faiss-cpu`

### 3. Dual-model Verification (Llama × Qwen)
To reduce hallucination and improve reliability, we run two independent models — a **Llama**-family model and a **Qwen**-family model — on every test example. The two predictions are compared:
- **Agreement** → the shared label is accepted as the final answer.
- **Conflict** → the example is flagged for manual review or resolved by a confidence-weighted tie-break.

- Script: [`src/judger.py`](src/judger.py)

---

## Project Structure

```
ArgFan/
├── data/                  # JSON / JSONL datasets (train, dev, test, corpus)
├── src/
│   ├── train_lora.py      # LoRA fine-tuning (PEFT + TRL)
│   ├── inference_rag.py   # RAG inference (FAISS + Sentence-Transformers)
│   └── judger.py          # Dual-model cross-validation (Llama + Qwen)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fine-tune with LoRA
python src/train_lora.py \
    --model_name_or_path meta-llama/Llama-3-8B-Instruct \
    --train_file data/train.jsonl \
    --output_dir models/lora_adapter

# 3. Run RAG inference
python src/inference_rag.py \
    --model_name_or_path models/lora_adapter \
    --corpus_file data/corpus.jsonl \
    --query "Is renewable energy beneficial?"

# 4. Dual-model verification
python src/judger.py \
    --llama_model models/lora_adapter \
    --qwen_model  Qwen/Qwen2-7B-Instruct \
    --test_file   data/test.jsonl \
    --output_file data/predictions.jsonl
```

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list. Core packages:

| Package | Purpose |
|---|---|
| `transformers` | Base model loading & generation |
| `peft` | LoRA / parameter-efficient fine-tuning |
| `trl` | SFT training loop |
| `sentence-transformers` | Passage encoding for RAG |
| `faiss-cpu` | Approximate nearest-neighbour search |

---

## Citation

If you use this code, please cite the ArgMining 2026 shared task proceedings (forthcoming).

