# TinyLlama LoRA Fine-Tuning Project

This project demonstrates how to fine-tune the [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model using LoRA (Low-Rank Adaptation) on a custom conversational dataset (`dataset.jsonl`).

## Requirements

- Python 3.8+
- torch
- transformers
- peft
- datasets
- trl
- bitsandbytes (optional, for QLoRA/4-bit training)

Install dependencies:

```bash
pip install torch transformers peft datasets trl bitsandbytes
```

## Dataset

The dataset should be in `dataset.jsonl` format, where each line is a JSON object with a `messages` field (list of chat messages with `role` and `content`).

## Training (LoRA Fine-Tuning)

1. **Prepare the training script** (see `train.py` or `lora_finetune.py`).
2. **Recommended settings:**

   - Epochs: 3–5
   - Learning Rate: ~5e-5
   - Batch size: 2–4 (adjust for your hardware)
   - LoRA rank (`r`): 8–16 (default: 8)
   - LoRA alpha: 16–32 (default: 16)
   - LoRA dropout: 0.1

3. **Run training:**
   ```bash
   python lora_finetune.py --dataset dataset.jsonl --epochs 3 --lr 5e-5
   ```
   The script will save the LoRA adapter weights in `output/`.

## Evaluation

1. **Select 5 representative prompts** from your dataset or use the provided examples.
2. **Run the model before and after fine-tuning** on these prompts:
   - Use the base model and the fine-tuned (LoRA) model to generate responses.
   - Save the results in `before_after.md` (see below).

## Side-by-Side Comparison

- Use `before_after.md` to present the prompt, base model response, and fine-tuned model response for each of the 5 prompts.
- Example format:

```
### Prompt 1: <prompt text>
- **Base Model:** <response>
- **Fine-Tuned Model:** <response>
```

## References

- [LoRA in Hugging Face PEFT](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

---

For more details, see the training and evaluation scripts in this repository.
