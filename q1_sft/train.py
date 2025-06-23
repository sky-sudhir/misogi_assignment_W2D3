import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "output_lora"
EPOCHS = 3
LR = 5e-5
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

# Flatten to (prompt, response) pairs
examples = []
for item in data:
    messages = item["messages"]
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
    if user_msg and assistant_msg:
        examples.append({"prompt": user_msg, "response": assistant_msg})

dataset = Dataset.from_list(examples)

# 2. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 3. LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)

# 4. Preprocess function
SYSTEM_PROMPT = "You are a helpful AI assistant."
def preprocess(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize the entire prompt
    model_inputs = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
    
    # Create the labels (same as input_ids, but with -100 for non-response tokens)
    labels = model_inputs["input_ids"].copy()
    
    # Find the assistant's response start
    response_tokens = tokenizer.encode(example["response"], add_special_tokens=False)
    response_start = None
    for i in range(len(labels) - len(response_tokens) + 1):
        if labels[i:i + len(response_tokens)] == response_tokens:
            response_start = i
            break
    
    # Set all non-response tokens to -100
    if response_start is not None:
        labels[:response_start] = [-100] * response_start
    
    # Set padding tokens to -100
    pad_token_id = tokenizer.pad_token_id
    labels = [label if label != pad_token_id else -100 for label in labels]
    
    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(preprocess, remove_columns=["prompt", "response"])

# 5. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",
    report_to=[],
    fp16=torch.cuda.is_available(),
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 7. Save 5 prompts for before/after comparison
EVAL_PROMPTS = [ex["prompt"] for ex in examples[:5]]

# 8. Function to generate responses
def generate_responses(model, tokenizer, prompts):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    results = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = pipe(chat_prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        results.append(output[0]["generated_text"])
    return results

# 9. Generate before fine-tuning
print("Generating responses BEFORE fine-tuning...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
before_responses = generate_responses(base_model, base_tokenizer, EVAL_PROMPTS)

# 10. Train
print("Starting LoRA fine-tuning...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)

# 11. Generate after fine-tuning
print("Generating responses AFTER fine-tuning...")
model.eval()
after_responses = generate_responses(model, tokenizer, EVAL_PROMPTS)

# 12. Write to before_after.md
with open("before_after.md", "w", encoding="utf-8") as f:
    f.write("# Before and After Fine-Tuning: Response Comparison\n\n")
    for i, prompt in enumerate(EVAL_PROMPTS):
        f.write(f"### Prompt {i+1}: {prompt}\n")
        f.write(f"- **Base Model:** {before_responses[i]}\n")
        f.write(f"- **Fine-Tuned Model:** {after_responses[i]}\n\n---\n\n")

print("Comparison written to before_after.md")